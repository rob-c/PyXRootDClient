"""The copy engine: streaming, verification, recursion, and third-party copy."""

from __future__ import annotations

import io
import os

import pytest

import xrd
from xrd.copy import engine
from xrd.copy import tpc as engine_tpc
from xrd.errors import ChecksumMismatchError, UnsupportedError, kXR_Unsupported
from xrd.proto import constants as c
from xrd.testing import FakeDAVServer, FakeServer, error, frame

PAYLOAD = bytes(range(256)) * 40  # 10 240 bytes, compresses nothing


@pytest.fixture
def src(tmp_path):
    """A local file with a known body."""
    path = tmp_path / "src.bin"
    path.write_bytes(PAYLOAD)
    return path


# ---------------------------------------------------------------------------
# The four directions
# ---------------------------------------------------------------------------


def test_download_writes_the_local_file(server, tmp_path):
    target = tmp_path / "out.root"
    result = xrd.copy(server.url / "data/a.root", target)
    assert target.read_bytes() == b"hello world"
    assert result.size == 11
    assert result.verified


def test_upload_writes_the_remote_file(src, server):
    result = xrd.copy(src, server.url / "up.bin")
    assert server.contents("/up.bin") == PAYLOAD
    assert result.size == len(PAYLOAD)


def test_remote_to_remote_streams_through_this_process(server):
    with FakeServer() as other:
        xrd.copy(server.url / "data/a.root", other.url / "copied.root")
        assert other.contents("/copied.root") == b"hello world"


def test_local_to_local_needs_no_server(src, tmp_path):
    result = xrd.copy(src, tmp_path / "clone.bin")
    assert (tmp_path / "clone.bin").read_bytes() == PAYLOAD
    assert result.checksum is None  # nothing to compare against


def test_an_open_stream_may_be_the_target(server):
    sink = io.BytesIO()
    result = xrd.copy(server.url / "data/a.root", sink)
    assert sink.getvalue() == b"hello world"
    assert "BytesIO" in result.target


def test_an_open_stream_may_be_the_source(server):
    xrd.copy(io.BytesIO(b"from memory"), server.url / "mem.bin")
    assert server.contents("/mem.bin") == b"from memory"


def test_a_path_object_works_on_either_side(server, tmp_path):
    remote = xrd.XRootDPath(server.url / "data/a.root")
    xrd.copy(remote, tmp_path / "p.root")
    assert (tmp_path / "p.root").read_bytes() == b"hello world"


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_progress_reports_every_chunk_and_ends_at_the_size(src, server):
    seen: list[tuple[int, int | None]] = []
    xrd.copy(src, server.url / "p.bin", chunk_size=1024, progress=lambda d, t: seen.append((d, t)))
    assert [d for d, _ in seen] == list(range(1024, len(PAYLOAD) + 1, 1024))
    assert {t for _, t in seen} == {len(PAYLOAD)}


def test_a_download_knows_the_total_up_front(server, tmp_path):
    totals = []
    xrd.copy(server.url / "data/a.root", tmp_path / "o", progress=lambda d, t: totals.append(t))
    assert totals == [11]


def test_a_small_chunk_size_still_copies_every_byte(src, server):
    xrd.copy(src, server.url / "chunky.bin", chunk_size=7)
    assert server.contents("/chunky.bin") == PAYLOAD


def test_overwrite_is_the_default_and_can_be_declined(src, tmp_path, server):
    target = tmp_path / "existing"
    target.write_bytes(b"old")
    xrd.copy(src, target)
    assert target.read_bytes() == PAYLOAD
    with pytest.raises(FileExistsError):
        xrd.copy(src, target, overwrite=False)
    with pytest.raises(FileExistsError):
        xrd.copy(src, server.url / "data/a.root", overwrite=False)


def test_a_missing_source_raises_the_usual_error(tmp_path, server):
    with pytest.raises(FileNotFoundError):
        xrd.copy(server.url / "nope.root", tmp_path / "x")
    with pytest.raises(FileNotFoundError):
        xrd.copy(tmp_path / "nope", server.url / "x")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_the_checksum_is_taken_against_the_target_when_it_is_remote(src, server):
    result = xrd.copy(src, server.url / "v.bin")
    assert result.checksum.algorithm == "adler32"
    assert result.checksum.value == xrd.crypto.checksum_bytes("adler32", PAYLOAD)


def test_a_corrupted_transfer_is_caught(src, server, monkeypatch):
    """The server's idea of the file must match the bytes we streamed."""
    monkeypatch.setattr("xrd.testing.server._checksum", lambda algorithm, data: "deadbeef")
    with pytest.raises(ChecksumMismatchError) as caught:
        xrd.copy(src, server.url / "bad.bin")
    assert caught.value.expected == "deadbeef"


def test_verification_can_be_declined(src, server, monkeypatch):
    monkeypatch.setattr("xrd.testing.server._checksum", lambda algorithm, data: "deadbeef")
    assert xrd.copy(src, server.url / "unchecked.bin", verify=False).checksum is None


def test_a_server_that_cannot_checksum_degrades_quietly(src, server, monkeypatch):
    def refuse(*args, **kwargs):
        raise UnsupportedError(kXR_Unsupported, "this server cannot checksum")

    monkeypatch.setattr(engine, "_server_checksum", refuse)
    assert xrd.copy(src, server.url / "q.bin").checksum is None
    with pytest.raises(UnsupportedError):
        xrd.copy(src, server.url / "q.bin", verify=True)


def test_the_algorithm_is_configurable(src, server):
    result = xrd.copy(src, server.url / "md5.bin", algorithm="md5")
    assert result.checksum.algorithm == "md5"


def test_a_download_verifies_against_the_source(server, tmp_path):
    result = xrd.copy(server.url / "data/a.root", tmp_path / "d.root")
    assert result.checksum.value == xrd.crypto.checksum_bytes("adler32", b"hello world")


# ---------------------------------------------------------------------------
# CopyResult
# ---------------------------------------------------------------------------


def test_the_result_describes_the_transfer(server, tmp_path):
    result = xrd.copy(server.url / "data/a.root", tmp_path / "r.root")
    assert result.size == 11
    assert result.rate > 0
    assert "->" in str(result)
    assert result.verified


def test_an_instantaneous_copy_has_an_infinite_rate():
    assert xrd.CopyResult("a", "b", 10, 0.0).rate == float("inf")


# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------


def test_copy_tree_uploads_a_whole_directory(tmp_path, server):
    (tmp_path / "sub").mkdir()
    (tmp_path / "top.txt").write_bytes(b"top")
    (tmp_path / "sub" / "deep.txt").write_bytes(b"deep")
    results = xrd.copy_tree(tmp_path, server.url / "tree")
    assert len(results) == 2
    assert server.contents("/tree/top.txt") == b"top"
    assert server.contents("/tree/sub/deep.txt") == b"deep"


def test_copy_tree_downloads_and_creates_local_directories(server, tmp_path):
    server.add_file("/data/empty/deep/x.bin", b"nested")
    xrd.copy_tree(server.url / "data", tmp_path / "down")
    assert (tmp_path / "down" / "a.root").read_bytes() == b"hello world"
    assert (tmp_path / "down" / "empty" / "deep" / "x.bin").read_bytes() == b"nested"


def test_copy_tree_passes_options_through(tmp_path, server):
    (tmp_path / "one.bin").write_bytes(PAYLOAD)
    results = xrd.copy_tree(tmp_path, server.url / "opts", chunk_size=8, verify=False)
    assert results[0].checksum is None
    assert server.contents("/opts/one.bin") == PAYLOAD


# ---------------------------------------------------------------------------
# Selection, synchronisation and moves
# ---------------------------------------------------------------------------


@pytest.fixture
def tree(tmp_path):
    """A small local tree: two ``.root`` files and one log, one level deep."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "one.root").write_bytes(b"one")
    (tmp_path / "notes.log").write_bytes(b"log")
    (tmp_path / "sub" / "two.root").write_bytes(b"two")
    return tmp_path


def copied(server, prefix):
    """The names the server holds under ``prefix``, relative to it."""
    return sorted(p[len(prefix) :].lstrip("/") for p in server.files if p.startswith(prefix))


def test_exclude_skips_what_matches_it(tree, server):
    xrd.copy_tree(tree, server.url / "sel", exclude=["*.log"])
    assert copied(server, "/sel") == ["one.root", "sub/two.root"]


def test_include_is_a_whitelist_and_exclude_beats_it(tree, server):
    xrd.copy_tree(tree, server.url / "inc", include=["*.root", "*.log"], exclude=["sub/*"])
    assert copied(server, "/inc") == ["notes.log", "one.root"]


def test_what_no_include_pattern_names_stays_behind(tree, server):
    """An include list nothing matches is an empty transfer, not a full one."""
    assert xrd.copy_tree(tree, server.url / "none", include=["*.parquet"]) == []
    assert copied(server, "/none") == []


def test_a_dry_run_reports_the_transfers_it_did_not_make(tree, server):
    results = xrd.copy_tree(tree, server.url / "dry", dry_run=True)
    assert sorted(r.size for r in results) == [3, 3, 3]
    assert copied(server, "/dry") == []
    assert "MB/s" not in str(results[0])


def test_sync_by_size_skips_what_is_already_there(tree, server):
    first = xrd.copy_tree(tree, server.url / "syn", sync="size")
    again = xrd.copy_tree(tree, server.url / "syn", sync="size")
    assert len(first) == 3 and again == []
    (tree / "one.root").write_bytes(b"longer than before")
    assert [r.size for r in xrd.copy_tree(tree, server.url / "syn", sync="size")] == [18]


def test_sync_by_checksum_notices_a_change_of_the_same_length(tree, server):
    xrd.copy_tree(tree, server.url / "cks", sync="checksum")
    assert xrd.copy_tree(tree, server.url / "cks", sync="checksum") == []
    (tree / "one.root").write_bytes(b"ONE")
    assert len(xrd.copy_tree(tree, server.url / "cks", sync="checksum")) == 1


def test_sync_by_mtime_copies_again_when_the_source_is_newer(tree, tmp_path_factory):
    """Local both sides: a copy carries no mtime, so the test sets them."""
    mirror = tmp_path_factory.mktemp("mirror")
    assert len(xrd.copy_tree(tree, mirror, sync="mtime")) == 3
    for path in mirror.rglob("*.*"):
        os.utime(path, (2_000_000_000, 2_000_000_000))
    assert xrd.copy_tree(tree, mirror, sync="mtime") == []
    os.utime(tree / "one.root", (2_100_000_000, 2_100_000_000))
    assert len(xrd.copy_tree(tree, mirror, sync="mtime")) == 1


def test_sync_copies_a_file_the_target_does_not_have_at_all(tree, server):
    assert len(xrd.copy_tree(tree, server.url / "fresh", sync="size")) == 3


def test_delete_removes_what_the_source_no_longer_has(tree, server):
    xrd.copy_tree(tree, server.url / "del")
    (tree / "notes.log").unlink()
    xrd.copy_tree(tree, server.url / "del", delete=True)
    assert copied(server, "/del") == ["one.root", "sub/two.root"]


def test_a_dry_run_deletes_nothing(tree, server):
    xrd.copy_tree(tree, server.url / "dd")
    (tree / "notes.log").unlink()
    xrd.copy_tree(tree, server.url / "dd", delete=True, dry_run=True)
    assert "notes.log" in copied(server, "/dd")


def test_delete_takes_an_exclusion_at_its_word(tree, server):
    """After the call the target holds the selection, and nothing else."""
    xrd.copy_tree(tree, server.url / "sel2")
    xrd.copy_tree(tree, server.url / "sel2", exclude=["*.log"], delete=True)
    assert copied(server, "/sel2") == ["one.root", "sub/two.root"]


def test_remove_source_makes_the_copy_a_move(src, server):
    result = xrd.copy(src, server.url / "moved.bin", remove_source=True)
    assert server.contents("/moved.bin") == PAYLOAD
    assert not src.exists()
    assert result.size == len(PAYLOAD)


def test_a_move_that_fails_verification_keeps_the_original(src, server):
    def wrong(conn, sid, params, body):
        yield frame(sid, c.kXR_ok, b"adler32 00000000\x00")

    server.handlers[c.kXR_query] = wrong
    with pytest.raises(ChecksumMismatchError):
        xrd.copy(src, server.url / "kept.bin", verify=True, remove_source=True)
    assert src.exists()


def test_a_remote_source_is_removed_too(server, tmp_path):
    xrd.copy(server.url / "data/a.root", tmp_path / "here.root", remove_source=True)
    assert (tmp_path / "here.root").read_bytes() == b"hello world"
    assert "/data/a.root" not in server.files


def test_a_dry_run_of_a_stream_has_nothing_to_measure(server):
    result = xrd.copy(io.BytesIO(b"data"), server.url / "never.bin", dry_run=True)
    assert result.size == 0
    assert "never.bin" in result.target


def test_a_probe_of_something_that_is_not_there_is_not_a_size(server, tmp_path):
    """The engine has to tell "absent" from "empty" to decide about syncing."""
    cfg = xrd.Config()
    assert engine._probe(xrd.parse(str(tmp_path / "absent")), cfg) is None
    assert engine._probe(server.url / "data/gone.root", cfg) is None
    assert engine._probe(server.url / "data/a.root", cfg)[0] == 11


# ---------------------------------------------------------------------------
# Third-party copy
# ---------------------------------------------------------------------------


def test_third_party_emits_the_stock_rendezvous(server):
    """The dialect is the contract: stock XRootD accepts only this order."""
    with FakeServer() as dst:
        result = xrd.third_party(server.url / "data/a.root", dst.url / "pulled.root")

    assert result.size == 11
    # Destination first (it is the puller), then the source registers the key.
    dst_open, = [p for p in dst.opened if "tpc.key" in p]
    src_open, = [p for p in server.opened if "tpc.key" in p]
    key = dst_open.split("tpc.key=")[1].split("&")[0]
    assert len(key) == 32
    assert dst_open == (
        f"/pulled.root?tpc.key={key}&tpc.src={server.address[0]}:{server.address[1]}"
        f"&tpc.lfn=/data/a.root&tpc.dlg={server.address[0]}:{server.address[1]}"
        "&tpc.spr=root&tpc.tpr=root&tpc.dlgon=0&oss.asize=11&tpc.stage=copy"
    )
    assert src_open == f"/data/a.root?tpc.key={key}&tpc.dst={dst.address[0]}&tpc.stage=copy"
    # Arm, register, trigger: two syncs on the destination, one open between.
    assert dst.seen.count(c.kXR_sync) == 2
    assert dst.seen.index(c.kXR_open) < dst.seen.index(c.kXR_sync)


def test_third_party_can_carry_a_token_mode(server):
    with FakeServer() as dst:
        xrd.third_party(server.url / "data/a.root", dst.url / "t.root", token_mode="delegate")
    assert all(p.endswith("&tpc.token_mode=delegate") for p in dst.opened if "tpc.key" in p)


def test_a_scheme_the_library_does_not_speak_is_reported_as_unsupported(server, tmp_path):
    """Both ends check the scheme, and both say so rather than raising TypeError."""
    with pytest.raises(UnsupportedError, match="cannot read from ftp"):
        xrd.copy("ftp://example.org/f.root", os.fspath(tmp_path / "out"))
    with pytest.raises(UnsupportedError, match="cannot write to ftp"):
        xrd.copy(server.url / "data/a.root", "ftp://example.org/f.root")


def test_third_party_refuses_a_non_root_endpoint(server, tmp_path):
    with pytest.raises(ValueError, match="root://"):
        xrd.third_party(server.url / "data/a.root", os.fspath(tmp_path / "x"))


def test_third_party_dispatches_an_http_pair_to_the_copy_dialect():
    """Two ``davs://`` endpoints get WLCG ``COPY``, not the XRootD rendezvous."""
    with FakeDAVServer(files={"/d/a.root": b"hello"}) as src, FakeDAVServer(dirs=["/d"]) as dst:
        result = xrd.third_party(src.url / "d/a.root", dst.url / "d/b.root")
    assert dst.contents("/d/b.root") == b"hello"
    assert result.size == 5


def test_third_party_refuses_to_mix_the_two_dialects(server):
    """Each dialect is one server asking another in a language it speaks."""
    with FakeDAVServer(dirs=["/d"]) as dst:
        with pytest.raises(ValueError, match="root:// and http://"):
            xrd.third_party(server.url / "data/a.root", dst.url / "d/b.root")


def test_the_root_only_options_are_refused_rather_than_ignored():
    """``tpc.token_mode`` has no HTTP spelling, so silently dropping it would lie."""
    with FakeDAVServer(files={"/d/a.root": b"x"}) as src, FakeDAVServer(dirs=["/d"]) as dst:
        with pytest.raises(ValueError, match="Credential header"):
            xrd.third_party(src.url / "d/a.root", dst.url / "d/b.root", token_mode="delegate")


def test_third_party_can_demand_an_exclusive_destination(server):
    with FakeServer(files={"/taken.root": b"x"}) as dst:
        with pytest.raises(FileExistsError):
            xrd.third_party(server.url / "data/a.root", dst.url / "taken.root", overwrite=False)


def test_a_third_party_copy_takes_the_timeout_it_was_given(server, monkeypatch):
    """``timeout`` is the request timeout for both sides of the rendezvous."""
    seen: list[float] = []
    original = engine_tpc.Router

    class Recording(original):  # type: ignore[misc, valid-type]
        def __init__(self, url, config=None, **kw):
            seen.append(config.request_timeout)
            super().__init__(url, config, **kw)

    monkeypatch.setattr(engine_tpc, "Router", Recording)
    with FakeServer(dirs=["/"]) as dst:
        xrd.third_party(server.url / "data/a.root", dst.url / "b.root", timeout=12.5)
    assert seen and set(seen) == {12.5}


def test_a_third_party_copy_survives_a_source_that_will_not_close(server):
    """The source close is a courtesy; failing it must not lose the copy."""
    with FakeServer(dirs=["/"]) as dst:
        server.handlers[c.kXR_close] = lambda conn, sid, params, body: iter(
            [error(sid, 3012, "close refused")]
        )
        result = xrd.third_party(server.url / "data/a.root", dst.url / "b.root")
        assert c.kXR_close in server.seen  # it was tried, and it failed
    assert result.size == 11


def test_a_source_that_only_knows_how_to_read_is_still_copied(server):
    """``readinto`` is an optimisation; ``read`` alone is the fallback path."""

    class ReadOnly:
        """Not an ``io`` object at all - just something with ``read``."""

        def __init__(self, data):
            self._data = io.BytesIO(data)

        def read(self, size=-1):
            return self._data.read(size)

    xrd.copy(ReadOnly(b"minimal" * 100), server.url / "minimal.bin")
    assert server.contents("/minimal.bin") == b"minimal" * 100


def test_a_source_of_unknown_size_leaves_the_allocation_hint_out():
    """``oss.asize`` is a promise; a size nobody knows is not one to make."""
    opaque = engine_tpc._dst_opaque("k" * 32, xrd.parse("root://s//f.root"), "s:1094", -1, "")
    assert "oss.asize" not in opaque
    assert opaque.endswith("&tpc.stage=copy")


def test_persist_on_close_can_be_declined(server):
    """Without ``posc`` the open carries no ``kXR_posc``; the rendezvous is the same."""
    with FakeServer() as dst:
        result = xrd.third_party(server.url / "data/a.root", dst.url / "plain.root", posc=False)
    assert result.size == 11
    assert "/plain.root" in dst.files
