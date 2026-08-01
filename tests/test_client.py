"""``FileSystem`` and ``File`` against a live loopback server.

Everything here goes over a real socket to :class:`xrd.testing.FakeServer`, so
a passing test means the request was framed correctly, the server understood
it, and the response parsed back into the right Python object.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from xrd.client.file import READV_MAX_BYTES, File, _batches, _write_batches
from xrd.client.filesystem import FileSystem
from xrd.errors import (
    ChecksumMismatchError,
    InvalidArgumentError,
    ProtocolError,
    UnsupportedError,
)
from xrd.flags import Access, DirListFlags, OpenFlags, StatInfoFlags
from xrd.proto import constants as c
from xrd.testing import FakeServer, error, frame
from xrd.types import CloneRange, ReadRange, WriteChunk


@pytest.fixture
def source(fs, server):
    """A readable handle on known bytes, sharing the session ``opened`` uses."""
    server.files["/data/src.bin"] = bytearray(b"0123456789")
    handle = File(fs.url.with_path("/data/src.bin"), fs.config, router=fs._router)
    handle.open(OpenFlags.READ)
    try:
        yield handle
    finally:
        handle.close()


@pytest.fixture
def opened(fs, server):
    """An open, writable handle on a fresh file."""
    handle = File(fs.url.with_path("/data/w.bin"), fs.config, router=fs._router)
    handle.open(OpenFlags.UPDATE | OpenFlags.NEW | OpenFlags.MAKEPATH, Access.OWNER_WRITE)
    try:
        yield handle
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# FileSystem: inspection
# ---------------------------------------------------------------------------


def test_ping_round_trips(fs):
    assert fs.ping() is None


def test_protocol_reports_what_the_server_announced(fs, server):
    info = fs.protocol()
    assert info.version == server.version
    assert info.flags == server.flags


def test_stat_returns_size_and_flags(fs):
    info = fs.stat("/data/a.root")
    assert info.st_size == 11
    assert info.is_file()
    assert not info.is_dir()
    assert info.path == "/data/a.root"


def test_stat_on_a_missing_path_is_a_file_not_found_error(fs):
    with pytest.raises(FileNotFoundError):
        fs.stat("/data/absent")


def test_statvfs_reports_space(fs):
    space = fs.statvfs("/")
    assert space.free_rw > 0
    assert space.f_bavail == space.free_rw


def test_statx_answers_for_every_path_at_once(fs):
    flags = fs.statx(["/data/a.root", "/data/empty"])
    assert len(flags) == 2
    assert not flags[0].is_dir()
    assert flags[1].is_dir()


def test_exists_isdir_isfile_and_getsize(fs):
    assert fs.exists("/data/a.root")
    assert not fs.exists("/data/absent")
    assert fs.isdir("/data")
    assert not fs.isdir("/data/a.root")
    assert fs.isfile("/data/a.root")
    assert not fs.isfile("/data")
    assert fs.getsize("/data/a.root") == 11


def test_a_relative_path_resolves_against_the_url(server, config):
    with FileSystem(server.url.with_path("/data"), config) as fs:
        assert fs.stat("a.root").st_size == 11


def test_cgi_survives_path_resolution(fs):
    assert fs._abs("a.root?xrd.k=1") == "/a.root?xrd.k=1"
    assert fs._abs("/data/../data/a.root") == "/data/a.root"


# ---------------------------------------------------------------------------
# FileSystem: listing
# ---------------------------------------------------------------------------


def test_listdir_gives_names_only(fs):
    assert fs.listdir("/data") == ["a.root", "empty"]


def test_scandir_attaches_stat(fs):
    entries = {e.name: e for e in fs.scandir("/data")}
    assert entries["a.root"].stat.st_size == 11
    assert entries["a.root"].is_file()
    assert entries["empty"].is_dir()
    assert entries["a.root"].path == "/data/a.root"


def test_scandir_without_stat_still_lists(fs):
    entries = fs.scandir("/data", flags=DirListFlags.NONE)
    assert [e.name for e in entries] == ["a.root", "empty"]
    assert all(e.stat is None for e in entries)


def test_a_dir_entry_is_os_pathlike(fs):
    import os

    entry = fs.scandir("/data")[0]
    assert os.fspath(entry) == "/data/a.root"


def test_iterdir_is_scandir(fs):
    assert [e.name for e in fs.iterdir("/data")] == ["a.root", "empty"]


def test_listing_a_missing_directory_raises(fs):
    with pytest.raises(FileNotFoundError):
        fs.listdir("/nowhere")


def test_walk_descends_top_down(fs):
    assert list(fs.walk("/")) == [
        ("/", ["data"], []),
        ("/data", ["empty"], ["a.root"]),
        ("/data/empty", [], []),
    ]


def test_walk_bottom_up_yields_the_leaves_first(fs):
    roots = [root for root, _, _ in fs.walk("/", topdown=False)]
    assert roots == ["/data/empty", "/data", "/"]


def test_walk_reports_errors_to_onerror_instead_of_raising(fs):
    seen: list[OSError] = []
    assert list(fs.walk("/nowhere", onerror=seen.append)) == []
    assert isinstance(seen[0], FileNotFoundError)


def test_walk_swallows_errors_when_asked_to(fs):
    assert list(fs.walk("/nowhere")) == []


def test_glob_matches_within_one_directory(fs):
    assert list(fs.glob("data/*.root")) == ["/data/a.root"]
    assert list(fs.glob("data/*.txt")) == []


def test_glob_crosses_directories_with_a_double_star(fs, server):
    server.add_file("/data/empty/deep.root", b"x")
    assert sorted(fs.glob("**/*.root")) == ["/data/a.root", "/data/empty/deep.root"]


def test_a_single_star_stays_inside_one_directory(fs, server):
    """``fnmatch`` would match the nested file too; a shell would not."""
    server.add_file("/data/empty/deep.root", b"x")
    assert list(fs.glob("/data/*.root")) == ["/data/a.root"]
    assert list(fs.glob("/*/*.root")) == ["/data/a.root"]


def test_glob_takes_an_absolute_pattern(fs, server):
    server.add_file("/data/empty/deep.root", b"x")
    assert sorted(fs.glob("/data/**/*.root")) == ["/data/a.root", "/data/empty/deep.root"]
    assert list(fs.glob("/data/**/deep.root")) == ["/data/empty/deep.root"]


def test_glob_matches_directories_as_well_as_files(fs, server):
    server.add_file("/data/empty/deep.root", b"x")
    assert list(fs.glob("/data/**/empty")) == ["/data/empty"]


def test_glob_understands_a_character_class(fs, server):
    server.add_file("/data/b.root", b"x")
    assert sorted(fs.glob("/data/[ab].root")) == ["/data/a.root", "/data/b.root"]
    assert list(fs.glob("/data/[!a].root")) == ["/data/b.root"]


def test_glob_only_walks_what_the_pattern_can_reach(fs, server):
    """A pattern rooted in one tree must not list the sibling trees."""
    server.add_file("/other/x.root", b"x")
    server.add_file("/data/deep/y.root", b"x")
    assert sorted(fs.glob("/data/**/*.root")) == ["/data/a.root", "/data/deep/y.root"]


@pytest.mark.parametrize(
    "pattern, path, matches",
    [
        ("/a/*.root", "/a/b.root", True),
        ("/a/*.root", "/a/b/c.root", False),
        ("/a/**/c.root", "/a/b/c.root", True),
        ("/a/**/c.root", "/a/c.root", True),
        ("/a/?.root", "/a/b.root", True),
        ("/a/?.root", "/a/bb.root", False),
        ("/a/[b-d].root", "/a/c.root", True),
        ("/a/[b.root", "/a/[b.root", True),  # an unclosed class is a literal
    ],
)
def test_the_glob_pattern_language(pattern, path, matches):
    from xrd.client.filesystem import _glob_regex

    assert bool(_glob_regex(pattern).fullmatch(path)) is matches


# ---------------------------------------------------------------------------
# FileSystem: mutation
# ---------------------------------------------------------------------------


def test_mkdir_and_rmdir(fs, server):
    fs.mkdir("/data/sub")
    assert "/data/sub" in server.dirs
    fs.rmdir("/data/sub")
    assert "/data/sub" not in server.dirs


def test_mkdir_on_an_existing_directory_raises_unless_allowed(fs):
    with pytest.raises(FileExistsError):
        fs.mkdir("/data")
    fs.mkdir("/data", exist_ok=True)


def test_makedirs_creates_the_whole_chain(fs, server):
    fs.makedirs("/a/b/c")
    assert {"/a", "/a/b", "/a/b/c"} <= server.dirs


def test_rmdir_refuses_a_populated_directory(fs):
    with pytest.raises(OSError):
        fs.rmdir("/data")


def test_remove_and_its_unlink_alias(fs, server):
    server.add_file("/data/gone.txt", b"x")
    fs.unlink("/data/gone.txt")
    assert "/data/gone.txt" not in server.files


def test_removing_a_missing_file_raises(fs):
    with pytest.raises(FileNotFoundError):
        fs.remove("/data/absent")


def test_rmtree_clears_a_populated_tree(fs, server):
    server.add_file("/tree/one/a", b"1")
    server.add_file("/tree/two/b", b"2")
    fs.rmtree("/tree")
    assert not [p for p in server.files if p.startswith("/tree")]
    assert not [p for p in server.dirs if p.startswith("/tree")]


def test_rmtree_can_ignore_errors(fs):
    fs.rmtree("/nowhere", ignore_errors=True)


def test_rename_moves_the_contents(fs, server):
    fs.rename("/data/a.root", "/data/b.root")
    assert server.contents("/data/b.root") == b"hello world"
    assert "/data/a.root" not in server.files


def test_move_is_rename(fs, server):
    fs.move("/data/a.root", "/data/c.root")
    assert "/data/c.root" in server.files


def test_chmod_is_accepted(fs):
    fs.chmod("/data/a.root", 0o640)


def test_truncate_by_path(fs, server):
    fs.truncate("/data/a.root", 5)
    assert server.contents("/data/a.root") == b"hello"


def test_touch_creates_an_empty_file(fs, server):
    fs.touch("/data/fresh")
    assert server.contents("/data/fresh") == b""


def test_touch_leaves_an_existing_file_alone(fs, server):
    """``kXR_new`` is refused for a file that is there - and that is fine."""
    fs.touch("/data/a.root")
    assert server.contents("/data/a.root") == b"hello world"


def test_touch_with_exist_ok_false_refuses_an_existing_file(fs):
    with pytest.raises(FileExistsError):
        fs.touch("/data/a.root", exist_ok=False)


# ---------------------------------------------------------------------------
# FileSystem: whole-file helpers
# ---------------------------------------------------------------------------


def test_read_bytes_and_write_bytes(fs, server):
    assert fs.read_bytes("/data/a.root") == b"hello world"
    assert fs.write_bytes("/data/new.bin", b"\x00\xff") == 2
    assert server.contents("/data/new.bin") == b"\x00\xff"


def test_read_text_and_write_text(fs, server):
    fs.write_text("/data/note.txt", "héllo")
    assert server.contents("/data/note.txt") == "héllo".encode()
    assert fs.read_text("/data/note.txt") == "héllo"


def test_open_takes_posc_by_name_and_nothing_else(fs, server):
    """``open`` mirrors the builtin plus ``posc``; a typo is a TypeError."""
    with fs.open("/data/posc.bin", "wb", posc=True) as fh:
        fh.write(b"x")
    assert server.contents("/data/posc.bin") == b"x"
    with pytest.raises(TypeError, match="posk"):
        fs.open("/data/posc.bin", "wb", posk=True)


def test_write_bytes_replaces_rather_than_appends(fs, server):
    fs.write_bytes("/data/a.root", b"short")
    assert server.contents("/data/a.root") == b"short"


# ---------------------------------------------------------------------------
# FileSystem: query, checksum, staging
# ---------------------------------------------------------------------------


def test_checksum_uses_the_servers_default_algorithm(fs):
    result = fs.checksum("/data/a.root")
    assert result.algorithm == "adler32"
    assert result.value


def test_checksum_can_ask_for_another_algorithm(fs):
    assert fs.checksum("/data/a.root", "md5").algorithm == "md5"


def test_query_config_answers_every_name(fs):
    assert fs.query_config("version", "role") == {"version": "v5.6.0", "role": "server"}


def test_query_config_defaults_to_the_version(fs):
    assert fs.query_config() == {"version": "v5.6.0"}


def test_the_config_attribute_is_the_client_config_not_a_query(fs, config):
    assert fs.config is config


def test_locate_names_the_holding_server(fs, server):
    locations = fs.locate("/data/a.root")
    assert locations[0].address == f"{server.address[0]}:{server.address[1]}"
    assert not locations[0].is_manager


def test_deep_locate_resolves_to_servers(fs):
    assert [loc.type for loc in fs.deep_locate("/data/a.root")] == ["S"]


def test_prepare_returns_a_request_handle(fs):
    assert fs.prepare(["/data/a.root"]).startswith("prep-")


def test_evict_is_a_prepare(fs, server):
    from xrd.proto import constants as c

    fs.evict(["/data/a.root"])
    assert c.kXR_prepare in server.seen
    assert server.evicted == ["/data/a.root"]


def test_a_plain_prepare_evicts_nothing(fs, server):
    fs.prepare(["/data/a.root"])
    assert server.evicted == []


def test_prepare_flags_split_across_the_two_option_fields(fs, server):
    """``EVICT`` is an ``optionX`` bit and ``NOTIFY`` an options-byte one, and
    asking for both has to reach the server as both."""
    from xrd.flags import PrepareFlags

    fs.prepare(["/data/a.root"], flags=PrepareFlags.EVICT | PrepareFlags.NOTIFY)
    assert server.evicted == ["/data/a.root"]
    assert server.prepare_options == [(c.kXR_notify, c.kXR_evict)]


def test_query_prepare_reports_on_each_file_of_the_request(fs):
    handle = fs.prepare(["/data/a.root"])
    reports = fs.query_prepare(handle, ["/data/a.root", "/data/missing.root"])
    assert [s.path for s in reports] == ["/data/a.root", "/data/missing.root"]
    assert (reports[0].online, reports[0].requested, reports[0].error) == (True, True, "")
    assert (bool(reports[0]), bool(reports[1])) == (True, False)
    assert str(reports[1]) == "/data/missing.root: nowhere (no such file)"


def test_a_staging_handle_the_server_never_issued_is_an_error(fs):
    with pytest.raises(InvalidArgumentError, match="not one of ours"):
        fs.query_prepare("prep-nobody", ["/data/a.root"])


def test_a_staging_query_with_no_paths_asks_about_the_request_itself(fs):
    """No path to route on, so it goes to whichever endpoint the client has."""
    assert fs.query_prepare(fs.prepare(["/data/a.root"]), []) == []


def test_a_file_on_tape_is_reported_as_waiting_rather_than_online(fs, server):
    server.nearline.add("/data/a.root")
    handle = fs.prepare(["/data/a.root"])
    waiting = fs.query_prepare(handle, ["/data/a.root"])[0]
    assert (waiting.on_tape, waiting.online) == (True, False)
    assert str(waiting) == "/data/a.root: on tape"


def test_a_file_on_tape_stats_as_offline(fs, server):
    server.nearline.add("/data/a.root")
    info = fs.stat("/data/a.root")
    assert info.is_offline() and info.is_readable()


def test_archive_info_says_where_each_file_lives(fs, server):
    server.nearline.add("/data/a.root")
    server.files["/data/b.root"] = bytearray(b"disk")
    on_tape, on_disk, gone = fs.archive_info(
        ["/data/a.root", "/data/b.root", "/data/none.root"]
    )
    assert (on_tape.on_tape, on_tape.online, on_tape.state) == (True, False, "NEARLINE")
    assert (on_disk.online, on_disk.state, bool(on_disk)) == (True, "ONLINE", True)
    assert (gone.exists, gone.error, gone.state) == (False, "no such file", "")


def test_archive_info_is_one_round_trip_for_the_lot(fs, server):
    from xrd.proto import constants as c

    server.seen.clear()
    fs.archive_info(["/data/a.root", "/data/a.root"])
    assert server.seen.count(c.kXR_statx) == 1


def test_cancel_prepare_withdraws_by_handle(fs, server):
    handle = fs.prepare(["/data/a.root"])
    fs.cancel_prepare(handle)
    assert server.cancelled_prepares == [handle]


def test_cancelling_a_checksum_names_the_file(fs, server):
    fs.checksum_cancel("/data/a.root")
    assert server.cancelled_checksums == ["/data/a.root"]


def test_query_stats_returns_what_the_server_said(fs):
    assert fs.query_stats() == '<statistics sel="a"/>'
    assert fs.query_stats("io") == '<statistics sel="io"/>'


def test_query_space_reads_the_oss_token(fs):
    space = fs.query_space("/data")
    assert (space.name, space.free, space.total) == ("public", 1500000, 2000000)
    assert space.largest_free == 1400000
    assert space.used == 500000
    assert space.unlimited


def test_a_pool_with_no_quota_is_unlimited_not_full(fs, server):
    """A missing ``oss.quota`` is "no limit", and zero would say the opposite."""
    server.space = "oss.cgroup=atlas&oss.space=10&oss.free=4"
    space = fs.query_space("/data")
    assert space.quota == -1
    assert space.unlimited
    assert str(space) == "atlas: 4 of 10 bytes free"


def test_a_space_field_that_is_not_a_number_is_refused(fs, server):
    server.space = "oss.cgroup=atlas&oss.free=lots"
    with pytest.raises(ProtocolError, match=r"oss\.free"):
        fs.query_space("/data")


def test_appid_labels_the_connection(fs, server):
    fs.appid("analysis-7")
    assert server.properties == ["appid analysis-7"]


def test_set_property_sends_the_directive_verbatim(fs, server):
    fs.set_property("monitor off")
    assert server.properties == ["monitor off"]


# ---------------------------------------------------------------------------
# FileSystem: extended attributes
# ---------------------------------------------------------------------------


def test_xattrs_round_trip_by_path(fs):
    fs.setxattr("/data/a.root", "user.owner", b"alice")
    fs.setxattr("/data/a.root", "user.run", b"42")
    assert fs.getxattr("/data/a.root", "user.owner") == b"alice"
    assert sorted(fs.listxattr("/data/a.root")) == ["user.owner", "user.run"]
    assert fs.xattrs("/data/a.root") == {"user.owner": b"alice", "user.run": b"42"}
    fs.removexattr("/data/a.root", "user.run")
    assert fs.listxattr("/data/a.root") == ["user.owner"]


def test_reading_a_missing_xattr_raises(fs):
    with pytest.raises(OSError):
        fs.getxattr("/data/a.root", "user.nothing")


# ---------------------------------------------------------------------------
# FileSystem: lifecycle
# ---------------------------------------------------------------------------


def test_the_filesystem_is_a_context_manager(server, config):
    with FileSystem(server.url, config) as fs:
        fs.ping()
    assert not fs._router.connected


def test_endpoint_names_host_and_port(fs, server):
    assert fs.endpoint == f"{server.address[0]}:{server.address[1]}"


def test_repr_is_the_url(fs, server):
    assert str(server.address[1]) in repr(fs)


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------


def test_a_file_opens_reads_and_closes(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with handle:
        assert handle.is_open
        assert handle.read() == b"hello world"
    assert not handle.is_open


def test_reading_a_closed_handle_is_a_value_error(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with pytest.raises(ValueError, match="closed file"):
        handle.read()


def test_opening_twice_is_refused(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with handle:
        with pytest.raises(ValueError, match="already open"):
            handle.open()


def test_close_is_idempotent(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    handle.open()
    handle.close()
    handle.close()


def test_open_volunteers_the_stat(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    info = handle.open()
    assert info is not None and info.st_size == 11
    assert handle.size == 11
    handle.close()


def test_an_open_that_fails_does_not_leave_a_connection_behind(server, config):
    """A caught FileExistsError in a loop used to cost a socket per attempt."""
    handle = File(server.url.with_path("/data/a.root"), config)
    with pytest.raises(FileExistsError):
        handle.open(OpenFlags.NEW | OpenFlags.WRITE)
    assert not handle._router.connected


def test_a_failed_open_leaves_a_borrowed_connection_alone(fs):
    """The router belongs to the FileSystem; the handle must not close it."""
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with pytest.raises(FileExistsError):
        handle.open(OpenFlags.NEW | OpenFlags.WRITE)
    assert fs.stat("/data/a.root").st_size == 11


def test_read_honours_size_and_offset(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with handle:
        assert handle.read(5) == b"hello"
        assert handle.read(5, 6) == b"world"
        assert handle.pread(5, 6) == b"world"
        assert handle.read(0) == b""
        assert handle.read(-1, 6) == b"world"


def test_a_large_read_is_split_into_chunks(server, config):
    from dataclasses import replace

    with FakeServer(files={"/big": b"a" * 100}) as srv:
        small = replace(config, chunk_size=16)
        handle = File(srv.url.with_path("/big"), small)
        with handle:
            assert handle.read() == b"a" * 100
        assert srv.seen.count(3013) >= 6  # kXR_read


def test_readinto_fills_a_buffer(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with handle:
        buffer = bytearray(5)
        assert handle.readinto(buffer) == 5
        assert bytes(buffer) == b"hello"


def test_iterating_a_file_yields_blocks(server, config):
    from dataclasses import replace

    with FakeServer(files={"/big": b"abcdefgh"}) as srv:
        handle = File(srv.url.with_path("/big"), replace(config, chunk_size=3))
        with handle:
            assert list(handle) == [b"abc", b"def", b"gh"]


def test_readv_returns_one_result_per_range_in_order(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with handle:
        assert handle.readv([(6, 5), (0, 5)]) == [b"world", b"hello"]
        assert handle.readv([ReadRange(0, 5)]) == [b"hello"]
        assert handle.readv([]) == []


def test_pgread_returns_verified_pages(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with handle:
        page = handle.pgread(11, 0)
        assert page.data == b"hello world"
        assert page.offset == 0
        assert page.corrupt_pages == ()
        assert handle.pgread(11, 0, verify=False).data == b"hello world"


def test_write_returns_the_count_and_updates_the_size(opened, server):
    assert opened.write(b"abcdef") == 6
    assert opened.size == 6
    assert server.contents("/data/w.bin") == b"abcdef"


def test_pwrite_takes_os_argument_order(opened, server):
    opened.write(b"......")
    opened.pwrite(b"XY", 2)
    assert server.contents("/data/w.bin") == b"..XY.."


def test_a_large_write_is_split(server, config):
    from dataclasses import replace

    with FakeServer() as srv:
        handle = File(srv.url.with_path("/big"), replace(config, chunk_size=8))
        handle.open(OpenFlags.UPDATE | OpenFlags.NEW | OpenFlags.MAKEPATH)
        assert handle.write(b"z" * 40) == 40
        handle.close()
        assert srv.contents("/big") == b"z" * 40


def test_writev_scatters_in_one_round_trip(opened, server):
    assert opened.writev([(0, b"abc"), (3, b"def")]) == 6
    assert server.contents("/data/w.bin") == b"abcdef"
    assert opened.writev([WriteChunk(6, b"gh")]) == 2
    assert opened.writev([]) == 0


def test_clone_copies_the_whole_source_without_the_data_crossing_the_wire(source, opened, server):
    assert opened.clone(source) == 10
    assert server.contents("/data/w.bin") == b"0123456789"
    assert server.seen.count(c.kXR_read) == 0  # nothing came back through us


def test_clone_takes_ranges_as_pairs_triples_or_the_dataclass(source, opened, server):
    copied = opened.clone(source, [(0, 2), (4, 2, 2), CloneRange(8, 2, target_offset=4)])
    assert copied == 6
    assert server.contents("/data/w.bin") == b"014589"


def test_a_clone_range_lands_at_its_own_offset_by_default(source, opened, server):
    assert opened.clone(source, [(6, 4)]) == 4
    assert server.contents("/data/w.bin") == b"\x00" * 6 + b"6789"


def test_cloning_nothing_asks_the_server_for_nothing(source, opened, server):
    assert opened.clone(source, []) == 0
    assert opened.clone(source, [(0, 0), CloneRange(4, 0)]) == 0
    assert server.seen.count(c.kXR_clone) == 0
    assert server.contents("/data/w.bin") == b""


def test_a_clone_of_more_ranges_than_fit_is_split(source, opened, server):
    from xrd.client.file import CLONE_MAX_RANGES

    spans = [(i % 10, 1, i) for i in range(CLONE_MAX_RANGES + 1)]
    assert opened.clone(source, spans) == CLONE_MAX_RANGES + 1
    assert server.seen.count(c.kXR_clone) == 2
    assert len(server.contents("/data/w.bin")) == CLONE_MAX_RANGES + 1


def test_a_clone_refreshes_the_size_it_reports(source, opened):
    opened.stat()
    opened.clone(source, [(0, 10, 90)])
    assert opened.size == 100


@pytest.mark.parametrize("span", [(-1, 4), (0, -4), (0, 4, -1)])
def test_a_clone_range_cannot_be_negative(source, opened, span):
    with pytest.raises(ValueError, match="none negative"):
        opened.clone(source, [span])


def test_cloning_across_two_connections_is_refused(source, opened, server):
    """A handle means nothing to a server that did not hand it out, and the
    two sessions may not even be talking to the same machine."""
    stranger = File(server.url.with_path("/data/src.bin"))
    stranger.open(OpenFlags.READ)
    try:
        with pytest.raises(ValueError, match="different connections"):
            opened.clone(stranger)
    finally:
        stranger.close()


def test_a_clone_cannot_be_checkpointed(source, opened):
    with opened.checkpoint():
        with pytest.raises(UnsupportedError, match="cannot be checkpointed"):
            opened.clone(source)


def test_a_clone_of_a_handle_the_server_never_opened_is_an_error(opened):
    from xrd.errors import XRootDError
    from xrd.proto import requests as r

    with pytest.raises(XRootDError, match="file is not open"):
        opened._router.execute(r.Clone(opened.handle, [(b"\xff\xff\xff\xff", 0, 1, 0)]))


def test_a_clone_list_that_is_not_whole_items_is_refused(opened):
    from xrd.proto import requests as r

    class Broken(r.Clone):
        def payload(self) -> bytes:
            return r.Clone.payload(self)[:20]

    with pytest.raises(InvalidArgumentError, match="Clone list is invalid"):
        opened._router.execute(Broken(opened.handle, [(opened.handle, 0, 1, 0)]))
    with pytest.raises(InvalidArgumentError, match="Clone list is invalid"):
        opened._router.execute(r.Clone(opened.handle, []))


def test_a_server_that_never_heard_of_opcode_3032_says_so_in_those_words(source, opened, server):
    """Opcode 3032 is past XProtocol.hh's fence, so a stock xrootd refuses it
    with "invalid request code" - which is true and useless. Say what it means."""
    server.handlers[c.kXR_clone] = _refuses(3006)  # kXR_InvalidRequest
    with pytest.raises(UnsupportedError, match="does not implement kXR_clone"):
        opened.clone(source)


def test_a_clone_that_fails_for_any_other_reason_is_left_alone(source, opened, server):
    server.handlers[c.kXR_clone] = _refuses(3010)  # kXR_NotAuthorized
    with pytest.raises(PermissionError, match="read-only export"):
        opened.clone(source)


def test_pgwrite_checksums_each_page(opened, server):
    assert opened.pgwrite(b"payload") == 7
    assert server.contents("/data/w.bin") == b"payload"


def test_truncate_on_the_handle_shortens_the_file(opened, server):
    opened.write(b"abcdef")
    assert opened.truncate(3) == 3
    assert server.contents("/data/w.bin") == b"abc"
    assert opened.size == 3


def test_sync_and_its_flush_alias(opened):
    opened.write(b"x")
    opened.sync()
    opened.flush()


def test_stat_on_the_handle_refreshes_on_demand(opened):
    opened.write(b"abc")
    assert opened.stat(refresh=True).st_size == 3


def test_checksum_and_verify_on_a_handle(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with handle:
        expected = handle.checksum().value
        handle.verify(expected)
        with pytest.raises(ChecksumMismatchError):
            handle.verify("00000000")


def test_visa_returns_server_metadata(fs):
    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    with handle:
        assert handle.visa()


def test_xattrs_on_the_open_handle(opened):
    opened.setxattr("user.k", b"v")
    assert opened.getxattr("user.k") == b"v"
    assert opened.listxattr() == ["user.k"]
    opened.removexattr("user.k")
    assert opened.listxattr() == []
    with pytest.raises(KeyError):
        opened.getxattr("user.k")


def test_a_checkpoint_commits_on_a_clean_exit(opened, server):
    with opened.checkpoint():
        opened.write(b"committed")
    assert server.contents("/data/w.bin") == b"committed"


def test_a_checkpoint_rolls_back_and_re_raises(opened, server):
    from xrd.proto import constants as c

    opened.write(b"before")
    with pytest.raises(RuntimeError):
        with opened.checkpoint():
            opened.write(b"doomed")
            raise RuntimeError("boom")
    assert server.contents("/data/w.bin") == b"before"
    # begin, the write that went through it, and the rollback.
    assert server.seen.count(c.kXR_chkpoint) == 3


def test_a_file_is_os_pathlike_and_reprs_its_state(fs):
    import os

    handle = File(fs.url.with_path("/data/a.root"), fs.config, router=fs._router)
    assert os.fspath(handle).endswith("/data/a.root")
    assert "closed" in repr(handle)
    with handle:
        assert "open" in repr(handle)


def test_opening_a_missing_file_for_reading_raises(fs):
    handle = File(fs.url.with_path("/data/absent"), fs.config, router=fs._router)
    with pytest.raises(FileNotFoundError):
        handle.open()


def test_opening_a_directory_as_a_file_raises(fs):
    handle = File(fs.url.with_path("/data"), fs.config, router=fs._router)
    with pytest.raises(OSError):
        handle.open()


# ---------------------------------------------------------------------------
# Vector batching
# ---------------------------------------------------------------------------


def test_read_batches_respect_the_chunk_ceiling():
    ranges = [ReadRange(i * 10, 10) for i in range(2500)]
    batches = list(_batches(ranges))
    assert all(len(b) <= 1024 for b in batches)
    assert sum(len(b) for b in batches) == 2500


def test_read_batches_respect_the_byte_ceiling():
    ranges = [ReadRange(0, READV_MAX_BYTES // 2) for _ in range(4)]
    assert [len(b) for b in _batches(ranges)] == [2, 2]


def test_a_single_oversized_range_is_refused():
    with pytest.raises(ProtocolError, match="use read"):
        list(_batches([ReadRange(0, READV_MAX_BYTES + 1)]))


def test_write_batches_respect_the_ceilings():
    chunks = [WriteChunk(0, b"x" * (READV_MAX_BYTES // 2)) for _ in range(4)]
    assert [len(b) for b in _write_batches(chunks)] == [2, 2]


# ---------------------------------------------------------------------------
# Namespace corners
# ---------------------------------------------------------------------------


def test_a_double_star_inside_a_component_still_crosses_directories(config):
    """``**`` need not be a whole component; ``/d/**.root`` means what it says."""
    files = {"/d/a.root": b"", "/d/sub/b.root": b"", "/d/sub/deep/c.root": b"", "/d/n.txt": b""}
    with FakeServer(files=files, dirs=["/d", "/d/sub", "/d/sub/deep"]) as srv:
        with FileSystem(srv.url, config) as fs:
            assert sorted(fs.glob("/d/**.root")) == [
                "/d/a.root",
                "/d/sub/b.root",
                "/d/sub/deep/c.root",
            ]


def test_a_filesystem_is_os_pathlike(fs):
    assert os.fspath(fs) == str(fs.url)
    assert pathlib.PurePosixPath(os.fspath(fs)).name


def test_statx_that_answers_the_wrong_number_of_flags_is_refused(fs, server):
    """One flags byte per path was asked for; anything else is a broken server."""

    def stingy(conn, sid, params, body):
        yield frame(sid, c.kXR_ok, b"\x00")

    server.handlers[c.kXR_statx] = stingy
    with pytest.raises(ProtocolError, match="2 paths"):
        fs.statx(["/data/a.root", "/data/empty"])


def test_isfile_is_false_for_a_path_that_is_not_there(fs):
    assert fs.isfile("/data/a.root")
    assert not fs.isfile("/data/nope")
    assert not fs.isdir("/data/nope")


def _refuses(sid_error: int = 3010):
    """A handler that answers every request with the same refusal."""
    return lambda conn, sid, params, body: iter([error(sid, sid_error, "read-only export")])


@pytest.mark.parametrize("opcode", [c.kXR_rm, c.kXR_rmdir])
def test_rmtree_reports_what_it_could_not_delete(config, opcode):
    """Files, subdirectories and the root are all reported, not skipped."""
    with FakeServer(files={"/d/f.root": b"x"}, dirs=["/d", "/d/sub"]) as srv:
        srv.handlers[opcode] = _refuses()
        with FileSystem(srv.url, config) as fs:
            with pytest.raises(PermissionError):
                fs.rmtree("/d")


def test_rmtree_reports_a_root_it_cannot_remove(config):
    with FakeServer(dirs=["/e"]) as srv:
        srv.handlers[c.kXR_rmdir] = _refuses()
        with FileSystem(srv.url, config) as fs:
            with pytest.raises(PermissionError):
                fs.rmtree("/e")


@pytest.mark.parametrize("opcode", [c.kXR_rm, c.kXR_rmdir])
def test_rmtree_can_be_told_to_ignore_what_it_cannot_delete(config, opcode):
    """``ignore_errors`` covers the files, the subdirectories and the root."""
    with FakeServer(files={"/d/f.root": b"x"}, dirs=["/d", "/d/sub"]) as srv:
        srv.handlers[opcode] = _refuses()
        with FileSystem(srv.url, config) as fs:
            fs.rmtree("/d", ignore_errors=True)


def test_deep_locate_resolves_a_manager_to_the_servers_behind_it(config):
    """A manager answers with more locations; only the endpoints come back."""
    holding = {"/d/f.root": b"x"}
    with FakeServer(files=holding) as data, FakeServer(files=holding) as mgr:
        host, port = data.address

        def refer(conn, sid, params, body):
            yield frame(sid, c.kXR_ok, f"Mw{host}:{port}".encode() + b"\x00")

        mgr.handlers[c.kXR_locate] = refer
        with FileSystem(mgr.url, config) as fs:
            found = fs.deep_locate("/d/f.root")
            assert [loc.address for loc in found] == [f"{host}:{port}"]
            assert not any(loc.is_manager for loc in found)


def test_deep_locate_skips_a_manager_that_cannot_be_reached(config, closed_port):
    """One dead tier must not lose the locations that did answer."""
    with FakeServer(files={"/d/f.root": b"x"}) as mgr:
        host, port = mgr.address

        def refer(conn, sid, params, body):
            yield frame(
                sid,
                c.kXR_ok,
                f"Mw{closed_port[0]}:{closed_port[1]} Sw{host}:{port}".encode() + b"\x00",
            )

        mgr.handlers[c.kXR_locate] = refer
        with FileSystem(mgr.url, config) as fs:
            assert [loc.address for loc in fs.deep_locate("/d/f.root")] == [f"{host}:{port}"]


# ---------------------------------------------------------------------------
# What the namespace says no with
# ---------------------------------------------------------------------------


def test_mkdir_needs_a_parent_unless_told_to_make_one(fs, server):
    with pytest.raises(FileNotFoundError):
        fs.mkdir("/absent/deep")
    fs.makedirs("/absent/deep")
    assert "/absent/deep" in server.dirs


def test_remove_is_for_files_and_says_so(fs):
    with pytest.raises(IsADirectoryError):
        fs.remove("/data")


def test_chmod_of_something_that_is_not_there_is_reported(fs):
    with pytest.raises(FileNotFoundError):
        fs.chmod("/absent.root", 0o644)


def test_truncate_past_the_end_zero_fills(fs, server):
    fs.truncate("/data/a.root", 16)
    assert server.contents("/data/a.root") == b"hello world" + bytes(5)


def test_creating_a_file_under_a_missing_collection_is_reported(server, config):
    handle = File(server.url.with_path("/nowhere/new.root"), config)
    with pytest.raises(FileNotFoundError):
        handle.open(OpenFlags.NEW | OpenFlags.WRITE)
    handle.open(OpenFlags.NEW | OpenFlags.WRITE | OpenFlags.MAKEPATH)
    handle.close()
    assert "/nowhere/new.root" in server.files


def test_statx_says_other_for_a_path_that_is_not_there(fs):
    entries = fs.statx(["/data/a.root", "/data", "/absent.root"])
    assert entries[0].is_file() and entries[1].is_dir()
    assert not entries[2].is_file() and not entries[2].is_dir()


def test_locating_something_that_is_not_there_is_reported(fs):
    with pytest.raises(FileNotFoundError):
        fs.locate("/absent.root")


def test_extended_attributes_need_the_file_to_exist(fs):
    with pytest.raises(FileNotFoundError):
        fs.getxattr("/absent.root", "user.tag")


def test_an_exclusive_xattr_is_not_overwritten(fs):
    fs.setxattr("/data/a.root", "user.tag", b"first", create_only=True)
    with pytest.raises(FileExistsError, match=r"user\.tag"):
        fs.setxattr("/data/a.root", "user.tag", b"second", create_only=True)
    assert fs.getxattr("/data/a.root", "user.tag") == b"first"


def test_removing_an_attribute_that_is_not_there_is_reported(fs):
    """``kXR_ok`` with a failure inside it is still a failure."""
    with pytest.raises(OSError, match=r"user\.absent"):
        fs.removexattr("/data/a.root", "user.absent")


def test_an_open_that_answers_with_no_stat_still_yields_a_handle(server, config):
    """``kXR_retstat`` is a request, not a promise; the size is asked for later."""

    from xrd.testing.server import _HANDLERS

    def bare(conn, sid, params, body):
        """The real open, with the optional stat trailer trimmed off."""
        for raw in _HANDLERS[c.kXR_open](conn, sid, params, body):
            yield frame(sid, c.kXR_ok, raw[8:12])

    server.handlers[c.kXR_open] = bare
    with File(server.url.with_path("/data/a.root"), config) as handle:
        assert len(handle.handle) == 4
        assert handle.read(5, 0) == b"hello"


def test_a_read_that_stops_early_stops_the_loop(server, config):
    """Past the end the server returns less than asked; that ends the read."""
    small = config.evolve(chunk_size=4)
    with File(server.url.with_path("/data/a.root"), small) as handle:
        assert handle.read(64, 0) == b"hello world"


def test_deep_locate_keeps_one_entry_per_address(config):
    """A server that answers twice for itself is one location, not two."""
    with FakeServer(files={"/d/f.root": b"x"}) as srv:
        host, port = srv.address

        def twice(conn, sid, params, body):
            yield frame(sid, c.kXR_ok, f"Sw{host}:{port} Sw{host}:{port}".encode() + b"\x00")

        srv.handlers[c.kXR_locate] = twice
        with FileSystem(srv.url, config) as fs:
            assert [loc.address for loc in fs.deep_locate("/d/f.root")] == [f"{host}:{port}"]


def test_batching_an_empty_request_asks_for_nothing():
    assert list(_batches([])) == []
    assert list(_write_batches([])) == []


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_a_write_inside_a_checkpoint_travels_as_one(opened, server):
    """The point of a checkpoint: the server journals the write, so it can
    undo it. A plain ``kXR_write`` alongside an open checkpoint would land on
    the file with nothing recorded, and the rollback would restore nothing."""
    opened.write(b"0123456789")
    with opened.checkpoint():
        opened.write(b"XX", 4)
        assert server.contents("/data/w.bin") == b"0123XX6789"
    assert server.contents("/data/w.bin") == b"0123XX6789"
    assert server.seen.count(c.kXR_write) == 1  # the one outside; the other was wrapped


def test_a_checkpoint_undoes_a_truncate_as_well_as_a_write(opened, server):
    opened.write(b"0123456789")
    with pytest.raises(RuntimeError):
        with opened.checkpoint():
            opened.truncate(2)
            assert server.contents("/data/w.bin") == b"01"
            raise RuntimeError("changed my mind")
    assert server.contents("/data/w.bin") == b"0123456789"


def test_a_checkpoint_reports_how_much_room_is_left(opened):
    from xrd.testing.server import CHECKPOINT_CAPACITY

    with opened.checkpoint() as checkpoint:
        assert checkpoint.query().used == 0
        opened.write(b"four")
        info = checkpoint.query()
    assert (info.capacity, info.used) == (CHECKPOINT_CAPACITY, 4)
    assert info.free == CHECKPOINT_CAPACITY - 4
    assert "4/" in str(info)


def test_a_checkpoint_says_which_file_it_belongs_to(opened):
    with opened.checkpoint() as checkpoint:
        assert "/data/w.bin" in repr(checkpoint)


def test_checkpoints_do_not_nest(opened):
    from xrd.errors import UnsupportedError

    with opened.checkpoint():
        with pytest.raises(UnsupportedError, match="already has a checkpoint"):
            with opened.checkpoint():
                pass


def test_a_writev_cannot_be_checkpointed_and_says_so(opened):
    from xrd.errors import UnsupportedError

    with opened.checkpoint():
        with pytest.raises(UnsupportedError, match="write, pgwrite and truncate"):
            opened.writev([WriteChunk(0, b"a")])


def test_a_pgwrite_inside_a_checkpoint_is_wrapped_too(opened, server):
    with opened.checkpoint():
        opened.pgwrite(b"page", 0)
    assert server.contents("/data/w.bin") == b"page"
    assert server.seen.count(c.kXR_pgwrite) == 0


def test_a_checkpoint_that_runs_out_of_room_raises(opened, monkeypatch):
    from xrd.errors import NoSpaceError
    from xrd.testing import server as fake

    monkeypatch.setattr(fake, "CHECKPOINT_CAPACITY", 4)
    with pytest.raises(NoSpaceError):
        with opened.checkpoint():
            opened.write(b"more than four bytes")


def test_only_a_write_or_a_truncate_can_be_checkpointed():
    from xrd.proto import requests as r

    with pytest.raises(ProtocolError, match="kXR_read"):
        r.ChkPoint.execute(b"HDL0", r.Read(b"HDL0", 0, 4))


def test_a_server_asked_for_an_unknown_checkpoint_subcode_complains(opened):
    from xrd.errors import InvalidArgumentError
    from xrd.proto import requests as r

    with opened.checkpoint():
        with pytest.raises(InvalidArgumentError, match="unknown checkpoint subcode"):
            opened._router.execute(r.ChkPoint(opened.handle, 99))


def test_a_checkpoint_operation_with_no_checkpoint_open_is_refused(opened):
    from xrd.errors import InvalidArgumentError
    from xrd.flags import ChkPointCode
    from xrd.proto import requests as r

    with pytest.raises(InvalidArgumentError, match="no checkpoint is open"):
        opened._router.execute(r.ChkPoint(opened.handle, int(ChkPointCode.QUERY)))


def test_a_checkpoint_payload_that_is_not_a_request_header_is_refused(opened):
    from xrd.errors import InvalidArgumentError
    from xrd.flags import ChkPointCode
    from xrd.proto import requests as r

    with opened.checkpoint():
        with pytest.raises(InvalidArgumentError, match="not one request header"):
            opened._router.execute(r.ChkPoint(opened.handle, int(ChkPointCode.XEQ), b"short"))


def test_a_server_refuses_to_checkpoint_an_operation_that_is_not_one(opened):
    from xrd.errors import UnsupportedError
    from xrd.flags import ChkPointCode
    from xrd.proto import requests as r
    from xrd.proto.frames import encode

    header = encode(r.Read(opened.handle, 0, 4), 0)[: c.REQUEST_HDRLEN]
    with opened.checkpoint():
        with pytest.raises(UnsupportedError, match="not checkpointable"):
            opened._router.execute(r.ChkPoint(opened.handle, int(ChkPointCode.XEQ), header))


# ---------------------------------------------------------------------------
# Links (vendor extension)
# ---------------------------------------------------------------------------


def test_a_symbolic_link_names_its_target(fs):
    fs.symlink("/data/a.root", "/data/link.root")
    assert fs.readlink("/data/link.root") == "/data/a.root"


def test_a_hard_link_is_the_same_bytes_under_another_name(fs):
    fs.link("/data/a.root", "/data/hard.root")
    assert fs.read_bytes("/data/hard.root") == fs.read_bytes("/data/a.root")
    fs.hardlink("/data/a.root", "/data/hard2.root")
    assert fs.exists("/data/hard2.root")


def test_linking_to_a_target_that_is_not_there_raises(fs):
    from xrd.errors import NotFoundError

    with pytest.raises(NotFoundError):
        fs.symlink("/data/absent.root", "/data/dangling.root")


def test_reading_a_link_that_is_not_one_raises(fs):
    from xrd.errors import NotFoundError

    with pytest.raises(NotFoundError):
        fs.readlink("/data/a.root")


def test_stat_follows_a_link_and_lstat_describes_it(fs):
    """``os.stat`` answers about the file, ``os.lstat`` about the link, and
    this is the same distinction one option bit down."""
    fs.symlink("/data/a.root", "/data/link.root")
    followed, itself = fs.stat("/data/link.root"), fs.lstat("/data/link.root")
    assert followed.size == fs.stat("/data/a.root").size
    assert itself.size == len("/data/a.root")
    assert StatInfoFlags.OTHER in itself.flags
    assert StatInfoFlags.OTHER not in followed.flags


def test_lstat_is_stat_with_the_option_off(fs, server):
    fs.symlink("/data/a.root", "/data/link.root")
    assert fs.lstat("/data/link.root") == fs.stat("/data/link.root", follow_symlinks=False)


def test_is_symlink_asks_the_only_question_with_an_unambiguous_answer(fs):
    """A followed stat of a link is indistinguishable from a stat of its
    target, so the link-ness comes from ``readlink`` rather than from flags."""
    fs.symlink("/data/a.root", "/data/link.root")
    assert fs.is_symlink("/data/link.root") is True
    assert fs.is_symlink("/data/a.root") is False


def test_a_server_without_the_link_extension_says_it_is_unsupported(server, config):
    from xrd.errors import UnsupportedError, kXR_Unsupported

    def refuse(conn, sid, params, body):
        yield error(sid, kXR_Unsupported, "kXR_symlink is not supported")

    server.handlers[c.kXR_symlink] = refuse
    with FileSystem(server.url, config) as filesystem:
        with pytest.raises(UnsupportedError):
            filesystem.symlink("/data/a.root", "/data/link.root")
