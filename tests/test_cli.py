"""The command-line tools: ``xrd-fs`` and ``xrd-cp``."""

from __future__ import annotations

import argparse
import io
import json

import pytest

from xrd import cli
from xrd.cli import Endpoints, config_from, dumps, size_arg
from xrd.cli import cp as cp_cli
from xrd.cli import fs as fs_cli
from xrd.errors import XRootDError
from xrd.testing import FakeDAVServer, FakeServer
from xrd.types import ChecksumInfo, StatInfo
from xrd.url import parse

BODY = b"hello world"


@pytest.fixture
def url(server):
    """The ``root://`` fixture server as a URL string, with a trailing slash."""
    return str(server.url)


@pytest.fixture
def dav():
    with FakeDAVServer(files={"/d/a.root": BODY}) as running:
        yield running


def run(argv, capsys):
    """Run ``xrd-fs`` and hand back ``(exit code, stdout, stderr)``."""
    code = fs_cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def test_one_endpoint_is_opened_once_however_many_paths_name_it(url):
    with Endpoints() as endpoints:
        first, path = endpoints.at(url + "data/a.root")
        second, other = endpoints.at(url + "data")
        assert first is second
        assert (path, other) == ("/data/a.root", "/data")


def test_a_local_path_is_not_an_endpoint(tmp_path):
    with Endpoints() as endpoints, pytest.raises(ValueError, match="local path"):
        endpoints.at(str(tmp_path))


@pytest.mark.parametrize(
    ("text", "expected"), [("4096", 4096), ("8k", 8192), ("2M", 2 << 20), ("1G", 1 << 30)]
)
def test_a_size_is_written_the_way_people_write_it(text, expected):
    assert size_arg(text) == expected


@pytest.mark.parametrize("text", ["", "0", "-1", "eight", "8Q"])
def test_a_size_that_is_not_one_is_a_usage_error(text):
    with pytest.raises(argparse.ArgumentTypeError):
        size_arg(text)


def test_json_output_covers_the_types_the_library_returns():
    payload = json.loads(dumps({"stat": StatInfo(st_size=3), "cks": ChecksumInfo("md5", "ab")}))
    assert payload["stat"]["st_size"] == 3
    assert payload["stat"]["flags"] == 0
    assert payload["cks"] == {"algorithm": "md5", "value": "ab"}
    # A URL is a dataclass too; it must serialise as itself, not as its fields.
    assert json.loads(dumps({"u": parse("root://h//p")}))["u"] == "root://h:1094//p"
    with pytest.raises(TypeError, match="cannot serialise"):
        dumps({"nope": object()})


def test_the_command_line_carries_the_configuration():
    args = argparse.Namespace(
        token="t", user="me", no_verify_tls=True, verbose=0, prompt=False, no_prompt=True
    )
    config = config_from(args)
    assert (config.token, config.username, config.verify_tls) == ("t", "me", False)
    assert config.prompt is False


# ---------------------------------------------------------------------------
# xrd-fs: reading
# ---------------------------------------------------------------------------


def test_ls_lists_names(url, capsys):
    code, out, _ = run(["ls", url + "data"], capsys)
    assert code == 0
    assert out.split() == ["a.root", "empty"]


def test_ls_long_shows_mode_size_and_time(url, capsys):
    _code, out, _ = run(["ls", "-l", url + "data"], capsys)
    first = out.splitlines()[0].split()
    assert first[0].startswith("-rw")
    assert first[1] == str(len(BODY))
    assert first[-1] == "a.root"


def test_ls_recursive_labels_each_directory(url, capsys):
    _code, out, _ = run(["ls", "-R", url + "data"], capsys)
    assert "/data:" in out and "/data/empty:" in out


def test_ls_json_is_a_map_of_directory_to_entries(url, capsys):
    _code, out, _ = run(["ls", "--json", url + "data"], capsys)
    payload = json.loads(out)
    assert [e["name"] for e in payload["/data"]] == ["a.root", "empty"]
    assert payload["/data"][1]["dir"] is True


def test_stat_prints_what_the_server_knows(url, capsys):
    code, out, _ = run(["stat", url + "data/a.root"], capsys)
    assert code == 0
    assert f"Size:  {len(BODY)}" in out


def test_stat_json_gives_the_whole_record(url, capsys):
    _code, out, _ = run(["stat", "--json", url + "data/a.root", url + "data"], capsys)
    sizes = [entry["st_size"] for entry in json.loads(out)]
    assert sizes[0] == len(BODY)


def test_cat_writes_bytes_not_text(url, capsysbinary):
    code = fs_cli.main(["cat", url + "data/a.root"])
    assert (code, capsysbinary.readouterr().out) == (0, BODY)


def test_checksum_asks_the_server(url, capsys):
    code, out, _ = run(["checksum", url + "data/a.root"], capsys)
    assert code == 0
    assert out.split() == ["adler32", "1a0b045d"]


def test_checksum_json_keys_by_url(url, capsys):
    _code, out, _ = run(["checksum", "--json", "-a", "adler32", url + "data/a.root"], capsys)
    assert json.loads(out)[url + "data/a.root"]["value"] == "1a0b045d"


def test_df_reports_space(url, capsys):
    code, out, _ = run(["df", url], capsys)
    assert code == 0
    assert "Read/write nodes" in out
    _code, payload, _ = run(["df", "--json", url], capsys)
    assert "free_rw" in json.loads(payload)


def test_locate_names_the_servers(url, capsys):
    code, out, _ = run(["locate", url + "data/a.root"], capsys)
    assert code == 0
    assert out.strip()
    _code, payload, _ = run(["locate", "--json", url + "data/a.root"], capsys)
    assert json.loads(payload)[0]["address"]


def test_ping_times_the_round_trip(url, capsys):
    code, out, _ = run(["ping", url], capsys)
    assert code == 0
    assert "responded in" in out
    _code, payload, _ = run(["ping", "--json", url], capsys)
    assert json.loads(payload)["ms"] >= 0


def test_query_reads_server_configuration(server, url, capsys):
    server.config_values["version"] = "v5.6.0"
    code, out, _ = run(["query", url, "version"], capsys)
    assert code == 0
    assert out.strip() == "version v5.6.0"
    _code, payload, _ = run(["query", "--json", url, "version"], capsys)
    assert json.loads(payload) == {"version": "v5.6.0"}


# ---------------------------------------------------------------------------
# xrd-fs: writing
# ---------------------------------------------------------------------------


def test_mkdir_makes_parents_on_request(server, url, capsys):
    assert run(["mkdir", "-p", url + "data/x/y"], capsys)[0] == 0
    assert "/data/x/y" in server.dirs
    assert run(["mkdir", url + "data/x/y"], capsys)[0] == 1  # already there


def test_touch_then_rm(server, url, capsys):
    assert run(["touch", url + "data/t.txt"], capsys)[0] == 0
    assert server.contents("/data/t.txt") == b""
    assert run(["rm", url + "data/t.txt"], capsys)[0] == 0
    assert "/data/t.txt" not in server.files


def test_rm_reports_what_is_not_there_unless_forced(url, capsys):
    code, _out, err = run(["rm", url + "data/nope"], capsys)
    assert code == 1
    assert "xrd-fs:" in err
    assert run(["rm", "-f", url + "data/nope"], capsys)[0] == 0


def test_rm_recursive_takes_the_tree(server, url, capsys):
    server.add_file("/data/tree/deep/f.bin", b"x")
    assert run(["rm", "-r", url + "data/tree"], capsys)[0] == 0
    assert not [p for p in server.files if p.startswith("/data/tree")]


def test_rmdir_keeps_the_emptiness_rule(server, url, capsys):
    assert run(["rmdir", url + "data"], capsys)[0] == 1  # not empty
    assert run(["rmdir", url + "data/empty"], capsys)[0] == 0
    assert "/data/empty" not in server.dirs


def test_mv_renames_within_one_endpoint(server, url, capsys):
    assert run(["mv", url + "data/a.root", url + "data/b.root"], capsys)[0] == 0
    assert server.contents("/data/b.root") == BODY


def test_mv_between_endpoints_says_to_use_xrd_cp(url, capsys):
    with FakeServer() as other:
        code, _out, err = run(["mv", url + "data/a.root", str(other.url) + "b.root"], capsys)
    assert code == 1
    assert "xrd-cp" in err


def test_xattr_gets_sets_and_removes(url, capsys):
    assert run(["xattr", url + "data/a.root", "--set", "run=42"], capsys)[0] == 0
    _code, out, _ = run(["xattr", url + "data/a.root"], capsys)
    assert out.strip() == "run=42"
    _code, payload, _ = run(["xattr", "--json", url + "data/a.root"], capsys)
    assert json.loads(payload) == {"run": "42"}
    assert run(["xattr", url + "data/a.root", "--remove", "run"], capsys)[0] == 0
    assert run(["xattr", "--json", url + "data/a.root"], capsys)[1].strip() == "{}"


# ---------------------------------------------------------------------------
# xrd-fs: failure modes
# ---------------------------------------------------------------------------


def test_a_missing_path_exits_one_and_says_why(url, capsys):
    code, _out, err = run(["stat", url + "data/nope"], capsys)
    assert code == 1
    assert "no such file" in err


def test_a_local_path_is_refused_with_a_useful_message(tmp_path, capsys):
    code, _out, err = run(["ls", str(tmp_path)], capsys)
    assert (code, "local path" in err) == (1, True)


def test_no_subcommand_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        fs_cli.main([])
    assert exit_info.value.code == 2


def test_webdav_urls_work_in_the_same_tool(dav, capsys):
    code, out, _ = run(["ls", str(dav.url) + "d"], capsys)
    assert (code, out.strip()) == (0, "a.root")
    assert run(["checksum", str(dav.url) + "d/a.root"], capsys)[1].split()[1] == "1a0b045d"


def test_what_webdav_cannot_do_is_reported_not_crashed(dav, capsys):
    code, _out, err = run(["df", str(dav.url)], capsys)
    assert code == 1
    assert "statvfs" in err


# ---------------------------------------------------------------------------
# xrd-cp
# ---------------------------------------------------------------------------


def test_a_download_names_its_target(url, tmp_path, capsys):
    target = tmp_path / "out.root"
    code = cp_cli.main([url + "data/a.root", str(target)])
    out = capsys.readouterr().out
    assert (code, target.read_bytes()) == (0, BODY)
    assert "11 bytes" in out


def test_a_trailing_slash_means_into_this_directory(url, tmp_path, capsys):
    code = cp_cli.main([url + "data/a.root", str(tmp_path) + "/"])
    capsys.readouterr()
    assert (code, (tmp_path / "a.root").read_bytes()) == (0, BODY)


def test_an_existing_directory_means_the_same(url, tmp_path, capsys):
    assert cp_cli.main(["-q", url + "data/a.root", str(tmp_path)]) == 0
    assert (tmp_path / "a.root").read_bytes() == BODY
    assert capsys.readouterr().out == ""


def test_several_sources_need_a_directory(server, url, tmp_path, capsys):
    server.add_file("/data/b.txt", b"second")
    assert cp_cli.main([url + "data/a.root", url + "data/b.txt", str(tmp_path)]) == 0
    capsys.readouterr()
    assert (tmp_path / "b.txt").read_bytes() == b"second"
    code = cp_cli.main([url + "data/a.root", url + "data/b.txt", str(tmp_path / "one.bin")])
    assert (code, "not a directory" in capsys.readouterr().err) == (2, True)


def test_recursive_copies_the_tree(server, url, tmp_path, capsys):
    server.add_file("/data/sub/deep.bin", b"deep")
    assert cp_cli.main(["-r", "-q", url + "data", str(tmp_path / "tree")]) == 0
    assert (tmp_path / "tree/sub/deep.bin").read_bytes() == b"deep"


def test_an_upload_goes_the_other_way(server, url, tmp_path, capsys):
    source = tmp_path / "up.bin"
    source.write_bytes(b"payload")
    assert cp_cli.main(["-q", str(source), url + "data/up.bin"]) == 0
    assert server.contents("/data/up.bin") == b"payload"


def test_no_clobber_refuses_an_existing_target(url, tmp_path, capsys):
    target = tmp_path / "out.root"
    target.write_bytes(b"keep")
    code = cp_cli.main(["-n", url + "data/a.root", str(target)])
    assert (code, target.read_bytes()) == (1, b"keep")
    assert "xrd-cp:" in capsys.readouterr().err


def test_json_reports_the_transfer(url, tmp_path, capsys):
    code = cp_cli.main(["--json", url + "data/a.root", str(tmp_path / "o.root")])
    record, = json.loads(capsys.readouterr().out)
    assert code == 0
    assert record["size"] == len(BODY)
    assert record["verified"] is True
    assert record["checksum"] == "adler32:1a0b045d"


def test_verification_can_be_demanded(url, tmp_path, capsys):
    argv = ["--verify", "-a", "adler32", "--json", url + "data/a.root", str(tmp_path / "a")]
    assert cp_cli.main(argv) == 0
    record, = json.loads(capsys.readouterr().out)
    assert record["checksum"] == "adler32:1a0b045d"


def test_verification_can_be_skipped(url, tmp_path, capsys):
    assert cp_cli.main(["--no-verify", "--json", url + "data/a.root", str(tmp_path / "b")]) == 0
    record, = json.loads(capsys.readouterr().out)
    assert (record["verified"], record["checksum"]) == (False, None)


def test_the_chunk_size_is_honoured(url, tmp_path, capsys):
    assert cp_cli.main(["-q", "--chunk-size", "4", url + "data/a.root", str(tmp_path / "c")]) == 0
    assert (tmp_path / "c").read_bytes() == BODY


def test_a_scheme_nobody_speaks_is_reported(tmp_path, capsys):
    code = cp_cli.main(["ftp://example.org/f", str(tmp_path / "x")])
    assert (code, "cannot read from ftp" in capsys.readouterr().err) == (1, True)


def test_third_party_is_one_flag(url, capsys):
    with FakeServer() as destination:
        code = cp_cli.main(["--tpc", "-q", url + "data/a.root", str(destination.url) + "pulled"])
        assert code == 0
        assert any("tpc.key" in path for path in destination.opened)


def test_third_party_between_webdav_endpoints_is_the_same_flag(dav, capsys):
    """One flag, either dialect: the URLs decide which one is spoken."""
    with FakeDAVServer(dirs=["/d"]) as destination:
        code = cp_cli.main(
            ["--tpc", "-q", str(dav.url) + "d/a.root", str(destination.url) + "d/pulled"]
        )
        assert code == 0
        assert destination.contents("/d/pulled") == BODY
        assert destination.copies[-1]["Source"].endswith("/d/a.root")


def test_a_copy_between_protocols_works_both_ways(url, dav, tmp_path, capsys):
    assert cp_cli.main(["-q", str(dav.url) + "d/a.root", url + "data/from-dav"]) == 0
    assert cp_cli.main(["-q", url + "data/a.root", str(dav.url) + "d/from-root"]) == 0
    capsys.readouterr()
    assert dav.contents("/d/from-root") == BODY


# ---------------------------------------------------------------------------
# The progress display
# ---------------------------------------------------------------------------


def test_the_bar_redraws_only_when_the_percentage_moves():
    stream = io.StringIO()
    bar = cp_cli.Bar("f.root", stream)
    for done in (0, 1, 2, 50, 100, 100):  # the last two say the same thing
        bar(done, 100)
    bar.finish()
    frames = [f for f in stream.getvalue().split("\r") if f]
    assert len(frames) == 5  # 0%, 1%, 2%, 50%, 100% - and 100% only once
    assert "100%" in frames[-1]
    assert frames[-1].endswith("\n")


def test_the_bar_copes_with_a_source_of_unknown_size():
    stream = io.StringIO()
    bar = cp_cli.Bar("stream", stream)
    bar(1024, None)
    bar.finish()
    assert "1.0 KiB" in stream.getvalue()


@pytest.mark.parametrize(
    ("size", "text"), [(0, "0 B"), (999, "999 B"), (1024, "1.0 KiB"), (5 << 20, "5.0 MiB")]
)
def test_byte_counts_are_human_readable(size, text):
    assert cp_cli._human(size) == text


def test_progress_is_off_when_stdout_is_not_a_terminal(url, tmp_path, capsys):
    """A pipe gets clean output; a tty gets a bar. Nothing else decides it."""
    assert cp_cli.main(["-p", "-q", url + "data/a.root", str(tmp_path / "p")]) == 0
    assert "%" in capsys.readouterr().err
    assert cp_cli.main(["-q", url + "data/a.root", str(tmp_path / "q")]) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("verbosity", "level"), [(1, "WARNING"), (2, "INFO"), (3, "DEBUG")])
def test_verbosity_flags_choose_a_logging_level(verbosity, level, monkeypatch):
    import logging

    chosen: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: chosen.update(kw))
    cli.configure_logging(verbosity)
    assert logging.getLevelName(chosen["level"]) == level


def test_no_verbosity_flag_leaves_logging_alone(monkeypatch):
    import logging

    monkeypatch.setattr(
        logging, "basicConfig", lambda **kw: pytest.fail("logging was configured anyway")
    )
    cli.configure_logging(0)


def test_the_json_encoder_refuses_what_it_cannot_render():
    with pytest.raises(TypeError, match="cannot serialise"):
        cli.dumps({"handle": object()})


def test_the_json_encoder_renders_flags_as_numbers_and_bytes_as_text():
    from xrd.flags import OpenFlags

    assert cli.dumps({"flags": OpenFlags.READ}) == f'{{\n  "flags": {int(OpenFlags.READ)}\n}}'
    assert '"payload": "hi"' in cli.dumps({"payload": b"hi"})


def test_a_bar_that_never_drew_anything_leaves_the_terminal_alone(capsys):
    """No percentage was ever printed, so there is no line to close."""
    cp_cli.Bar("f.root").finish()
    assert capsys.readouterr().err == ""


def test_ping_can_be_asked_to_say_nothing(url, capsys):
    code, out, err = run(["ping", "--quiet", url], capsys)
    assert (code, out, err) == (0, "", "")


def test_a_trailing_slash_is_a_directory_without_asking(url):
    """``cp f d/`` means *into* ``d`` even before anyone has looked."""
    assert cp_cli._is_dir(parse(url + "data/"), Endpoints(cli.Config())) is True


def test_a_destination_the_server_will_not_discuss_is_not_a_directory():
    """``isdir`` failing is an answer: treat the name as a file to write."""

    class Grumpy:
        def isdir(self, _path):
            raise XRootDError("no")

    class Only:
        def at(self, _url):
            return Grumpy(), "/d"

    assert cp_cli._is_dir(parse("root://h//d"), Only()) is False
