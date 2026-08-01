from __future__ import annotations

import errno
import pickle

import pytest

from xrd import errors as e


@pytest.mark.parametrize(
    "code, cls, builtin",
    [
        (e.kXR_NotFound, e.NotFoundError, FileNotFoundError),
        (e.kXR_NotAuthorized, e.PermissionError_, PermissionError),
        (e.kXR_isDirectory, e.IsADirectoryError_, IsADirectoryError),
        (e.kXR_NoSpace, e.NoSpaceError, OSError),
        (e.kXR_FSError, e.IOError_, OSError),
        (e.kXR_Unsupported, e.UnsupportedError, OSError),
        (e.kXR_InvalidRequest, e.InvalidArgumentError, OSError),
    ],
)
def test_server_codes_map_to_builtin_oserrors(code, cls, builtin):
    """A caller that only knows ``FileNotFoundError`` still catches ours."""
    with pytest.raises(builtin) as info:
        e.raise_for_status(code, "nope", path="/a/b")
    assert isinstance(info.value, cls)
    assert isinstance(info.value, e.ServerError)


def test_oserror_attributes_are_populated():
    with pytest.raises(e.NotFoundError) as info:
        e.raise_for_status(e.kXR_NotFound, "no such file", path="/a/b")
    exc = info.value
    assert exc.errno == errno.ENOENT
    assert exc.filename == "/a/b"
    assert "no such file" in str(exc)
    assert exc.code == e.kXR_NotFound


def test_file_exists_maps_to_the_builtin():
    with pytest.raises(FileExistsError):
        e.raise_for_status(e.kXR_ItExists, "already there", path="/a")


def test_unknown_codes_still_raise_a_server_error():
    with pytest.raises(e.ServerError) as info:
        e.raise_for_status(31337, "mystery")
    assert info.value.code == 31337


def test_ok_does_not_raise():
    assert e.raise_for_status(0, "") is None


def test_everything_descends_from_xrootderror():
    for cls in (
        e.ProtocolError,
        e.ConnectionError,
        e.TimeoutError,
        e.TransientError,
        e.AuthenticationError,
        e.RedirectLimitError,
        e.ChecksumMismatchError,
        e.ServerError,
    ):
        assert issubclass(cls, e.XRootDError)


def test_connection_and_timeout_are_also_builtins():
    assert issubclass(e.ConnectionError, OSError)
    assert issubclass(e.TimeoutError, OSError)


def test_no_mechanism_error_reports_what_was_offered_and_why():
    exc = e.NoMechanismError(offered=["gsi", "unix"], tried={"gsi": "no proxy"})
    assert exc.offered == ["gsi", "unix"]
    assert exc.tried == {"gsi": "no proxy"}
    assert "gsi" in str(exc)


def test_checksum_mismatch_shows_both_values():
    exc = e.ChecksumMismatchError("adler32", "deadbeef", "cafebabe")
    assert exc.algorithm == "adler32"
    assert "deadbeef" in str(exc) and "cafebabe" in str(exc)


def test_transient_error_records_progress():
    exc = e.TransientError("gave up", attempts=4, committed=8192)
    assert (exc.attempts, exc.committed) == (4, 8192)


@pytest.mark.parametrize(
    "exc",
    [
        e.NotFoundError(e.kXR_NotFound, "gone", path="/p"),
        e.ServerError(e.kXR_ServerError, "boom", path="/p"),
        e.TransientError("gave up", attempts=2, committed=7),
        e.NoMechanismError(offered=["gsi"], tried={"gsi": "no proxy"}),
        e.ChecksumMismatchError("adler32", "aa", "bb"),
    ],
)
def test_errors_survive_pickling(exc):
    """Process pools re-raise in the parent; the payload must travel intact."""
    back = pickle.loads(pickle.dumps(exc))
    assert type(back) is type(exc)
    assert str(back) == str(exc)


def test_pickled_server_error_keeps_code_and_path():
    back = pickle.loads(pickle.dumps(e.NotFoundError(e.kXR_NotFound, "gone", path="/p")))
    assert isinstance(back, FileNotFoundError)
    assert (back.code, back.path, back.errno) == (e.kXR_NotFound, "/p", errno.ENOENT)
