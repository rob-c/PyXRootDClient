"""File objects for remote paths.

:func:`open_url` has :func:`open`'s signature and returns the same stack of
objects the builtin does - a buffered binary reader, a buffered writer, a
random-access buffer, or a text wrapper - so remote files behave exactly like
local ones, including ``for line in f``.
"""

from __future__ import annotations

import io
from typing import IO, Any, BinaryIO, Literal, TextIO, overload

from ..client.file import File
from ..config import Config
from ..session.router import Router
from ..url import XRootDURL, parse
from .raw import OpenBinaryMode, OpenTextMode, XRootDRawIO, flags_for_mode, parse_mode

__all__ = ["open_url", "XRootDRawIO", "flags_for_mode", "parse_mode"]

DEFAULT_BUFFER_SIZE = 1 << 20


# The overloads are the builtin's, narrowed the same way typeshed narrows
# ``open``: a literal mode string tells the caller whether ``read`` gives
# ``bytes`` or ``str``, and ``buffering=0`` hands back the raw layer - which
# is where :attr:`XRootDRawIO.file` lives. The first deliberately overlaps the
# second, because ``buffering=0`` is an ``int`` like any other and only its
# literal value says which layer the caller wants.
@overload
def open_url(  # type: ignore[overload-overlap]
    url: str | XRootDURL,
    mode: OpenBinaryMode = ...,
    *,
    buffering: Literal[0],
    encoding: None = ...,
    errors: None = ...,
    newline: None = ...,
    config: Config | None = ...,
    router: Router | None = ...,
    posc: bool = ...,
) -> XRootDRawIO: ...


@overload
def open_url(
    url: str | XRootDURL,
    mode: OpenBinaryMode = ...,
    *,
    buffering: int = ...,
    encoding: None = ...,
    errors: None = ...,
    newline: None = ...,
    config: Config | None = ...,
    router: Router | None = ...,
    posc: bool = ...,
) -> BinaryIO: ...


@overload
def open_url(
    url: str | XRootDURL,
    mode: OpenTextMode,
    *,
    buffering: int = ...,
    encoding: str | None = ...,
    errors: str | None = ...,
    newline: str | None = ...,
    config: Config | None = ...,
    router: Router | None = ...,
    posc: bool = ...,
) -> TextIO: ...


@overload
def open_url(
    url: str | XRootDURL,
    mode: str,
    *,
    buffering: int = ...,
    encoding: str | None = ...,
    errors: str | None = ...,
    newline: str | None = ...,
    config: Config | None = ...,
    router: Router | None = ...,
    posc: bool = ...,
) -> IO[Any]: ...


def open_url(
    url: str | XRootDURL,
    mode: str = "rb",
    *,
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    config: Config | None = None,
    router: Router | None = None,
    posc: bool = False,
) -> IO[Any] | io.RawIOBase:
    """Open a remote file. Mirrors :func:`open`, including its return types.

    An ``http``/``https``/``dav``/``davs`` URL is handed to
    :func:`~xrd.http.open_http`, which returns the same stack of objects.

    ``buffering`` of 0 gives the raw layer (binary only), a positive value
    sets the buffer size, and ``-1`` uses
    :data:`DEFAULT_BUFFER_SIZE` - a megabyte, because a wide-area round trip
    costs far more than a local one.
    """
    _base, binary, updating = parse_mode(mode)
    if not binary and buffering == 0:
        raise ValueError("can't have unbuffered text I/O")
    if binary and encoding is not None:
        raise ValueError("binary mode doesn't take an encoding argument")

    target = parse(url)
    if target.is_http:
        from ..http import open_http

        return open_http(
            target,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            config=config,
        )

    handle = File(target, config, router=router)
    raw = XRootDRawIO(handle, mode, posc=posc)

    if buffering == 0:
        return raw
    size = DEFAULT_BUFFER_SIZE if buffering < 0 else buffering

    stream: BinaryIO
    if updating or (raw.readable() and raw.writable()):
        stream = io.BufferedRandom(raw, size)
    elif raw.writable():
        stream = io.BufferedWriter(raw, size)
    else:
        stream = io.BufferedReader(raw, size)

    if binary:
        return stream
    return io.TextIOWrapper(stream, encoding=encoding, errors=errors, newline=newline)
