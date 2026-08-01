"""One encoder per ``kXR_*`` opcode.

Layouts follow nginx-xrootd ``wire_core_requests.h``; cross-checked against
go-hep ``xrootd/xrdproto/<op>`` and XRootD.jl ``Wire/requests.jl``. Every
``params`` method writes exactly 16 bytes (header offsets 4..19).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from . import constants as c
from .buffer import Writer
from .frames import Request

__all__ = [
    "Protocol", "Login", "Auth", "Ping", "EndSession", "Bind",
    "Stat", "StatVFS", "Statx", "Dirlist", "Locate", "Query", "Prepare",
    "Mkdir", "Rm", "Rmdir", "Mv", "Chmod", "Truncate", "Set",
    "Open", "Close", "Read", "Write", "Sync",
    "ReadV", "WriteV", "PgRead", "PgWrite", "ChkPoint", "Fattr", "Sigver",
]


def _encode_path(path: str) -> bytes:
    return path.encode("utf-8")


# --------------------------------------------------------------------------
# Session bring-up
# --------------------------------------------------------------------------


class Protocol(Request):
    """``kXR_protocol`` - capability negotiation."""

    __slots__ = ("flags",)
    opcode = c.kXR_protocol

    def __init__(self, flags: int | None = None) -> None:
        self.flags = c.kXR_secreqs | c.kXR_ableTLS if flags is None else flags

    def params(self, w: Writer) -> None:
        w.u32(c.kXR_PROTOCOLVERSION).u8(self.flags).u8(c.kXR_ExpLogin).zeros(10)

    def __repr__(self) -> str:
        return f"Protocol(flags=0x{self.flags:02x})"


class Login(Request):
    """``kXR_login``."""

    __slots__ = ("username", "pid", "capver", "token")
    opcode = c.kXR_login

    def __init__(
        self, username: str, pid: int | None = None, capver: int | None = None, token: str = ""
    ) -> None:
        self.username = username
        self.pid = os.getpid() if pid is None else pid
        self.capver = c.kXR_ver005 | c.kXR_asyncap if capver is None else capver
        self.token = token

    def params(self, w: Writer) -> None:
        # pid[4] username[8] reserved[1] ability[1] capver[1] role[1]
        w.i32(self.pid).padded(self.username, 8).zeros(2).u8(self.capver).u8(0)

    def payload(self) -> bytes:
        return self.token.encode("utf-8")

    def __repr__(self) -> str:
        return f"Login(username={self.username!r}, pid={self.pid})"


class Auth(Request):
    """``kXR_auth`` - one round of a credential exchange."""

    __slots__ = ("credtype", "cred")
    opcode = c.kXR_auth
    idempotent = False

    def __init__(self, credtype: str, cred: bytes) -> None:
        if len(credtype.encode()) > 4:
            raise ValueError(f"credtype must be <= 4 bytes, got {credtype!r}")
        self.credtype = credtype
        self.cred = cred

    def params(self, w: Writer) -> None:
        w.zeros(12).padded(self.credtype, 4)

    def payload(self) -> bytes:
        return self.cred

    def __repr__(self) -> str:
        return f"Auth(credtype={self.credtype!r}, cred=<{len(self.cred)} bytes>)"


class Ping(Request):
    """``kXR_ping``."""

    __slots__ = ()
    opcode = c.kXR_ping


class EndSession(Request):
    """``kXR_endsess`` - graceful session teardown."""

    __slots__ = ("sessid",)
    opcode = c.kXR_endsess
    idempotent = False

    def __init__(self, sessid: bytes = b"") -> None:
        self.sessid = sessid

    def params(self, w: Writer) -> None:
        w.padded(self.sessid, 16)


class Bind(Request):
    """``kXR_bind`` - attach an extra data connection to a session."""

    __slots__ = ("sessid",)
    opcode = c.kXR_bind
    idempotent = False

    def __init__(self, sessid: bytes) -> None:
        self.sessid = sessid

    def params(self, w: Writer) -> None:
        w.padded(self.sessid, 16)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


class Stat(Request):
    """``kXR_stat``."""

    __slots__ = ("path", "options", "fhandle")
    opcode = c.kXR_stat

    def __init__(
        self, path: str = "", options: int = 0, fhandle: bytes = c.NULL_FHANDLE
    ) -> None:
        self.path = path
        self.options = options
        self.fhandle = fhandle

    def params(self, w: Writer) -> None:
        w.u8(self.options).zeros(11).padded(self.fhandle, 4)

    def payload(self) -> bytes:
        return _encode_path(self.path)

    def __repr__(self) -> str:
        # By type, not by name: ``StatVFS`` is a ``Stat`` with an option set,
        # and a log line saying so is the point of the repr.
        return f"{type(self).__name__}(path={self.path!r}, options=0x{self.options:02x})"


class StatVFS(Stat):
    """``kXR_stat`` with ``kXR_vfs`` - filesystem space."""

    __slots__ = ()

    def __init__(self, path: str) -> None:
        super().__init__(path, options=c.kXR_vfs)


class Statx(Request):
    """``kXR_statx`` - flag-only stat of many paths at once."""

    __slots__ = ("paths",)
    opcode = c.kXR_statx

    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = list(paths)

    def payload(self) -> bytes:
        return "\n".join(self.paths).encode("utf-8")

    def __repr__(self) -> str:
        return f"Statx(paths={self.paths!r})"


class Dirlist(Request):
    """``kXR_dirlist``."""

    __slots__ = ("path", "options")
    opcode = c.kXR_dirlist

    def __init__(self, path: str, options: int | None = None) -> None:
        self.path = path
        self.options = c.kXR_dstat if options is None else options

    def params(self, w: Writer) -> None:
        w.zeros(15).u8(self.options)

    def payload(self) -> bytes:
        return _encode_path(self.path)

    def __repr__(self) -> str:
        return f"Dirlist(path={self.path!r}, options=0x{self.options:02x})"


class Locate(Request):
    """``kXR_locate``."""

    __slots__ = ("path", "options")
    opcode = c.kXR_locate

    def __init__(self, path: str, options: int = 0) -> None:
        self.path = path
        self.options = options

    def params(self, w: Writer) -> None:
        w.u16(self.options).zeros(14)

    def payload(self) -> bytes:
        return _encode_path(self.path)

    def __repr__(self) -> str:
        return f"Locate(path={self.path!r}, options=0x{self.options:04x})"


class Query(Request):
    """``kXR_query``."""

    __slots__ = ("infotype", "args", "fhandle")
    opcode = c.kXR_query

    def __init__(
        self, infotype: int, args: str | bytes = b"", fhandle: bytes = c.NULL_FHANDLE
    ) -> None:
        self.infotype = infotype
        self.args = args
        self.fhandle = fhandle

    def params(self, w: Writer) -> None:
        w.u16(self.infotype).zeros(2).padded(self.fhandle, 4).zeros(8)

    def payload(self) -> bytes:
        return self.args.encode("utf-8") if isinstance(self.args, str) else self.args

    def __repr__(self) -> str:
        return f"Query(infotype={self.infotype}, args={self.args!r})"


class Prepare(Request):
    """``kXR_prepare`` - stage or evict files."""

    __slots__ = ("paths", "options", "priority", "port")
    opcode = c.kXR_prepare
    signed = True
    idempotent = False

    def __init__(
        self, paths: Sequence[str], options: int = 0, priority: int = 0, port: int = 0
    ) -> None:
        self.paths = list(paths)
        self.options = options
        self.priority = priority
        self.port = port

    def params(self, w: Writer) -> None:
        w.u8(self.options).u8(self.priority).u16(self.port).zeros(12)

    def payload(self) -> bytes:
        return "\n".join(self.paths).encode("utf-8")

    def __repr__(self) -> str:
        return f"Prepare(paths={len(self.paths)}, options=0x{self.options:02x})"


# --------------------------------------------------------------------------
# Namespace mutation
# --------------------------------------------------------------------------


class Mkdir(Request):
    """``kXR_mkdir``."""

    __slots__ = ("path", "mode", "mkpath")
    opcode = c.kXR_mkdir
    signed = True
    idempotent = False

    def __init__(self, path: str, mode: int = 0o755, mkpath: bool = False) -> None:
        self.path = path
        self.mode = mode
        self.mkpath = mkpath

    def params(self, w: Writer) -> None:
        w.u8(c.kXR_mkdirpath if self.mkpath else 0).zeros(13).u16(self.mode)

    def payload(self) -> bytes:
        return _encode_path(self.path)

    def __repr__(self) -> str:
        return f"Mkdir(path={self.path!r}, mode=0o{self.mode:o}, mkpath={self.mkpath})"


class Rm(Request):
    """``kXR_rm``."""

    __slots__ = ("path",)
    opcode = c.kXR_rm
    signed = True
    idempotent = False

    def __init__(self, path: str) -> None:
        self.path = path

    def payload(self) -> bytes:
        return _encode_path(self.path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path!r})"  # ``Rmdir`` is one of these


class Rmdir(Rm):
    """``kXR_rmdir``."""

    __slots__ = ()
    opcode = c.kXR_rmdir


class Mv(Request):
    """``kXR_mv`` - the payload is ``"<src> <dst>"``, split by ``arg1len``."""

    __slots__ = ("src", "dst")
    opcode = c.kXR_mv
    signed = True
    idempotent = False

    def __init__(self, src: str, dst: str) -> None:
        self.src = src
        self.dst = dst

    def params(self, w: Writer) -> None:
        w.zeros(14).u16(len(self.src.encode("utf-8")))

    def payload(self) -> bytes:
        return f"{self.src} {self.dst}".encode()

    def __repr__(self) -> str:
        return f"Mv(src={self.src!r}, dst={self.dst!r})"


class Chmod(Request):
    """``kXR_chmod``."""

    __slots__ = ("path", "mode")
    opcode = c.kXR_chmod
    signed = True
    idempotent = False

    def __init__(self, path: str, mode: int) -> None:
        self.path = path
        self.mode = mode

    def params(self, w: Writer) -> None:
        w.zeros(14).u16(self.mode)

    def payload(self) -> bytes:
        return _encode_path(self.path)

    def __repr__(self) -> str:
        return f"Chmod(path={self.path!r}, mode=0o{self.mode:o})"


class Truncate(Request):
    """``kXR_truncate`` - by path, or by handle when ``fhandle`` is set."""

    __slots__ = ("path", "size", "fhandle")
    opcode = c.kXR_truncate
    signed = True
    idempotent = False

    def __init__(self, path: str = "", size: int = 0, fhandle: bytes = c.NULL_FHANDLE) -> None:
        self.path = path
        self.size = size
        self.fhandle = fhandle

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).i64(self.size).zeros(4)

    def payload(self) -> bytes:
        return _encode_path(self.path)

    def __repr__(self) -> str:
        return f"Truncate(path={self.path!r}, size={self.size})"


class Set(Request):
    """``kXR_set`` - set a server-side client property."""

    __slots__ = ("data",)
    opcode = c.kXR_set
    signed = True
    idempotent = False

    def __init__(self, data: str) -> None:
        self.data = data

    def payload(self) -> bytes:
        return self.data.encode("utf-8")


# --------------------------------------------------------------------------
# File I/O
# --------------------------------------------------------------------------


class Open(Request):
    """``kXR_open``."""

    __slots__ = ("path", "options", "mode")
    opcode = c.kXR_open
    signed = True
    idempotent = False

    def __init__(self, path: str, options: int, mode: int = 0) -> None:
        self.path = path
        self.options = options
        self.mode = mode

    def params(self, w: Writer) -> None:
        w.u16(self.mode).u16(self.options).zeros(12)

    def payload(self) -> bytes:
        return _encode_path(self.path)

    def __repr__(self) -> str:
        return f"Open(path={self.path!r}, options=0x{self.options:04x}, mode=0o{self.mode:o})"


class Close(Request):
    """``kXR_close``."""

    __slots__ = ("fhandle",)
    opcode = c.kXR_close
    idempotent = False

    def __init__(self, fhandle: bytes) -> None:
        self.fhandle = fhandle

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).zeros(12)


class Read(Request):
    """``kXR_read``."""

    __slots__ = ("fhandle", "offset", "length")
    opcode = c.kXR_read

    def __init__(self, fhandle: bytes, offset: int, length: int) -> None:
        self.fhandle = fhandle
        self.offset = offset
        self.length = length

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).i64(self.offset).i32(self.length)

    def __repr__(self) -> str:
        return f"Read(offset={self.offset}, length={self.length})"


class Write(Request):
    """``kXR_write``."""

    __slots__ = ("fhandle", "offset", "data")
    opcode = c.kXR_write
    signed = True
    idempotent = False

    def __init__(self, fhandle: bytes, offset: int, data: bytes) -> None:
        self.fhandle = fhandle
        self.offset = offset
        self.data = data

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).i64(self.offset).zeros(4)

    def payload(self) -> bytes:
        return self.data

    def __repr__(self) -> str:
        return f"Write(offset={self.offset}, len={len(self.data)})"


class Sync(Request):
    """``kXR_sync``."""

    __slots__ = ("fhandle",)
    opcode = c.kXR_sync
    idempotent = False

    def __init__(self, fhandle: bytes) -> None:
        self.fhandle = fhandle

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).zeros(12)


class ReadV(Request):
    """``kXR_readv`` - many scattered ranges in one round trip.

    Payload is an array of ``readahead_list``:
    ``fhandle[4] rlen[4] roffset[8]``.
    """

    __slots__ = ("chunks", "pathid")
    opcode = c.kXR_readv

    def __init__(self, chunks: Sequence[tuple[bytes, int, int]], pathid: int = 0) -> None:
        #: ``(fhandle, offset, length)`` triples.
        self.chunks = list(chunks)
        self.pathid = pathid

    def params(self, w: Writer) -> None:
        w.zeros(15).u8(self.pathid)

    def payload(self) -> bytes:
        w = Writer()
        for fhandle, offset, length in self.chunks:
            w.padded(fhandle, 4).i32(length).i64(offset)
        return w.bytes()

    def __repr__(self) -> str:
        return f"ReadV(chunks={len(self.chunks)})"


class WriteV(Request):
    """``kXR_writev``.

    ``dlen`` covers the array of ``write_list`` (``fhandle[4] wlen[4]
    offset[8]``) and nothing else - servers check that it divides by 16 and
    answer ``kXR_ArgInvalid`` when it does not. The concatenated data streams
    after the frame as a trailer.
    """

    __slots__ = ("chunks", "options")
    opcode = c.kXR_writev
    signed = True
    idempotent = False

    def __init__(self, chunks: Sequence[tuple[bytes, int, bytes]], sync: bool = False) -> None:
        #: ``(fhandle, offset, data)`` triples.
        self.chunks = list(chunks)
        self.options = c.kXR_wv_doSync if sync else 0

    def params(self, w: Writer) -> None:
        w.u8(self.options).zeros(15)

    def payload(self) -> bytes:
        w = Writer()
        for fhandle, offset, data in self.chunks:
            w.padded(fhandle, 4).i32(len(data)).i64(offset)
        return w.bytes()

    def trailer(self) -> bytes:
        return b"".join(data for _, _, data in self.chunks)

    def __repr__(self) -> str:
        return f"WriteV(chunks={len(self.chunks)})"


class PgRead(Request):
    """``kXR_pgread`` - read with a per-4KiB-page CRC32c."""

    __slots__ = ("fhandle", "offset", "length", "reqflags", "pathid")
    opcode = c.kXR_pgread

    def __init__(
        self, fhandle: bytes, offset: int, length: int, retry: bool = False, pathid: int = 0
    ) -> None:
        self.fhandle = fhandle
        self.offset = offset
        self.length = length
        self.reqflags = c.kXR_pgRetry if retry else 0
        self.pathid = pathid

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).i64(self.offset).i32(self.length)

    def payload(self) -> bytes:
        if not self.reqflags and not self.pathid:
            return b""
        return Writer().u8(self.pathid).u8(self.reqflags).zeros(2).bytes()

    def __repr__(self) -> str:
        return f"PgRead(offset={self.offset}, length={self.length})"


class PgWrite(Request):
    """``kXR_pgwrite`` - write with a per-page CRC32c prefix on each page."""

    __slots__ = ("fhandle", "offset", "data", "reqflags", "pathid")
    opcode = c.kXR_pgwrite
    signed = True
    idempotent = False

    def __init__(
        self, fhandle: bytes, offset: int, data: bytes, retry: bool = False, pathid: int = 0
    ) -> None:
        self.fhandle = fhandle
        self.offset = offset
        #: Already CRC-interleaved payload; build it with :func:`pack_pages`.
        self.data = data
        self.reqflags = c.kXR_pgRetry if retry else 0
        self.pathid = pathid

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).i64(self.offset).u8(self.pathid).u8(self.reqflags).zeros(2)

    def payload(self) -> bytes:
        return self.data

    def __repr__(self) -> str:
        return f"PgWrite(offset={self.offset}, len={len(self.data)})"


class ChkPoint(Request):
    """``kXR_chkpoint`` - transactional write checkpointing."""

    __slots__ = ("fhandle", "subcode", "data")
    opcode = c.kXR_chkpoint
    signed = True
    idempotent = False

    def __init__(self, fhandle: bytes, subcode: int, data: bytes = b"") -> None:
        self.fhandle = fhandle
        self.subcode = subcode
        self.data = data

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).zeros(11).u8(self.subcode)

    def payload(self) -> bytes:
        return self.data

    def __repr__(self) -> str:
        return f"ChkPoint(subcode={self.subcode})"


class Fattr(Request):
    """``kXR_fattr`` - extended attributes.

    Body layout (go-hep ``xrdproto/fattr``): ``path\\0`` then, per attribute,
    a 2-byte rc placeholder, ``name\\0``, and for ``SET`` a 4-byte length
    followed by the value.
    """

    __slots__ = ("subcode", "numattr", "options", "fhandle", "body", "idempotent")
    opcode = c.kXR_fattr
    signed = True

    def __init__(
        self,
        subcode: int,
        body: bytes,
        numattr: int = 0,
        options: int = 0,
        fhandle: bytes = c.NULL_FHANDLE,
    ) -> None:
        self.subcode = subcode
        self.numattr = numattr
        self.options = options
        self.fhandle = fhandle
        self.body = body
        self.idempotent = subcode in (c.kXR_fattrGet, c.kXR_fattrList)

    def params(self, w: Writer) -> None:
        w.padded(self.fhandle, 4).u8(self.subcode).u8(self.numattr).u8(self.options).zeros(9)

    def payload(self) -> bytes:
        return self.body

    # -- body builders --------------------------------------------------

    @staticmethod
    def _path_name(path: str, name: str) -> bytes:
        return Writer().text(path, nul=True).u16(0).text(name, nul=True).bytes()

    @classmethod
    def get(cls, path: str, name: str, fhandle: bytes = c.NULL_FHANDLE) -> Fattr:
        return cls(c.kXR_fattrGet, cls._path_name(path, name), numattr=1, fhandle=fhandle)

    @classmethod
    def delete(cls, path: str, name: str, fhandle: bytes = c.NULL_FHANDLE) -> Fattr:
        return cls(c.kXR_fattrDel, cls._path_name(path, name), numattr=1, fhandle=fhandle)

    @classmethod
    def set(
        cls,
        path: str,
        name: str,
        value: bytes,
        create_only: bool = False,
        fhandle: bytes = c.NULL_FHANDLE,
    ) -> Fattr:
        body = cls._path_name(path, name) + Writer().u32(len(value)).raw(value).bytes()
        return cls(
            c.kXR_fattrSet,
            body,
            numattr=1,
            options=c.kXR_fattrIsNew if create_only else 0,
            fhandle=fhandle,
        )

    @classmethod
    def list(cls, path: str, values: bool = False, fhandle: bytes = c.NULL_FHANDLE) -> Fattr:
        return cls(
            c.kXR_fattrList,
            Writer().text(path, nul=True).bytes(),
            options=c.kXR_fattrAData if values else 0,
            fhandle=fhandle,
        )

    def __repr__(self) -> str:
        return f"Fattr(subcode={self.subcode}, numattr={self.numattr})"


class Sigver(Request):
    """``kXR_sigver`` - the signature frame that prefixes a signed request."""

    __slots__ = ("expectrid", "seqno", "signature", "crypto", "nodata")
    opcode = c.kXR_sigver

    def __init__(
        self,
        expectrid: int,
        seqno: int,
        signature: bytes,
        crypto: int | None = None,
        nodata: bool = False,
    ) -> None:
        self.expectrid = expectrid
        self.seqno = seqno
        self.signature = signature
        self.crypto = c.kXR_SHA256_sig if crypto is None else crypto
        self.nodata = nodata

    def params(self, w: Writer) -> None:
        w.u16(self.expectrid).u8(0).u8(c.kXR_nodata_sig if self.nodata else 0)
        w.u64(self.seqno).u8(self.crypto).zeros(3)

    def payload(self) -> bytes:
        return self.signature

    def __repr__(self) -> str:
        return f"Sigver(expectrid={c.request_name(self.expectrid)}, seqno={self.seqno})"
