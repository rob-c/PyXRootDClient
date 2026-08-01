"""Response body decoders.

Each class decodes the bytes *after* the 8-byte ``ServerResponseHdr``. The
header itself is handled by :mod:`xrd.proto.frames`; status dispatch is the
:class:`~xrd.proto.machine.SessionMachine`'s job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..errors import ProtocolError
from ..flags import StatInfoFlags
from ..types import (
    CheckpointInfo,
    ChecksumInfo,
    DirEntry,
    LocationInfo,
    PrepareStatus,
    ProtocolInfo,
    SpaceInfo,
    StatInfo,
    VFSInfo,
)
from . import constants as c
from .buffer import Reader

__all__ = [
    "ErrorInfo", "RedirectInfo", "WaitInfo", "AttnInfo", "StatusInfo",
    "LoginInfo", "ReadVSegment", "FattrItem", "FattrResult",
    "parse_protocol", "parse_login", "parse_bind", "parse_stat", "parse_statvfs", "parse_statx",
    "parse_dirlist", "parse_locate", "parse_open", "parse_checksum",
    "parse_checkpoint", "parse_readlink", "parse_space", "parse_prepare_status",
    "parse_error", "parse_redirect", "parse_wait", "parse_waitresp", "parse_attn",
    "parse_status", "parse_readv", "parse_fattr",
]


# --------------------------------------------------------------------------
# Control-status bodies
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """Body of a ``kXR_error`` response."""

    code: int
    message: str


def parse_error(data: bytes) -> ErrorInfo:
    r = Reader(data, "kXR_error")
    return ErrorInfo(r.i32(), r.cstring())


@dataclass(frozen=True, slots=True)
class RedirectInfo:
    """Body of a ``kXR_redirect`` response."""

    host: str
    port: int
    token: str = ""

    @property
    def url(self) -> str:
        scheme = "roots" if self.port < 0 else "root"
        return f"{scheme}://{self.host}:{abs(self.port)}/"


def parse_redirect(data: bytes) -> RedirectInfo:
    r = Reader(data, "kXR_redirect")
    port = r.i32()
    target = r.rest().split(b"\x00", 1)[0].decode("utf-8", "replace")
    host, sep, token = target.partition("?")
    if not host:
        raise ProtocolError("kXR_redirect names no host to redirect to")
    return RedirectInfo(host, port, token if sep else "")


@dataclass(frozen=True, slots=True)
class WaitInfo:
    """Body of a ``kXR_wait`` or ``kXR_waitresp`` response."""

    seconds: int
    message: str = ""


def parse_wait(data: bytes) -> WaitInfo:
    r = Reader(data, "kXR_wait")
    return WaitInfo(r.i32(), r.cstring())


def parse_waitresp(data: bytes) -> WaitInfo:
    r = Reader(data, "kXR_waitresp")
    return WaitInfo(r.i32())


@dataclass(frozen=True, slots=True)
class AttnInfo:
    """Body of an unsolicited ``kXR_attn`` response."""

    action: int
    params: bytes = b""

    @property
    def message(self) -> str:
        return self.params.split(b"\x00", 1)[0].decode("utf-8", "replace")


def parse_attn(data: bytes) -> AttnInfo:
    r = Reader(data, "kXR_attn")
    return AttnInfo(r.i32(), r.rest())


@dataclass(frozen=True, slots=True)
class StatusInfo:
    """Body of a ``kXR_status`` response.

    ``crc32c`` covers every byte of the body after itself. ``info`` is the
    request-specific tail (for ``kXR_pgread`` an 8-byte offset); ``dlen`` is
    the length of the raw data that follows the body on the wire.
    """

    crc32c: int
    streamid: int
    requestid: int
    resptype: int
    dlen: int
    info: bytes = b""

    @property
    def is_final(self) -> bool:
        return self.resptype == c.kXR_FinalResult

    @property
    def offset(self) -> int:
        """Data offset, for paged I/O responses."""
        if len(self.info) < 8:
            raise ProtocolError("kXR_status body carries no offset")
        return Reader(self.info, "kXR_status.info").i64()


def parse_status(data: bytes) -> StatusInfo:
    r = Reader(data, "kXR_status")
    crc = r.u32()
    streamid = r.u16()
    requestid = r.u8() + c.kXR_1stRequest
    resptype = r.u8()
    r.skip(4)
    dlen = r.i32()
    return StatusInfo(crc, streamid, requestid, resptype, dlen, r.rest())


# --------------------------------------------------------------------------
# Session bring-up
# --------------------------------------------------------------------------


def parse_protocol(data: bytes) -> ProtocolInfo:
    """``kXR_protocol`` - version, flags and the optional security requirements."""
    r = Reader(data, "kXR_protocol")
    version = r.i32()
    flags = r.u32()
    seclvl = c.kXR_secNone
    secver = 0
    secopt = 0
    overrides: dict[int, int] = {}
    if r.remaining >= 6:
        tag = r.u8()
        if tag != ord("S"):
            raise ProtocolError(f"kXR_protocol security block has tag {tag!r}, expected 'S'")
        r.skip(1)
        secver = r.u8()
        secopt = r.u8()
        seclvl = r.u8()
        for _ in range(r.u8()):
            if r.remaining < 2:
                break
            # Read into locals: a subscript assignment evaluates its
            # right-hand side first, which would swap opcode and level.
            opcode = r.u8() + c.kXR_1stRequest
            overrides[opcode] = r.u8()
    return ProtocolInfo(
        version=version,
        flags=flags,
        security_level=seclvl,
        security_version=secver,
        security_options=secopt,
        security_overrides=overrides,
    )


@dataclass(frozen=True, slots=True)
class LoginInfo:
    """``kXR_login`` - the session id plus the server's security continuation."""

    sessid: bytes
    sec: str = ""

    @property
    def mechanisms(self) -> tuple[str, ...]:
        """Protocol names offered by the server, most preferred first."""
        return tuple(
            part[2:].split(",", 1)[0]
            for part in self.sec.split("&")
            if part.startswith("P=")
        )


def parse_login(data: bytes) -> LoginInfo:
    r = Reader(data, "kXR_login")
    sessid = r.bytes(min(c.SESSION_ID_LEN, r.remaining))
    return LoginInfo(sessid, r.rest().split(b"\x00", 1)[0].decode("utf-8", "replace"))


def parse_bind(data: bytes) -> int:
    """``kXR_bind`` - the path id the server assigned to this connection.

    Zero is not a path: it is how every request spells "the control link", so
    a server that answers with it has told us nothing usable.
    """
    r = Reader(data, "kXR_bind")
    pathid = r.u8()
    if pathid == 0:
        raise ProtocolError("kXR_bind returned path id 0, which is the control link")
    return pathid


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def _stat_fields(text: str, path: str) -> StatInfo:
    parts = text.split()
    if len(parts) < 4:
        raise ProtocolError(f"kXR_stat returned {len(parts)} fields, expected >= 4: {text!r}")
    mtime = int(parts[3])
    return StatInfo(
        id=parts[0],
        st_size=int(parts[1]),
        flags=StatInfoFlags(int(parts[2])),
        st_mtime=mtime,
        st_ctime=mtime,
        st_atime=mtime,
        path=path,
    )


def parse_stat(data: bytes, path: str = "") -> StatInfo:
    """``kXR_stat`` - a space-separated ``id size flags modtime`` line."""
    return _stat_fields(data.split(b"\x00", 1)[0].decode("utf-8", "replace"), path)


def parse_statvfs(data: bytes) -> VFSInfo:
    """``kXR_stat`` with ``kXR_vfs``."""
    parts = data.split(b"\x00", 1)[0].decode("utf-8", "replace").split()
    if len(parts) < 6:
        raise ProtocolError(f"kXR_stat vfs returned {len(parts)} fields, expected 6")
    return VFSInfo(
        nodes_rw=int(parts[0]),
        free_rw=int(parts[1]),
        utilization_rw=int(parts[2]),
        nodes_staging=int(parts[3]),
        free_staging=int(parts[4]),
        utilization_staging=int(parts[5]),
    )


def parse_statx(data: bytes) -> tuple[StatInfoFlags, ...]:
    """``kXR_statx`` - one flags byte per requested path."""
    return tuple(StatInfoFlags(byte) for byte in data)


def _checked(name: str, path: str) -> str:
    """Refuse a listing entry that is not a single path component.

    Every consumer joins these names onto a directory - ``walk`` to recurse,
    ``copy_tree`` to build a local destination - so a server that answers
    ``../../.ssh/authorized_keys`` would have a recursive download write
    outside the directory it was pointed at. Names are components; anything
    else is a broken or hostile server and neither deserves the benefit of
    the doubt.
    """
    if "/" in name or name == "..":
        raise ProtocolError(f"kXR_dirlist entry {name!r} in {path!r} is not a name")
    return name


def parse_dirlist(data: bytes, path: str = "", with_stat: bool = True) -> list[DirEntry]:
    """``kXR_dirlist``.

    Plain mode is one name per line. With ``kXR_dstat`` the server emits a
    leading ``".\\n<stat>"`` entry followed by ``name\\n<stat>`` pairs; the
    dot entry describes the directory itself and is dropped.

    The dot entry is also how a server says it honoured the request: one that
    ignores ``kXR_dstat`` answers with plain names, and reading those in pairs
    would pass every second name off as a stat line. So the reply decides,
    not the flag we sent.
    """
    text = data.split(b"\x00", 1)[0].decode("utf-8", "replace")
    lines = [ln for ln in text.split("\n") if ln]
    if not with_stat or lines[:1] != ["."]:
        return [DirEntry(name=_checked(n, path), parent=path) for n in lines]

    entries: list[DirEntry] = []
    for name, statline in zip(lines[::2], lines[1::2], strict=False):
        if name == ".":
            continue
        _checked(name, path)
        entries.append(
            DirEntry(
                name=name,
                parent=path,
                stat=_stat_fields(statline, f"{path.rstrip('/')}/{name}" if path else name),
            )
        )
    return entries


def parse_locate(data: bytes) -> list[LocationInfo]:
    """``kXR_locate`` - space-separated ``XY<host:port>`` tokens.

    ``X`` is the server type (``S`` server, ``M`` manager, lower case when
    pending) and ``Y`` the access mode (``r`` read-only, ``w`` writable).
    """
    text = data.split(b"\x00", 1)[0].decode("utf-8", "replace")
    out: list[LocationInfo] = []
    for token in text.split():
        if len(token) < 3:
            continue
        out.append(LocationInfo(address=token[2:], type=token[0], access=token[1]))
    return out


def parse_open(data: bytes, path: str = "") -> tuple[bytes, StatInfo | None]:
    """``kXR_open`` - the handle plus, with ``kXR_retstat``, a stat line."""
    r = Reader(data, "kXR_open")
    fhandle = r.bytes(c.FHANDLE_LEN)
    if r.remaining < 8:
        return fhandle, None
    r.skip(8)  # cpsize[4] cptype[4]
    tail = r.rest().split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    return fhandle, _stat_fields(tail, path) if tail else None


def parse_checksum(data: bytes) -> ChecksumInfo:
    """``kXR_query`` with ``kXR_Qcksum`` - ``"<type> <value>"``."""
    text = data.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    name, _, value = text.partition(" ")
    if not value:
        raise ProtocolError(f"malformed checksum response: {text!r}")
    return ChecksumInfo(algorithm=name.lower(), value=value.strip().lower())


def _truth(value: object) -> bool:
    """One field of the prepare JSON as a boolean, whichever way it was spelt.

    The document is generated by the server, and versions of it have written
    these flags as ``true``, as ``1`` and as ``"1"``; all three mean yes, and
    a client that only understood one of them would report a staged file as
    still on tape.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def parse_prepare_status(data: bytes) -> list[PrepareStatus]:
    """``kXR_query`` with ``kXR_QPrep`` - the staging state of each file asked about.

    Unlike every other query, this one answers with a JSON document rather
    than a packed structure or a CGI string. Fields the server did not send
    keep the dataclass default, and ones it sent that this does not know about
    are ignored, because the format has gained keys between releases.
    """
    text = data.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    try:
        document = json.loads(text or "{}")
        entries = document.get("responses") if isinstance(document, dict) else document
        return [
            PrepareStatus(
                path=str(entry.get("path", "")),
                exists=_truth(entry.get("path_exists")),
                on_tape=_truth(entry.get("on_tape")),
                online=_truth(entry.get("online")),
                requested=_truth(entry.get("requested")),
                has_request_id=_truth(entry.get("has_reqid")),
                requested_at=str(entry.get("req_time", "")),
                error=str(entry.get("error_text", "")),
            )
            for entry in entries or ()
        ]
    except (ValueError, AttributeError, TypeError) as exc:
        raise ProtocolError(f"prepare status is not a JSON document: {text[:120]!r}") from exc


def parse_space(data: bytes) -> SpaceInfo:
    """``kXR_query`` with ``kXR_Qspace`` - ``oss.*`` CGI, in bytes.

    The reply is a query string rather than a structure, and a server is free
    to answer with a subset of the keys: a pool with no quota omits
    ``oss.quota`` rather than sending a zero. Missing keys therefore keep the
    dataclass default, which for the quota is ``-1`` - "no limit" - because a
    zero would read as a pool nobody may write a byte to.
    """
    text = data.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    fields: dict[str, str] = {}
    for pair in text.replace("\n", "&").split("&"):
        key, sep, value = pair.partition("=")
        if sep:
            fields[key.strip()] = value.strip()

    def number(key: str, default: int) -> int:
        raw = fields.get(key)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ProtocolError(f"kXR_Qspace {key}={raw!r} is not a number") from exc

    return SpaceInfo(
        name=fields.get("oss.cgroup", ""),
        total=number("oss.space", 0),
        free=number("oss.free", 0),
        largest_free=number("oss.maxf", 0),
        used=number("oss.used", 0),
        quota=number("oss.quota", -1),
    )


def parse_checkpoint(data: bytes) -> CheckpointInfo:
    """``kXR_chkpoint`` with ``kXR_ckpQuery`` - capacity then bytes used."""
    r = Reader(data, "kXR_ckpQuery")
    return CheckpointInfo(capacity=r.u32(), used=r.u32())


def parse_readlink(data: bytes) -> str:
    """``kXR_readlink`` (vendor extension) - the target, NUL-padded."""
    target = data.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    if not target:
        raise ProtocolError("kXR_readlink named no target")
    return target


# --------------------------------------------------------------------------
# Vector and paged I/O
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReadVSegment:
    """One element of a ``kXR_readv`` response."""

    fhandle: bytes
    offset: int
    data: bytes


def parse_readv(data: bytes) -> list[ReadVSegment]:
    """``kXR_readv`` - repeated ``readahead_list`` headers each followed by data."""
    r = Reader(data, "kXR_readv")
    out: list[ReadVSegment] = []
    while r.remaining:
        fhandle = r.bytes(c.FHANDLE_LEN)
        length = r.i32()
        offset = r.i64()
        if length < 0:
            raise ProtocolError(f"kXR_readv segment declares a negative length of {length}")
        out.append(ReadVSegment(fhandle, offset, r.bytes(length)))
    return out


# --------------------------------------------------------------------------
# Extended attributes
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FattrItem:
    """One attribute in a ``kXR_fattr`` response."""

    name: str
    code: int = 0
    value: bytes | None = None


@dataclass(frozen=True, slots=True)
class FattrResult:
    """``kXR_fattr`` response body."""

    errors: int = 0
    items: list[FattrItem] = field(default_factory=list)

    def as_dict(self) -> dict[str, bytes]:
        return {i.name: i.value for i in self.items if i.value is not None}


def _read_name(r: Reader) -> str:
    raw = bytearray()
    while True:
        b = r.u8()
        if b == 0:
            break
        raw.append(b)
    return raw.decode("utf-8", "replace")


def parse_fattr(data: bytes, values: bool = True) -> FattrResult:
    """``kXR_fattr`` - ``nerrs[1] nattr[1]`` then ``rc[2] name\\0 [len[4] value]``."""
    if len(data) < 2:
        return FattrResult()
    r = Reader(data, "kXR_fattr")
    errors = r.u8()
    count = r.u8()
    items: list[FattrItem] = []
    for _ in range(count):
        if r.remaining < 3:
            break
        code = r.u16()
        name = _read_name(r)
        value: bytes | None = None
        if values and r.remaining >= 4:
            value = r.bytes(r.i32())
        items.append(FattrItem(name, code, value))
    return FattrResult(errors, items)
