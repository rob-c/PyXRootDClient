"""Namespace operations against one storage endpoint.

:class:`FileSystem` is the explicit, per-operation layer. It reads like
``os`` and ``shutil`` because those are the names Python programmers already
know: :meth:`stat`, :meth:`listdir`, :meth:`makedirs`, :meth:`remove`,
:meth:`rename`, :meth:`walk`.

    >>> fs = FileSystem("root://eos.example.org")
    >>> fs.makedirs("/store/user/me/out", exist_ok=True)
    >>> for entry in fs.scandir("/store/user/me"):
    ...     print(entry.name, entry.stat.st_size)
"""

from __future__ import annotations

import errno
import os
import posixpath
import re
import urllib.parse
from collections.abc import Iterator, Sequence
from typing import IO, Any

from .._log import get_logger
from ..config import Config
from ..errors import (
    AttrNotFoundError,
    ExistsError,
    IOError_,
    NotFoundError,
    PermissionError_,
    ProtocolError,
    ServerError,
    kXR_AttrNotFound,
    kXR_IOError,
    kXR_ItExists,
    kXR_NotAuthorized,
    kXR_NotFound,
)
from ..flags import Access, DirListFlags, LocateFlags, OpenFlags, PrepareFlags, QueryCode
from ..proto import constants as c
from ..proto import requests as r
from ..proto import responses as rp
from ..session.router import Router
from ..types import ChecksumInfo, DirEntry, LocationInfo, ProtocolInfo, StatInfo, VFSInfo
from ..url import XRootDURL, parse

__all__ = ["FileSystem"]

_log = get_logger(__name__)

_MAGIC = "*?["


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob pattern, slash-aware.

    :func:`fnmatch.fnmatch` is no use here because its ``*`` swallows path
    separators, which makes ``/store/*/file`` match three levels down and
    ``**`` mean nothing in particular. This is the pathlib reading: ``**/``
    is zero or more directories, everything else stays in its component.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            end = pattern.find("]", i + 1)
            if end < 0:  # an unclosed bracket is a literal one, as in fnmatch
                out.append(re.escape("["))
                i += 1
                continue
            body = pattern[i + 1 : end].replace("\\", "\\\\")
            out.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def _literal_prefix(pattern: str) -> str:
    """The deepest directory of ``pattern`` that contains no wildcard."""
    parts = pattern.split("/")[:-1]
    keep: list[str] = []
    for part in parts:
        if any(ch in part for ch in _MAGIC):
            break
        keep.append(part)
    return "/".join(keep) or "/"


def _split_cgi(path: str) -> tuple[str, str]:
    """A resolved path split into the path itself and its opaque suffix."""
    base, sep, cgi = path.partition("?")
    return base, (sep + cgi if sep else "")


def _cgi(explicit: str, inherited: dict[str, str]) -> str:
    """The opaque suffix for a path: what was asked for, plus what was implied."""
    if not inherited:
        return f"?{explicit}" if explicit else ""
    named = {key for key, _ in urllib.parse.parse_qsl(explicit, keep_blank_values=True)}
    extra = urllib.parse.urlencode({k: v for k, v in inherited.items() if k not in named})
    if not extra:
        return f"?{explicit}" if explicit else ""
    return f"?{explicit}&{extra}" if explicit else f"?{extra}"


class FileSystem:
    """Namespace and administrative operations on one endpoint."""

    def __new__(cls, url: str | XRootDURL, config: Config | None = None) -> FileSystem:
        """Pick the implementation the scheme calls for.

        ``root://`` is this class; ``https://`` and the WebDAV spellings are
        :class:`~xrd.http.HTTPFileSystem`, which offers the same methods. The
        dispatch lives in ``__new__`` for the same reason
        :class:`pathlib.Path`'s does: callers should name what they want, not
        which implementation provides it.
        """
        if cls is FileSystem and parse(url).is_http:
            from ..http.dav import HTTPFileSystem

            return object.__new__(HTTPFileSystem)
        return object.__new__(cls)

    def __init__(self, url: str | XRootDURL, config: Config | None = None) -> None:
        self.url = parse(url) if isinstance(url, str) else url
        self.config = config or Config()
        self._router = Router(self.url, self.config)

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _abs(self, path: str) -> str:
        """Resolve ``path`` against the URL's path, carrying the opaque data.

        Opaque data is how ``root://`` passes an authorisation token, so a
        token on the filesystem's own URL belongs on every path derived from
        it - and a caller who spells the same key out for one call means that
        one, so their value wins.
        """
        base, sep, cgi = path.partition("?")
        if not base.startswith("/"):
            base = posixpath.join(self.url.path or "/", base)
        return posixpath.normpath(base) + _cgi(cgi if sep else "", self.url.query)

    def _url_for(self, path: str) -> XRootDURL:
        """The URL of ``path`` under this filesystem, with the CGI on it once.

        :meth:`_abs` has already folded in whatever the filesystem's own URL
        carried, so the query is cleared rather than applied a second time.
        """
        return self.url.evolve(path=self._abs(path), query={})

    @property
    def endpoint(self) -> str:
        return self._router.endpoint

    def close(self) -> None:
        self._router.close()

    def __enter__(self) -> FileSystem:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"FileSystem({str(self.url)!r})"

    def __fspath__(self) -> str:
        return str(self.url)

    # ------------------------------------------------------------------
    # Interrogation
    # ------------------------------------------------------------------

    def ping(self) -> None:
        """Round-trip the server. Raises if it is unhealthy."""
        self._router.execute(r.Ping())

    def protocol(self) -> ProtocolInfo:
        """Capabilities of the endpoint, from the connection's negotiation."""
        info = self._router.session.protocol
        assert isinstance(info, ProtocolInfo)
        return info

    def stat(self, path: str) -> StatInfo:
        """``kXR_stat``. Raises :class:`FileNotFoundError` if absent."""
        target = self._abs(path)
        res = self._router.execute(r.Stat(target), path=target)
        return rp.parse_stat(res.data, target)

    def statvfs(self, path: str = "/") -> VFSInfo:
        """Space and staging utilisation, in ``os.statvfs`` spirit."""
        target = self._abs(path)
        res = self._router.execute(r.StatVFS(target), path=target)
        return rp.parse_statvfs(res.data)

    def statx(self, paths: Sequence[str]) -> list[StatInfo]:
        """Flags-only stat of many paths in one round trip."""
        targets = [self._abs(p) for p in paths]
        res = self._router.execute(r.Statx(targets))
        flags = rp.parse_statx(res.data)
        if len(flags) != len(targets):
            raise ProtocolError(
                f"statx returned {len(flags)} flags for {len(targets)} paths"
            )
        return [StatInfo(flags=f, path=p) for f, p in zip(flags, targets, strict=True)]

    def exists(self, path: str) -> bool:
        """``True`` if ``path`` resolves. Never raises for a missing file."""
        try:
            self.stat(path)
        except NotFoundError:
            return False
        return True

    def isdir(self, path: str) -> bool:
        try:
            return self.stat(path).is_dir()
        except NotFoundError:
            return False

    def isfile(self, path: str) -> bool:
        try:
            return self.stat(path).is_file()
        except NotFoundError:
            return False

    def getsize(self, path: str) -> int:
        return self.stat(path).st_size

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def scandir(self, path: str = "", *, flags: DirListFlags = DirListFlags.STAT) -> list[DirEntry]:
        """Directory entries with stat information attached."""
        target = self._abs(path or "/")
        with_stat = bool(flags & DirListFlags.STAT)
        res = self._router.execute(
            r.Dirlist(target, int(flags) & ~int(DirListFlags.RECURSIVE)), path=target
        )
        return rp.parse_dirlist(res.data, target, with_stat=with_stat)

    def listdir(self, path: str = "") -> list[str]:
        """Entry names only, like :func:`os.listdir`."""
        return [e.name for e in self.scandir(path, flags=DirListFlags.NONE)]

    def iterdir(self, path: str = "") -> Iterator[DirEntry]:
        """Iterate entries. The protocol has no cursor, so this reads it all."""
        yield from self.scandir(path)

    def walk(
        self, top: str = "", *, topdown: bool = True, onerror: object = None
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        """``os.walk`` over the remote namespace."""
        root = self._abs(top or "/")
        try:
            entries = self.scandir(root)
        except OSError as exc:
            if callable(onerror):
                onerror(exc)
            return
        # Opaque data belongs on the request, not in the name of a directory:
        # what is yielded is a path, and what descends carries the token.
        base, cgi = _split_cgi(root)
        dirs = [e.name for e in entries if e.is_dir()]
        files = [e.name for e in entries if not e.is_dir()]
        if topdown:
            yield base, dirs, files
        for name in list(dirs):
            child = posixpath.join(base, name) + cgi
            yield from self.walk(child, topdown=topdown, onerror=onerror)
        if not topdown:
            yield base, dirs, files

    def glob(self, pattern: str, *, root: str = "") -> Iterator[str]:
        """Match ``pattern`` against the namespace, absolute paths out.

        The semantics are :meth:`pathlib.Path.glob`'s, because that is what a
        caller writing ``**/*.root`` means: ``*`` and ``?`` stay inside one
        path component, ``**`` crosses them, and directories match as well as
        files. A relative pattern is taken from ``root``, an absolute one as
        it stands.

        Only the directories a pattern can actually reach are listed: the
        literal prefix is walked, not the whole namespace, so
        ``glob("/store/mc/**/*.root")`` never asks about ``/store/data``.
        """
        base, cgi = _split_cgi(self._abs(root or "/"))
        target = pattern if pattern.startswith("/") else posixpath.join(base, pattern)
        match = _glob_regex(target).fullmatch
        start = _literal_prefix(target)
        deep = "**" in posixpath.basename(target)  # ``/d/**.root`` still descends
        if not deep and start == posixpath.dirname(target):  # magic in the last component only
            for entry in self.scandir(start + cgi, flags=DirListFlags.NONE):
                full = posixpath.join(start, entry.name)
                if match(full):
                    yield full
            return
        for dirpath, dirs, files in self.walk(start + cgi):
            for name in sorted(dirs + files):
                full = posixpath.join(dirpath, name)
                if match(full):
                    yield full

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mkdir(
        self,
        path: str,
        mode: int = 0o755,
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        """Create a directory, with ``pathlib.Path.mkdir`` semantics."""
        target = self._abs(path)
        try:
            self._router.execute(
                r.Mkdir(target, int(mode) & 0o777, mkpath=parents), path=target
            )
        except FileExistsError:
            if not exist_ok:
                raise

    def makedirs(self, path: str, mode: int = 0o755, exist_ok: bool = False) -> None:
        """:func:`os.makedirs`."""
        self.mkdir(path, mode, parents=True, exist_ok=exist_ok)

    def rmdir(self, path: str) -> None:
        """Remove an empty directory."""
        target = self._abs(path)
        self._router.execute(r.Rmdir(target), path=target)

    def remove(self, path: str) -> None:
        """Remove a file."""
        target = self._abs(path)
        self._router.execute(r.Rm(target), path=target)

    #: :func:`os.unlink` is the same call.
    unlink = remove

    def rmtree(self, path: str, *, ignore_errors: bool = False) -> None:
        """Recursively remove a directory. There is no server-side primitive."""
        target = self._abs(path)
        _, cgi = _split_cgi(target)
        for dirpath, dirs, files in self.walk(target, topdown=False):
            for name in files:
                try:
                    self.remove(posixpath.join(dirpath, name) + cgi)
                except OSError:
                    if not ignore_errors:
                        raise
            for name in dirs:
                try:
                    self.rmdir(posixpath.join(dirpath, name) + cgi)
                except OSError:
                    if not ignore_errors:
                        raise
        try:
            self.rmdir(target)
        except OSError:
            if not ignore_errors:
                raise

    def rename(self, src: str, dst: str) -> None:
        """Rename within the same storage element."""
        source, destination = self._abs(src), self._abs(dst)
        self._router.execute(r.Mv(source, destination), path=source)

    #: ``shutil``'s spelling.
    move = rename

    def symlink(self, target: str, link: str) -> None:
        """``os.symlink`` order: make ``link`` point at ``target``.

        A vendor extension (``kXR_symlink``), not part of XProtocol - the same
        opcode XRootD.jl and XrdRust use. A server that has not been taught it
        answers :class:`~xrd.errors.UnsupportedError`, which is the honest
        answer for a namespace that has no links.
        """
        source, destination = self._abs(target), self._abs(link)
        self._router.execute(r.Symlink(source, destination), path=destination)

    def link(self, src: str, dst: str) -> None:
        """``os.link`` order: hard-link ``dst`` to ``src``. Vendor extension."""
        source, destination = self._abs(src), self._abs(dst)
        self._router.execute(r.Link(source, destination), path=destination)

    #: ``os``'s spelling of the same call.
    hardlink = link

    def readlink(self, path: str) -> str:
        """What a symbolic link points at. Vendor extension."""
        target = self._abs(path)
        result = self._router.execute(r.Readlink(target), path=target)
        return rp.parse_readlink(result.data)

    def chmod(self, path: str, mode: int) -> None:
        target = self._abs(path)
        self._router.execute(r.Chmod(target, int(mode) & 0o777), path=target)

    def truncate(self, path: str, size: int) -> None:
        """Resize a file by path, without opening it."""
        target = self._abs(path)
        self._router.execute(r.Truncate(target, size), path=target)

    def touch(self, path: str, *, exist_ok: bool = True) -> None:
        """Create an empty file if it is not there.

        ``kXR_new`` is the only flag that creates without truncating, and a
        server refuses it when the file exists - which is precisely the case
        ``exist_ok`` is about, so it is caught rather than pre-checked. The
        mtime of an existing file is left alone: no request in the protocol
        moves it, and quietly rewriting the file to fake one would be worse
        than not doing it.
        """
        from .file import File

        flags = OpenFlags.NEW | OpenFlags.UPDATE | OpenFlags.MAKEPATH
        fh = File(self._url_for(path), self.config, router=self._router)
        try:
            fh.open(flags=flags, mode=Access.OWNER_READ | Access.OWNER_WRITE)
        except ExistsError:
            if not exist_ok:
                raise
            return
        fh.close()

    # ------------------------------------------------------------------
    # Query, checksums, staging, location
    # ------------------------------------------------------------------

    def query(self, code: QueryCode | int, args: str = "") -> bytes:
        """Raw ``kXR_query``."""
        res = self._router.execute(r.Query(int(code), args), path=args)
        return res.data

    def checksum(self, path: str, algorithm: str | None = None) -> ChecksumInfo:
        """Server-computed checksum. ``algorithm`` selects via CGI when given."""
        target = self._abs(path)
        if algorithm:
            target += ("&" if "?" in target else "?") + f"cks.type={algorithm}"
        res = self._router.execute(r.Query(c.kXR_Qcksum, target), path=target)
        return rp.parse_checksum(res.data)

    def query_config(self, *names: str) -> dict[str, str]:
        """``kXR_query`` config lookup; one value per requested name.

        Named ``query_config`` and not ``config`` because :attr:`config` is
        this filesystem's own :class:`~xrd.config.Config`.

        A name the server has no value for is absent from the result, the
        way a missing key is absent from a :class:`dict`. Splitting on
        ``\\n`` rather than by lines keeps the remaining names lined up with
        their values when an earlier one comes back empty.
        """
        wanted = list(names) or ["version"]
        res = self._router.execute(r.Query(c.kXR_Qconfig, "\n".join(wanted)))
        body = res.data.split(b"\x00", 1)[0].decode("utf-8", "replace")
        values = body.split("\n")
        return {name: value for name, value in zip(wanted, values, strict=False) if value}

    def locate(
        self, path: str, *, flags: LocateFlags = LocateFlags.NONE
    ) -> list[LocationInfo]:
        """Which servers hold ``path``."""
        target = self._abs(path)
        res = self._router.execute(r.Locate(target, int(flags)), path=target)
        return rp.parse_locate(res.data)

    def deep_locate(self, path: str) -> list[LocationInfo]:
        """Locate, resolving managers down to the servers behind them."""
        seen: dict[str, LocationInfo] = {}
        pending = list(self.locate(path))
        while pending:
            loc = pending.pop()
            known = seen.get(loc.address)
            if known is not None:
                # A supervisor answers as a manager to the tier above it and as
                # a server to the tier below; keeping only the first answer
                # would drop a node that does hold the file.
                if known.is_manager and not loc.is_manager:
                    seen[loc.address] = loc
                continue
            seen[loc.address] = loc
            if loc.is_manager:
                child = FileSystem(self.url.evolve(host=loc.host, port=loc.port), self.config)
                try:
                    pending.extend(child.locate(path))
                except OSError:
                    pass
                finally:
                    child.close()
        return [v for v in seen.values() if not v.is_manager]

    def prepare(
        self,
        paths: Sequence[str],
        *,
        flags: PrepareFlags = PrepareFlags.STAGE,
        priority: int = 0,
    ) -> str:
        """Stage, evict or co-locate files. Returns the request handle."""
        targets = [self._abs(p) for p in paths]
        res = self._router.execute(r.Prepare(targets, int(flags), priority))
        return res.data.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()

    def evict(self, paths: Sequence[str]) -> str:
        """Ask the server to drop its cached copies."""
        return self.prepare(paths, flags=PrepareFlags.EVICT)

    # ------------------------------------------------------------------
    # Extended attributes
    # ------------------------------------------------------------------

    def getxattr(self, path: str, name: str) -> bytes:
        """One attribute value, in ``os.getxattr`` spirit."""
        target = self._abs(path)
        res = self._router.execute(r.Fattr.get(target, name), path=target)
        result = rp.parse_fattr(res.data)
        for item in result.items:
            if item.code == 0 and item.value is not None:
                return item.value
        raise _attr_error(name, target)

    def setxattr(self, path: str, name: str, value: bytes, *, create_only: bool = False) -> None:
        target = self._abs(path)
        res = self._router.execute(
            r.Fattr.set(target, name, value, create_only=create_only), path=target
        )
        _check_fattr(rp.parse_fattr(res.data, values=False), target)

    def removexattr(self, path: str, name: str) -> None:
        target = self._abs(path)
        res = self._router.execute(r.Fattr.delete(target, name), path=target)
        _check_fattr(rp.parse_fattr(res.data, values=False), target)

    def listxattr(self, path: str) -> list[str]:
        target = self._abs(path)
        res = self._router.execute(r.Fattr.list(target), path=target)
        return [item.name for item in rp.parse_fattr(res.data, values=False).items]

    def xattrs(self, path: str) -> dict[str, bytes]:
        """Every attribute and its value, in one round trip."""
        target = self._abs(path)
        res = self._router.execute(r.Fattr.list(target, values=True), path=target)
        return rp.parse_fattr(res.data).as_dict()

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def open(
        self,
        path: str,
        mode: str = "rb",
        *,
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        posc: bool = False,
    ) -> IO[Any]:
        """Open a remote file with :func:`open`'s signature.

        The element type follows the mode, so a caller that wants ``bytes``
        or ``str`` statically should reach for :func:`xrd.open`, whose
        overloads know the difference.
        """
        from ..io import open_url

        return open_url(
            self._url_for(path),
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            config=self.config,
            router=self._router,
            posc=posc,
        )

    def read_bytes(self, path: str) -> bytes:
        """Whole-file read, like :meth:`pathlib.Path.read_bytes`."""
        with self.open(path, "rb") as fh:
            data: bytes = fh.read()
        return data

    def write_bytes(self, path: str, data: bytes) -> int:
        """Whole-file write, like :meth:`pathlib.Path.write_bytes`."""
        with self.open(path, "wb") as fh:
            written: int = fh.write(data)
        return written

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        with self.open(path, "r", encoding=encoding) as fh:
            text: str = fh.read()
        return text

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> int:
        with self.open(path, "w", encoding=encoding) as fh:
            written: int = fh.write(text)
        return written


def _attr_error(name: str, path: str) -> ServerError:
    return AttrNotFoundError(kXR_AttrNotFound, f"no attribute {name!r}", path=path)


#: ``kXR_fattr`` reports one ``errno`` per attribute, not a ``kXR_`` code, and
#: a whole-request ``kXR_ok`` can still carry a failed attribute inside it.
_ATTR_ERRNO: dict[int, tuple[type[ServerError], int]] = {
    errno.EEXIST: (ExistsError, kXR_ItExists),
    errno.ENODATA: (AttrNotFoundError, kXR_AttrNotFound),
    errno.ENOENT: (NotFoundError, kXR_NotFound),
    errno.EACCES: (PermissionError_, kXR_NotAuthorized),
}


def _check_fattr(result: rp.FattrResult, path: str) -> None:
    """Raise for the first attribute the server refused."""
    for item in result.items:
        if item.code:
            kind, code = _ATTR_ERRNO.get(item.code, (IOError_, kXR_IOError))
            raise kind(code, f"{os.strerror(item.code)}: attribute {item.name!r}", path=path)
