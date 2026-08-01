"""A fake XRootD server, backed by a dictionary.

:class:`FakeServer` speaks enough of the binary protocol to exercise a real
client end to end - handshake, login, the namespace, file I/O, vector and
paged I/O, checksums, xattrs - over a loopback socket, with no daemon, no
configuration file and no privileges::

    with FakeServer(files={"/data/a.root": b"hello"}) as server:
        fs = xrd.FileSystem(server.url)
        assert fs.read_bytes("/data/a.root") == b"hello"

It is deliberately a *fake*, not a simulator: it stores files in memory and
authorises everything. What it is faithful about is the wire format, which is
the part a client can get wrong. Redirects, waits and chunked responses can be
injected (:attr:`redirects`, :attr:`waits`, :attr:`chunk_reads`) so the
awkward paths get exercised too.

TLS is not implemented; point a client at it with ``root://``, not
``roots://``.
"""

from __future__ import annotations

import posixpath
import socket
import socketserver
import struct
import threading
import zlib
from collections.abc import Callable, Iterable, Iterator
from typing import cast

from ..errors import kXR_ArgInvalid
from ..proto import constants as c
from ..proto.buffer import Reader, Writer
from ..url import XRootDURL, parse

__all__ = ["FakeServer", "frame", "error"]

#: Requests whose body is payload rather than a path, so nothing about them
#: belongs in :attr:`FakeServer.arguments`.
_DATA_REQUESTS = frozenset(
    {c.kXR_write, c.kXR_writev, c.kXR_pgwrite, c.kXR_auth, c.kXR_login, c.kXR_sigver}
)

#: What :attr:`FakeServer.handlers` entries look like.
Handler = Callable[["_Connection", int, bytes, bytes], Iterator[bytes]]

_REQ = struct.Struct(">HH16sI")
_RESP = struct.Struct(">HHI")
_NULL_HANDLE = b"\x00\x00\x00\x00"


def frame(streamid: int, status: int, body: bytes = b"") -> bytes:
    """One complete response frame: the 8-byte header and its body.

    Public because :attr:`FakeServer.handlers` is: answering badly means
    building the frame yourself.
    """
    return _RESP.pack(streamid, status, len(body)) + body


def error(streamid: int, code: int, message: str) -> bytes:
    """A ``kXR_error`` frame carrying a server error code and its text."""
    return frame(streamid, c.kXR_error, struct.pack(">i", code) + message.encode() + b"\x00")


_frame = frame
_error = error


def _clean(path: str) -> str:
    """Strip any CGI and normalise, the way a server resolves a path."""
    base = path.partition("?")[0]
    return posixpath.normpath("/" + base.lstrip("/")) if base else "/"


def _endpoint(server: socketserver.BaseServer) -> tuple[str, int]:
    """The bound ``(host, port)``. These servers are always AF_INET, which is
    narrower than the address family ``socketserver`` is typed for.
    """
    host, port = cast("tuple[str, int]", server.server_address)
    return host, port


class _NotFound(Exception):
    """Raised inside a handler to produce a ``kXR_NotFound``."""

    def __init__(self, path: str, code: int = 3011, message: str = "no such file or directory"):
        self.path = path
        self.code = code
        self.message = message


class FakeServer:
    """An in-memory XRootD server listening on an ephemeral loopback port."""

    def __init__(
        self,
        *,
        files: dict[str, bytes] | None = None,
        dirs: Iterable[str] = (),
        sec: str = "",
        host: str = "127.0.0.1",
        port: int = 0,
        flags: int = c.kXR_isServer,
        sessid: bytes = b"\x11" * 16,
        version: int = 0x0500_0000,
    ) -> None:
        #: Path to contents. Mutated by the client, readable by the test.
        self.files: dict[str, bytearray] = {}
        #: Directories that exist, including the parents of every file.
        self.dirs: set[str] = {"/"}
        #: Extended attributes, per path.
        self.xattrs: dict[str, dict[str, bytes]] = {}
        #: Values ``kXR_query`` config lookups answer with.
        self.config_values: dict[str, str] = {"version": "v5.6.0", "role": "server"}
        #: Opcodes seen, in order - what a test asserts round trips on.
        self.seen: list[int] = []
        #: Raw path argument of every ``kXR_open``, CGI included. The opaque
        #: data is the whole protocol for third-party copy, so it is kept.
        self.opened: list[str] = []
        #: ``(opcode, argument)`` for every request that named one, again with
        #: the CGI left on - which is how a test checks that opaque data
        #: reached the operation it was meant for.
        self.arguments: list[tuple[int, str]] = []
        #: ``opcode -> (host, port, token)``, consumed once each.
        self.redirects: dict[int, tuple[str, int, str]] = {}
        #: ``opcode -> count`` of ``kXR_wait`` replies to send first.
        self.waits: dict[int, int] = {}
        #: Split read responses into ``kXR_oksofar`` chunks of this size.
        self.chunk_reads = 0
        #: Rounds of ``kXR_authmore`` to demand before accepting a credential.
        self.auth_rounds = 0
        #: ``opcode -> handler`` overrides, tried before the built-in ones.
        #: A handler takes ``(connection, streamid, params, body)`` and yields
        #: raw frames, so it can answer with anything at all - including
        #: something no real server would send.
        self.handlers: dict[int, Handler] = {}

        self._live: set[object] = set()
        self._live_lock = threading.Lock()

        self.sec = sec
        self.flags = flags
        self.sessid = sessid
        self.version = version

        for path, data in (files or {}).items():
            self.add_file(path, data)
        for path in dirs:
            self.add_dir(path)

        self._wanted = (host, port)
        self._last: tuple[str, int] | None = None
        self._bound: _TCPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Contents
    # ------------------------------------------------------------------

    def add_dir(self, path: str) -> None:
        """Create a directory and every parent of it."""
        target = _clean(path)
        while target not in self.dirs:
            self.dirs.add(target)
            target = posixpath.dirname(target) or "/"

    def add_file(self, path: str, data: bytes = b"") -> None:
        """Create or replace a file, creating its parents."""
        target = _clean(path)
        self.files[target] = bytearray(data)
        self.add_dir(posixpath.dirname(target))

    def contents(self, path: str) -> bytes:
        """What the server currently holds at ``path``."""
        return bytes(self.files[_clean(path)])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def _server(self) -> _TCPServer:
        """The listening socket, bound on first use.

        Binding lazily means a server built only to hold contents - a
        fixture that a test never connects to - never claims a port.
        """
        if self._bound is None:
            self._bound = _TCPServer(self._wanted, _Handler)
            self._bound.fake = self
        return self._bound

    @property
    def address(self) -> tuple[str, int]:
        """Where the server listens; reading it before ``start`` binds the port.

        After :meth:`stop` it reports where the server *was*, rather than
        quietly claiming a fresh port to answer the question.
        """
        if self._bound is None and self._last is not None:
            return self._last
        return _endpoint(self._server)

    @property
    def url(self) -> XRootDURL:
        """``root://host:port/`` - what a client connects to."""
        host, port = self.address
        return parse(f"root://{host}:{port}/")

    def start(self) -> FakeServer:
        """Serve in a background thread. Idempotent."""
        if self._thread is None:
            # A short poll interval keeps stop() from costing half a second,
            # which is the difference between a fast test suite and a slow one.
            self._thread = threading.Thread(
                target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            self._thread.start()
        return self

    def disconnect(self) -> None:
        """Drop every live connection, as a restarting server would.

        The listening socket stays open, so a client that reconnects gets
        served again - which is exactly the reconnect path worth testing.
        """
        with self._live_lock:
            live, self._live = set(self._live), set()
        for sock in live:
            try:
                sock.shutdown(socket.SHUT_RDWR)  # type: ignore[attr-defined]
            except OSError:
                pass

    def stop(self) -> None:
        """Stop serving and release the port. Idempotent."""
        if self._bound is None:
            return
        self._last = _endpoint(self._bound)
        if self._thread is not None:
            self._bound.shutdown()
            self._thread = None
        self._bound.server_close()
        self._bound = None
        self.disconnect()

    def __enter__(self) -> FakeServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        # Reading the address would bind the port; a repr must not do that.
        bound = _endpoint(self._bound) if self._bound else self._last
        where = f"{bound[0]}:{bound[1]}" if bound else "unbound"
        return f"FakeServer({where}, files={len(self.files)}, dirs={len(self.dirs)})"


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    fake: FakeServer


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        _Connection(self.server.fake, self.request).run()  # type: ignore[attr-defined]


class _Connection:
    """One client connection's worth of protocol."""

    def __init__(self, server: FakeServer, sock: object) -> None:
        self.s = server
        self.sock = sock
        self.rfile = sock.makefile("rb")  # type: ignore[attr-defined]
        self.handles: dict[bytes, str] = {}
        self.next_handle = 1
        self.auth_seen = 0

    # -- plumbing -------------------------------------------------------

    def run(self) -> None:
        with self.s._live_lock:
            self.s._live.add(self.sock)
        try:
            if len(self.rfile.read(20) or b"") < 20:
                return
            self._send(_frame(0, c.kXR_ok, struct.pack(">ii", 0, c.ROOTD_PQ)))
            while True:
                header = self.rfile.read(c.REQUEST_HDRLEN)
                if not header or len(header) < c.REQUEST_HDRLEN:
                    return
                sid, opcode, params, dlen = _REQ.unpack(header)
                body = self.rfile.read(dlen) if dlen else b""
                self.s.seen.append(opcode)
                for chunk in self._dispatch(sid, opcode, params, body):
                    self._send(chunk)
                if opcode == c.kXR_endsess:
                    return
        except (OSError, ValueError):
            pass
        finally:
            with self.s._live_lock:
                self.s._live.discard(self.sock)
            try:
                self.rfile.close()
                self.sock.close()  # type: ignore[attr-defined]
            except OSError:
                pass

    def _send(self, data: bytes) -> None:
        self.sock.sendall(data)  # type: ignore[attr-defined]

    def _dispatch(self, sid: int, opcode: int, params: bytes, body: bytes) -> Iterator[bytes]:
        if opcode == c.kXR_sigver:
            return  # the signature prefixes the next frame; nothing to answer

        target = self.s.redirects.pop(opcode, None)
        if target is not None:
            host, port, token = target
            where = f"{host}{'?' + token if token else ''}".encode()
            yield _frame(sid, c.kXR_redirect, struct.pack(">i", port) + where + b"\x00")
            return

        if opcode not in _DATA_REQUESTS:
            text = body.split(b"\x00", 1)[0].decode("utf-8", "replace")
            if text:
                self.s.arguments.append((opcode, text))

        if self.s.waits.get(opcode, 0):
            self.s.waits[opcode] -= 1
            yield _frame(sid, c.kXR_wait, struct.pack(">i", 0) + b"try again\x00")
            return

        handler = self.s.handlers.get(opcode) or _HANDLERS.get(opcode)
        if handler is None:
            yield _error(sid, 3013, f"request {opcode} is not supported")
            return
        try:
            yield from handler(self, sid, params, body)
        except _NotFound as missing:
            yield _error(sid, missing.code, f"{missing.message}: {missing.path}")

    # -- namespace helpers ----------------------------------------------

    def _path(self, params: bytes, body: bytes, at: slice = slice(12, 16)) -> str:
        """The path a request names, by body or by open handle."""
        text = body.split(b"\x00", 1)[0].decode("utf-8", "replace")
        if text:
            return _clean(text)
        handle = params[at]
        if handle != _NULL_HANDLE and handle in self.handles:
            return self.handles[handle]
        raise _NotFound("", 3001, "no path and no handle")

    def _stat_line(self, path: str) -> bytes:
        if path in self.s.dirs:
            flags = c.kXR_isDir | c.kXR_readable | c.kXR_writable
            size = 4096
        elif path in self.s.files:
            flags = c.kXR_readable | c.kXR_writable
            size = len(self.s.files[path])
        else:
            raise _NotFound(path)
        return f"{abs(hash(path)) % 10**9} {size} {flags} 1700000000".encode()

    def _children(self, path: str) -> list[str]:
        if path not in self.s.dirs:
            raise _NotFound(path)
        prefix = path.rstrip("/") + "/"
        names = {
            entry[len(prefix) :].split("/", 1)[0]
            for entry in list(self.s.files) + list(self.s.dirs)
            if entry.startswith(prefix) and entry != path
        }
        return sorted(names)

    def _file(self, path: str) -> bytearray:
        try:
            return self.s.files[path]
        except KeyError:
            raise _NotFound(path) from None


# --------------------------------------------------------------------------
# Handlers. Each yields the frames to send back.
# --------------------------------------------------------------------------


def _h_protocol(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    yield _frame(sid, c.kXR_ok, struct.pack(">iI", conn.s.version, conn.s.flags))


def _h_login(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    trailer = conn.s.sec.encode() + b"\x00" if conn.s.sec else b""
    yield _frame(sid, c.kXR_ok, conn.s.sessid + trailer)


def _h_auth(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    conn.auth_seen += 1
    if conn.auth_seen <= conn.s.auth_rounds:
        yield _frame(sid, c.kXR_authmore, b"challenge")
    else:
        yield _frame(sid, c.kXR_ok)


def _h_ping(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    yield _frame(sid, c.kXR_ok)


def _h_endsess(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    yield _frame(sid, c.kXR_ok)


def _h_stat(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    if params[0] & c.kXR_vfs:
        yield _frame(sid, c.kXR_ok, b"1 1000000 50 1 500000 20\x00")
        return
    path = conn._path(params, body)
    yield _frame(sid, c.kXR_ok, conn._stat_line(path) + b"\x00")


def _h_statx(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    out = bytearray()
    for raw in body.split(b"\x00", 1)[0].decode().split("\n"):
        path = _clean(raw)
        if path in conn.s.dirs:
            out.append(c.kXR_isDir)
        elif path in conn.s.files:
            out.append(c.kXR_readable | c.kXR_writable)
        else:
            out.append(c.kXR_other)
    yield _frame(sid, c.kXR_ok, bytes(out))


def _h_dirlist(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    path = _clean(body.split(b"\x00", 1)[0].decode())
    names = conn._children(path)
    if not params[15] & c.kXR_dstat:
        yield _frame(sid, c.kXR_ok, "\n".join(names).encode() + b"\x00")
        return
    out = bytearray(b".\n" + conn._stat_line(path) + b"\n")
    for name in names:
        full = posixpath.join(path, name)
        out += name.encode() + b"\n" + conn._stat_line(full) + b"\n"
    yield _frame(sid, c.kXR_ok, bytes(out) + b"\x00")


def _h_mkdir(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    path = _clean(body.split(b"\x00", 1)[0].decode())
    if path in conn.s.dirs or path in conn.s.files:
        yield _error(sid, 3018, f"already exists: {path}")
        return
    parent = posixpath.dirname(path)
    if not params[0] & c.kXR_mkdirpath and parent not in conn.s.dirs:
        yield _error(sid, 3011, f"no such file or directory: {parent}")
        return
    conn.s.add_dir(path)
    yield _frame(sid, c.kXR_ok)


def _h_rm(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    path = _clean(body.split(b"\x00", 1)[0].decode())
    if path in conn.s.dirs:
        yield _error(sid, 3016, f"is a directory: {path}")
        return
    conn._file(path)
    del conn.s.files[path]
    conn.s.xattrs.pop(path, None)
    yield _frame(sid, c.kXR_ok)


def _h_rmdir(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    path = _clean(body.split(b"\x00", 1)[0].decode())
    if path not in conn.s.dirs:
        raise _NotFound(path)
    if conn._children(path):
        yield _error(sid, 3005, f"directory not empty: {path}")
        return
    conn.s.dirs.discard(path)
    yield _frame(sid, c.kXR_ok)


def _h_mv(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    split = struct.unpack(">H", params[14:16])[0]
    text = body.split(b"\x00", 1)[0].decode()
    src, dst = _clean(text[:split]), _clean(text[split + 1 :])
    conn.s.files[dst] = conn._file(src)
    del conn.s.files[src]
    conn.s.add_dir(posixpath.dirname(dst))
    yield _frame(sid, c.kXR_ok)


def _h_chmod(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    path = _clean(body.split(b"\x00", 1)[0].decode())
    if path not in conn.s.files and path not in conn.s.dirs:
        raise _NotFound(path)
    yield _frame(sid, c.kXR_ok)


def _h_truncate(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    size = struct.unpack(">q", params[4:12])[0]
    path = conn._path(params, body, at=slice(0, 4))
    data = conn._file(path)
    if size < len(data):
        del data[size:]
    else:
        data.extend(bytes(size - len(data)))
    yield _frame(sid, c.kXR_ok)


def _h_set(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    yield _frame(sid, c.kXR_ok)


def _h_open(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    _mode, options = struct.unpack(">HH", params[:4])
    raw = body.split(b"\x00", 1)[0].decode()
    conn.s.opened.append(raw)
    path = _clean(raw)
    exists = path in conn.s.files
    if path in conn.s.dirs:
        yield _error(sid, 3016, f"is a directory: {path}")
        return
    if options & c.kXR_new and exists:
        yield _error(sid, 3018, f"already exists: {path}")
        return
    # Only kXR_new and kXR_delete create; kXR_open_updt and kXR_open_apnd on
    # their own are "open what is there", exactly as stock xrootd has it. The
    # fake used to create for any write flag, which let a client that never
    # asked for creation pass here and fail against a real server.
    if not exists:
        if not options & (c.kXR_new | c.kXR_delete):
            raise _NotFound(path)
        parent = posixpath.dirname(path)
        if parent not in conn.s.dirs and not options & c.kXR_mkpath:
            yield _error(sid, 3011, f"no such file or directory: {parent}")
            return
        conn.s.add_file(path)
    elif options & c.kXR_delete:
        conn.s.files[path] = bytearray()

    handle = struct.pack(">I", conn.next_handle)
    conn.next_handle += 1
    conn.handles[handle] = path
    reply = handle
    if options & c.kXR_retstat:
        reply += bytes(8) + conn._stat_line(path) + b"\x00"
    yield _frame(sid, c.kXR_ok, reply)


def _h_close(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    if conn.handles.pop(params[:4], None) is None:
        yield _error(sid, 3004, "file is not open")
        return
    yield _frame(sid, c.kXR_ok)


def _h_read(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    offset, length = struct.unpack(">qi", params[4:16])
    data = conn._file(conn._path(params, b"", at=slice(0, 4)))[offset : offset + length]
    step = conn.s.chunk_reads
    if step and len(data) > step:
        for start in range(0, len(data) - step, step):
            yield _frame(sid, c.kXR_oksofar, bytes(data[start : start + step]))
        last = ((len(data) - 1) // step) * step
        yield _frame(sid, c.kXR_ok, bytes(data[last:]))
        return
    yield _frame(sid, c.kXR_ok, bytes(data))


def _h_write(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    offset = struct.unpack(">q", params[4:12])[0]
    _splice(conn._file(conn._path(params, b"", at=slice(0, 4))), offset, body)
    yield _frame(sid, c.kXR_ok)


def _h_sync(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    conn._path(params, b"", at=slice(0, 4))
    yield _frame(sid, c.kXR_ok)


def _h_readv(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    out = Writer()
    reader = Reader(body, "kXR_readv")
    while reader.remaining >= c.READ_LIST_ENTRY_LEN:
        handle = reader.bytes(4)
        length = reader.i32()
        offset = reader.i64()
        data = conn._file(conn.handles[handle])[offset : offset + length]
        out.raw(handle).i32(len(data)).i64(offset).raw(bytes(data))
    yield _frame(sid, c.kXR_ok, out.bytes())


def _h_writev(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    # dlen counts the write_list and nothing else, so the body is exactly N
    # descriptors and the data for them arrives afterwards, uncounted.
    if len(body) % c.READ_LIST_ENTRY_LEN:
        yield _error(sid, kXR_ArgInvalid, "Write vector is invalid")
        return
    reader = Reader(body, "kXR_writev")
    entries: list[tuple[bytes, int, int]] = []
    while reader.remaining >= c.READ_LIST_ENTRY_LEN:
        handle = reader.bytes(4)
        length = reader.i32()
        offset = reader.i64()
        entries.append((handle, offset, length))
    for handle, offset, length in entries:
        _splice(conn._file(conn.handles[handle]), offset, conn.rfile.read(length))
    yield _frame(sid, c.kXR_ok)


def _h_pgread(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    from ..crypto.crc32c import pack_pages

    offset, length = struct.unpack(">qi", params[4:16])
    data = bytes(conn._file(conn._path(params, b"", at=slice(0, 4)))[offset : offset + length])
    packed = pack_pages(data, offset)
    status = (
        struct.pack(">I", 0)
        + struct.pack(">H", sid)
        + bytes([c.kXR_pgread - c.kXR_1stRequest, c.kXR_FinalResult])
        + bytes(4)
        + struct.pack(">i", len(packed))
        + struct.pack(">q", offset)
    )
    yield _frame(sid, c.kXR_status, status) + packed


def _h_pgwrite(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    from ..crypto.crc32c import unpack_pages

    offset = struct.unpack(">q", params[4:12])[0]
    data, corrupt = unpack_pages(body, offset)
    if corrupt:
        yield _error(sid, 3019, f"checksum error on pages {list(corrupt)}")
        return
    _splice(conn._file(conn._path(params, b"", at=slice(0, 4))), offset, data)
    yield _frame(sid, c.kXR_ok)


def _h_chkpoint(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    yield _frame(sid, c.kXR_ok)


def _h_query(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    infotype = struct.unpack(">H", params[:2])[0]
    args = body.split(b"\x00", 1)[0].decode("utf-8", "replace")
    if infotype == c.kXR_Qcksum:
        path = _clean(args)
        algorithm = "adler32"
        for pair in args.partition("?")[2].split("&"):
            if pair.startswith("cks.type="):
                algorithm = pair.split("=", 1)[1]
        value = _checksum(algorithm, bytes(conn._file(path)))
        yield _frame(sid, c.kXR_ok, f"{algorithm} {value}".encode() + b"\x00")
    elif infotype == c.kXR_Qconfig:
        names = args.split("\n")
        values = [conn.s.config_values.get(n, "") for n in names]
        yield _frame(sid, c.kXR_ok, "\n".join(values).encode() + b"\x00")
    elif infotype == c.kXR_Qvisa:
        yield _frame(sid, c.kXR_ok, b"visa\x00")
    else:
        yield _error(sid, 3013, f"query {infotype} is not supported")


def _h_locate(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    path = _clean(body.split(b"\x00", 1)[0].decode())
    if path not in conn.s.files and path not in conn.s.dirs:
        raise _NotFound(path)
    host, port = conn.s.address
    yield _frame(sid, c.kXR_ok, f"Sw{host}:{port}".encode() + b"\x00")


def _h_prepare(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    yield _frame(sid, c.kXR_ok, b"prep-0001\x00")


def _h_fattr(conn: _Connection, sid: int, params: bytes, body: bytes) -> Iterator[bytes]:
    subcode, numattr, options = params[4], params[5], params[6]
    reader = Reader(body, "kXR_fattr")
    path = reader.cstring()
    if not path:
        path = conn.handles.get(params[:4], "")
    path = _clean(path)
    if path not in conn.s.files and path not in conn.s.dirs:
        raise _NotFound(path)
    store = conn.s.xattrs.setdefault(path, {})

    if subcode == c.kXR_fattrList:
        names = sorted(store)
        values = bool(options & c.kXR_fattrAData)
        reply = _fattr_reply([(n, 0, store[n] if values else None) for n in names])
        yield _frame(sid, c.kXR_ok, reply)
        return

    items: list[tuple[str, int, bytes | None]] = []
    for _ in range(max(numattr, 1)):
        if reader.remaining < 3:
            break
        reader.u16()
        name = reader.cstring()
        if subcode == c.kXR_fattrSet:
            value = reader.bytes(reader.i32())
            if options & c.kXR_fattrIsNew and name in store:
                items.append((name, 17, None))
                continue
            store[name] = bytes(value)
            items.append((name, 0, None))
        elif subcode == c.kXR_fattrGet:
            current = store.get(name)
            items.append((name, 0 if current is not None else 61, current or b""))
        elif subcode == c.kXR_fattrDel:
            items.append((name, 0 if store.pop(name, None) is not None else 61, None))
    yield _frame(sid, c.kXR_ok, _fattr_reply(items))


_HANDLERS = {
    c.kXR_protocol: _h_protocol,
    c.kXR_login: _h_login,
    c.kXR_auth: _h_auth,
    c.kXR_ping: _h_ping,
    c.kXR_endsess: _h_endsess,
    c.kXR_stat: _h_stat,
    c.kXR_statx: _h_statx,
    c.kXR_dirlist: _h_dirlist,
    c.kXR_mkdir: _h_mkdir,
    c.kXR_rm: _h_rm,
    c.kXR_rmdir: _h_rmdir,
    c.kXR_mv: _h_mv,
    c.kXR_chmod: _h_chmod,
    c.kXR_truncate: _h_truncate,
    c.kXR_set: _h_set,
    c.kXR_open: _h_open,
    c.kXR_close: _h_close,
    c.kXR_read: _h_read,
    c.kXR_write: _h_write,
    c.kXR_sync: _h_sync,
    c.kXR_readv: _h_readv,
    c.kXR_writev: _h_writev,
    c.kXR_pgread: _h_pgread,
    c.kXR_pgwrite: _h_pgwrite,
    c.kXR_chkpoint: _h_chkpoint,
    c.kXR_query: _h_query,
    c.kXR_locate: _h_locate,
    c.kXR_prepare: _h_prepare,
    c.kXR_fattr: _h_fattr,
}


def _splice(data: bytearray, offset: int, payload: bytes) -> None:
    """Write ``payload`` at ``offset``, zero-filling any hole.

    Writing nothing does nothing, as for :func:`os.pwrite`: an empty write
    past the end does not extend the file to reach it.
    """
    if not payload:
        return
    if offset > len(data):
        data.extend(bytes(offset - len(data)))
    data[offset : offset + len(payload)] = payload


def _fattr_reply(items: list[tuple[str, int, bytes | None]]) -> bytes:
    w = Writer().u8(sum(1 for _, code, _ in items if code)).u8(len(items))
    for name, code, value in items:
        w.u16(code).text(name, nul=True)
        if value is not None:
            w.i32(len(value)).raw(value)
    return w.bytes()


def _checksum(algorithm: str, data: bytes) -> str:
    from ..crypto.checksum import checksum_bytes

    try:
        return checksum_bytes(algorithm, data)
    except ValueError:
        return f"{zlib.adler32(data):08x}"
