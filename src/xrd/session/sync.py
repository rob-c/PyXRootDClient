"""The blocking session driver.

One :class:`Session` is one connection: it owns a transport and a
:class:`~xrd.proto.machine.SessionMachine`, and turns the machine's events
into ordinary Python returns and exceptions. Redirects are *not* followed
here - a redirect means a different server, which is a policy decision the
caller makes; :class:`Session` reports it as :class:`RedirectRequired`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .._log import get_logger
from ..config import Config
from ..errors import ConnectionError as XrdConnectionError
from ..errors import ProtocolError, WaitLimitError, XRootDError
from ..proto import machine as m
from ..proto import responses as rp
from ..proto.frames import Request
from ..transport.base import Transport
from ..transport.sync import SocketTransport
from ..url import XRootDURL, parse

__all__ = ["Session", "Result", "RedirectRequired"]

_log = get_logger(__name__)

_RECV = 1 << 18


class RedirectRequired(XRootDError):
    """The server redirected; the caller must re-issue elsewhere."""

    def __init__(self, target: rp.RedirectInfo) -> None:
        self.target = target
        super().__init__(f"redirected to {target.url}")


@dataclass(frozen=True, slots=True)
class Result:
    """A completed request."""

    data: bytes
    status: rp.StatusInfo | None = None

    def __len__(self) -> int:
        return len(self.data)

    def __bytes__(self) -> bytes:
        return self.data


class Session:
    """An authenticated connection to one XRootD server."""

    def __init__(self, transport: Transport, machine: m.SessionMachine, config: Config) -> None:
        self._t = transport
        self._m = machine
        self.config = config
        self._lock = threading.RLock()
        self._inbox: dict[int, list[m.Event]] = {}
        self._notices: list[rp.AttnInfo] = []
        self._paths: dict[int, Transport] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def connect(
        cls,
        url: str | XRootDURL,
        *,
        config: Config | None = None,
        username: str = "",
    ) -> Session:
        """Connect, negotiate, upgrade to TLS if required, log in, authenticate."""
        target = parse(url) if isinstance(url, str) else url
        config = config or Config()
        transport = SocketTransport.connect(target.host, target.port, config)
        machine = m.SessionMachine(
            host=target.host,
            port=target.port,
            config=config,
            username=username or target.username or config.username,
            want_tls=target.use_tls or config.require_tls,
        )
        session = cls(transport, machine, config)
        try:
            machine.start()
            session._bringup()
        except BaseException:
            transport.close()
            raise
        return session

    def _bringup(self) -> None:
        while self._m.state not in (m.State.READY, m.State.FAILED, m.State.CLOSED):
            for event in self._pump():
                if isinstance(event, m.NeedTLS):
                    self._flush()
                    self._t.start_tls(self._m.host, self.config)
                    self._m.tls_established()
                elif isinstance(event, m.Failed):
                    raise event.error
                elif isinstance(event, m.Disconnected):
                    raise XrdConnectionError(f"connection lost during bring-up: {event.reason}")
        if self._m.state is not m.State.READY:
            raise XrdConnectionError(f"bring-up ended in state {self._m.state.name}")
        _log.debug(
            "session ready with %s:%d via %s%s",
            self._m.host,
            self._m.port,
            self._m.mechanism or "no auth",
            " over TLS" if self._m.tls_active else "",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def host(self) -> str:
        return self._m.host

    @property
    def port(self) -> int:
        return self._m.port

    @property
    def endpoint(self) -> str:
        return f"{self._m.host}:{self._m.port}"

    @property
    def protocol(self) -> object:
        return self._m.protocol_info

    @property
    def mechanism(self) -> str:
        return self._m.mechanism

    @property
    def is_tls(self) -> bool:
        return self._m.tls_active

    @property
    def closed(self) -> bool:
        return self._m.state in (m.State.CLOSED, m.State.FAILED)

    @property
    def data_paths(self) -> list[int]:
        """The path ids bound to this session, in the order they were bound."""
        return list(self._paths)

    @property
    def has_data_path(self) -> bool:
        return bool(self._paths)

    def notices(self) -> list[rp.AttnInfo]:
        """Drain any unsolicited ``kXR_attn`` messages the server sent."""
        out, self._notices = self._notices, []
        return out

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------

    def bind_data_path(self) -> int:
        """Open a second connection to this server and bind it to the session.

        Returns the path id the server assigned. Pass it to a request that
        takes one - :class:`~xrd.proto.requests.Read`, ``ReadV``, ``PgRead``,
        ``Write``, ``PgWrite`` - and that request's bulk bytes travel on the
        new socket instead of competing with control traffic on this one.

        The new connection does not log in and is not authenticated: it says
        which session it belongs to and inherits that session's identity,
        which is exactly why the server will only accept it from the same
        client. It therefore costs a handshake, not a login.
        """
        with self._lock:
            if self.closed:
                raise XrdConnectionError(f"session to {self.endpoint} is closed")
            if not self._m.session_id:
                raise XRootDError(
                    f"{self.endpoint} gave this session no id, so it cannot "
                    f"be bound to a second connection"
                )
            transport = SocketTransport.connect(self._m.host, self._m.port, self.config)
            machine = m.SessionMachine(
                host=self._m.host,
                port=self._m.port,
                config=self.config,
                username=self._m.username,
                want_tls=self.is_tls,
                bind_to=self._m.session_id,
            )
            try:
                # The bring-up of a data path is a session's worth of work
                # minus the login, so it is driven by a session - which is
                # then thrown away, leaving only the socket.
                bringup = Session(transport, machine, self.config)
                machine.start()
                bringup._bringup()
                pathid = machine.pathid
                if pathid in self._paths:
                    raise ProtocolError(
                        f"{self.endpoint} handed out path id {pathid} twice"
                    )
            except BaseException:
                transport.close()
                raise
            self._paths[pathid] = transport
            _log.debug("bound data path %d to %s", pathid, self.endpoint)
            return pathid

    def _close_path(self, pathid: int) -> None:
        transport = self._paths.pop(pathid, None)
        if transport is not None:
            transport.close()

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def execute(
        self,
        request: Request,
        *,
        path: str = "",
        on_chunk: Callable[[bytes], None] | None = None,
    ) -> Result:
        """Send ``request`` and block until it completes.

        ``on_chunk`` receives the body in pieces as they arrive, including the
        final piece, so concatenating them gives exactly
        :attr:`Result.data` - a caller can stream straight to a file. The
        full body is accumulated for the return value either way.
        """
        with self._lock:
            if self.closed:
                raise XrdConnectionError(f"session to {self.endpoint} is closed")
            if request.pathid and request.pathid not in self._paths:
                raise ValueError(
                    f"data path {request.pathid} is not bound to {self.endpoint}"
                )
            sid = self._m.submit(request, path=path)
            answers_on = request.pathid if request.reply_on_path else 0
            return self._await(sid, on_chunk, answers_on)

    def _await(
        self, sid: int, on_chunk: Callable[[bytes], None] | None, pathid: int = 0
    ) -> Result:
        waits = 0
        streamed = 0
        while True:
            for event in self._events_for(sid, pathid):
                if isinstance(event, m.Completed):
                    # The machine's Completed carries the whole body; hand the
                    # streaming caller only the tail it has not seen yet.
                    if on_chunk is not None and len(event.data) > streamed:
                        on_chunk(event.data[streamed:])
                    return Result(event.data, event.status)
                if isinstance(event, m.Failed):
                    raise event.error
                if isinstance(event, m.Chunk):
                    if on_chunk is not None:
                        on_chunk(event.data)
                    streamed += len(event.data)
                elif isinstance(event, m.Redirected):
                    self._m.release(sid)
                    raise RedirectRequired(event.target)
                elif isinstance(event, m.Waiting):
                    if event.resend:
                        waits += 1
                        if waits > self.config.redirect_limit:
                            raise WaitLimitError(
                                f"server kept asking to wait for "
                                f"{type(event.request).__name__}",
                                attempts=waits,
                            )
                        _log.debug("server asked to wait %.1fs: %s", event.seconds, event.message)
                        time.sleep(event.seconds)
                        self._m.resume(sid)
                        self._flush()
                        streamed = 0  # the body starts over on the resend

    # ------------------------------------------------------------------
    # I/O pump
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        data = self._m.data_to_send()
        if data:
            self._t.send(data)
        for pathid, transport in self._paths.items():
            queued = self._m.path_data_to_send(pathid)
            if queued:
                transport.send(queued)

    def _pump(self, pathid: int = 0) -> list[m.Event]:
        """One send/receive turn; returns whatever events it produced."""
        self._flush()
        transport = self._paths[pathid] if pathid else self._t
        self._m.receive_data(transport.receive(_RECV), pathid=pathid)
        return list(self._m.events())

    def _events_for(self, sid: int, pathid: int = 0) -> list[m.Event]:
        """Block until at least one event for ``sid`` is available.

        ``pathid`` says which link the answer is expected on: a read that
        named a data path is answered there, and waiting on the control link
        for it would wait forever.
        """
        queued = self._inbox.pop(sid, None)
        if queued:
            return queued
        while True:
            mine: list[m.Event] = []
            for event in self._pump(pathid):
                target = getattr(event, "streamid", None)
                if target == sid:
                    mine.append(event)
                elif isinstance(event, m.Attention):
                    self._notices.append(event.info)
                elif isinstance(event, m.PathLost):
                    self._close_path(event.pathid)
                elif isinstance(event, m.Disconnected):
                    raise XrdConnectionError(f"connection lost: {event.reason}")
                elif isinstance(event, m.Failed) and target is None:
                    raise event.error
                elif target is not None:
                    self._inbox.setdefault(target, []).append(event)
            if mine:
                return mine

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """End the session and drop the connection."""
        with self._lock:
            if self._m.state is m.State.READY:
                try:
                    self._m.close()
                    self._flush()
                except (XRootDError, OSError):
                    pass
            self._t.close()
            for transport in self._paths.values():
                transport.close()
            self._paths.clear()
            self._m.state = m.State.CLOSED

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        paths = f", {len(self._paths)} data path" if self._paths else ""
        if len(self._paths) > 1:
            paths += "s"
        return (
            f"Session({self.endpoint}, {self._m.state.name.lower()}"
            f"{', tls' if self.is_tls else ''}"
            f"{', ' + self._m.mechanism if self._m.mechanism else ''}{paths})"
        )
