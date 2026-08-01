"""One HTTP/1.1 connection pool, spelled the way the rest of the package is.

Built on :mod:`http.client` because the whole point of this package is that a
plain interpreter can talk to a storage element: no ``requests``, no
``httpx``, no wheels. What this adds on top of the stdlib is the part every
grid client needs anyway - persistent connections keyed by endpoint, bearer
tokens (macaroons and SciTokens are just bearer tokens), X.509 proxies,
redirect following across hosts, one retry when a pooled connection turns out
to have gone stale, and HTTP status codes translated into the same exceptions
the ``root://`` side raises.
"""

from __future__ import annotations

import http.client
import os
import socket
import ssl
import urllib.parse
from dataclasses import dataclass, field

from .._log import get_logger
from ..config import Config
from ..errors import (
    ConnectionError as XRDConnectionError,
)
from ..errors import (
    RedirectLimitError,
    kXR_ArgInvalid,
    kXR_FileLocked,
    kXR_ItExists,
    kXR_NoSpace,
    kXR_NotAuthorized,
    kXR_NotFound,
    kXR_ServerError,
    kXR_Unsupported,
    raise_for_status,
)
from ..errors import (
    TimeoutError as XRDTimeoutError,
)
from ..transport.base import tls_context
from ..url import XRootDURL, parse

__all__ = ["HTTPClient", "Response", "bearer_token", "check_status", "status_code"]

_log = get_logger(__name__)

#: Everything the client sends is safe to repeat, so every verb may be retried
#: once against a fresh connection. ``PUT`` and ``DELETE`` are idempotent by
#: definition; ``POST`` is not, and is excluded.
_RETRYABLE = frozenset(
    {"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "PROPFIND", "MKCOL", "MOVE", "COPY"}
)

_REDIRECTS = frozenset({301, 302, 303, 307, 308})

#: HTTP status to the ``kXR_*`` code that means the same thing, so one status
#: table feeds the one exception table the whole package already has.
_ERRORS: dict[int, int] = {
    400: kXR_ArgInvalid,
    401: kXR_NotAuthorized,
    403: kXR_NotAuthorized,
    404: kXR_NotFound,
    405: kXR_Unsupported,
    409: kXR_NotFound,  # RFC 4918: the parent collection is missing
    410: kXR_NotFound,
    412: kXR_ItExists,  # a failed If-None-Match is "it is already there"
    416: kXR_ArgInvalid,
    423: kXR_FileLocked,
    429: kXR_FileLocked,
    501: kXR_Unsupported,
    503: kXR_FileLocked,
    507: kXR_NoSpace,
}

#: What a body is truncated to when a caller asks for the whole thing. A
#: PROPFIND against a huge collection is the realistic way to blow up memory.
MAX_BODY = 1 << 26


@dataclass(slots=True)
class Response:
    """A finished HTTP response: status line, headers, and the whole body."""

    status: int
    reason: str
    headers: http.client.HTTPMessage
    body: bytes = b""
    url: str = ""

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name, default)

    @property
    def content_length(self) -> int | None:
        raw = self.headers.get("Content-Length")
        return int(raw) if raw is not None and raw.isdigit() else None

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, "replace")


def bearer_token(config: Config, url: XRootDURL | None = None) -> str | None:
    """The token to present, in the order a grid user expects it to be found.

    A token in the URL wins because it was written at the call site; then the
    explicit :attr:`~xrd.Config.token`, then the token file that
    ``BEARER_TOKEN_FILE`` (or :attr:`~xrd.Config.token_file`) names, then
    ``$BEARER_TOKEN``.
    """
    if url is not None:
        from_url = url.query.get("authz") or url.query.get("access_token")
        if from_url:
            return from_url.removeprefix("Bearer ").strip()
    if config.token:
        return config.token
    if config.token_file:
        try:
            with open(config.token_file, encoding="utf-8") as fh:
                return fh.read().strip() or None
        except OSError:
            _log.debug("token file %s is unreadable", config.token_file)
    return os.environ.get("BEARER_TOKEN") or None


def _context(config: Config) -> ssl.SSLContext:
    """A TLS context — the same one ``roots://`` gets.

    ``davs://`` and ``roots://`` must trust the same CAs and present the same
    X.509 proxy, so this defers to :func:`xrd.transport.base.tls_context`
    rather than building a second context that could drift from it.
    """
    return tls_context(config)


@dataclass(slots=True)
class HTTPClient:
    """Connections to one or more HTTP endpoints, reused between requests.

        >>> client = HTTPClient(config)
        >>> client.request("HEAD", parse("https://dav.example.org/store/f")).status
        200

    Not thread-safe, exactly like the :mod:`http.client` connections it holds:
    give each thread its own.
    """

    config: Config = field(default_factory=Config)
    _pool: dict[tuple[str, str, int], http.client.HTTPConnection] = field(default_factory=dict)

    def close(self) -> None:
        """Drop every pooled connection. Idempotent."""
        for conn in self._pool.values():
            conn.close()
        self._pool.clear()

    def __enter__(self) -> HTTPClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"HTTPClient({len(self._pool)} connection(s))"

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def connection(self, url: XRootDURL) -> http.client.HTTPConnection:
        """The pooled connection for ``url``'s endpoint, opened on demand."""
        key = _key(url)
        conn = self._pool.get(key)
        if conn is None:
            conn = self._connect(url)
            self._pool[key] = conn
        return conn

    def _connect(self, url: XRootDURL) -> http.client.HTTPConnection:
        scheme, host, port = _key(url)
        timeout = self.config.connect_timeout
        if scheme == "https":
            return http.client.HTTPSConnection(
                host, port, timeout=timeout, context=_context(self.config)
            )
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def _discard(self, url: XRootDURL) -> None:
        conn = self._pool.pop(_key(url), None)
        if conn is not None:
            conn.close()

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def headers_for(self, url: XRootDURL, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Standing headers for ``url``, with ``extra`` layered on top."""
        headers = {"User-Agent": "xrd/1.0 (pure python)", "Accept": "*/*"}
        token = bearer_token(self.config, url)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if extra:
            headers.update(extra)
        return headers

    def open(
        self,
        method: str,
        url: XRootDURL,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        expect: tuple[int, ...] = (),
        errors: dict[int, int] | None = None,
    ) -> http.client.HTTPResponse:
        """Issue a request and hand back the *unread* response.

        The connection stays busy until the caller reads and closes it, which
        is what makes a ranged ``GET`` a stream rather than a buffer.
        """
        target = url
        for _ in range(self.config.redirect_limit + 1):
            response = self._once(method, target, body, headers)
            if response.status in _REDIRECTS and response.getheader("Location"):
                location = response.getheader("Location", "")
                response.read()
                response.close()
                target = _redirect(target, location)
                if response.status == 303:
                    method, body = "GET", None
                _log.debug("%s redirected to %s", method, target)
                continue
            try:
                check_status(response.status, response.reason, target, expect, errors)
            except Exception:
                # Drain before unwinding: an error response left unread makes
                # the pooled connection unusable, and the next request on it
                # would be silently re-sent on a fresh one.
                try:
                    response.read(MAX_BODY)
                finally:
                    response.close()
                raise
            return response
        raise RedirectLimitError(f"more than {self.config.redirect_limit} redirects for {url}")

    def request(
        self,
        method: str,
        url: XRootDURL,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        expect: tuple[int, ...] = (),
        errors: dict[int, int] | None = None,
    ) -> Response:
        """Issue a request and read the whole response."""
        response = self.open(method, url, body=body, headers=headers, expect=expect, errors=errors)
        try:
            payload = b"" if method == "HEAD" else response.read(MAX_BODY)
        finally:
            response.close()
        return Response(response.status, response.reason, response.msg, payload, str(url))

    def _once(
        self,
        method: str,
        url: XRootDURL,
        body: bytes | None,
        headers: dict[str, str] | None,
    ) -> http.client.HTTPResponse:
        """One request, retried once if a pooled connection had gone stale."""
        target = request_target(url)
        sent = self.headers_for(url, headers)
        # A conditional request is not safe to repeat: the first attempt may
        # have been applied, which makes the second one fail its condition.
        repeatable = method in _RETRYABLE and not any(k.lower().startswith("if-") for k in sent)
        for attempt in (0, 1):
            conn = self.connection(url)
            try:
                conn.request(method, target, body=body, headers=sent)
                return conn.getresponse()
            except (http.client.HTTPException, OSError) as exc:
                self._discard(url)
                if attempt == 0 and repeatable:
                    _log.debug("retrying %s %s after %s", method, url, exc)
                    continue
                raise _wrap(exc, method, url) from exc
        raise AssertionError("unreachable")  # pragma: no cover


def _key(url: XRootDURL) -> tuple[str, str, int]:
    scheme = "https" if url.use_tls else "http"
    return (scheme, url.host, url.port)


def request_target(url: XRootDURL) -> str:
    """The origin-form request target: percent-encoded path plus query."""
    target = urllib.parse.quote(url.path or "/", safe="/:@!$&'()*+,;=~")
    query = {k: v for k, v in url.query.items() if k not in ("authz", "access_token")}
    if query:
        target += "?" + urllib.parse.urlencode(query)
    return target


def _redirect(current: XRootDURL, location: str) -> XRootDURL:
    """Resolve a ``Location`` against the URL that produced it."""
    if "://" in location:
        return parse(location)
    joined = urllib.parse.urljoin(f"{_key(current)[0]}://{current.netloc}{current.path}", location)
    return parse(joined).evolve(username=current.username, password=current.password)


def check_status(
    status: int,
    reason: str,
    url: XRootDURL,
    expect: tuple[int, ...],
    errors: dict[int, int] | None,
) -> None:
    """Turn a status the caller did not ask for into the right exception.

    ``errors`` overrides the table per verb, because WebDAV reuses statuses:
    a ``405`` from ``MKCOL`` means the collection is already there, while
    everywhere else it means the server does not implement the verb.
    """
    if status in expect if expect else status < 400:
        return
    raise_for_status(status_code(status, errors), f"HTTP {status} {reason}", path=url.path)


def status_code(status: int, errors: dict[int, int] | None = None) -> int:
    """The ``kXR_*`` code an HTTP status means.

    Separate from :func:`check_status` because a status does not always
    arrive in a status line: a third-party copy reports its outcome in the
    response body, and quotes the status the far side gave it.
    """
    code = (errors or {}).get(status) or _ERRORS.get(status)
    if code is not None:
        return code
    return kXR_ServerError if status >= 500 else kXR_ArgInvalid


def _wrap(exc: Exception, method: str, url: XRootDURL) -> Exception:
    """Present a transport failure as this package's error, not the stdlib's."""
    if isinstance(exc, socket.timeout | TimeoutError):
        return XRDTimeoutError(f"{method} {url} timed out")
    return XRDConnectionError(f"{method} {url} failed: {exc}")
