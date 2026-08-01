"""Client configuration.

Resolution order for any setting: explicit argument > the active
``contextvars`` override > environment variable > default.
"""

from __future__ import annotations

import contextlib
import contextvars
import getpass
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .auth.prompt import Prompter

__all__ = ["Config", "current", "configure", "override"]


def _env_flag(name: str) -> bool | None:
    """A tri-state environment switch: ``None`` when the variable is unset."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() not in ("0", "no", "off", "false")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _default_user() -> str:
    for var in ("XRD_USER", "USER", "LOGNAME"):
        val = os.environ.get(var)
        if val:
            return val
    try:
        return getpass.getuser()
    except Exception:
        return "nobody"


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable client settings. Use :meth:`evolve` to derive a variant."""

    username: str = field(default_factory=_default_user)

    # -- timeouts and retries (XRD_* names match the official client) --
    connect_timeout: float = field(
        default_factory=lambda: _env_float("XRD_CONNECTIONWINDOW", 30.0)
    )
    request_timeout: float = field(default_factory=lambda: _env_float("XRD_REQUESTTIMEOUT", 300.0))
    stream_timeout: float = field(default_factory=lambda: _env_float("XRD_STREAMTIMEOUT", 60.0))
    connect_retries: int = field(default_factory=lambda: _env_int("XRD_CONNECTIONRETRY", 3))
    #: Seconds to wait before the first reconnection attempt, doubling for
    #: each one after it and capped by :attr:`wait_cap`. Zero retries flat out.
    retry_backoff: float = field(default_factory=lambda: _env_float("XRD_STREAMERRORWINDOW", 0.5))
    redirect_limit: int = field(default_factory=lambda: _env_int("XRD_REDIRECTLIMIT", 16))
    wait_cap: float = 600.0
    keepalive_interval: float = 60.0

    # -- transfer tuning ----------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("XRD_CPCHUNKSIZE", 1 << 22))
    readahead: int = field(default_factory=lambda: _env_int("XRD_READAHEAD", 1 << 20))
    parallel_chunks: int = field(default_factory=lambda: _env_int("XRD_CPPARALLELCHUNKS", 4))

    # -- pooling -------------------------------------------------------
    pool_size: int = field(default_factory=lambda: _env_int("XRD_POOLSIZE", 8))
    pool_idle_ttl: float = 120.0

    # -- security ------------------------------------------------------
    token: str | None = None
    token_file: str | None = field(default_factory=lambda: os.environ.get("BEARER_TOKEN_FILE"))
    keytab: str | None = field(
        default_factory=lambda: os.environ.get("XrdSecSSSKT") or os.environ.get("XrdSecsssKT")
    )
    proxy: str | None = field(default_factory=lambda: os.environ.get("X509_USER_PROXY"))
    ca_path: str | None = field(default_factory=lambda: os.environ.get("X509_CERT_DIR"))
    ca_file: str | None = field(default_factory=lambda: os.environ.get("SSL_CERT_FILE"))
    auth_order: Sequence[str] = ("gsi", "ztn", "krb5", "sss", "unix", "host")
    verify_tls: bool = True
    require_tls: bool = False
    #: Ask for missing credentials rather than failing. ``None`` - the default,
    #: overridable with ``$XRD_PROMPT`` - means "only if somebody is there",
    #: which is a terminal on both stdin and stderr. See :mod:`xrd.auth.prompt`.
    prompt: bool | None = field(default_factory=lambda: _env_flag("XRD_PROMPT"))
    #: Where those questions go. ``None`` uses the terminal; a callable taking
    #: an :class:`~xrd.auth.prompt.Ask` puts them in a dialog or a notebook.
    prompter: Prompter | None = None

    # -- behaviour -----------------------------------------------------
    #: Re-open a read-only file and carry on when its data server disappears
    #: mid-read. Off means a lost connection surfaces as a
    #: :class:`~xrd.errors.TransientError` at the call that hit it.
    recover_handles: bool = True
    verify_checksums: bool = True
    preferred_checksum: str = "adler32"

    def evolve(self, **changes: object) -> Config:
        """A copy with ``changes`` applied."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        secret = {"token"}
        parts = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            parts.append(f"{f}=" + ("'<redacted>'" if f in secret and val else repr(val)))
        return f"Config({', '.join(parts)})"


_default = Config()
_current: contextvars.ContextVar[Config] = contextvars.ContextVar("xrd_config", default=_default)


def current() -> Config:
    """The configuration in effect right now."""
    return _current.get()


def configure(**changes: object) -> Config:
    """Mutate the process-wide default configuration and return it."""
    global _default
    _default = _default.evolve(**changes)
    _current.set(_default)
    return _default


@contextlib.contextmanager
def override(**changes: object) -> Iterator[Config]:
    """Temporarily apply ``changes`` to the ambient configuration.

    >>> with override(request_timeout=5.0):
    ...     ...
    """
    cfg = current().evolve(**changes)
    token = _current.set(cfg)
    try:
        yield cfg
    finally:
        _current.reset(token)
