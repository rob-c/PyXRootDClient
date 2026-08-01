"""Client configuration.

Resolution order for any setting: explicit argument > the active
``contextvars`` override > configuration file > environment variable >
default.

The file is optional and only read when asked for - :meth:`Config.from_file`,
which the command line calls for you. It is ``configparser`` INI, the same
shape the C tools' ``~/.xrdrc`` has::

    [defaults]
    connect_timeout = 10
    auth_order = gsi, ztn, unix

    [alias eos]
    proxy = ~/.globus/eos-proxy
    require_tls = yes
"""

from __future__ import annotations

import configparser
import contextlib
import contextvars
import difflib
import getpass
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .auth.prompt import Prompter

__all__ = ["Config", "current", "configure", "override", "find_config_file", "CONFIG_PATHS"]

#: Where :meth:`Config.from_file` looks when it is not told, in order.
#: ``$XRD_CONFIG`` beats all of them and, being explicit, must exist.
CONFIG_PATHS = ("~/.config/xrd/config.ini", "~/.xrdrc")

#: The section every configuration file may have, applied before any alias.
DEFAULTS_SECTION = "defaults"

#: Settings a file has no business carrying: a callable cannot be spelled in
#: INI, and a literal token in a dotfile is a secret in every backup of it -
#: ``token_file`` says the same thing without copying the bearer around.
_NOT_IN_FILES = {"prompter": "it is a callable", "token": "use token_file instead"}


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


def find_config_file() -> str | None:
    """The configuration file that would be read, or ``None`` if there is none.

    ``$XRD_CONFIG`` wins and is taken at its word: pointing it at a file that
    is not there is a mistake worth hearing about, not a reason to fall back
    to somebody else's dotfile.
    """
    named = os.environ.get("XRD_CONFIG")
    if named:
        return os.path.expanduser(named)
    for candidate in CONFIG_PATHS:
        path = os.path.expanduser(candidate)
        if os.path.isfile(path):
            return path
    return None


def _as_flag(name: str, raw: str) -> bool:
    """INI truth, ``configparser``'s vocabulary."""
    try:
        return configparser.ConfigParser.BOOLEAN_STATES[raw.strip().lower()]
    except KeyError:
        raise ValueError(f"{name}: {raw!r} is not a yes/no value") from None


def _coerce(name: str, raw: str, annotation: str) -> object:
    """One INI string, as the type the field is declared with."""
    text = raw.strip()
    if "Sequence" in annotation:
        return tuple(part.strip() for part in text.replace(",", " ").split() if part.strip())
    if annotation.startswith("bool"):
        if not text and "None" in annotation:
            return None
        return _as_flag(name, text)
    if not text and "None" in annotation:
        return None
    try:
        if annotation.startswith("int"):
            return int(text)
        if annotation.startswith("float"):
            return float(text)
    except ValueError:
        raise ValueError(f"{name}: {raw!r} is not a number") from None
    return os.path.expanduser(text)


def _settings_from(
    parser: configparser.ConfigParser, section: str, where: str
) -> dict[str, object]:
    """One section, validated against the fields :class:`Config` actually has."""
    known = {f.name: str(f.type) for f in fields(Config)}
    settings: dict[str, object] = {}
    for key, raw in parser.items(section):
        name = key.strip().lower().replace("-", "_")
        refusal = _NOT_IN_FILES.get(name)
        if refusal is not None:
            raise ValueError(f"{where}: {name} cannot be set in a file - {refusal}")
        if name not in known:
            close = difflib.get_close_matches(name, known, n=1)
            hint = f", did you mean {close[0]}?" if close else ""
            raise ValueError(f"{where}: unknown setting {name!r}{hint}")
        settings[name] = _coerce(f"{where}: {name}", raw, known[name])
    return settings


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
    parallel_files: int = field(default_factory=lambda: _env_int("XRD_CPPARALLELFILES", 1))

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

    @classmethod
    def from_file(
        cls, path: str | os.PathLike[str] | None = None, *, alias: str | None = None
    ) -> Config:
        """Build a configuration from an INI file.

            >>> config = Config.from_file(alias="eos")   # doctest: +SKIP

        ``[defaults]`` is applied first, then ``[alias <name>]`` on top of it,
        so an alias only has to say what it changes. With no ``path``, the
        first of :data:`CONFIG_PATHS` that exists is read - and if none does,
        the defaults are what you get, which is what an absent dotfile should
        mean. A named ``alias`` that the file does not define is an error: a
        typo there would silently connect as somebody else.
        """
        target = str(path) if path is not None else find_config_file()
        if target is None:
            if alias is not None:
                raise FileNotFoundError(f"no configuration file, so no alias {alias!r}")
            return cls()
        parser = configparser.ConfigParser()
        try:
            with open(target, encoding="utf-8") as handle:
                parser.read_file(handle, source=target)
        except configparser.Error as exc:
            raise ValueError(f"{target}: {exc}") from None
        settings: dict[str, object] = {}
        if parser.has_section(DEFAULTS_SECTION):
            settings.update(_settings_from(parser, DEFAULTS_SECTION, target))
        if alias is not None:
            section = f"alias {alias}"
            if not parser.has_section(section):
                offered = sorted(
                    name[len("alias ") :] for name in parser.sections() if name.startswith("alias ")
                )
                known = f"; this file defines {', '.join(offered)}" if offered else ""
                raise KeyError(f"{target} has no alias {alias!r}{known}")
            settings.update(_settings_from(parser, section, f"{target} [{section}]"))
        return cls(**settings)  # type: ignore[arg-type]

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
