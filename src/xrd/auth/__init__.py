"""Authentication mechanisms and the ladder that picks between them.

The server advertises what it accepts in the ``kXR_login`` security trailer;
:func:`select` intersects that with what this machine can actually produce,
ordered by :attr:`~xrd.config.Config.auth_order`. Every mechanism registers
unconditionally, because a zero-dependency install can genuinely attempt
``gsi``, ``ztn``, ``sss``, ``unix`` and ``host`` — all five are pure Python.

``krb5`` registers too, and reads your credential cache without help, but the
exchange needs :mod:`gssapi`. With no ticket and no module it stays quiet and
the ladder moves on; with a live ticket and no module it raises, so that
:func:`select` records *why* rather than falling through to ``unix`` with a
perfectly good ticket sitting there.

A mechanism that raises from ``available()`` is never fatal: :func:`select`
catches it, records the reason, and carries on to the next one.
"""

from __future__ import annotations

from collections.abc import Iterator

from .._log import get_logger
from ..config import Config
from ..errors import NoMechanismError
from .base import Credential, Offer, parse_security_trailer
from .gsi import GSICredential
from .krb5 import KerberosCredential
from .simple import HostCredential, UnixCredential
from .sss import SSSCredential
from .ztn import TokenCredential, discover_token

__all__ = [
    "Credential",
    "Offer",
    "parse_security_trailer",
    "register",
    "registry",
    "require",
    "select",
    "UnixCredential",
    "HostCredential",
    "SSSCredential",
    "TokenCredential",
    "GSICredential",
    "KerberosCredential",
    "discover_token",
]

_log = get_logger(__name__)

_REGISTRY: dict[str, type[Credential]] = {}


def register(cls: type[Credential]) -> type[Credential]:
    """Add a mechanism to the registry, keyed on its wire name."""
    _REGISTRY[cls.name] = cls
    return cls


def registry() -> dict[str, type[Credential]]:
    """Every mechanism this installation can attempt."""
    return dict(_REGISTRY)


for _cls in (
    UnixCredential,
    HostCredential,
    SSSCredential,
    TokenCredential,
    GSICredential,
    KerberosCredential,
):
    register(_cls)


def select(
    sec: str | list[Offer],
    config: Config | None = None,
    *,
    username: str = "",
    host: str = "",
    rejected: dict[str, str] | None = None,
) -> Iterator[Credential]:
    """Yield usable credentials, most preferred first.

    Iteration is lazy: building a GSI proxy costs a file read and a parse, so
    it only happens if the mechanisms ahead of it were rejected. Pass a dict
    as ``rejected`` to collect why each skipped mechanism was unusable.
    """
    config = config or Config()
    offers = parse_security_trailer(sec) if isinstance(sec, str) else list(sec)
    by_name = {o.name: o for o in offers}
    order = [n for n in config.auth_order if n in by_name]
    order += [o.name for o in offers if o.name not in order]
    why = {} if rejected is None else rejected

    for name in order:
        cls = _REGISTRY.get(name)
        if cls is None:
            why[name] = "not supported by this client"
            continue
        try:
            cred = cls.available(by_name[name], config, username=username, host=host)
        except Exception as exc:  # a broken credential must not mask the rest
            why[name] = f"{type(exc).__name__}: {exc}"
            _log.debug("%s unusable: %s", name, exc)
            continue
        if cred is None:
            why[name] = "no credential material found"
            continue
        yield cred


def require(
    sec: str | list[Offer],
    config: Config | None = None,
    *,
    username: str = "",
    host: str = "",
) -> list[Credential]:
    """Like :func:`select`, but raises when nothing is usable."""
    rejected: dict[str, str] = {}
    creds = list(select(sec, config, username=username, host=host, rejected=rejected))
    if not creds:
        offers = parse_security_trailer(sec) if isinstance(sec, str) else list(sec)
        raise NoMechanismError(offered=[o.name for o in offers], tried=rejected)
    return creds
