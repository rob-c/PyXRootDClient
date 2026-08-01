"""``ztn`` - WLCG bearer tokens and SciTokens.

Discovery follows the WLCG Bearer Token Discovery specification, which is
also what the C client uses: an explicit token, then ``$BEARER_TOKEN``, then
``$BEARER_TOKEN_FILE``, then ``$XDG_RUNTIME_DIR/bt_u$UID``, then
``/tmp/bt_u$UID``.
"""

from __future__ import annotations

import base64
import json
import os
import time

from .._log import get_logger
from ..config import Config
from ..errors import TokenExpiredError
from .base import Credential, Offer

__all__ = ["TokenCredential", "discover_token", "token_claims", "token_expiry"]

_log = get_logger(__name__)


def _token_paths(config: Config) -> list[str]:
    paths = []
    if config.token_file:
        paths.append(config.token_file)
    uid = os.getuid()
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        paths.append(os.path.join(runtime, f"bt_u{uid}"))
    paths.append(f"/tmp/bt_u{uid}")
    return paths


def discover_token(config: Config | None = None) -> str | None:
    """Locate a bearer token, or ``None`` if there is not one to be had."""
    config = config or Config()
    if config.token:
        return config.token.strip()
    env = os.environ.get("BEARER_TOKEN")
    if env and env.strip():
        return env.strip()
    for path in _token_paths(config):
        try:
            with open(path, encoding="utf-8") as fh:
                content = fh.read().strip()
        except OSError:
            continue
        if content:
            _log.debug("using bearer token from %s", path)
            return content
    return None


def token_claims(token: str) -> dict[str, object]:
    """Decode a JWT's claim set without verifying it.

    Signature verification is the server's job; the client only reads the
    expiry so it can fail fast with a useful message instead of a 3010.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        return dict(json.loads(base64.urlsafe_b64decode(payload)))
    except (ValueError, TypeError):
        return {}


def token_expiry(token: str) -> float | None:
    """The ``exp`` claim as a UNIX timestamp, if the token carries one."""
    exp = token_claims(token).get("exp")
    return float(exp) if isinstance(exp, (int, float, str)) and str(exp).isdigit() else None


class TokenCredential(Credential):
    """``ztn`` - ``"ztn\\0<token>"``."""

    __slots__ = ("token", "expires_at")
    name = "ztn"

    def __init__(self, token: str) -> None:
        self.token = token
        self.expires_at = token_expiry(token)

    def initial(self) -> bytes:
        if self.expires_at is not None and self.expires_at <= time.time():
            when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.expires_at))
            raise TokenExpiredError(f"bearer token expired at {when}")
        return b"ztn\x00" + self.token.encode("ascii")

    @classmethod
    def available(
        cls, offer: Offer, config: Config, *, username: str, host: str
    ) -> TokenCredential | None:
        token = discover_token(config)
        return cls(token) if token else None

    def __repr__(self) -> str:
        return f"TokenCredential(len={len(self.token)}, token=<redacted>)"
