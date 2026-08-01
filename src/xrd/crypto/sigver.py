"""``kXR_sigver`` request signing.

When the server advertises a security level at or above ``kXR_secStandard``,
covered opcodes must be preceded by a ``kXR_sigver`` frame carrying
``HMAC-SHA256(key, seqno_be64 || request_header || payload)``. Conformant
servers answer the signature frame only when it fails to verify.
"""

from __future__ import annotations

import hashlib
import hmac
import struct

from ..proto import constants as c

__all__ = ["SIGNED_OPCODES", "LEVEL_OPCODES", "Signer", "sigver_hmac", "is_signed"]

#: Opcodes that mutate state, and so are signed from ``kXR_secStandard`` up.
SIGNED_OPCODES = frozenset(
    {
        c.kXR_chmod, c.kXR_fattr, c.kXR_mkdir, c.kXR_mv, c.kXR_open,
        c.kXR_pgwrite, c.kXR_prepare, c.kXR_rm, c.kXR_rmdir, c.kXR_set,
        c.kXR_truncate, c.kXR_write, c.kXR_writev, c.kXR_chkpoint, c.kXR_clone,
    }
)

#: Additional opcodes each security level brings in over the one below it.
LEVEL_OPCODES: dict[int, frozenset[int]] = {
    c.kXR_secNone: frozenset(),
    c.kXR_secCompatible: frozenset(),
    c.kXR_secStandard: SIGNED_OPCODES,
    c.kXR_secIntense: SIGNED_OPCODES | {c.kXR_close, c.kXR_dirlist, c.kXR_locate, c.kXR_stat},
    c.kXR_secPedantic: frozenset(range(c.kXR_1stRequest, c.kXR_clone + 1)),
}


def is_signed(opcode: int, level: int, overrides: dict[int, int] | None = None) -> bool:
    """Whether ``opcode`` needs a signature at security ``level``.

    ``overrides`` is the per-opcode table from the ``kXR_protocol`` security
    block; a value of ``kXR_secNone`` there exempts an otherwise-covered
    opcode, and any other value forces one in.
    """
    if overrides and opcode in overrides:
        return overrides[opcode] != c.kXR_secNone
    if level <= c.kXR_secCompatible:
        return False
    return opcode in LEVEL_OPCODES.get(level, SIGNED_OPCODES)


def sigver_hmac(key: bytes, seqno: int, header: bytes, payload: bytes) -> bytes:
    """The HMAC a ``kXR_sigver`` frame carries."""
    msg = struct.pack(">Q", seqno) + bytes(header) + bytes(payload)
    return hmac.new(bytes(key), msg, hashlib.sha256).digest()


class Signer:
    """Per-connection signing state: the session key and the sequence number.

    The sequence number is monotonic per connection and must never repeat, so
    :meth:`sign` is the only way to advance it.
    """

    __slots__ = ("key", "level", "overrides", "_seqno")

    def __init__(
        self,
        key: bytes,
        level: int = c.kXR_secNone,
        overrides: dict[int, int] | None = None,
    ) -> None:
        self.key = key
        self.level = level
        self.overrides = overrides or {}
        self._seqno = 0

    @property
    def seqno(self) -> int:
        return self._seqno

    def required(self, opcode: int) -> bool:
        return bool(self.key) and is_signed(opcode, self.level, self.overrides)

    def sign(self, frame: bytes) -> tuple[int, bytes] | None:
        """Signature for an encoded request frame, or ``None`` if unneeded.

        Returns ``(seqno, mac)``; the caller wraps them in a
        :class:`~xrd.proto.requests.Sigver` on the same stream.
        """
        opcode = struct.unpack_from(">H", frame, 2)[0]
        if not self.required(opcode):
            return None
        self._seqno += 1
        header = frame[: c.REQUEST_HDRLEN]
        # Exactly what dlen declares, not everything after the header: a
        # kXR_writev frame is followed by data the server reads separately and
        # does not sign, so signing it would fail verification on every write.
        dlen = struct.unpack_from(">i", frame, c.REQUEST_HDRLEN - 4)[0]
        payload = frame[c.REQUEST_HDRLEN : c.REQUEST_HDRLEN + dlen]
        return self._seqno, sigver_hmac(self.key, self._seqno, header, payload)

    def __repr__(self) -> str:
        return f"Signer(level={self.level}, seqno={self._seqno}, key=<redacted>)"
