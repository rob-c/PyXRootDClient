"""Public flag and enum types.

``IntFlag``/``IntEnum`` so they compose with ``|``, compare as ints on the
wire, and have a readable ``repr`` in tracebacks.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag

__all__ = [
    "OpenFlags",
    "Access",
    "DirListFlags",
    "QueryCode",
    "MkDirFlags",
    "StatInfoFlags",
    "PrepareFlags",
    "LocateFlags",
    "FattrCode",
    "ChkPointCode",
]


class OpenFlags(IntFlag):
    """``kXR_open`` options."""

    NONE = 0
    COMPRESS = 0x0001
    DELETE = 0x0002
    FORCE = 0x0004
    NEW = 0x0008
    READ = 0x0010
    UPDATE = 0x0020
    REFRESH = 0x0080
    MAKEPATH = 0x0100
    APPEND = 0x0200
    RETSTAT = 0x0400
    REPLICA = 0x0800
    POSC = 0x1000
    NOWAIT = 0x2000
    SEQIO = 0x4000
    WRITE = 0x8000


class Access(IntFlag):
    """``kXR_open`` / ``kXR_mkdir`` / ``kXR_chmod`` mode bits (POSIX order)."""

    NONE = 0
    OTHER_EXEC = 0o001
    OTHER_WRITE = 0o002
    OTHER_READ = 0o004
    GROUP_EXEC = 0o010
    GROUP_WRITE = 0o020
    GROUP_READ = 0o040
    OWNER_EXEC = 0o100
    OWNER_WRITE = 0o200
    OWNER_READ = 0o400


class DirListFlags(IntFlag):
    NONE = 0
    ONLINE = 0x01
    STAT = 0x02
    CKSUM = 0x04
    RECURSIVE = 0x08


class MkDirFlags(IntFlag):
    NONE = 0
    MAKEPATH = 0x01


class QueryCode(IntEnum):
    STATS = 1
    PREPARE = 2
    CHECKSUM = 3
    XATTR = 4
    SPACE = 5
    CHECKSUM_CANCEL = 6
    CONFIG = 7
    VISA = 8
    OPAQUE = 16
    OPAQUE_FILE = 32
    OPAQUE_GROUP = 64


class StatInfoFlags(IntFlag):
    NONE = 0
    X_SET = 0x01
    IS_DIR = 0x02
    OTHER = 0x04
    OFFLINE = 0x08
    IS_READABLE = 0x10
    IS_WRITABLE = 0x20
    POSC_PENDING = 0x40
    BACKUP_EXISTS = 0x80


class PrepareFlags(IntFlag):
    """``kXR_prepare`` options.

    Everything up to ``USE_TCP`` is a bit of the options byte. ``EVICT``
    arrived later and lives in ``optionX``, the extended half-word, so it is
    spelled here one byte up and :class:`~xrd.proto.requests.Prepare` puts it
    where the protocol wants it. Combining the two still works::

        fs.prepare(paths, flags=PrepareFlags.EVICT | PrepareFlags.NOTIFY)
    """

    NONE = 0
    CANCEL = 1
    NOTIFY = 2
    NO_ERRORS = 4
    STAGE = 8
    WRITE_MODE = 16
    COLOCATE = 32
    FRESH = 64
    USE_TCP = 128
    EVICT = 1 << 8


class LocateFlags(IntFlag):
    NONE = 0
    ADD_PEERS = 1 << 0
    REFRESH = 1 << 7
    PREFER_NAME = 1 << 8
    NO_WAIT = 1 << 13


class FattrCode(IntEnum):
    DEL = 0
    GET = 1
    LIST = 2
    SET = 3


class ChkPointCode(IntEnum):
    BEGIN = 0
    COMMIT = 1
    QUERY = 2
    ROLLBACK = 3
    XEQ = 4
