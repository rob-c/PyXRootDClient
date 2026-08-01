"""A pure-Python XRootD client.

Speaks the native ``root://`` binary protocol and HTTP/WebDAV, with no
compiled extension and no XRootD installation required.

    >>> import xrd
    >>> with xrd.open("root://eos.example.org//store/data.root") as f:
    ...     header = f.read(1024)

The four levels of the API, from most to least convenient:

``xrd.open`` / :class:`~xrd.path.XRootDPath`
    File objects and ``pathlib`` semantics.
:class:`~xrd.client.FileSystem` / :class:`~xrd.client.File`
    Explicit per-operation control, sync or async.
:class:`~xrd.session.Session`
    A single authenticated connection.
:mod:`xrd.proto`
    The sans-io protocol machinery.
"""

from __future__ import annotations

from .client import Checkpoint, File, FileSystem
from .config import Config, configure, current, find_config_file, override
from .copy import CopyResult, SyncMode, copy, copy_tree, third_party
from .errors import (
    AttrNotFoundError,
    AuthenticationError,
    BusyError,
    ChecksumMismatchError,
    ConnectionError,
    CredentialError,
    InvalidArgumentError,
    NoMechanismError,
    NoSpaceError,
    NotFoundError,
    ProtocolError,
    QuotaError,
    ReadOnlyError,
    RedirectLimitError,
    ServerError,
    TimeoutError,
    TokenExpiredError,
    TransientError,
    UnsupportedError,
    WaitLimitError,
    XRootDError,
)
from .flags import Access, DirListFlags, MkDirFlags, OpenFlags, QueryCode, StatInfoFlags
from .io import open_url as open
from .path import XRootDPath
from .path import XRootDPath as Path  # ``xrd.Path`` reads the way pathlib does
from .types import (
    CheckpointInfo,
    ChecksumInfo,
    CloneRange,
    DirEntry,
    LocationInfo,
    PageResult,
    PrepareStatus,
    ProtocolInfo,
    ReadRange,
    SpaceInfo,
    StatInfo,
    VFSInfo,
    WriteChunk,
)
from .url import XRootDURL, parse

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    """Expose ``xrd.aio`` without importing :mod:`asyncio` for everyone else.

    The async facade is one attribute access away, but the synchronous path
    never pays for it: nothing under ``xrd`` imports ``asyncio`` until someone
    asks for ``xrd.aio``.
    """
    if name == "aio":
        import importlib

        # Not ``from . import aio``: the import system resolves that by asking
        # this very function for the attribute, and the recursion is infinite.
        return importlib.import_module(f"{__name__}.aio")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    # the async facade (imported on first use)
    "aio",
    # configuration
    "Config",
    "configure",
    "current",
    "override",
    "find_config_file",
    # copying
    "copy",
    "copy_tree",
    "third_party",
    "CopyResult",
    "SyncMode",
    # files and paths
    "open",
    "XRootDPath",
    "Path",
    "FileSystem",
    "File",
    "Checkpoint",
    # urls
    "XRootDURL",
    "parse",
    # values
    "CheckpointInfo",
    "ChecksumInfo",
    "CloneRange",
    "DirEntry",
    "LocationInfo",
    "PageResult",
    "ProtocolInfo",
    "ReadRange",
    "StatInfo",
    "SpaceInfo",
    "PrepareStatus",
    "VFSInfo",
    "WriteChunk",
    # flags
    "Access",
    "DirListFlags",
    "MkDirFlags",
    "OpenFlags",
    "QueryCode",
    "StatInfoFlags",
    # errors
    "XRootDError",
    "ProtocolError",
    "ConnectionError",
    "TimeoutError",
    "TransientError",
    "AuthenticationError",
    "NoMechanismError",
    "CredentialError",
    "TokenExpiredError",
    "RedirectLimitError",
    "WaitLimitError",
    "ChecksumMismatchError",
    "ServerError",
    "NotFoundError",
    "NoSpaceError",
    "UnsupportedError",
    "ReadOnlyError",
    "QuotaError",
    "AttrNotFoundError",
    "BusyError",
    "InvalidArgumentError",
]
