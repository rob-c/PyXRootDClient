"""Helpers for testing code that talks to XRootD.

    from xrd.testing import FakeServer

    with FakeServer(files={"/data/a.root": b"hello"}) as server:
        fs = xrd.FileSystem(server.url)
        assert fs.read_bytes("/data/a.root") == b"hello"

:class:`FakeDAVServer` is the same idea for ``http``, ``https`` and WebDAV,
and :class:`FaultProxy` is the network those servers do not have: put it in
front of either one and the connection can be made to drop, stall, corrupt or
reorder on cue.

Nothing here is imported by the library itself, so it costs nothing at run
time; it is a supported part of the public API all the same, because a client
library that cannot be tested without a storage element is not much use.
"""

from __future__ import annotations

from .faults import FaultProxy
from .http import FakeDAVServer
from .server import FakeServer, error, frame

__all__ = ["FakeServer", "FakeDAVServer", "FaultProxy", "frame", "error"]
