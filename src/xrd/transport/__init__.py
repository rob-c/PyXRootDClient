"""Byte transports: a real socket, and an in-memory pipe for tests."""

from __future__ import annotations

from .base import Transport, tls_context
from .memory import MemoryTransport, pipe
from .sync import SocketTransport

__all__ = ["Transport", "SocketTransport", "MemoryTransport", "pipe", "tls_context"]
