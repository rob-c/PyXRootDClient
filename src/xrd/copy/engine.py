"""The copy pump and the endpoint adapters that feed it.

The engine is deliberately thin. Every endpoint - a local path, a remote URL,
or an already-open file object - is reduced to a binary stream first, so the
loop in the middle knows nothing about XRootD, and adding a protocol means
teaching :func:`_reader`/`_writer` one more scheme rather than touching the
transfer logic.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import IO, Any, cast

from ..config import Config
from ..crypto import new as new_checksum
from ..errors import ChecksumMismatchError, UnsupportedError, kXR_Unsupported
from ..types import ChecksumInfo
from ..url import XRootDURL, parse

__all__ = ["copy", "copy_tree", "CopyResult"]

#: Called with ``(bytes_done, total_or_None)`` after every chunk.
Progress = Callable[[int, "int | None"], None]

#: What :func:`copy` accepts on either side.
Endpoint = "str | os.PathLike[str] | XRootDURL | IO[bytes]"


@dataclass(frozen=True, slots=True)
class CopyResult:
    """What one completed transfer did."""

    source: str
    target: str
    size: int
    seconds: float
    checksum: ChecksumInfo | None = None

    @property
    def rate(self) -> float:
        """Bytes per second; ``inf`` if the copy took no measurable time."""
        return self.size / self.seconds if self.seconds > 0 else float("inf")

    @property
    def verified(self) -> bool:
        return self.checksum is not None

    def __str__(self) -> str:
        return f"{self.source} -> {self.target} ({self.size} bytes, {self.rate / 1e6:.1f} MB/s)"


# ---------------------------------------------------------------------------
# Endpoint adapters
# ---------------------------------------------------------------------------


def _is_stream(obj: object) -> bool:
    """True for something already open - a file object, a socket wrapper."""
    return hasattr(obj, "read") or hasattr(obj, "write")


def _target_url(obj: object) -> XRootDURL | None:
    """The URL ``obj`` names, or ``None`` if it is an open stream."""
    return None if _is_stream(obj) else parse(obj)  # type: ignore[arg-type]


def _reader(url: XRootDURL, config: Config, stack: ExitStack) -> tuple[IO[bytes], int | None]:
    """A binary reader for ``url``, plus its size when the endpoint knows it."""
    if url.is_local:
        return stack.enter_context(open(url.path, "rb")), os.path.getsize(url.path)
    if url.is_root:
        from ..io import open_url

        raw = stack.enter_context(open_url(url, "rb", buffering=0, config=config))
        # A RawIOBase reads bytes but is not an ``IO[bytes]`` as far as
        # typeshed is concerned; the pump only ever calls ``read``.
        return cast("IO[bytes]", raw), raw.file.size
    if url.is_http:
        from ..http import open_http

        return stack.enter_context(open_http(url, "rb", config=config)), None
    raise UnsupportedError(kXR_Unsupported, f"cannot read from {url.scheme}://")


def _writer(url: XRootDURL, config: Config, stack: ExitStack, *, overwrite: bool) -> IO[bytes]:
    """A binary writer for ``url``. ``overwrite=False`` means exclusive create."""
    mode = "wb" if overwrite else "xb"
    if url.is_local:
        return stack.enter_context(open(url.path, mode))
    if url.is_root:
        from ..io import open_url

        raw = stack.enter_context(open_url(url, mode, buffering=0, config=config))
        return cast("IO[bytes]", raw)
    if url.is_http:
        from ..http import open_http

        return stack.enter_context(open_http(url, mode, config=config))
    raise UnsupportedError(kXR_Unsupported, f"cannot write to {url.scheme}://")


def _server_checksum(url: XRootDURL, config: Config, algorithm: str) -> ChecksumInfo:
    """Ask whichever server owns ``url`` what it thinks the checksum is."""
    if url.is_http:
        from ..http import digest

        return digest(url, algorithm, config=config)
    from ..client import FileSystem

    with FileSystem(url.with_path("/"), config) as fs:
        return fs.checksum(url.path, algorithm)


# ---------------------------------------------------------------------------
# The pump
# ---------------------------------------------------------------------------


def _pump(
    reader: IO[bytes],
    writer: IO[bytes],
    total: int | None,
    chunk_size: int,
    progress: Progress | None,
    digest: Any,
) -> int:
    """Move every byte, digesting and reporting as it goes."""
    buffer = bytearray(chunk_size)
    view = memoryview(buffer)
    done = 0
    readinto = getattr(reader, "readinto", None)
    while True:
        if readinto is not None:
            count = readinto(buffer)
            piece = view[:count] if count else view[:0]
        else:  # a stream that only implements read()
            data = reader.read(chunk_size)
            count = len(data)
            piece = memoryview(data)
        if not count:
            break
        writer.write(piece)
        if digest is not None:
            digest.update(piece)
        done += count
        if progress is not None:
            progress(done, total)
    return done


def copy(
    source: Any,
    target: Any,
    *,
    chunk_size: int | None = None,
    verify: bool | None = None,
    algorithm: str | None = None,
    overwrite: bool = True,
    progress: Progress | None = None,
    config: Config | None = None,
) -> CopyResult:
    """Copy ``source`` to ``target`` and report what happened.

    Both sides may be a URL, a local path, an :class:`~xrd.XRootDPath`, or an
    already-open binary file object:

        >>> copy("root://host//store/f.root", "/scratch/f.root")     # download
        >>> copy("/scratch/f.root", "root://host//store/f.root")     # upload
        >>> copy("root://a//f", "root://b//f")                       # via here
        >>> with open("/scratch/f", "wb") as fh:
        ...     copy("root://host//store/f.root", fh)                # into a stream

    ``verify`` compares a digest taken while streaming against the server's
    own checksum - of the target if it is remote, otherwise of the source.
    Left at ``None`` it follows ``config.verify_checksums`` and degrades
    quietly when the server cannot checksum; set it to ``True`` to make an
    unverifiable copy an error.

    ``overwrite=False`` creates the target exclusively, raising
    :class:`FileExistsError` if it is already there.
    """
    cfg = config or Config()
    chunk = chunk_size or cfg.chunk_size
    algo = algorithm or cfg.preferred_checksum
    src_url, dst_url = _target_url(source), _target_url(target)

    wanted = cfg.verify_checksums if verify is None else verify
    # A digest of the bytes in flight is only worth taking if some server can
    # be asked to compare it with what it holds.
    checkable = next((u for u in (dst_url, src_url) if u is not None and not u.is_local), None)
    digest = new_checksum(algo) if wanted and checkable is not None else None

    started = time.monotonic()
    with ExitStack() as stack:
        reader, total = (source, None) if src_url is None else _reader(src_url, cfg, stack)
        writer = target if dst_url is None else _writer(dst_url, cfg, stack, overwrite=overwrite)
        size = _pump(reader, writer, total, chunk, progress, digest)
    elapsed = time.monotonic() - started

    checksum = None
    if digest is not None and checkable is not None:
        checksum = _compare(checkable, cfg, algo, digest.hexdigest(), strict=verify is True)

    return CopyResult(
        source=str(src_url) if src_url else repr(source),
        target=str(dst_url) if dst_url else repr(target),
        size=size,
        seconds=elapsed,
        checksum=checksum,
    )


def _compare(
    url: XRootDURL, config: Config, algorithm: str, ours: str, *, strict: bool
) -> ChecksumInfo | None:
    """Compare our streaming digest with the server's, or explain why not."""
    try:
        theirs = _server_checksum(url, config, algorithm)
    except OSError:
        if strict:
            raise
        return None  # the server cannot checksum; the copy still happened
    if theirs.value.lower() != ours.lower():
        raise ChecksumMismatchError(algorithm, theirs.value, ours)
    return theirs


# ---------------------------------------------------------------------------
# Recursive copies
# ---------------------------------------------------------------------------


def _walk(url: XRootDURL, config: Config) -> Iterator[str]:
    """Every file under ``url``, as paths relative to it."""
    if url.is_local:
        for root, _, names in os.walk(url.path):
            rel = os.path.relpath(root, url.path)
            for name in names:
                yield name if rel == "." else os.path.join(rel, name)
        return
    from ..client import FileSystem

    with FileSystem(url.with_path("/"), config) as fs:
        base = url.path.rstrip("/")
        for root, _, names in fs.walk(url.path):
            rel = root[len(base) :].strip("/")
            for name in names:
                yield f"{rel}/{name}" if rel else name


def copy_tree(
    source: Any,
    target: Any,
    *,
    config: Config | None = None,
    progress: Progress | None = None,
    **options: Any,
) -> list[CopyResult]:
    """Copy a directory recursively, returning one result per file.

    Local target directories are created as needed; remote ones are, too,
    because a remote write asks the server for ``kXR_mkpath``. Options other
    than ``progress`` and ``config`` are passed through to :func:`copy`.
    """
    cfg = config or Config()
    src_url, dst_url = parse(source), parse(target)
    results = []
    for rel in _walk(src_url, cfg):
        destination = dst_url / rel
        if destination.is_local:
            os.makedirs(os.path.dirname(destination.path), exist_ok=True)
        results.append(
            copy(src_url / rel, destination, config=cfg, progress=progress, **options)
        )
    return results
