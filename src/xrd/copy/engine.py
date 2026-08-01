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
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import IO, Any, Literal, cast

from .._log import get_logger
from ..config import Config
from ..crypto import new as new_checksum
from ..errors import ChecksumMismatchError, UnsupportedError, kXR_Unsupported
from ..types import ChecksumInfo
from ..url import XRootDURL, parse

__all__ = ["copy", "copy_tree", "CopyResult", "SyncMode"]

#: Called with ``(bytes_done, total_or_None)`` after every chunk.
Progress = Callable[[int, "int | None"], None]

#: How ``copy_tree(sync=...)`` decides a file is already at the target:
#: ``"size"`` trusts the length, ``"mtime"`` also wants the copy to be no
#: older than the original, and ``"checksum"`` compares the bytes themselves.
SyncMode = Literal["size", "mtime", "checksum"]

_log = get_logger(__name__)

#: What :func:`copy` accepts on either side.
Endpoint = "str | os.PathLike[str] | XRootDURL | IO[bytes]"


@dataclass(frozen=True, slots=True)
class CopyResult:
    """What one completed transfer did."""

    source: str
    target: str
    #: Bytes this call moved - which is not the size of the file when the
    #: transfer resumed part way through it. See :attr:`resumed_at`.
    size: int
    seconds: float
    checksum: ChecksumInfo | None = None
    #: The offset a resumed transfer started from, and 0 for a whole copy, so
    #: ``resumed_at + size`` is the length of the finished file either way.
    resumed_at: int = 0

    @property
    def rate(self) -> float:
        """Bytes per second; ``inf`` if the copy took no measurable time."""
        return self.size / self.seconds if self.seconds > 0 else float("inf")

    @property
    def resumed(self) -> bool:
        return self.resumed_at > 0

    @property
    def verified(self) -> bool:
        return self.checksum is not None

    def __str__(self) -> str:
        # A dry run, or a copy too quick for the clock, has no rate to quote.
        rate = f", {self.rate / 1e6:.1f} MB/s" if self.seconds > 0 else ""
        resumed = f", resumed at {self.resumed_at}" if self.resumed_at else ""
        return f"{self.source} -> {self.target} ({self.size} bytes{resumed}{rate})"


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


def _resumer(url: XRootDURL, config: Config, stack: ExitStack, offset: int) -> IO[bytes]:
    """A writer for ``url`` positioned at ``offset``, keeping what is there.

    An HTTP target has no such thing: a ``PUT`` replaces the whole resource,
    so a partial upload can only be repeated, never continued.
    """
    if url.is_local:
        handle = stack.enter_context(open(url.path, "r+b"))
        handle.seek(offset)
        return handle
    if url.is_root:
        from ..io import open_url

        raw = stack.enter_context(open_url(url, "r+b", buffering=0, config=config))
        raw.seek(offset)
        return cast("IO[bytes]", raw)
    raise UnsupportedError(kXR_Unsupported, f"cannot resume a copy into {url.scheme}://")


@dataclass(frozen=True, slots=True)
class _Resume:
    """A transfer that picks up where an interrupted one stopped."""

    offset: int
    source: XRootDURL
    target: XRootDURL


def _continue_from(
    source: XRootDURL | None, target: XRootDURL | None, config: Config, *, overwrite: bool
) -> _Resume | None:
    """Where to continue ``target`` from, or ``None`` if there is nothing to."""
    if not overwrite:
        raise ValueError("resume continues a partial target, which overwrite=False forbids")
    if source is None or target is None:
        raise ValueError("resume needs a URL on both sides, not an already-open stream")
    there = _probe(target, config)
    if there is None or there[0] == 0:
        return None  # nothing there yet, so this is an ordinary copy
    here = _probe(source, config)
    if here is not None and there[0] > here[0]:
        raise ValueError(f"{target} is longer than {source}, so it is not a partial copy of it")
    return _Resume(there[0], source, target)


def _shifted(progress: Progress | None, offset: int) -> Progress | None:
    """``progress`` reporting where in the *file* it is, not where in the tail."""
    if progress is None:
        return None
    report = progress
    return lambda done, total: report(done + offset, total)


def _probe(url: XRootDURL, config: Config) -> tuple[int, int] | None:
    """``(size, mtime)`` for ``url``, or ``None`` when there is nothing there."""
    if url.is_local:
        try:
            info = os.stat(url.path)
        except OSError:
            return None
        return info.st_size, int(info.st_mtime)
    from ..client import FileSystem

    with FileSystem(url.with_path("/"), config) as fs:
        try:
            stat = fs.stat(url.path)
        except OSError:
            return None
    return stat.st_size, stat.st_mtime


def _remove(url: XRootDURL, config: Config) -> None:
    """Delete ``url`` - what turns a copy into a move."""
    if url.is_local:
        os.remove(url.path)
        return
    from ..client import FileSystem

    with FileSystem(url.with_path("/"), config) as fs:
        fs.remove(url.path)


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
    dry_run: bool = False,
    remove_source: bool = False,
    resume: bool = False,
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

    ``dry_run`` reports the transfer it would have made without moving a byte;
    ``remove_source`` deletes the source once the copy is on disk and has
    passed whatever verification was asked for, which together make a move.

    ``resume`` continues an interrupted transfer: whatever is already at the
    target is kept and the copy starts at the end of it, which
    :attr:`CopyResult.resumed_at` reports. A target that is not there yet is
    copied whole, so the flag is safe to set unconditionally on a retry. Since
    a continued transfer never reads the bytes it did not move, verification
    switches from digesting the stream to comparing the two files afterwards -
    which costs a read of whichever end is local. An HTTP target cannot be
    resumed at all, because a ``PUT`` replaces the whole resource.
    """
    cfg = config or Config()
    chunk = chunk_size or cfg.chunk_size
    algo = algorithm or cfg.preferred_checksum
    src_url, dst_url = _target_url(source), _target_url(target)

    if dry_run:
        known = _probe(src_url, cfg) if src_url is not None else None
        return CopyResult(
            source=str(src_url) if src_url else repr(source),
            target=str(dst_url) if dst_url else repr(target),
            size=known[0] if known else 0,
            seconds=0.0,
        )

    resuming = _continue_from(src_url, dst_url, cfg, overwrite=overwrite) if resume else None
    wanted = cfg.verify_checksums if verify is None else verify
    # A digest of the bytes in flight is only worth taking if some server can
    # be asked to compare it with what it holds - and if they are all the
    # bytes, which for a continued transfer they are not.
    checkable = next((u for u in (dst_url, src_url) if u is not None and not u.is_local), None)
    digest = new_checksum(algo) if wanted and checkable is not None and not resuming else None

    started = time.monotonic()
    with ExitStack() as stack:
        reader, total = (source, None) if src_url is None else _reader(src_url, cfg, stack)
        if resuming is not None:
            reader.seek(resuming.offset)
            writer = _resumer(resuming.target, cfg, stack, resuming.offset)
        elif dst_url is None:
            writer = target
        else:
            writer = _writer(dst_url, cfg, stack, overwrite=overwrite)
        report = progress if resuming is None else _shifted(progress, resuming.offset)
        size = _pump(reader, writer, total, chunk, report, digest)
    elapsed = time.monotonic() - started

    checksum = None
    if resuming is not None:
        if wanted:
            checksum = _compare_ends(
                resuming.source, resuming.target, cfg, algo, strict=verify is True
            )
    elif digest is not None and checkable is not None:
        checksum = _compare(checkable, cfg, algo, digest.hexdigest(), strict=verify is True)

    if remove_source and src_url is not None:
        _remove(src_url, cfg)  # only now: a failed verification kept the original

    return CopyResult(
        source=str(src_url) if src_url else repr(source),
        target=str(dst_url) if dst_url else repr(target),
        size=size,
        seconds=elapsed,
        checksum=checksum,
        resumed_at=resuming.offset if resuming is not None else 0,
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


def _compare_ends(
    source: XRootDURL, target: XRootDURL, config: Config, algorithm: str, *, strict: bool
) -> ChecksumInfo | None:
    """Verify a resumed copy by comparing the two files rather than the stream.

    A transfer that started part way through never saw the beginning of the
    file, so the digest taken in flight covers a tail and proves nothing. The
    only honest check left is to ask both ends what they hold.
    """
    try:
        ours = _digest_of(source, config, algorithm)
        theirs = _digest_of(target, config, algorithm)
    except OSError:
        if strict:
            raise
        return None  # an end that cannot checksum; the copy still happened
    if ours != theirs:
        raise ChecksumMismatchError(algorithm, theirs, ours)
    return ChecksumInfo(algorithm, theirs)


# ---------------------------------------------------------------------------
# Recursive copies
# ---------------------------------------------------------------------------


def _digest_of(url: XRootDURL, config: Config, algorithm: str) -> str:
    """``algorithm`` over ``url``: the server's answer, or ours if it is local."""
    if not url.is_local:
        return _server_checksum(url, config, algorithm).value.lower()
    digest = new_checksum(algorithm)
    with open(url.path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest().lower()


def _up_to_date(
    source: XRootDURL, target: XRootDURL, config: Config, mode: SyncMode, algorithm: str
) -> bool:
    """Is ``target`` already the copy of ``source`` that ``mode`` asks for?

    A different length always means a different file, whatever the mode, so
    that comparison comes first and saves the expensive one behind it.
    """
    theirs, ours = _probe(target, config), _probe(source, config)
    if theirs is None or ours is None or theirs[0] != ours[0]:
        return False
    if mode == "size":
        return True
    if mode == "mtime":
        return theirs[1] >= ours[1]
    return _digest_of(source, config, algorithm) == _digest_of(target, config, algorithm)


def _selected(rel: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    """``rsync``'s rule: an explicit include list is a whitelist, exclude wins."""
    if include and not any(fnmatch(rel, pattern) for pattern in include):
        return False
    return not any(fnmatch(rel, pattern) for pattern in exclude)


def _prune(target: XRootDURL, config: Config, keep: set[str], *, dry_run: bool) -> list[str]:
    """Remove everything under ``target`` that ``keep`` does not name.

    Each removal is logged, because a deletion nobody asked for by name is
    the one thing in a copy worth being able to read back afterwards.
    """
    removed = []
    for rel in _walk(target, config):
        if rel in keep:
            continue
        removed.append(rel)
        _log.info("%s %s", "would delete" if dry_run else "deleting", target / rel)
        if not dry_run:
            _remove(target / rel, config)
    return removed


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
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    sync: SyncMode | None = None,
    delete: bool = False,
    **options: Any,
) -> list[CopyResult]:
    """Copy a directory recursively, returning one result per file copied.

    Local target directories are created as needed; remote ones are, too,
    because a remote write asks the server for ``kXR_mkpath``. Options other
    than the ones named here - ``dry_run`` and ``verify`` among them - are
    passed through to :func:`copy`.

    ``include`` and ``exclude`` are :mod:`fnmatch` patterns matched against
    each path relative to ``source``; an include list is a whitelist and an
    exclusion always wins. ``sync`` skips files already at the target, and
    ``delete`` removes files under the target that the source does not have -
    excluded ones among them, since after this call the target is meant to
    hold what the selection describes and nothing else.
    """
    cfg = config or Config()
    src_url, dst_url = parse(source), parse(target)
    algo = options.get("algorithm") or cfg.preferred_checksum
    wanted = [rel for rel in _walk(src_url, cfg) if _selected(rel, include, exclude)]
    results = []
    for rel in wanted:
        destination = dst_url / rel
        if sync is not None and _up_to_date(src_url / rel, destination, cfg, sync, algo):
            continue
        if destination.is_local and not options.get("dry_run"):
            os.makedirs(os.path.dirname(destination.path), exist_ok=True)
        results.append(
            copy(src_url / rel, destination, config=cfg, progress=progress, **options)
        )
    if delete:
        _prune(dst_url, cfg, set(wanted), dry_run=bool(options.get("dry_run")))
    return results
