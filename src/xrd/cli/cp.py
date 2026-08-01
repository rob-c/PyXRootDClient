"""``xrd-cp`` - copy files between anything and anything.

    $ xrd-cp root://eos.example.org//store/data.root /tmp/
    $ xrd-cp -r /tmp/results davs://dav.example.org/store/results
    $ xrd-cp --tpc root://a//store/f.root root://b//store/f.root

The tool is ``cp``: the last argument is the destination, several sources are
allowed when it is a directory, and a trailing ``/`` means "into this
directory". Everything it does is one call into :func:`xrd.copy`.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import sys
from collections.abc import Sequence
from typing import TextIO, TypedDict

from ..config import Config
from ..copy import CopyResult, copy, copy_tree, third_party
from ..errors import XRootDError
from ..url import XRootDURL, parse
from . import OK, USAGE, Endpoints, common_flags, config_from, dumps, fail, size_arg

__all__ = ["main"]

PROGRAM = "xrd-cp"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Copy files between root://, https://, and the local filesystem.",
    )
    parser.add_argument("source", nargs="+", help="what to copy; several when DEST is a directory")
    parser.add_argument("dest", help="where to put it")
    parser.add_argument("-r", "--recursive", action="store_true", help="copy directories")
    parser.add_argument(
        "-n",
        "--no-clobber",
        action="store_true",
        help="fail rather than overwrite an existing destination",
    )
    parser.add_argument(
        "--tpc", action="store_true", help="third-party copy: let the servers move the data"
    )
    parser.add_argument(
        "--verify", dest="verify", action="store_true", help="require a checksum match"
    )
    parser.add_argument(
        "--no-verify", dest="verify", action="store_false", help="skip checksum verification"
    )
    parser.set_defaults(verify=None)
    parser.add_argument("-a", "--algorithm", metavar="NAME", help="checksum to verify with")
    parser.add_argument("--chunk-size", type=size_arg, metavar="N", help="transfer chunk, e.g. 8M")
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        default=None,
        help="show progress (default on a tty)",
    )
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false", help="never show progress"
    )
    common_flags(parser)
    return parser


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


class Bar:
    """A one-line progress display for a terminal, and nothing else.

    Deliberately not a dependency on ``tqdm``: the callback protocol
    :func:`xrd.copy` takes is ``(done, total)``, so anyone who wants ``tqdm``
    passes ``tqdm(...).update`` themselves.
    """

    def __init__(self, label: str, stream: TextIO | None = None) -> None:
        self.label = label
        self.stream = stream or sys.stderr
        self._last = -1

    def __call__(self, done: int, total: int | None) -> None:
        if total:
            percent = int(100 * done / total)
            if percent == self._last:
                return
            self._last = percent
            text = f"{self.label} {percent:3d}% {_human(done)}/{_human(total)}"
        else:
            text = f"{self.label}     {_human(done)}"
        print(f"\r{text}", end="", file=self.stream, flush=True)

    def finish(self) -> None:
        if self._last >= 0 or self.stream is not sys.stderr:
            print(file=self.stream)


class _CopyOptions(TypedDict, total=False):
    """What the flags may override; anything absent keeps the library default."""

    overwrite: bool
    verify: bool
    algorithm: str
    chunk_size: int


def _human(size: int) -> str:
    """A byte count the way a person reads it."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Destination resolution
# ---------------------------------------------------------------------------


def _is_dir(url: XRootDURL, endpoints: Endpoints) -> bool:
    """Does ``url`` name a directory that already exists?"""
    if url.path.endswith("/"):
        return True
    if url.is_local:
        return os.path.isdir(url.path)
    filesystem, path = endpoints.at(url)
    try:
        return bool(filesystem.isdir(path))
    except XRootDError:
        return False


def _destination(source: XRootDURL, dest: XRootDURL, *, into: bool) -> XRootDURL:
    """``cp f d`` writes ``d``; ``cp f d/`` and ``cp f dir`` write ``dir/f``."""
    if not into:
        return dest
    name = posixpath.basename(source.path.rstrip("/")) or "root"
    return dest / name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = config_from(args)
    sources = [parse(s) for s in args.source]
    dest = parse(args.dest)
    show = args.progress if args.progress is not None else (sys.stderr.isatty() and not args.quiet)

    try:
        with Endpoints(config) as endpoints:
            into = _is_dir(dest, endpoints)
            if len(sources) > 1 and not into:
                print(f"{PROGRAM}: {args.dest} is not a directory", file=sys.stderr)
                return USAGE
            results = _run(sources, dest, args, config, into=into, show=show)
    except (XRootDError, OSError, ValueError) as exc:
        return fail(PROGRAM, exc)

    if args.json:
        print(dumps([_record(r) for r in results]))
    elif not args.quiet:
        for result in results:
            print(result)
    return OK


def _run(
    sources: list[XRootDURL],
    dest: XRootDURL,
    args: argparse.Namespace,
    config: Config,
    *,
    into: bool,
    show: bool,
) -> list[CopyResult]:
    """Do the transfers the parsed command line asks for."""
    options: _CopyOptions = {"overwrite": not args.no_clobber}
    if args.verify is not None:
        options["verify"] = args.verify
    if args.algorithm:
        options["algorithm"] = args.algorithm
    if args.chunk_size:
        options["chunk_size"] = args.chunk_size

    results: list[CopyResult] = []
    for source in sources:
        target = _destination(source, dest, into=into)
        bar = Bar(posixpath.basename(source.path.rstrip("/")) or str(source)) if show else None
        try:
            if args.tpc:
                results.append(
                    third_party(source, target, config=config, overwrite=not args.no_clobber)
                )
            elif args.recursive:
                results.extend(
                    copy_tree(source, target, config=config, progress=bar, **options)
                )
            else:
                results.append(copy(source, target, config=config, progress=bar, **options))
        finally:
            if bar is not None:
                bar.finish()
    return results


def _record(result: CopyResult) -> dict[str, object]:
    """One transfer as JSON: the dataclass plus what it computes."""
    return {
        "source": result.source,
        "target": result.target,
        "size": result.size,
        "seconds": round(result.seconds, 6),
        "rate": round(result.rate, 3) if result.seconds > 0 else None,
        "verified": result.verified,
        "checksum": None if result.checksum is None else str(result.checksum),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
