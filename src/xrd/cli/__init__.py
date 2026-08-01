"""The command-line tools, and the pieces both of them share.

``xrd-cp`` and ``xrd-fs`` are thin wrappers: everything they do is a call into
the library, so anything the CLI can do is something a program can do in one
line. Exit codes are the usual three - ``0`` success, ``1`` a runtime failure,
``2`` a usage error - and every command takes ``--json`` so a shell script can
consume the output without parsing columns.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Any

from ..config import Config
from ..url import XRootDURL, parse

__all__ = ["OK", "ERROR", "USAGE", "Endpoints", "dumps", "fail", "size_arg", "common_flags"]

OK, ERROR, USAGE = 0, 1, 2


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _plain(obj: Any) -> Any:
    """Make the library's own types serialisable without teaching them JSON."""
    if isinstance(obj, XRootDURL):
        return str(obj)  # before the dataclass branch: a URL is one, and reads badly as one
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def dumps(payload: object) -> str:
    """``--json`` output: stable key order, one document per invocation."""
    return json.dumps(payload, default=_plain, indent=2, sort_keys=False)


def fail(program: str, exc: BaseException) -> int:
    """Report ``exc`` the way a Unix tool does, and give back the exit code."""
    print(f"{program}: {exc}", file=sys.stderr)
    return ERROR


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

_SUFFIXES = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}


def size_arg(text: str) -> int:
    """``argparse`` type for a byte count written the way humans write it."""
    raw = text.strip().lower().removesuffix("b")
    scale = _SUFFIXES.get(raw[-1:], 1)
    digits = raw[:-1] if scale > 1 else raw
    try:
        value = int(digits)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a size: {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return value * scale


def common_flags(parser: argparse.ArgumentParser) -> None:
    """The options every command in both tools understands."""
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("-q", "--quiet", action="store_true", help="say nothing on success")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="log more (repeatable)")
    parser.add_argument("--token", metavar="TOKEN", help="bearer token to present")
    parser.add_argument("--user", metavar="NAME", help="username to authenticate as")
    parser.add_argument(
        "--no-verify-tls", action="store_true", help="do not verify the server certificate"
    )


def configure_logging(verbosity: int) -> None:
    """``-v`` is warnings, ``-vv`` info, ``-vvv`` the wire itself."""
    import logging

    if not verbosity:
        return
    level = {1: logging.WARNING, 2: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def config_from(args: argparse.Namespace) -> Config:
    """A :class:`~xrd.Config` carrying whatever the command line asked for."""
    configure_logging(args.verbose)
    settings: dict[str, object] = {}
    if args.token:
        settings["token"] = args.token
    if args.user:
        settings["username"] = args.user
    if args.no_verify_tls:
        settings["verify_tls"] = False
    return Config(**settings)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class Endpoints:
    """One :class:`~xrd.FileSystem` per endpoint, opened once and shared.

        >>> with Endpoints(config) as endpoints:
        ...     fs, path = endpoints.at("root://host//store/f.root")

    Commands take a list of URLs that often name the same server; without this
    a five-argument ``ls`` would open five connections.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._open: dict[tuple[str, str, int], Any] = {}

    def at(self, url: str | XRootDURL) -> tuple[Any, str]:
        """The filesystem for ``url``'s endpoint, and the path within it."""
        from ..client import FileSystem

        target = parse(url)
        if target.is_local:
            raise ValueError(f"{url} is a local path, not a remote endpoint")
        key = (target.scheme, target.host, target.port)
        found = self._open.get(key)
        if found is None:
            found = self._open[key] = FileSystem(target.with_path("/"), self.config)
        return found, target.path or "/"

    def close(self) -> None:
        for filesystem in self._open.values():
            filesystem.close()
        self._open.clear()

    def __enter__(self) -> Endpoints:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
