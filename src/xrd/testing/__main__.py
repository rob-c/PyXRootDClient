"""``python -m xrd.testing`` - share a directory over ``root://``.

Kept out of :mod:`xrd.testing.server` so that running it does not import that
module a second time under another name, which is what ``python -m`` does to a
module its own package has already imported.
"""

from __future__ import annotations

from .server import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
