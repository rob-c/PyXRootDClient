"""Reading ROOT files, in Python, over the same connection as everything else.

    >>> import xrd.root
    >>> with xrd.root.open_root("root://eos.example.org//store/events.root") as f:  # doctest: +SKIP
    ...     tree = f["Events"]
    ...     for batch in tree.iterate(["pt", "eta"], step=10_000):
    ...         train(batch)

ROOT is how experimental physics stores its data, and until now getting at it
from Python meant a C++ toolchain or a wheel with a compiler behind it. This
reads the format itself - keys, directories, trees, baskets and all four of
ROOT's compression algorithms - with nothing but the standard library, and it
reads it a basket at a time, so a tree on the other side of the world costs
the entries you asked for rather than the file.

What it does not do is every ROOT class ever written: histograms, split C++
objects and the truncated float types are refused by name, with the class in
the message, because a plausible misreading of physics data is worse than a
refusal. Plain numeric and string branches - which is what a training set
looks like - are read, jagged rows and all.

:mod:`xrd.root.ml` turns what comes out into tensors, if PyTorch is there.
"""

from __future__ import annotations

from .errors import FormatError, ROOTError, UnsupportedFeatureError
from .file import Directory, Key, ROOTFile, open_root
from .tree import Branch, Jagged, TTree

__all__ = [
    # opening
    "open_root",
    "ROOTFile",
    "Directory",
    "Key",
    # data
    "TTree",
    "Branch",
    "Jagged",
    # errors
    "ROOTError",
    "FormatError",
    "UnsupportedFeatureError",
]
