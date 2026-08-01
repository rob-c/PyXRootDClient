"""What reading a ROOT file can go wrong with.

Both are :class:`~xrd.errors.XRootDError`, so one ``except`` still covers
everything this package raises, and they are two rather than one because the
answers differ: a :class:`FormatError` means the bytes are wrong, and a
:class:`UnsupportedFeatureError` means they are right and this reader is the
one that is missing something.
"""

from __future__ import annotations

from ..errors import XRootDError

__all__ = ["ROOTError", "FormatError", "UnsupportedFeatureError"]


class ROOTError(XRootDError):
    """Base of everything :mod:`xrd.root` raises."""


class FormatError(ROOTError):
    """The bytes are not the ROOT format they claim to be."""


class UnsupportedFeatureError(ROOTError):
    """A valid ROOT file using something this reader does not do.

    Raised by name, with the class or algorithm in the message, rather than
    guessed at: a plausible misreading of physics data is worse than a
    refusal.
    """
