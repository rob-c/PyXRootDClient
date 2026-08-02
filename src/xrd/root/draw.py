"""Pictures without a canvas: matplotlib when it is there, characters when not.

ROOT draws through a ``TCanvas``, which this library does not carry. What it
has instead is the two ways Python usually looks at data: matplotlib axes,
made on demand and returned so the caller can keep styling them, and plain
text, which needs nothing installed and goes anywhere a string goes.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import UnsupportedFeatureError

__all__ = ["axes", "bar", "shade"]

#: A bar grown one eighth of a character at a time, then a full block.
EIGHTHS = "▏▎▍▌▋▊▉█"
#: A grid cell, darker the fuller it is.
SHADES = " ░▒▓█"


def axes() -> Any:
    """A fresh matplotlib axes to draw on, or a refusal that says what to do."""
    try:
        from matplotlib import pyplot
    except ImportError:
        raise UnsupportedFeatureError(
            "drawing this needs matplotlib, which is not installed: "
            "pip install matplotlib - or use .text(), which draws in the "
            "terminal with nothing installed at all"
        ) from None
    return pyplot.subplots()[1]


def bar(fraction: float, width: int) -> str:
    """A horizontal bar filling ``fraction`` of ``width`` characters.

        >>> bar(0.5, 4)
        '██'
    """
    eighths = round(max(0.0, min(1.0, fraction)) * width * 8)
    text = "█" * (eighths // 8)
    if eighths % 8:
        text += EIGHTHS[eighths % 8 - 1]
    return text


def shade(fraction: float) -> str:
    """One cell of a grid, in five steps from empty to full.

        >>> shade(0.9)
        '█'
    """
    if fraction <= 0:
        return SHADES[0]
    return SHADES[max(1, min(4, math.ceil(fraction * 4)))]


def missing_picture(what: str, dimensions: int) -> UnsupportedFeatureError:
    """The refusal for a histogram with more axes than a picture has."""
    return UnsupportedFeatureError(
        f"a {dimensions}-dimensional {what} has no honest flat picture; take "
        f".values() and slice it down to the two dimensions you want to see"
    )
