"""Graphs, as points rather than as the arrays they are written as.

A graph in a ROOT file is two arrays of the same length and, if whoever wrote
it kept them, two or four more holding the error bars. This turns that into
the thing it is - points, and what the bars round them are - while leaving
every member it was written with in reach under :attr:`Graph.members`.
"""

from __future__ import annotations

import array
from typing import Any

from .errors import FormatError, UnsupportedFeatureError

__all__ = ["GRAPHS", "Graph"]

#: The graph classes this reads: points, points with error bars, points whose
#: bars are a different length each side, and points whose y errors are kept
#: in more than one layer.
GRAPHS = ("TGraph", "TGraphErrors", "TGraphAsymmErrors", "TGraphMultiErrors")


def _core(row: dict[str, Any]) -> dict[str, Any] | None:
    """The ``TGraph`` part of a graph, however far down it is inherited."""
    if "fNpoints" in row:
        return row
    for value in row.values():
        if isinstance(value, dict):
            found = _core(value)
            if found is not None:
                return found
    return None


class Graph:
    """A graph: its points, and the error bars round them.

        >>> for x, y in graph:                     # doctest: +SKIP
        ...     print(x, y)

    ``x`` and ``y`` are :class:`array.array` of one value per point, which
    :func:`numpy.asarray` takes without copying. :attr:`xerr` and :attr:`yerr`
    are the bars either side of each point, or ``None`` for a graph written
    without them; a graph keeping its errors in layers has them in
    :attr:`layers`.
    """

    __slots__ = ("classname", "members", "x", "y", "_core")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        core = _core(members)
        if core is None or "fX" not in core or "fY" not in core:
            raise FormatError(f"a {classname} was written without its points")
        points = int(core["fNpoints"])
        if len(core["fX"]) < points or len(core["fY"]) < points:
            raise FormatError(
                f"a {classname} of {points} points holds only "
                f"{min(len(core['fX']), len(core['fY']))} of them"
            )
        #: The class the file says this is, such as ``TGraphErrors``.
        self.classname = classname
        #: Every member, as it was written, for whatever is not here by name.
        self.members = members
        self._core = core
        #: Where each point is, one value per point.
        self.x: array.array[float] = array.array("d", core["fX"][:points])
        self.y: array.array[float] = array.array("d", core["fY"][:points])

    @property
    def name(self) -> str:
        """What the graph is called, which is the key it was written under."""
        return str(self._core["TNamed"]["fName"])

    @property
    def title(self) -> str:
        """The title it is drawn with, which is usually a sentence about it."""
        return str(self._core["TNamed"]["fTitle"])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[float, float]:
        return (self.x[index], self.y[index])

    def __iter__(self) -> Any:
        return iter(zip(self.x, self.y, strict=True))

    def points(self) -> list[tuple[float, float]]:
        """Every point as a pair, which is what a graph is a picture of."""
        return list(self)

    def _pair(self, low: Any, high: Any) -> tuple[array.array[float], array.array[float]]:
        """Two arrays of bars, cut to the points the graph says it has."""
        points = len(self)
        return (array.array("d", low[:points]), array.array("d", high[:points]))

    def _bars(
        self, axis: str, *spellings: tuple[str, str]
    ) -> tuple[array.array[float], array.array[float]] | None:
        """The bars either side along one axis, or ``None`` if there are none.

        Each spelling is the pair of names one class gives the low and the
        high bar; a graph whose bars are the same length both sides keeps a
        single array instead, and that array is both sides of the point.
        """
        for below, above in spellings:
            low, high = self.members.get(below), self.members.get(above)
            if low is not None and high is not None:
                return self._pair(low, high)
        same = self.members.get(f"fE{axis}")
        return None if same is None else self._pair(same, same)

    @property
    def xerr(self) -> tuple[array.array[float], array.array[float]] | None:
        """The bars left and right of each point, or ``None`` if none were kept."""
        return self._bars("X", ("fEXlow", "fEXhigh"), ("fExL", "fExH"))

    @property
    def layers(self) -> tuple[tuple[array.array[float], array.array[float]], ...]:
        """The bars below and above each point, one pair per layer of them.

        A graph told to keep its statistical and its systematic errors apart
        keeps a layer of each, in the order they were added. An ordinary graph
        keeps one layer, and a graph written without y errors keeps none.
        """
        low, high = self.members.get("fEyL"), self.members.get("fEyH")
        if low is None or high is None:
            bars = self._bars("Y", ("fEYlow", "fEYhigh"))
            return () if bars is None else (bars,)
        return tuple(self._pair(a, b) for a, b in zip(low, high, strict=True))

    @property
    def yerr(self) -> tuple[array.array[float], array.array[float]] | None:
        """The bars below and above each point, or ``None`` if none were kept.

        A graph keeping more than one layer of them refuses here rather than
        answer with one of the layers or with a sum of them it made up: which
        of those you want is yours to say, and :attr:`layers` has them all.
        """
        layers = self.layers
        if len(layers) > 1:
            raise UnsupportedFeatureError(
                f"this {self.classname} keeps its y errors in {len(layers)} layers, "
                f"which are one pair of bars only once you have said how to add them "
                f"up; they are each of them in .layers"
            )
        return layers[0] if layers else None

    def __repr__(self) -> str:
        return f"<{self.classname} {self.name!r} of {len(self)} points>"
