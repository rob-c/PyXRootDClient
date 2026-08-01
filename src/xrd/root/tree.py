"""What a tree looks like once it has been read: columns you can ask for.

The unit of I/O is the basket - a compressed block of consecutive entries for
one branch - and everything here is arranged around that. Asking for entries
100 to 200 reads the baskets that hold them and nothing else, which is what
makes it reasonable to iterate a hundred-gigabyte tree over the network from a
laptop: the bytes that cross the wire are the ones asked for.
"""

from __future__ import annotations

import array
import sys
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from .buffer import Buffer
from .compression import decompress
from .errors import UnsupportedFeatureError

if TYPE_CHECKING:
    from .file import Source
    from .objects import BranchRecord, LeafRecord

__all__ = ["Jagged", "Branch", "TTree"]

#: Entries per step when iterating, if nobody says otherwise.
DEFAULT_STEP = 10_000

if sys.byteorder == "little":  # pragma: no cover - one of two, decided by the machine

    def to_native(values: array.array[Any]) -> array.array[Any]:
        """ROOT writes big-endian; make it the order this machine reads."""
        values.byteswap()
        return values

else:  # pragma: no cover - big-endian machines, where ROOT's order is ours

    def to_native(values: array.array[Any]) -> array.array[Any]:
        return values


class Jagged(Sequence[Any]):
    """Rows of different lengths: one flat array, and where each row starts.

        >>> jets.tolist()                          # doctest: +SKIP
        [[12.5, 3.5], [], [88.0, 1.25, 0.5]]

    This is the shape a physics file is usually in - a variable number of
    particles per collision - and it is kept flat because that is how it
    arrives and how a tensor wants it back.
    """

    __slots__ = ("content", "offsets")

    def __init__(self, content: array.array[Any], offsets: array.array[Any]) -> None:
        self.content = content
        self.offsets = offsets

    def __repr__(self) -> str:
        return f"<Jagged {len(self)} rows of {len(self.content)} {self.content.typecode} values>"

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("row out of range")
        return self.content[self.offsets[index] : self.offsets[index + 1]]

    def lengths(self) -> list[int]:
        """How long each row is."""
        return [self.offsets[i + 1] - self.offsets[i] for i in range(len(self))]

    def tolist(self) -> list[list[Any]]:
        return [list(row) for row in self]

    def padded(self, width: int | None = None, fill: float = 0.0) -> tuple[array.array[Any], int]:
        """A rectangle: every row cut or filled to the same width.

        Gives back the flat values and the width, which is what a tensor of
        shape ``(rows, width)`` is made of. ``width`` defaults to the longest
        row, so nothing is lost unless a number is asked for.
        """
        if width is None:
            width = max(self.lengths(), default=0)
        code = self.content.typecode
        out = array.array(code, [fill if code in "fd" else int(fill)]) * (len(self) * width)
        for row in range(len(self)):
            start, end = self.offsets[row], self.offsets[row + 1]
            end = min(end, start + width)
            out[row * width : row * width + (end - start)] = self.content[start:end]
        return out, width


class Basket:
    """One compressed block of entries, decompressed and ready to slice."""

    __slots__ = ("keylen", "nevsize", "nevbuf", "last", "data", "offsets")

    def __init__(self, source: Source, seek: int, nbytes: int, has_offsets: bool) -> None:
        from .file import Key

        raw = source.read(seek, nbytes)
        head = Buffer(raw)
        key = Key(head)
        head.i16()  # the basket's own version, which adds nothing we use
        head.i32()  # the buffer size it was built with
        self.nevsize = head.i32()
        if self.nevsize < 0:  # a negative size means feature bits follow
            self.nevsize = -self.nevsize
            head.skip_record()
        self.nevbuf = head.i32()
        self.last = head.i32()

        self.keylen = key.keylen
        payload = raw[key.keylen :]
        self.data = decompress(payload, key.objlen) if key.compressed else payload
        self.offsets: list[int] = []
        if has_offsets:
            at = Buffer(self.data, key.keylen)
            at.pos = self.last
            self.offsets = at.i32s(at.i32())

    def start_of(self, entry: int, offset: int) -> int:
        """Where a leaf's bytes for one entry begin, in the payload."""
        if self.offsets:
            return self.offsets[entry] - self.keylen + offset
        return entry * self.nevsize + offset

    def end_of(self, entry: int) -> int:
        """Where the entry after this one begins, for a row of unknown length."""
        boundary = self.offsets[entry + 1] if entry + 1 < self.nevbuf else self.last
        return boundary - self.keylen


class Branch:
    """One column of a tree: a name, a type, and the entries under it.

        >>> tree["pt"].array(0, 1000)              # doctest: +SKIP
        array('f', [22.5, 19.0, ...])

    A branch holding several leaves appears once per leaf, named
    ``branch.leaf``, which is how ROOT itself writes such a name.
    """

    __slots__ = ("name", "record", "leaf", "_source", "_cached")

    def __init__(self, name: str, record: BranchRecord, leaf: LeafRecord, source: Source) -> None:
        self.name = name
        self.record = record
        self.leaf = leaf
        self._source = source
        self._cached: tuple[int, Basket] | None = None

    def __repr__(self) -> str:
        return f"<Branch {self.name!r} of {self.typename or self.leaf.classname}>"

    def __len__(self) -> int:
        return self.num_entries

    @property
    def title(self) -> str:
        return self.leaf.title

    @property
    def num_entries(self) -> int:
        return self.record.entries

    @property
    def typename(self) -> str | None:
        """``'float32'``, ``'str'``, or ``None`` when this reader cannot decode it."""
        return self.leaf.typename

    @property
    def length(self) -> int:
        """How many values each entry holds, for a column of fixed-size arrays.

        ``1`` for a plain number, ``10`` for ``x[10]``; the flat array a fixed
        column gives back is this wide per entry, which is the shape to give a
        tensor. A variable column says ``1`` here and means the lengths in
        :class:`Jagged` instead.
        """
        return self.leaf.length

    @property
    def is_jagged(self) -> bool:
        """Does the number of values per entry change from entry to entry?"""
        return self.leaf.count is not None

    @property
    def num_baskets(self) -> int:
        return len(self.record.basket_seek)

    def _refuse_if_unreadable(self) -> None:
        if self.typename is None:
            raise UnsupportedFeatureError(
                f"{self.name!r} holds {self.leaf.reason}; this reader does the plain numeric "
                f"and string leaves, and tree.unreadable lists the rest with the reason"
            )
        if self.is_jagged and len(self.record.leaves) > 1:
            raise UnsupportedFeatureError(
                f"{self.name!r} is a variable-length leaf sharing a branch with "
                f"{len(self.record.leaves) - 1} others, where the file does not say alone how "
                f"long each row is; write it as its own branch and it reads"
            )

    def _bounds(self, entry_start: int, entry_stop: int | None) -> tuple[int, int]:
        total = self.num_entries
        start = total + entry_start if entry_start < 0 else entry_start
        stop = total if entry_stop is None else entry_stop
        if stop < 0:
            stop += total
        return max(start, 0), min(stop, total)

    def _spans(self, start: int, stop: int) -> Iterator[tuple[int, int, int]]:
        """Which baskets hold ``[start, stop)``, and the part of each to take."""
        bounds = self.record.basket_entry
        for index in range(self.num_baskets):
            low, high = bounds[index], bounds[index + 1]
            if high <= start or low >= stop:
                continue
            yield index, max(start, low) - low, min(stop, high) - low

    def basket(self, index: int) -> Basket:
        """Read one basket, remembering the last so a small step is not a reread."""
        if self._cached is not None and self._cached[0] == index:
            return self._cached[1]
        basket = Basket(
            self._source,
            self.record.basket_seek[index],
            self.record.basket_bytes[index],
            self.record.entry_offset_len > 0,
        )
        self._cached = (index, basket)
        return basket

    def array(self, entry_start: int = 0, entry_stop: int | None = None) -> Any:
        """The values for a range of entries.

        Fixed-size columns come back as an :class:`array.array`, variable ones
        as :class:`Jagged`, and character leaves as a list of strings.
        """
        self._refuse_if_unreadable()
        start, stop = self._bounds(entry_start, entry_stop)
        if self.leaf.classname == "TLeafC":
            return self._strings(start, stop)

        leaf = self.leaf
        size = leaf.length * leaf.itemsize
        content = bytearray()
        counts: list[int] = []
        for index, low, high in self._spans(start, stop):
            basket = self.basket(index)
            if self.is_jagged:
                for entry in range(low, high):
                    at, end = basket.start_of(entry, leaf.offset), basket.end_of(entry)
                    counts.append((end - at) // leaf.itemsize)
                    content += basket.data[at:end]
            elif not basket.offsets and leaf.offset == 0 and basket.nevsize == size:
                content += basket.data[low * size : high * size]  # the whole run at once
            else:
                for entry in range(low, high):
                    at = basket.start_of(entry, leaf.offset)
                    content += basket.data[at : at + size]

        values = to_native(array.array(leaf.typecode, bytes(content)))
        if not self.is_jagged:
            return values
        offsets = array.array("q", [0]) * (len(counts) + 1)
        total = 0
        for row, count in enumerate(counts):
            total += count
            offsets[row + 1] = total
        return Jagged(values, offsets)

    def _strings(self, start: int, stop: int) -> list[str]:
        out = []
        for index, low, high in self._spans(start, stop):
            basket = self.basket(index)
            reader = Buffer(basket.data, basket.keylen)
            for entry in range(low, high):
                reader.pos = basket.start_of(entry, self.leaf.offset) + basket.keylen
                out.append(reader.string())
        return out


class TTree:
    """A tree, and the columns in it.

        >>> tree                                   # doctest: +SKIP
        <TTree 'events' with 4 branches and 1000 entries>
        >>> for batch in tree.iterate(["pt", "eta"], step=1000):   # doctest: +SKIP
        ...     train(batch["pt"], batch["eta"])

    ``len(tree)`` is the number of entries, because that is what a tree is a
    lot of; the number of columns is ``len(tree.branches)``.
    """

    __slots__ = ("name", "title", "num_entries", "branches", "unreadable", "_source")

    def __init__(
        self,
        name: str,
        title: str,
        num_entries: int,
        records: list[BranchRecord],
        source: Source,
    ) -> None:
        self.name = name
        self.title = title
        self.num_entries = num_entries
        self._source = source
        self.branches: dict[str, Branch] = {}
        self.unreadable: dict[str, str] = {}
        for top in records:
            for record in top.walk():
                many = len(record.leaves) > 1
                for leaf in record.leaves:
                    label = f"{record.name}.{leaf.name}" if many else record.name
                    self.branches[label] = Branch(label, record, leaf, source)
                    if leaf.typename is None:
                        self.unreadable[label] = leaf.reason

    def __repr__(self) -> str:
        return (
            f"<TTree {self.name!r} with {len(self.branches)} branches "
            f"and {self.num_entries} entries>"
        )

    def __len__(self) -> int:
        return self.num_entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.branches)

    def __contains__(self, name: object) -> bool:
        return name in self.branches

    def __getitem__(self, name: str) -> Branch:
        try:
            return self.branches[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a branch of {self.name!r}; there is "
                + ", ".join(self.branches)
            ) from None

    def keys(self) -> list[str]:
        """Every column, readable here or not."""
        return list(self.branches)

    def readable(self) -> list[str]:
        """The columns this reader can decode, which is what ``arrays`` defaults to."""
        return [name for name in self.branches if name not in self.unreadable]

    def typenames(self) -> dict[str, str]:
        """What each column holds, in Python's words."""
        return {
            name: branch.typename or f"? ({branch.leaf.classname})"
            for name, branch in self.branches.items()
        }

    def show(self) -> str:
        """A one-line-per-column summary, for looking before leaping.

            >>> print(tree.show())                 # doctest: +SKIP
            pt        float32   variable
            nmuon     int32
        """
        lines = []
        for name, branch in self.branches.items():
            kind = branch.typename or f"? ({branch.leaf.classname})"
            lines.append(f"{name:<24} {kind:<10}{' variable' if branch.is_jagged else ''}".rstrip())
        return "\n".join(lines)

    def arrays(
        self,
        names: Sequence[str] | None = None,
        entry_start: int = 0,
        entry_stop: int | None = None,
    ) -> dict[str, Any]:
        """Several columns at once, over the same range of entries.

        With no names, every column this reader can decode; the ones it cannot
        are in :attr:`unreadable` with the reason, rather than quietly missing.
        """
        wanted = self.readable() if names is None else list(names)
        return {name: self[name].array(entry_start, entry_stop) for name in wanted}

    def iterate(
        self,
        names: Sequence[str] | None = None,
        *,
        step: int = DEFAULT_STEP,
        entry_start: int = 0,
        entry_stop: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Walk the tree in batches, reading only what each batch needs.

            >>> for batch in tree.iterate(step=50_000):        # doctest: +SKIP
            ...     ...

        This is the one to reach for over a network: memory is one step, not
        one file, and a tree far larger than the machine goes through it.
        """
        if step <= 0:
            raise ValueError("step must be at least one entry")
        stop = self.num_entries if entry_stop is None else min(entry_stop, self.num_entries)
        at = entry_start
        while at < stop:
            yield self.arrays(names, at, min(at + step, stop))
            at += step
