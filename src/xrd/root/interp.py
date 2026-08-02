"""From what a branch says it holds to what Python gets back.

A column is one of three shapes and no more: a flat run of numbers, rows of
numbers of differing lengths, or one Python object per entry. Everything
ROOT can write into a branch - a member split out of a C++ class, a
``vector<float>``, a ``map<string,short>`` - lands in one of the three, and
whatever does not land in one of them is refused by name rather than read
approximately. A plausible misreading of physics data is worse than a
refusal, which is why this module says no as precisely as it says yes.
"""

from __future__ import annotations

import array
import math
import struct
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .buffer import Buffer, to_native
from .cxx import Mapping, Prim, Seq, Str, parse, py_name
from .errors import FormatError, UnsupportedFeatureError

if TYPE_CHECKING:
    from .file import Source
    from .objects import BranchRecord, LeafRecord
    from .tree import Basket

__all__ = ["Column", "Flat", "Rows", "Values", "Members", "Refused", "build"]

#: Set in a record's version when a container was written field by field
#: rather than object by object: all the keys, then all the values.
MEMBER_WISE = 0x4000

#: A record header - four bytes of byte count, two of version.
RECORD = 6

#: The streamer type of a member, offset by the shape it was declared in:
#: ``fType`` is the fundamental type plus one of these.
OFFSET_L = 20  # a fixed-size array, ``x[10]``
OFFSET_P = 40  # a pointer to a counted one, ``x[n]``

#: The two fundamental types that are not their own width: a float squeezed
#: into a range the declaration's comment spells out. ``True`` for the one
#: that is a ``double`` when the comment asks for nothing in particular.
PACKED = {9: True, 19: False}

#: The fundamental types, as ROOT numbers them in a streamer.
BASIC: dict[int, Prim] = {
    1: Prim("int8", "b", 1),
    2: Prim("int16", "h", 2),
    3: Prim("int32", "i", 4),
    4: Prim("int64", "q", 8),
    5: Prim("float32", "f", 4),
    6: Prim("int32", "i", 4),  # a counter, which is an int that another branch uses
    8: Prim("float64", "d", 8),
    11: Prim("uint8", "B", 1),
    12: Prim("uint16", "H", 2),
    13: Prim("uint32", "I", 4),
    14: Prim("uint64", "Q", 8),
    15: Prim("uint32", "I", 4),  # a bit field, which is written as the integer it is
    16: Prim("int64", "q", 8),
    17: Prim("uint64", "Q", 8),
    18: Prim("bool", "b", 1),
}

#: What the streamer types this reader will not decode actually are, so that
#: a refusal names the thing rather than the number.
KINDS = {
    0: "a base class written into the entry",
    7: "a C string pointer",
    10: "a legacy char",
    61: "an object",
    62: "an object without a dictionary",
    63: "a pointer to an object",
    64: "a pointer to an object",
    66: "a TObject",
    67: "a TNamed",
    365: "an STL string",
    500: "a class with a streamer of its own",
    501: "a loop over a member of another class",
}

#: How a packed float column turns its bytes into the doubles they stand for.
Unpack = Callable[[bytes], "array.array[Any]"]

_COUNT = struct.Struct(">I")
_BITS = struct.Struct(">I")
_FLOAT = struct.Struct(">f")

#: How ROOT lets a packing range be written in units of pi, longest first.
_PI = (
    ("2pi", 2 * math.pi),
    ("2*pi", 2 * math.pi),
    ("twopi", 2 * math.pi),
    ("pi/2", math.pi / 2),
    ("pi/4", math.pi / 4),
    ("pi", math.pi),
)


class Column:
    """How one branch's bytes become values, and what to call them."""

    __slots__ = ("typename",)
    #: ``flat``, ``rows``, ``values`` or ``refused``.
    kind = "refused"

    def __init__(self, typename: str | None) -> None:
        self.typename = typename

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.typename or 'unreadable'}>"


class Numeric(Column):
    """A column of numbers: how wide one is on disk, and how it decodes.

    Nearly every column is the machine's own numbers in the other byte order,
    which is a byte swap and nothing else. A ``Double32_t`` is not: it is a
    double squeezed into three or four bytes, and ``unpack`` is how those
    bytes come back out as doubles.
    """

    __slots__ = ("typecode", "itemsize", "unpack")

    def __init__(self, prim: Prim, unpack: Unpack | None = None) -> None:
        super().__init__(prim.typename)
        self.typecode = prim.typecode
        #: Bytes one value takes in the file, which is not always its width here.
        self.itemsize = prim.itemsize
        self.unpack = unpack

    def decode(self, raw: bytes) -> array.array[Any]:
        """A run of values, from the bytes the file holds them in."""
        if self.unpack is not None:
            return self.unpack(raw)
        return to_native(array.array(self.typecode, raw))


class Flat(Numeric):
    """The same number of values in every entry, one after another."""

    __slots__ = ("length",)
    kind = "flat"

    def __init__(
        self, prim: Prim, length: int, unpack: Unpack | None = None
    ) -> None:
        super().__init__(prim, unpack)
        self.length = length


class Rows(Numeric):
    """A different number of values in each entry, and where each row is."""

    __slots__ = ("header", "counted")
    kind = "rows"

    def __init__(
        self,
        prim: Prim,
        header: int,
        counted: bool,
        unpack: Unpack | None = None,
    ) -> None:
        super().__init__(prim, unpack)
        #: Bytes in front of the values: none, a marker byte, or a record.
        self.header = header
        #: Does the entry say how many values it holds, or only where it ends?
        self.counted = counted

    def span(self, basket: Basket, entry: int, offset: int) -> tuple[int, int]:
        """Where this entry's values start and stop inside the basket."""
        start = basket.start_of(entry, offset) + self.header
        end = basket.end_of(entry)
        if not self.counted:
            return start, end
        count = _COUNT.unpack_from(basket.data, start - 4)[0]
        stop = start + count * self.itemsize
        if stop > end:
            raise FormatError(
                f"an entry says it holds {count} values, which do not fit in the "
                f"{end - start} bytes it was written into"
            )
        return start, stop


class Values(Column):
    """One Python object per entry: a string, a list, a dict."""

    __slots__ = ("_read",)
    kind = "values"

    def __init__(self, typename: str, read: Callable[[Buffer], Any]) -> None:
        super().__init__(typename)
        self._read = read

    def value(self, buf: Buffer, at: int) -> Any:
        buf.pos = at
        return self._read(buf)


class Members(Column):
    """A branch with no bytes of its own, whose members are the branches under it.

    This is what ROOT leaves behind when it splits a C++ object: the object
    itself is a branch holding nothing, and each member is a branch of its own
    beneath it. There is nothing here to decode - the values come from the
    members - so this class carries only the name of the shape they go back
    into, which is a plain :class:`dict` per entry.
    """

    __slots__ = ()
    kind = "members"

    def __init__(self) -> None:
        super().__init__("dict")


class Refused(Column):
    """A column this reader will not guess at, and the reason in words."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        super().__init__(None)
        self.reason = reason


# -- reading one value ------------------------------------------------------


def _items(node: Any, buf: Buffer, count: int) -> Any:
    """``count`` values of one type, written one after another."""
    if isinstance(node, Prim):
        return to_native(array.array(node.typecode, buf.take(count * node.itemsize)))
    if isinstance(node, Str):
        return [buf.string() for _ in range(count)]
    return [_items(node.item, buf, buf.u32()) for _ in range(count)]


def _block(node: Any, buf: Buffer, count: int) -> Any:
    """One member-wise block: a run of values of one type, all together.

    A block of a class is introduced by a record of its own; a block of
    numbers, or of ``TString``, is simply the values.
    """
    if not isinstance(node, Prim) and not (isinstance(node, Str) and not node.record):
        buf.header()
    return _items(node, buf, count)


def _string(buf: Buffer) -> str:
    """A string standing alone in the entry, as a top-level branch writes it."""
    return buf.string()


def _member_string(buf: Buffer) -> str:
    """A string that is a member of a class, which carries a record header."""
    buf.header()
    return buf.string()


def _sequence(item: Any) -> Callable[[Buffer], Any]:
    def read(buf: Buffer) -> Any:
        version, _end = buf.header()
        if version & MEMBER_WISE:
            raise UnsupportedFeatureError(
                "this container was written field by field, which happens for a "
                "container of C++ objects and is not a shape this reader decodes"
            )
        return _items(item, buf, buf.u32())

    return read


def _mapping(node: Mapping) -> Callable[[Buffer], Any]:
    def read(buf: Buffer) -> Any:
        version, _end = buf.header()
        if not version & MEMBER_WISE:
            raise UnsupportedFeatureError(
                "this map was written pair by pair, which this reader has never "
                "seen a file do and will not decode on a guess"
            )
        if buf.u16() == 0:
            buf.u32()  # a class with no version of its own says so with a checksum
        count = buf.u32()
        keys = _block(node.key, buf, count)
        values = _block(node.value, buf, count)
        return dict(zip(keys, values, strict=True))

    return read


# -- deciding what a column is ----------------------------------------------


def _nested(node: Any) -> str:
    """Why a type below the top of an entry cannot be read, if it cannot."""
    if isinstance(node, Mapping):
        return "a map inside another container, which no file this reader has met writes"
    if isinstance(node, Seq):
        return _nested(node.item)
    return ""


def _unusable(node: Any) -> str:
    """Why a whole column cannot be read, if it cannot."""
    if isinstance(node, Mapping):
        if not isinstance(node.key, Prim | Str):
            return "a map keyed by a container, which is not a thing a dict can be keyed by"
        return _nested(node.value)
    return _nested(node)


def _number(text: str) -> float:
    """One end of a packing range, which ROOT lets you write in units of pi."""
    for spelling, value in _PI:
        if spelling in text:
            return -value if "-" in text else value
    return float(text)


def _range(title: str) -> tuple[float, float, float] | None:
    """``xmin``, ``xmax`` and the scale factor a title asks for.

    All zero means the title says nothing, which is itself a recipe: the
    default packing for the type. ``None`` means it says something this
    reader cannot make sense of, which is worth refusing over.
    """
    beg, end = title.rfind("["), title.rfind("]")
    if beg < 0 or end < 0:
        return 0.0, 0.0, 0.0
    if beg > 0:
        slash = title.rfind("/", 0, beg)
        if slash < 0 or slash + 2 != beg:
            return 0.0, 0.0, 0.0  # a title like ``x[10]``: an array, not a range
    parts = [part.strip().lower() for part in title[beg + 1 : end].split(",")]
    if len(parts) == 1:
        return 0.0, 0.0, 0.0
    if len(parts) > 3:
        return None
    nbits = 32
    try:
        if len(parts) == 3:
            nbits = int(parts[2])
        xmin, xmax = _number(parts[0]), _number(parts[1])
    except ValueError:
        return None
    if not 2 <= nbits <= 32:
        return None
    if xmin >= xmax:
        # No range: the bit count is all there is, and ROOT keeps it in xmin.
        return (float(nbits) + 0.1 if nbits < 15 else 0.0), xmax, 0.0
    return xmin, xmax, float((1 << nbits) if nbits < 32 else 0xFFFFFFFF) / (xmax - xmin)


def _scaled(factor: float, xmin: float) -> Unpack:
    """Values written as a whole number of steps across a known range."""

    def unpack(raw: bytes) -> array.array[Any]:
        count = len(raw) // 4
        return array.array("d", [v / factor + xmin for v in struct.unpack(f">{count}I", raw)])

    return unpack


def _truncated(nbits: int) -> Unpack:
    """Values written as a float with its mantissa cut short."""
    mask = (1 << (nbits + 1)) - 1
    sign = 1 << (nbits + 1)

    def unpack(raw: bytes) -> array.array[Any]:
        out = array.array("d")
        for exp, man in struct.iter_unpack(">BH", raw):
            bits = ((exp << 23) | ((man & mask) << (23 - nbits))) & 0xFFFFFFFF
            value = _FLOAT.unpack(_BITS.pack(bits))[0]
            out.append(-value if man & sign else value)
        return out

    return unpack


def _widened(raw: bytes) -> array.array[Any]:
    """A ``Double32_t`` with no recipe at all, which is a plain ``float``."""
    return array.array("d", to_native(array.array("f", raw)))


def _packing(title: str, double32: bool) -> tuple[Prim, Unpack] | None:
    """How wide a packed float is on disk and how it comes back out.

    ``None`` means the title says something about the packing that this reader
    cannot turn into numbers, which is worth refusing over: the alternative is
    a column of plausible wrong values.
    """
    spec = _range(title)
    if spec is None:
        return None
    xmin, _xmax, factor = spec
    if factor:
        return Prim("float64", "d", 4), _scaled(factor, xmin)
    if int(xmin) == 0 and double32:
        return Prim("float64", "d", 4), _widened
    return Prim("float64", "d", 3), _truncated(int(xmin) or 12)


def _unreadable_range(title: str) -> Refused:
    return Refused(
        f"a packed float whose range is written {title!r}, which is not a "
        f"spelling this reader can turn into numbers"
    )


def _packed(leaf: LeafRecord) -> Column:
    """A ``Double32_t`` or ``Float16_t``, whose title says how it was squeezed."""
    found = _packing(leaf.title, leaf.classname == "TLeafD32")
    if found is None:
        return _unreadable_range(leaf.title)
    prim, unpack = found
    if leaf.count is not None:
        return Rows(prim, 0, False, unpack)
    return Flat(prim, leaf.length, unpack)


def _packed_member(
    branch: BranchRecord, leaf: LeafRecord, source: Source, kind: int, counted: bool
) -> Column:
    """A packed float split out of a class, whose range the streamer holds.

    A branch of its own carries the recipe in the leaf's title; a member does
    not, because ROOT put it in the declaration's trailing comment instead.
    Without that comment there is no honest way to read the bytes, so a class
    the file does not describe is refused rather than read at a guess.
    """
    member = source.streamers().get(branch.classname, {}).get(leaf.name)
    if member is None:
        return Refused(
            f"a packed float member of {branch.classname or 'a class'} that this file's "
            f"streamer information does not describe, so the range it was squeezed into "
            f"is not knowable"
        )
    found = _packing(member.title, PACKED[kind])
    if found is None:
        return _unreadable_range(member.title)
    prim, unpack = found
    if counted:
        return Rows(prim, 1, False, unpack)  # one marker byte, then the packed values
    return Flat(prim, leaf.length, unpack)


def _plain(leaf: LeafRecord) -> Column:
    """A ``TLeafI`` and its kind: the type is the class of the leaf itself."""
    if leaf.classname == "TLeafC":
        return Values("str", _string)
    prim = Prim(str(leaf.typename), leaf.typecode, leaf.itemsize)
    if leaf.count is not None:
        return Rows(prim, 0, False)
    return Flat(prim, leaf.length)


def _declared(branch: BranchRecord, leaf: LeafRecord, source: Source) -> Column:
    """A column whose type is a class name the file has to spell out."""
    if leaf.ltype < 0:
        name, header = branch.classname, False  # a whole object: the branch names it
    else:
        member = source.streamers().get(branch.classname, {}).get(leaf.name)
        if member is None:
            return Refused(
                f"a member of {branch.classname or 'a class'} that this file's streamer "
                f"information does not describe, so its type is not knowable"
            )
        name, header = member.typename, True

    node = parse(name)
    if node is None:
        return Refused(
            f"{name or 'an unnamed type'}, which is a C++ type this reader does not "
            f"decode; a split file has its members as branches of their own"
        )
    if isinstance(node, Str):
        return Values("str", _member_string if header and node.record else _string)
    if isinstance(node, Seq) and isinstance(node.item, Prim):
        return Rows(node.item, RECORD + 4, True)
    reason = _unusable(node)
    if reason:
        return Refused(reason)
    if isinstance(node, Mapping):
        return Values(py_name(node), _mapping(node))
    assert isinstance(node, Seq)
    return Values(py_name(node), _sequence(node.item))


def build(branch: BranchRecord, leaf: LeafRecord, source: Source) -> Column:
    """What this column is, or why it cannot be read.

    Everything else in this module is reached through here: a branch record
    and one of its leaves go in, and a :class:`Column` comes out - a readable
    one, or a :class:`Refused` carrying the sentence that says why not.
    """
    from .objects import LEAF_TYPES, PACKED_LEAVES

    if leaf.classname in LEAF_TYPES:
        return _plain(leaf)
    if leaf.classname in PACKED_LEAVES:
        return _packed(leaf)
    if leaf.classname != "TLeafElement":
        return Refused(leaf.reason)
    if leaf.ltype == 65:
        return Values("str", _string)  # a TString member, written with no header
    for base in (0, OFFSET_L, OFFSET_P):
        kind = leaf.ltype - base
        prim = BASIC.get(kind)
        if prim is None:
            if kind in PACKED:
                return _packed_member(branch, leaf, source, kind, base == OFFSET_P)
            continue
        if base == OFFSET_P:
            return Rows(prim, 1, False)  # one marker byte, then the counted values
        return Flat(prim, leaf.length)
    if leaf.ltype in KINDS:
        return Refused(f"{KINDS[leaf.ltype]}, which this reader does not decode")
    return _declared(branch, leaf, source)
