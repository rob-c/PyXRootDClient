"""Reading the three classes a tree is made of: TTree, TBranch, TLeaf.

Every one of them is versioned, and the version decides which fields are
there - a file from 2008 and a file from last week are both valid and are not
the same bytes. The conditionals below are that history, and they are the
whole reason a reader can be short: after the fields it wants, it jumps to the
byte count the record started with and never has to know the rest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .buffer import Buffer
from .errors import UnsupportedFeatureError

if TYPE_CHECKING:
    from .file import Source

__all__ = ["TREE_CLASSES", "read_tree"]

#: Key classes this reader will open as a tree.
TREE_CLASSES = ("TTree", "TNtuple", "TNtupleD")

#: ROOT leaf class to the Python name, ``array`` type code and item size.
LEAF_TYPES = {
    "TLeafO": ("bool", "b", 1),
    "TLeafB": ("int8", "b", 1),
    "TLeafS": ("int16", "h", 2),
    "TLeafI": ("int32", "i", 4),
    "TLeafL": ("int64", "q", 8),
    "TLeafG": ("int64", "q", 8),
    "TLeafF": ("float32", "f", 4),
    "TLeafD": ("float64", "d", 8),
    "TLeafC": ("str", "", 0),
}

#: Leaves holding floats squeezed into fewer bytes, by the recipe in the title.
PACKED_LEAVES = ("TLeafF16", "TLeafD32")

#: Leaves that are real and are not plain numbers behind a name.
LEAF_REASONS = {
    "TLeafElement": "a split C++ object, which needs the file's streamer information",
    "TLeafObject": "a whole object per entry, which needs the file's streamer information",
}

UNSIGNED = {"b": "B", "h": "H", "i": "I", "q": "Q"}


class LeafRecord:
    """One column's description, as the file states it."""

    __slots__ = (
        "classname",
        "name",
        "title",
        "length",
        "etype",
        "offset",
        "unsigned",
        "count",
        "ltype",
    )

    def __init__(self, classname: str) -> None:
        self.classname = classname
        self.name = self.title = ""
        self.length = 1
        self.etype = self.offset = 0
        self.unsigned = False
        self.count: LeafRecord | None = None
        #: The streamer type a ``TLeafElement`` names; ``-1`` for a whole object.
        self.ltype = -1

    def __repr__(self) -> str:
        return f"<LeafRecord {self.name!r} of class {self.classname}>"

    @property
    def typename(self) -> str | None:
        """``'float32'``, ``'str'``, or ``None`` if this reader cannot decode it."""
        found = LEAF_TYPES.get(self.classname)
        if found is None:
            return None
        name = found[0]
        return f"u{name}" if self.unsigned and name.startswith("int") else name

    @property
    def typecode(self) -> str:
        """The :mod:`array` type code the values come back in."""
        _name, code, _size = LEAF_TYPES[self.classname]
        return UNSIGNED[code] if self.unsigned and code in UNSIGNED else code

    @property
    def itemsize(self) -> int:
        return LEAF_TYPES[self.classname][2]

    @property
    def reason(self) -> str:
        """Why this column cannot be read, for a message that names it."""
        unknown = f"{self.classname}, which is not a kind of leaf this reader knows"
        return LEAF_REASONS.get(self.classname, unknown)


class BranchRecord:
    """Where a branch keeps its baskets, and which columns are in them."""

    __slots__ = (
        "name",
        "title",
        "classname",
        "entry_offset_len",
        "entries",
        "first_entry",
        "basket_bytes",
        "basket_entry",
        "basket_seek",
        "leaves",
        "branches",
    )

    def __init__(self) -> None:
        self.name = self.title = ""
        #: The C++ class a ``TBranchElement`` belongs to; empty for a plain branch.
        self.classname = ""
        self.entry_offset_len = 0
        self.entries = self.first_entry = 0
        self.basket_bytes: list[int] = []
        self.basket_entry: list[int] = []
        self.basket_seek: list[int] = []
        self.leaves: list[LeafRecord] = []
        self.branches: list[BranchRecord] = []

    def __repr__(self) -> str:
        return f"<BranchRecord {self.name!r} with {len(self.basket_seek)} baskets>"

    def walk(self) -> Any:
        """This branch and every branch under it, depth first."""
        yield self
        for child in self.branches:
            yield from child.walk()


def read_leaf(buf: Buffer, classname: str) -> LeafRecord:
    """A ``TLeaf`` of any class: the part every leaf shares, then a jump."""
    _version, end = buf.header()
    _base, inner = buf.header()
    leaf = LeafRecord(classname)
    leaf.name, leaf.title = buf.named()
    leaf.length = buf.i32() or 1
    leaf.etype = buf.i32()
    leaf.offset = buf.i32()
    buf.bool()  # whether a range was recorded, which only matters for writing
    leaf.unsigned = buf.bool()
    count = buf.any(CLASSES)
    if isinstance(count, LeafRecord):
        leaf.count = count
    buf.resume(inner)
    if classname == "TLeafElement":
        buf.i32()  # which member of the class this is, which its name says too
        leaf.ltype = buf.i32()
    buf.resume(end)
    return leaf


def read_branch(buf: Buffer) -> BranchRecord:
    """A ``TBranch``, as written by any ROOT since 2007."""
    version, end = buf.header()
    if version < 10:
        raise UnsupportedFeatureError(
            f"this file has a branch written in the ROOT 4 layout (TBranch version {version}); "
            f"copy it forward with hadd from any ROOT 5 or 6 and it will open"
        )
    branch = BranchRecord()
    branch.name, branch.title = buf.named()
    buf.skip_record()  # TAttFill: the colour it is drawn in
    buf.i32()  # compression, which the baskets state for themselves
    buf.i32()  # the size a basket was aimed at
    branch.entry_offset_len = buf.i32()
    write_basket = buf.i32()
    buf.i64()  # the entry number the next basket would start at
    if version >= 13:
        buf.skip_record()  # TIOFeatures
    buf.i32()  # this branch's offset inside its parent's entry
    max_baskets = buf.i32()
    buf.i32()  # split level
    branch.entries = buf.i64()
    if version >= 11:
        branch.first_entry = buf.i64()
    buf.i64(), buf.i64()  # bytes written, before and after compression

    branch.branches = [b for b in buf.objarray(CLASSES) if isinstance(b, BranchRecord)]
    branch.leaves = [leaf for leaf in buf.objarray(CLASSES) if isinstance(leaf, LeafRecord)]
    buf.objarray(CLASSES)  # baskets still in memory: none, in a file that was closed

    buf.u8()
    branch.basket_bytes = buf.i32s(max_baskets)[:write_basket]
    buf.u8()
    branch.basket_entry = buf.i64s(max_baskets)[: write_basket + 1]
    buf.u8()
    branch.basket_seek = buf.i64s(max_baskets)[:write_basket]
    buf.resume(end)
    return branch


def read_branch_element(buf: Buffer) -> BranchRecord:
    """A ``TBranchElement``: a branch, plus the C++ class it was split out of.

    The class name is what tells a top-level branch apart from a member of
    something bigger - ``vector<float>`` says everything about how to read the
    column, where ``Event`` says to look at the branches under it instead.
    """
    version, end = buf.header()
    branch = read_branch(buf)
    branch.classname = buf.string()
    if version > 1:
        buf.string(), buf.string()  # the parent class, and the TClonesArray class
        buf.u32()  # the checksum of the class this was written from
    buf.u16() if version >= 10 else buf.u32()  # that class's version
    buf.resume(end)
    return branch


def read_derived_branch(buf: Buffer) -> BranchRecord:
    """A ``TBranchObject``, ``TBranchClones`` or ``TBranchSTL``.

    The C++ part on top is what this reader cannot follow; the branch part
    underneath is what makes the column appear in the tree at all, with a name
    and a reason instead of silently not being there.
    """
    _version, end = buf.header()
    branch = read_branch(buf)
    buf.resume(end)
    return branch


def _leaf_class(classname: str) -> Callable[[Buffer], LeafRecord]:
    def read(buf: Buffer) -> LeafRecord:
        return read_leaf(buf, classname)

    return read


#: What :meth:`Buffer.any` knows how to build; anything else is stepped over.
CLASSES: dict[str, Any] = {
    name: _leaf_class(name) for name in (*LEAF_TYPES, *LEAF_REASONS, *PACKED_LEAVES)
}
CLASSES["TBranch"] = read_branch
CLASSES["TBranchElement"] = read_branch_element
for _derived in ("TBranchObject", "TBranchClones", "TBranchSTL"):
    CLASSES[_derived] = read_derived_branch


def read_tree(buf: Buffer, source: Source, name: str, classname: str = "TTree") -> Any:
    """Read a ``TTree`` record and hand back something that can be iterated."""
    from .tree import TTree

    if classname != "TTree":
        buf.header()  # a TNtuple is a TTree with a field after it

    version, end = buf.header()
    if version <= 5:
        raise UnsupportedFeatureError(
            f"this tree was written by ROOT 4 or older (TTree version {version}), which this "
            f"reader does not go back to; hadd it forward first"
        )
    title = buf.named()[1]
    for _ in range(3):
        buf.skip_record()  # line, fill and marker attributes

    entries = buf.i64()
    buf.i64(), buf.i64(), buf.i64()  # bytes written, three ways
    if version >= 18:
        buf.i64()
    if version >= 16:
        buf.f64()  # the weight given to each entry
    buf.i32(), buf.i32(), buf.i32()  # timer interval, scan field, update
    if version >= 17:
        buf.i32()
    clusters = buf.i32() if version >= 19 else 0
    buf.i64()  # the entry count it was told to stop at
    buf.i64(), buf.i64(), buf.i64()  # max entry loop, max virtual size, autosave
    if version >= 18:
        buf.i64()
    buf.i64()  # the estimated number of entries
    if version >= 19:
        buf.u8()
        buf.i64s(clusters)  # where each cluster of baskets ends
        buf.u8()
        buf.i64s(clusters)
    if version >= 20:
        buf.skip_record()  # TIOFeatures

    branches = [b for b in buf.objarray(CLASSES) if isinstance(b, BranchRecord)]
    buf.resume(end)
    return TTree(name, title, entries, branches, source)
