"""What the file says its own classes look like.

Every ROOT file carries a description of the C++ classes it was written from:
for each class, the members in declaration order, each with its type spelled
out. A split branch names only the member, so this is the only place the type
of ``StlVecI16`` can be learnt - and learning it is the difference between a
column that reads and one that is refused.

It is read once, on the first branch that needs it, and not at all by a file
whose branches all say what they hold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .buffer import Buffer

if TYPE_CHECKING:
    from .file import Source

__all__ = ["Member", "read_streamers"]

#: Enough of the record at ``fSeekInfo`` to read its key.
KEY_WINDOW = 512


class Member:
    """One data member of one class, as the file describes it."""

    __slots__ = ("name", "title", "stype", "typename", "length")

    def __init__(self, name: str, title: str, stype: int, typename: str, length: int) -> None:
        self.name = name
        #: The declaration's trailing comment, where a packed float keeps its range.
        self.title = title
        self.stype = stype
        self.typename = typename
        self.length = length

    def __repr__(self) -> str:
        return f"<Member {self.typename} {self.name}>"


def read_element(buf: Buffer, depth: int = 1) -> Member:
    """A ``TStreamerElement``, under whichever subclass wrote it.

    Every one of them is the same record with more fields bolted underneath,
    and the fields this reader wants are in the part they share. ``depth`` is
    how many records deep that shared part is: one for nearly every subclass,
    two for the single one that derives from another subclass rather than
    from the element itself.
    """
    _version, end = buf.header()
    for _ in range(depth - 1):
        buf.header()
    base, inner = buf.header()
    name, title = buf.named()
    stype = buf.i32()
    buf.i32()  # how many bytes the member takes in memory
    length = buf.i32()
    buf.i32()  # how many dimensions, which the maximum indices below repeat
    buf.i32s(buf.i32() if base == 1 else 5)
    typename = buf.string()
    buf.resume(inner)
    buf.resume(end)
    return Member(name, title, stype, typename, length)


def read_info(buf: Buffer) -> tuple[str, dict[str, Member]]:
    """A ``TStreamerInfo``: one class, and the members it was declared with."""
    _version, end = buf.header()
    name, _title = buf.named()
    buf.u32()  # the checksum ROOT uses to tell two versions of a class apart
    buf.i32()  # the class version
    elements = buf.any(CLASSES)
    members = {m.name: m for m in elements if isinstance(m, Member)} if elements else {}
    buf.resume(end)
    return name, members


#: The ``TStreamerElement`` subclasses, all read as the element under them.
ELEMENTS = (
    "TStreamerElement",
    "TStreamerBase",
    "TStreamerBasicType",
    "TStreamerBasicPointer",
    "TStreamerLoop",
    "TStreamerObject",
    "TStreamerObjectAny",
    "TStreamerObjectPointer",
    "TStreamerObjectAnyPointer",
    "TStreamerString",
    "TStreamerSTL",
    "TStreamerArtificial",
)


def read_stl_string(buf: Buffer) -> Member:
    """A ``TStreamerSTLstring``, which is a ``TStreamerSTL`` with a lid on it.

    It is the one subclass that derives from another subclass, so its shared
    part sits one record deeper than everything else's.
    """
    return read_element(buf, 2)


def read_objarray(buf: Buffer) -> list[Any]:
    """The ``TObjArray`` a ``TStreamerInfo`` keeps its members in."""
    return buf.objarray(CLASSES)


CLASSES: dict[str, Any] = dict.fromkeys(ELEMENTS, read_element)
CLASSES["TStreamerSTLstring"] = read_stl_string
CLASSES["TStreamerInfo"] = read_info
CLASSES["TObjArray"] = read_objarray


def read_streamers(source: Source) -> dict[str, dict[str, Member]]:
    """Every class this file describes, by name.

    A file written without streamer information - which a tree of plain
    numbers does not need - gives back nothing, and the branches that would
    have wanted it say so for themselves.
    """
    from .file import Key

    seek, nbytes = source.info
    if seek <= 0 or nbytes <= 0:
        return {}
    key = Key(Buffer(source.read(seek, min(nbytes, KEY_WINDOW))))
    found: dict[str, dict[str, Member]] = {}
    for info in key.buffer(source).tlist(CLASSES):
        if isinstance(info, tuple):
            found.setdefault(info[0], info[1])
    return found
