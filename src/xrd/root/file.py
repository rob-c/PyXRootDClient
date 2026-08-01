"""The container: a ROOT file, its keys, and the directories they sit in.

A ROOT file is a header, then a chain of records each introduced by a key
saying what class it holds, how long it is, and where its payload starts. A
directory is nothing more than a list of those keys. That is all this module
knows, and it is enough to find a tree in a file on the other side of the
world for the cost of three reads.
"""

from __future__ import annotations

import datetime
import struct
from typing import IO, TYPE_CHECKING, Any

from ..url import parse
from .buffer import Buffer
from .compression import decompress
from .errors import FormatError, UnsupportedFeatureError

if TYPE_CHECKING:
    from ..config import Config

__all__ = ["Key", "Directory", "ROOTFile", "open_root"]

MAGIC = b"root"
#: Enough for a header and the longest sensible class, name and title.
KEY_WINDOW = 1024


class Source:
    """Random access to the bytes of a file, wherever they are.

    Anything with ``seek`` and ``read`` will do, which is what makes a remote
    tree the same code as a local one: the XRootD, HTTP and S3 file objects
    all have both.
    """

    __slots__ = ("handle", "name", "owned")

    def __init__(self, handle: IO[bytes], name: str, owned: bool) -> None:
        self.handle = handle
        self.name = name
        self.owned = owned

    def read(self, offset: int, size: int) -> bytes:
        self.handle.seek(offset)
        data = self.handle.read(size)
        if len(data) != size:
            raise FormatError(
                f"{self.name} ends at {offset + len(data)}, where {size} bytes were expected "
                f"from {offset}: the file is truncated"
            )
        return data

    def close(self) -> None:
        if self.owned:
            self.handle.close()


class Key:
    """The label in front of every record in a ROOT file.

        >>> key.classname, key.name, key.nbytes    # doctest: +SKIP
        ('TTree', 'events', 4711)

    The payload begins ``keylen`` bytes after :attr:`seek_key` and is
    compressed exactly when it is shorter than :attr:`objlen`.
    """

    __slots__ = (
        "nbytes",
        "version",
        "objlen",
        "datime",
        "keylen",
        "cycle",
        "seek_key",
        "seek_pdir",
        "classname",
        "name",
        "title",
    )

    def __init__(self, buf: Buffer) -> None:
        self.nbytes = buf.i32()
        self.version = buf.u16()
        self.objlen = buf.i32()
        self.datime = buf.u32()
        self.keylen = buf.i16()
        self.cycle = buf.i16()
        big = self.version > 1000
        self.seek_key = buf.i64() if big else buf.i32()
        self.seek_pdir = buf.i64() if big else buf.i32()
        self.classname = buf.string()
        self.name = buf.string()
        self.title = buf.string()

    def __repr__(self) -> str:
        return f"<Key {self.name!r};{self.cycle} of class {self.classname}>"

    @property
    def time(self) -> datetime.datetime:
        """When this record was written, out of ROOT's packed date word."""
        packed = self.datime
        return datetime.datetime(
            (packed >> 26) + 1995,
            (packed >> 22) & 0xF,
            (packed >> 17) & 0x1F,
            (packed >> 12) & 0x1F,
            (packed >> 6) & 0x3F,
            packed & 0x3F,
        )

    @property
    def compressed(self) -> bool:
        return self.objlen != self.nbytes - self.keylen

    def payload(self, source: Source) -> bytes:
        """The object this key introduces, decompressed if it needs to be."""
        raw = source.read(self.seek_key + self.keylen, self.nbytes - self.keylen)
        return decompress(raw, self.objlen) if self.compressed else raw

    def buffer(self, source: Source) -> Buffer:
        """The payload as a cursor, counting from where ROOT counts from."""
        return Buffer(self.payload(source), self.keylen)


def read_keys(source: Source, seek: int, nbytes: int) -> list[Key]:
    """The key list a directory keeps, at the place the directory says."""
    buf = Buffer(source.read(seek, nbytes))
    Key(buf)  # the list is itself a record, introduced by its own key
    return [Key(buf) for _ in range(buf.i32())]


class Directory:
    """A mapping from name to whatever that name holds.

        >>> list(directory)                        # doctest: +SKIP
        ['events', 'histograms']
        >>> directory["events"]                    # doctest: +SKIP
        <TTree 'events' with 100 entries and 4 branches>

    Names are the plain ones. A file that has kept several cycles of the same
    name gives up the newest, and an older one is still reachable by asking
    for ``"events;1"`` outright.
    """

    __slots__ = ("_source", "_keys", "path")

    def __init__(self, source: Source, keys: list[Key], path: str = "") -> None:
        self._source = source
        self._keys = keys
        self.path = path

    def __repr__(self) -> str:
        return f"<Directory {self.path or '/'} with {len(self._keys)} keys>"

    def _key(self, name: str) -> Key:
        wanted, _, cycle = name.partition(";")
        found = None
        for key in self._keys:
            if key.name != wanted:
                continue
            if cycle:
                if str(key.cycle) == cycle:
                    return key
            elif found is None or key.cycle > found.cycle:
                found = key
        if found is None:
            raise KeyError(f"{name!r} is not in {self.path or '/'}; there is {', '.join(self)}")
        return found

    def keys(self) -> list[str]:
        """Every name here, newest cycle only, in the order they were written."""
        seen: dict[str, None] = {}
        for key in self._keys:
            seen[key.name] = None
        return list(seen)

    def classnames(self) -> dict[str, str]:
        """What each name is, without reading any of them.

        The first thing to try when something will not open: it says whether
        a name is a ``TTree`` this reader knows or an object it does not.
        """
        return {key.name: key.classname for key in self._keys}

    def __iter__(self) -> Any:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, name: object) -> bool:
        try:
            self._key(str(name))
        except KeyError:
            return False
        return True

    def __getitem__(self, name: str) -> Any:
        head, slash, rest = name.partition("/")
        if slash:
            return self[head][rest]
        return self._read(self._key(head))

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except KeyError:
            return default

    def _read(self, key: Key) -> Any:
        if key.classname in ("TDirectory", "TDirectoryFile"):
            return self._subdirectory(key)
        from .objects import TREE_CLASSES, read_tree

        if key.classname in TREE_CLASSES:
            return read_tree(key.buffer(self._source), self._source, key.name, key.classname)
        raise UnsupportedFeatureError(
            f"{key.name!r} is a {key.classname}, and this reader does trees rather than "
            f"every ROOT class; ask the file what it holds with .classnames()"
        )

    def _subdirectory(self, key: Key) -> Directory:
        record = _directory_record(self._source, key.seek_key + key.keylen)
        return Directory(
            self._source,
            read_keys(self._source, record["seek_keys"], record["nbytes_keys"]),
            f"{self.path}/{key.name}".lstrip("/"),
        )

    def trees(self) -> list[str]:
        """The names here that are trees, which is usually the one you want."""
        from .objects import TREE_CLASSES

        return [key.name for key in self._keys if key.classname in TREE_CLASSES]

    def tree(self, name: str | None = None) -> Any:
        """The named tree, or the only one, so that a file with one just opens.

            >>> events = xrd.root.open_root(url).tree()   # doctest: +SKIP
        """
        if name is not None:
            return self[name]
        found = self.trees()
        if len(found) != 1:
            raise KeyError(
                f"{self.path or 'this file'} holds {len(found)} trees "
                f"({', '.join(found) or 'none'}): name the one you want"
            )
        return self[found[0]]


def _directory_record(source: Source, seek: int) -> dict[str, int]:
    """The fixed part of a ``TDirectory``, which says where its keys live."""
    buf = Buffer(source.read(seek, 42))
    version = buf.u16()
    buf.u32(), buf.u32()  # written and modified, both in ROOT's packed form
    nbytes_keys, _nbytes_name = buf.i32(), buf.i32()
    big = version > 1000
    buf.i64() if big else buf.i32()  # this directory
    buf.i64() if big else buf.i32()  # its parent
    seek_keys = buf.i64() if big else buf.i32()
    if seek_keys <= 0 or nbytes_keys <= 0:
        raise FormatError("a directory in this file has no key list to read")
    return {"seek_keys": seek_keys, "nbytes_keys": nbytes_keys}


class ROOTFile(Directory):
    """An open ROOT file.

        >>> with xrd.root.open_root("root://host//store/data.root") as f:  # doctest: +SKIP
        ...     tree = f["events"]

    It is a :class:`Directory`, so the file itself is the top of the tree of
    names, and closing it closes whatever it was opened over.
    """

    __slots__ = ("version", "compression", "uuid")

    def __init__(self, source: Source) -> None:
        header = source.read(0, 100)
        if not header.startswith(MAGIC):
            raise FormatError(
                f"{source.name} does not start with {MAGIC!r}: it is not a ROOT file "
                f"(a truncated download and an HTML error page both look like this)"
            )
        version, begin = struct.unpack_from(">ii", header, 4)
        wide = version >= 1000000
        form = ">qqiiiBiqi" if wide else ">iiiiiBiii"
        (_end, _free, _nfree_bytes, _nfree, nbytes_name, _units, compression, _info, _ninfo) = (
            struct.unpack_from(form, header, 12)
        )
        self.version = version % 1000000
        self.compression = compression
        record = _directory_record(source, begin + nbytes_name)
        super().__init__(source, read_keys(source, record["seek_keys"], record["nbytes_keys"]))

    def __repr__(self) -> str:
        return f"<ROOTFile {self._source.name!r} written by ROOT {self.version}>"

    @property
    def name(self) -> str:
        """Where this file came from."""
        return self._source.name

    def close(self) -> None:
        """Close the underlying file, if this object opened it."""
        self._source.close()

    def __enter__(self) -> ROOTFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_root(target: Any, *, config: Config | None = None) -> ROOTFile:
    """Open a ROOT file, wherever it is.

        >>> import xrd.root                                       # doctest: +SKIP
        >>> with xrd.root.open_root("root://host//store/x.root") as f:
        ...     print(f.keys())

    ``target`` is a URL of any scheme this library speaks, a local path, or an
    already-open binary file - a file object is used as it is and left open,
    on the principle that whoever opened it closes it.
    """
    if hasattr(target, "read") and hasattr(target, "seek"):
        return ROOTFile(Source(target, getattr(target, "name", "<file>"), owned=False))

    url = parse(target)
    if url.is_local:
        handle: IO[bytes] = open(url.path, "rb")
    else:
        from ..io import open_url

        handle = open_url(url, "rb", config=config)
    try:
        return ROOTFile(Source(handle, str(target), owned=True))
    except BaseException:
        handle.close()  # a file that will not open should not leak the handle
        raise
