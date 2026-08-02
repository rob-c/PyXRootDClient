"""The datasets machine learning is taught with, written as ROOT files.

    >>> from xrd.root import datasets                        # doctest: +SKIP
    >>> datasets.convert("cifar10", "cifar10.root")          # doctest: +SKIP
    {'train_airplane': 5000, 'train_automobile': 5000, ...}

Physics keeps its data in ROOT and reads it over XRootD; everyone else keeps
theirs in whatever the framework of the year unpacks. That is a shame in one
direction only - the format is perfectly good at images and tables - so this
module converts the sets people actually teach and benchmark with into ROOT
files, laid out the way a training loop wants them: one tree per class, a
fixed-size array per entry, the label beside it, and the row's place in the
original file so any number can be traced back to where it came from.

Nothing is redistributed here. Each dataset is fetched from the people who
publish it, on the machine doing the converting, and :attr:`DATASETS` records
what each one is and what its licence says. The converted file carries that
same statement in an ``about`` key, so a file that outlives this program still
says where it came from.

Every archive is read with the standard library and nothing else - IDX, tar,
zip, a zip inside a zip, gzip, WAV, ARFF, CSV and the XML a spreadsheet keeps
inside its own zip - and what comes out is pictures, sound, sentences, tables
and plain blocks of numbers, because a training loop should not have to care
which of those it is reading. Some of these sets have a class to sort a row
into and some have a number to predict from it; both are here. The CIFAR
sets are taken in their binary distribution rather than the Python one on
purpose: that one is a pickle, and unpickling a download is a way to run
somebody else's code.
"""

from __future__ import annotations

import csv
import gzip
import io
import struct
import tarfile
import wave
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from math import nan
from typing import Any

from ..url import parse
from .writer import WritableFile, create

__all__ = [
    "CIFAR",
    "DATASETS",
    "IDX_FILES",
    "MNIST_MIRROR",
    "IDX_TYPES",
    "IMAGE_BASKET",
    "MISSING",
    "TABLE_BASKET",
    "Audio",
    "Dataset",
    "Images",
    "Matrix",
    "Table",
    "convert",
    "describe",
    "fetch",
    "read_arff",
    "read_idx",
    "read_table",
    "read_xlsx",
]

#: How many bytes of a column gather before a basket goes out. Images get a
#: large one on purpose: a training loop reads thousands of rows at a time,
#: and at this size that is one read rather than twenty.
IMAGE_BASKET = 512 * 1024
TABLE_BASKET = 64 * 1024

#: What the type byte of an IDX file says its values are. These datasets hold
#: unsigned bytes; the others are named so a wrong file is refused by what it
#: actually is rather than read as pixels.
IDX_TYPES = {
    0x08: "unsigned bytes",
    0x09: "signed bytes",
    0x0B: "16-bit integers",
    0x0C: "32-bit integers",
    0x0D: "floats",
    0x0E: "doubles",
}

#: Where MNIST is served from. The original site stopped offering it; this
#: is the mirror PyTorch's own downloader uses.
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

#: The four files an IDX-format image set comes in. MNIST chose these names
#: and the two datasets built to drop into its place kept them.
IDX_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}

#: How the tabular sets spell a value nobody measured. A missing number is
#: written as a NaN, which is what every reader downstream already means by
#: it; a missing category is written as -1, because there is no such code.
MISSING = frozenset({"", "?", "NA", "na", "N/A", "nan", "NaN", "null"})

#: The namespace everything in a spreadsheet's XML is in.
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

#: The day a ``"date"`` column counts from, as an ordinal: 1970-01-01.
EPOCH = date(1970, 1, 1).toordinal()

#: The moment a ``"time"`` column counts from: midnight starting that day.
MIDNIGHT = datetime(1970, 1, 1)

#: One row on its way to a tree: which class it belongs to, and its columns.
Rows = Iterator[tuple[int, dict[str, Any]]]

#: What a reader hands back: the class names, the columns, and the rows.
Loaded = tuple[tuple[str, ...], dict[str, Any], Rows]


def read_idx(raw: bytes) -> tuple[tuple[int, ...], bytes]:
    """The shape and the values of one IDX file, unzipped if it arrived zipped.

        >>> read_idx(open("train-labels-idx1-ubyte.gz", "rb").read())  # doctest: +SKIP
        ((60000,), b'\\x05\\x00\\x04...')

    IDX is four magic bytes - two zeros, a type, and how many dimensions -
    then one big-endian length per dimension, then the values.
    """
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if len(raw) < 4 or raw[:2] != b"\x00\x00":
        raise ValueError(
            "this is not an IDX file: one starts with two zero bytes, a type and a rank, "
            "and this starts with something else"
        )
    kind, rank = raw[2], raw[3]
    if kind != 0x08:
        what = IDX_TYPES.get(kind, f"values of type {kind:#04x}")
        raise ValueError(f"this IDX file holds {what}, and these image sets hold unsigned bytes")
    if len(raw) < 4 + 4 * rank:
        raise ValueError(
            f"this IDX file says it has {rank} dimensions and stops before saying them"
        )
    shape = struct.unpack_from(f">{rank}i", raw, 4)
    wanted = 1
    for size in shape:
        wanted *= size
    values = raw[4 + 4 * rank :]
    if len(values) != wanted:
        raise ValueError(
            f"this IDX file says {'x'.join(str(size) for size in shape)}, which is {wanted} "
            f"values, and holds {len(values)}"
        )
    return shape, values


def read_table(
    raw: bytes,
    *,
    delimiter: str | None = ",",
    header: bool = False,
    comment: str = "",
    quoted: bool = True,
    tail: int = 0,
) -> Iterator[list[str]]:
    """Every row of a delimited text file, as stripped strings, blanks dropped.

    ``delimiter`` is what separates the fields, or ``None`` for any run of
    whitespace, which is how the older sets are written. ``comment`` drops
    every line that starts with it, and ``header`` drops the first line that
    survives that - the names are as likely to come after a preamble as before
    one. ``quoted`` is whether a double quote groups a field, as it does in a CSV;
    a file of English sentences means its quotes literally, and setting this
    ``False`` keeps them rather than reading them as punctuation.

    ``tail`` is how many fields a whitespace-separated line has when the last
    of them is free text with spaces in it - a car's name at the end of a row
    of numbers. The line is split that many times and no further, and the last
    field is unwrapped if it arrived in quotes, which is what a CSV reader
    would have done with it.
    """
    text = io.StringIO(raw.decode("utf-8-sig"), newline="")
    rows: Iterator[list[str]] = (
        ((line.split(None, tail - 1) if tail else line.split()) for line in text)
        if delimiter is None
        else csv.reader(
            text,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL if quoted else csv.QUOTE_NONE,
        )
    )
    naming = header
    for row in rows:
        if not any(cell.strip() for cell in row) or (comment and row[0].startswith(comment)):
            continue
        if naming:
            naming = False
            continue
        cells = [cell.strip() for cell in row]
        if tail and quoted:
            cells[-1] = _unquoted(cells[-1])
        yield cells


def _unquoted(cell: str) -> str:
    """A field with the quotes around it taken off, if that is what they are."""
    return cell[1:-1] if len(cell) > 1 and cell[0] == cell[-1] == '"' else cell


def read_xlsx(raw: bytes, *, sheet: int = 1, header: bool = False) -> Iterator[list[str]]:
    """Every row of one sheet of a spreadsheet, as the strings it shows.

        >>> next(read_xlsx(open("ENB2012_data.xlsx", "rb").read()))  # doctest: +SKIP
        ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'Y1', 'Y2']

    A modern spreadsheet is a zip of XML, so this needs nothing the standard
    library does not already have. Text is usually kept once in a shared table
    and referred to by number, which is why a sheet on its own reads as
    nonsense and this does not.

    A row of nothing at all is not data - spreadsheets are full of them, below
    and beside what was typed - and neither are the empty cells trailing a
    row, so both are dropped. A gap in the middle of a row is kept, as an empty
    string, because the fields after it still have to line up.
    """
    book = zipfile.ZipFile(io.BytesIO(raw))
    inside = f"xl/worksheets/sheet{sheet}.xml"
    if inside not in book.namelist():
        sheets = sum(name.startswith("xl/worksheets/sheet") for name in book.namelist())
        raise ValueError(f"this spreadsheet has {sheets} sheets in it, and no sheet {sheet}")
    shared: list[str] = []
    if "xl/sharedStrings.xml" in book.namelist():
        shared = [
            "".join(part.text or "" for part in entry.iter(XLSX_NS + "t"))
            for entry in ET.fromstring(book.read("xl/sharedStrings.xml"))
        ]
    rows = _sheet(book.open(inside), shared)
    if header:
        next(rows, None)
    yield from rows


def _sheet(stream: Any, shared: Sequence[str]) -> Iterator[list[str]]:
    """The rows of one sheet's XML, each cell put back where its reference says."""
    with stream:
        for _, element in ET.iterparse(stream):
            if element.tag != XLSX_NS + "row":
                continue
            cells: list[str] = []
            for cell in element:
                while len(cells) < _at(cell.get("r", "")):
                    cells.append("")
                value = cell.find(XLSX_NS + "v")
                if value is None:
                    cells.append("".join(part.text or "" for part in cell.iter(XLSX_NS + "t")))
                elif cell.get("t") == "s":
                    cells.append(shared[int(value.text or "0")])
                else:
                    cells.append(value.text or "")
            while cells and not cells[-1]:
                cells.pop()
            element.clear()
            if cells:
                yield cells


def _at(ref: str) -> int:
    """Which column a cell reference like ``A1`` or ``BC7`` names, from zero."""
    place = 0
    for letter in ref.rstrip("0123456789"):
        place = place * 26 + ord(letter.upper()) - 64
    return place - 1


def read_arff(raw: bytes) -> Iterator[list[str]]:
    """The rows of an ARFF file's ``@data`` section, its header and comments gone.

    ARFF is what Weka wrote and half of UCI still publishes: a header of
    ``@attribute`` lines saying what the columns are, then ``@data`` and a
    comma-separated table. The header is documentation - what the columns
    actually hold is declared in :attr:`Table.fields` - so only the rows come
    back.
    """
    lines = raw.decode("utf-8-sig").splitlines()
    for at, line in enumerate(lines):
        if line.strip().lower().startswith("@data"):
            return read_table("\n".join(lines[at + 1 :]).encode(), comment="%")
    raise ValueError(
        "this is not an ARFF file: one has a @data line and its rows after it, "
        "and this has no @data line at all"
    )


def fetch(source: Any, *, config: Any = None) -> bytes:
    """Everything at ``source``: bytes as they are, or a path or URL read whole.

    Read through this library, so ``root://``, ``https://`` and the rest all
    work, and so does an already-open file.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if hasattr(source, "read"):
        return bytes(source.read())
    url = parse(source)
    if url.is_local:
        with open(url.path, "rb") as handle:
            return handle.read()
    from ..io import open_url

    with open_url(url, "rb", config=config) as handle:
        return bytes(handle.read())


def _member(raw: bytes, member: str) -> bytes:
    """One file out of whatever arrived: a zip, a gzip, or the bytes themselves."""
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            held = archive.namelist()
            if member not in held:
                raise ValueError(f"this zip holds {', '.join(held)}, and not {member!r}")
            raw = archive.read(member)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def _number(cell: str, cast: Any, name: str, index: int, label: str) -> Any:
    """One field as a number, saying where it was when it is not one."""
    try:
        return cast(cell)
    except ValueError:
        raise ValueError(
            f"row {index} of {label} has {cell!r} in {name}, which is not a number"
        ) from None


def _numbered(kinds: Mapping[str, int] | Sequence[str]) -> Mapping[str, int]:
    """A field's categories as name to code, however they were written down."""
    if isinstance(kinds, Mapping):
        return kinds
    return {name: at for at, name in enumerate(kinds)}


def _plain(names: Sequence[str], role: str = "i") -> tuple[tuple[str, str], ...]:
    """A run of fields that all hold the same sort of value, named in order."""
    return tuple((name, role) for name in names)


def _tidy(text: str) -> tuple[str, ...]:
    """The class names in a file of them, one per line, made safe to name a tree."""
    return tuple(
        "".join(letter if letter.isalnum() else "_" for letter in line.strip().lower())
        for line in text.splitlines()
        if line.strip()
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Dataset:
    """Where a dataset comes from, what it holds, and what may be done with it.

    The subclasses below add how to read one: :class:`Images` for the IDX sets
    shaped like MNIST, :class:`CIFAR` for the two tiny-image archives,
    :class:`Audio` for the recordings, :class:`Matrix` for the plain blocks of
    numbers, and :class:`Table` for the delimited ones.
    """

    #: How this module is asked for it: ``convert("cifar10", ...)``.
    name: str
    #: What its authors call it, for messages.
    label: str
    #: One line saying what is in it.
    title: str
    #: What its licence is. Read it before you redistribute what comes out.
    licence: str
    #: Where it is published, for anyone who wants the paper or the terms.
    source: str
    #: The classes, in label order. One tree is written per class, named here.
    classes: tuple[str, ...]
    #: The splits it comes in. Single-split sets use ``("all",)``.
    splits: tuple[str, ...] = ("all",)
    #: How big a basket to fill before writing one out.
    basket_size: int = TABLE_BASKET

    def about(self, split: str) -> str:
        """What goes in the file's ``about`` key, so it can say what it is."""
        where = f"\nsplit: {split}" if len(self.splits) > 1 else ""
        return (
            f"{self.label}: {self.title}{where}\nlicence: {self.licence}\n"
            f"source: {self.source}\nconverted by xrd.root.datasets, {self.layout()}"
        )

    def entry_title(self, split: str, cls: str) -> str:
        """The title of the tree holding one class."""
        where = f"{split} " if len(self.splits) > 1 else ""
        return f"{self.label} {where}rows labelled {cls}"

    def sorting(self) -> str:
        """How many classes it has, the way :func:`describe` puts it."""
        return f"{len(self.classes)} classes"

    def layout(self) -> str:
        """How the trees are laid out, the way :meth:`about` puts it."""
        return "one tree per class"


@dataclass(frozen=True, slots=True, kw_only=True)
class Images(Dataset):
    """An image set in IDX format: MNIST, and the two sets built to replace it."""

    #: Where the four files are served from, when they are served separately.
    base: str = ""
    #: The one archive holding them, when they are not.
    archive: str = ""
    #: Which two of them each split is, images first.
    files: Mapping[str, tuple[str, str]] = field(default_factory=lambda: IDX_FILES)
    #: How wide and tall one picture is.
    side: int = 28
    #: What the pixels mean, for the tree title.
    shade: str = "greyscale"

    def urls(self, split: str) -> dict[str, str]:
        """The images and the labels of one split, or the archive holding both."""
        if self.archive:
            return {"archive": self.archive}
        images, labels = self.files[split]
        return {"images": self.base + images, "labels": self.base + labels}

    def entry_title(self, split: str, cls: str) -> str:
        where = f"{split} " if len(self.splits) > 1 else ""
        return f"{self.label} {where}images of class {cls}, {self.side}x{self.side} {self.shade}"

    def rows(self, raw: Mapping[str, bytes], split: str) -> Loaded:
        """The pixels and the labels, checked against each other."""
        if self.archive:
            pictures, names = self.files[split]
            held = raw["archive"]
            raw = {"images": _member(held, pictures), "labels": _member(held, names)}
        shape, pixels = read_idx(raw["images"])
        counts, labels = read_idx(raw["labels"])
        if len(shape) != 3 or shape[1:] != (self.side, self.side):
            raise ValueError(
                f"these images are {'x'.join(str(size) for size in shape)}, and {self.label} "
                f"images come as a count of {self.side}x{self.side} pictures"
            )
        if len(counts) != 1 or counts[0] != shape[0]:
            raise ValueError(
                f"there are {shape[0]} images and {counts[0] if len(counts) == 1 else counts} "
                f"labels, and every image needs exactly one"
            )
        width = self.side * self.side
        columns: dict[str, Any] = {"image": ("B", width), "label": "i", "index": "i"}
        return self.classes, columns, self._entries(pixels, labels, width)

    def _entries(self, pixels: bytes, labels: bytes, width: int) -> Rows:
        view = memoryview(pixels)
        for index, label in enumerate(labels):
            at = index * width
            yield label, {"image": view[at : at + width], "label": label, "index": index}


@dataclass(frozen=True, slots=True, kw_only=True)
class CIFAR(Dataset):
    """One of the two tiny-image archives, in its binary rather than pickled form.

    A record is its labels then 3072 bytes: 1024 red, 1024 green, 1024 blue,
    each of them a 32 by 32 picture read along its rows. That is the layout
    PyTorch wants, so ``.view(-1, 3, 32, 32)`` is all the reshaping there is.
    """

    #: The one archive holding every split.
    archive: str
    #: Which members of it each split is.
    files: Mapping[str, tuple[str, ...]]
    #: The member naming the classes, one per line, in label order.
    meta: str
    #: The member naming the coarse classes, when there are two labels a row.
    coarse: str = ""
    #: How wide and tall one picture is.
    side: int = 32

    def urls(self, split: str) -> dict[str, str]:
        """The single archive, whichever split was asked for."""
        return {"archive": self.archive}

    def sorting(self) -> str:
        """How many classes it has; CIFAR-100 keeps its hundred in the archive."""
        return f"{len(self.classes) or 100} classes"

    def entry_title(self, split: str, cls: str) -> str:
        return (
            f"{self.label} {split} images of class {cls}, {self.side}x{self.side} colour, "
            f"three planes"
        )

    def rows(self, raw: Mapping[str, bytes], split: str) -> Loaded:
        """The class names out of the archive's own list, then the records."""
        tar = tarfile.open(fileobj=io.BytesIO(raw["archive"]), mode="r:gz")
        try:
            held = {member.name.rsplit("/", 1)[-1]: member for member in tar.getmembers()}
            classes = _tidy(_extract(tar, held, self.meta).decode())
            if self.classes and classes != self.classes:
                raise ValueError(
                    f"this archive says its classes are {', '.join(classes)}, and {self.label} "
                    f"has {', '.join(self.classes)}"
                )
            width = 3 * self.side * self.side
            columns: dict[str, Any] = {"image": ("B", width), "label": "i"}
            if self.coarse:
                columns["coarse"] = "i"
            columns["index"] = "i"
        except Exception:
            tar.close()
            raise
        return classes, columns, self._entries(tar, held, split, width)

    def _entries(
        self, tar: tarfile.TarFile, held: Mapping[str, Any], split: str, width: int
    ) -> Rows:
        stride = width + (2 if self.coarse else 1)
        index = 0
        try:
            for member in self.files[split]:
                data = _extract(tar, held, member)
                if len(data) % stride:
                    raise ValueError(
                        f"{member} is {len(data)} bytes, and {self.label} records are "
                        f"{stride} bytes each"
                    )
                view = memoryview(data)
                for at in range(0, len(data), stride):
                    row: dict[str, Any] = {}
                    if self.coarse:
                        row["coarse"], label = data[at], data[at + 1]
                    else:
                        label = data[at]
                    start = at + stride - width
                    row["image"] = view[start : start + width]
                    row["label"] = label
                    row["index"] = index
                    index += 1
                    yield label, row
        finally:
            tar.close()


def _extract(tar: tarfile.TarFile, held: Mapping[str, Any], member: str) -> bytes:
    """One member of a tar as bytes, refusing what is not there or not a file."""
    found = held.get(member)
    handle = tar.extractfile(found) if found is not None else None
    if handle is None:
        raise ValueError(f"this archive holds {', '.join(sorted(held))}, and not {member!r}")
    with handle:
        return handle.read()


@dataclass(frozen=True, slots=True, kw_only=True)
class Audio(Dataset):
    """Recordings in an archive of WAV files, one clip an entry.

    A tree column holds a fixed number of values, so every clip is written
    into one :attr:`samples` long and the ``length`` column says how much of
    it is real - the rest is silence this module put there. Nothing is
    resampled, mixed down or trimmed: a clip that is not what :attr:`rate`
    says, or is longer than the column, is refused by name.
    """

    #: The archive holding the recordings.
    archive: str
    #: The folder inside it they are in.
    folder: str
    #: What the first part of a file name means, as an index into
    #: :attr:`~Dataset.classes`.
    labels: Mapping[str, int]
    #: Who is speaking, in code order, when the file names say. A speaker
    #: nobody wrote down is refused rather than numbered on the spot.
    speakers: tuple[str, ...] = ()
    #: How many samples a second every recording must be.
    rate: int = 8000
    #: How long the column is. A clip is padded to it, never cut down to it.
    samples: int = 20000

    def urls(self, split: str) -> dict[str, str]:
        """The single archive, whichever split was asked for."""
        return {"archive": self.archive}

    def entry_title(self, split: str, cls: str) -> str:
        return (
            f"{self.label} recordings of class {cls}, {self.rate} Hz mono, "
            f"{self.samples} samples an entry"
        )

    def rows(self, raw: Mapping[str, bytes], split: str) -> Loaded:
        """The recordings under the folder, in name order, and their columns."""
        archive = zipfile.ZipFile(io.BytesIO(raw["archive"]))
        try:
            held = sorted(
                name
                for name in archive.namelist()
                if name.endswith(".wav") and self.folder in name.split("/")[:-1]
            )
            if not held:
                raise ValueError(
                    f"this archive has no .wav files in a {self.folder!r} folder, and that "
                    f"is where {self.label} keeps its recordings"
                )
            columns: dict[str, Any] = {"audio": ("h", self.samples), "length": "i", "label": "i"}
            if self.speakers:
                columns["speaker"] = "i"
            columns["index"] = "i"
        except Exception:
            archive.close()
            raise
        return self.classes, columns, self._entries(archive, held)

    def _entries(self, archive: zipfile.ZipFile, held: Sequence[str]) -> Rows:
        speakers = {name: at for at, name in enumerate(self.speakers)}
        try:
            for index, path in enumerate(held):
                stem = path.rsplit("/", 1)[-1][: -len(".wav")]
                parts = stem.split("_")
                if parts[0] not in self.labels:
                    raise ValueError(
                        f"{stem} of {self.label} is labelled {parts[0]!r}, and its classes "
                        f"are {', '.join(sorted(self.labels))}"
                    )
                which = self.labels[parts[0]]
                values, count = self._clip(archive.read(path), stem)
                row: dict[str, Any] = {
                    "audio": values,
                    "length": count,
                    "label": which,
                    "index": index,
                }
                if self.speakers:
                    who = parts[1] if len(parts) > 1 else ""
                    if who not in speakers:
                        raise ValueError(
                            f"{stem} of {self.label} is spoken by {who!r}, and the speakers "
                            f"it knows are {', '.join(self.speakers)}"
                        )
                    row["speaker"] = speakers[who]
                yield which, row
        finally:
            archive.close()

    def _clip(self, raw: bytes, stem: str) -> tuple[tuple[int, ...], int]:
        """One recording's samples, padded out to the width of the column."""
        with wave.open(io.BytesIO(raw)) as clip:
            spoken = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate())
            if spoken != (1, 2, self.rate):
                channels, width, rate = spoken
                raise ValueError(
                    f"{stem} of {self.label} is {channels}-channel {8 * width}-bit at "
                    f"{rate} Hz, and these recordings are mono 16-bit at {self.rate} Hz"
                )
            frames = clip.readframes(clip.getnframes())
        count = len(frames) // 2
        if count > self.samples:
            raise ValueError(
                f"{stem} of {self.label} is {count} samples long, and the column holds "
                f"{self.samples}"
            )
        return struct.unpack(f"<{count}h", frames) + (0,) * (self.samples - count), count


@dataclass(frozen=True, slots=True, kw_only=True)
class Table(Dataset):
    """A delimited table: one row an example, one field a feature or its label."""

    #: Where it is served from.
    url: str
    #: Which member of the archive it is, when it arrives in one.
    member: str = ""
    #: Which member of the outer archive holds the inner one, when it arrives
    #: in a zip inside a zip.
    inner: str = ""
    #: Which member each split is, when the archive holds more than one.
    files: Mapping[str, str] = field(default_factory=dict)
    #: What separates the fields, or ``None`` for any run of whitespace.
    delimiter: str | None = ","
    #: Whether the first line names them rather than holding one.
    header: bool = False
    #: What a line that is not data starts with, when the file has any.
    comment: str = ""
    #: Whether it is an ARFF file, whose rows come after its ``@data`` line.
    arff: bool = False
    #: Whether it is a spreadsheet rather than a text file, read with
    #: :func:`read_xlsx`.
    xlsx: bool = False
    #: How many fields a whitespace-separated row has when the last of them is
    #: free text with spaces in it. Zero splits on every run of whitespace.
    tail: int = 0
    #: How a ``"date"`` or ``"time"`` field is written, in
    #: :func:`~datetime.datetime.strptime` terms. A ``"date"`` column holds the
    #: days since 1970 and a ``"time"`` column the seconds, so a set that
    #: measures something every quarter of an hour keeps its clock.
    dates: str = "%Y-%m-%d"
    #: Whether a double quote groups a field. A table of sentences means its
    #: quotes literally and sets this ``False``.
    quoted: bool = True
    #: How many bytes a ``"text"`` field is given. A tree column is a fixed
    #: size, so the text is written into one this wide with a ``_length``
    #: column beside it; anything longer is refused rather than cut short.
    text_size: int = 0
    #: The fields in the order the file writes them: a name, and either a
    #: typecode, ``"label"``, ``"text"``, ``"date"``, ``"time"``, ``"target"``,
    #: or the name of an entry in :attr:`codes`. A set with no ``"label"`` and no
    #: :attr:`classes` is a regression set: it has a ``"target"`` to predict
    #: instead of a class to sort into, and its rows go to one tree.
    fields: tuple[tuple[str, str], ...]
    #: What each label in the file means, as an index into :attr:`classes`.
    labels: Mapping[str, int] = field(default_factory=dict)
    #: How the categorical fields are numbered, either as a mapping or as the
    #: categories in code order - a string of them where each is one letter.
    #: A category nobody wrote down is refused rather than guessed at.
    codes: Mapping[str, Mapping[str, int] | Sequence[str]] = field(default_factory=dict)

    def urls(self, split: str) -> dict[str, str]:
        """The one file it comes in."""
        return {"table": self.url}

    def entry_title(self, split: str, cls: str) -> str:
        """The title of the tree, which holds every row when there are no classes."""
        where = f"{split} " if len(self.splits) > 1 else ""
        return f"{self.label} {where}rows" + (f" labelled {cls}" if self.classes else "")

    def sorting(self) -> str:
        """How many classes it has, or that it has a number to predict instead."""
        return f"{len(self.classes)} classes" if self.classes else "no classes, a number to predict"

    def layout(self) -> str:
        """One tree a class, unless there are none and every row shares one."""
        return "one tree per class" if self.classes else "one tree of every row"

    def rows(self, raw: Mapping[str, bytes], split: str) -> Loaded:
        """The columns the fields become, then the rows themselves."""
        columns: dict[str, Any] = {}
        for name, role in self.fields:
            if role == "label":
                continue
            if role == "text":
                columns[name] = ("B", self.text_size)
                columns[f"{name}_length"] = "i"
            elif role in ("d", "target"):
                columns[name] = "d"
            elif role == "time":
                columns[name] = "q"
            else:
                columns[name] = "i"
        if self.classes:
            columns["label"] = "i"
        columns["index"] = "i"
        return self.classes or ("rows",), columns, self._entries(raw["table"], split)

    def _text(self, cell: str, name: str, index: int) -> bytes:
        """One field as the bytes its column holds, padded out to the width of it."""
        written = cell.encode()
        if len(written) > self.text_size:
            raise ValueError(
                f"row {index} of {self.label} has {len(written)} bytes in {name}, and the "
                f"column holds {self.text_size}"
            )
        return written + bytes(self.text_size - len(written))

    def _when(self, cell: str, name: str, index: int) -> datetime:
        """One field read as the moment it is written as."""
        try:
            return datetime.strptime(cell, self.dates)
        except ValueError:
            raise ValueError(
                f"row {index} of {self.label} has {cell!r} in {name}, and the dates in it are "
                f"written {self.dates}"
            ) from None

    def _day(self, cell: str, name: str, index: int) -> int:
        """One date as the days since 1970 its column holds."""
        return self._when(cell, name, index).date().toordinal() - EPOCH

    def _moment(self, cell: str, name: str, index: int) -> int:
        """One timestamp as the whole seconds since 1970 its column holds."""
        return int((self._when(cell, name, index) - MIDNIGHT).total_seconds())

    def _cell(
        self, cell: str, name: str, role: str, index: int, coded: Mapping[str, Mapping[str, int]]
    ) -> Any:
        """One field as the number its column holds, or what a gap means there."""
        if role in ("d", "target"):
            return nan if cell in MISSING else _number(cell, float, name, index, self.label)
        if role == "date":
            return -1 if cell in MISSING else self._day(cell, name, index)
        if role == "time":
            return -1 if cell in MISSING else self._moment(cell, name, index)
        if role == "i":
            return -1 if cell in MISSING else _number(cell, int, name, index, self.label)
        if cell in MISSING:
            return -1
        if cell not in coded[role]:
            raise ValueError(
                f"row {index} of {self.label} has {cell!r} in {name}, and the ones it knows "
                f"are {', '.join(sorted(coded[role]))}"
            )
        return coded[role][cell]

    def _read(self, held: bytes) -> Iterator[list[str]]:
        """The rows of the file, however this one happens to be written."""
        if self.arff:
            return read_arff(held)
        if self.xlsx:
            return read_xlsx(held, header=self.header)
        return read_table(
            held,
            delimiter=self.delimiter,
            header=self.header,
            comment=self.comment,
            quoted=self.quoted,
            tail=self.tail,
        )

    def _entries(self, raw: bytes, split: str) -> Rows:
        if self.inner:
            raw = _member(raw, self.inner)
        table = self._read(_member(raw, self.files.get(split, self.member)))
        coded = {role: _numbered(kinds) for role, kinds in self.codes.items()}
        for index, cells in enumerate(table):
            if len(cells) != len(self.fields):
                raise ValueError(
                    f"row {index} of {self.label} has {len(cells)} fields, and its columns "
                    f"are {len(self.fields)}"
                )
            which = -1 if self.classes else 0
            row: dict[str, Any] = {}
            for cell, (name, role) in zip(cells, self.fields, strict=True):
                if role == "text":
                    row[name] = self._text(cell, name, index)
                    row[f"{name}_length"] = len(cell.encode())
                elif role != "label":
                    row[name] = self._cell(cell, name, role, index, coded)
                elif cell in self.labels:
                    which = self.labels[cell]
                else:
                    raise ValueError(
                        f"row {index} of {self.label} is labelled {cell!r}, and its classes "
                        f"are {', '.join(sorted(self.labels))}"
                    )
            if self.classes:
                row["label"] = which
            row["index"] = index
            yield which, row


@dataclass(frozen=True, slots=True, kw_only=True)
class Matrix(Dataset):
    """A file that is one long row of numbers an example, and nothing else.

    The feature-vector sets are written this way: every row the same width, no
    header naming anything, and the label kept wherever the publisher found
    convenient - in a file of its own beside the numbers, as a run of columns
    at the end of the row, or nowhere at all except a count at the top saying
    how many rows each class has. All three are read here, because the
    alternative is asking everyone to reshape the file first.

    The numbers become one fixed-size column rather than one column a feature,
    which is what a training loop wants of 561 of them, and is what
    :func:`~.ml.iter_tensors` hands straight to a tensor.
    """

    #: Where it is served from.
    url: str
    #: The archive inside the archive, when it arrives wrapped twice.
    inner: str = ""
    #: Which member of it each split's numbers are.
    files: Mapping[str, str]
    #: Which member each split's labels are, when they are kept apart.
    label_files: Mapping[str, str] = field(default_factory=dict)
    #: What each label in such a file means, as an index into
    #: :attr:`~Dataset.classes`.
    labels: Mapping[str, int] = field(default_factory=dict)
    #: Any other column kept in a file of its own, as name to member a split.
    beside: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: How many numbers a row holds, checked against every one of them.
    width: int
    #: What the column of them is called.
    column: str = "features"
    #: What one of them is stored as.
    kind: str = "d"
    #: How many columns at the end of a row are a one-hot label, if it is
    #: written there rather than in a file of its own.
    onehot: int = 0
    #: Whether the first line counts the rows of each class in turn, which is
    #: how a file with no labels in it at all still says what its rows are.
    counts: bool = False

    def urls(self, split: str) -> dict[str, str]:
        """The one archive everything comes in."""
        return {"archive": self.url}

    def entry_title(self, split: str, cls: str) -> str:
        where = f"{split} " if len(self.splits) > 1 else ""
        return f"{self.label} {where}rows labelled {cls}, {self.width} numbers each"

    def rows(self, raw: Mapping[str, bytes], split: str) -> Loaded:
        """The fixed-size column of numbers, the label, and whatever is beside it."""
        columns: dict[str, Any] = {self.column: (self.kind, self.width), "label": "i"}
        for name in self.beside:
            columns[name] = "i"
        columns["index"] = "i"
        return self.classes, columns, self._entries(raw["archive"], split)

    def _held(self, raw: bytes, member: str) -> bytes:
        """One member of the archive, out of the archive inside it when there is one."""
        return _member(_member(raw, self.inner) if self.inner else raw, member)

    def _column(self, raw: bytes, member: str) -> list[str]:
        """A file holding one value a line, which is how a label or a subject arrives."""
        return [cells[0] for cells in read_table(self._held(raw, member), delimiter=None)]

    def _runs(self, counted: Sequence[str]) -> list[int]:
        """Where each class ends, from the line that says how many rows each has."""
        if len(counted) != len(self.classes):
            raise ValueError(
                f"{self.label} starts by counting {len(counted)} classes, and it has "
                f"{len(self.classes)}"
            )
        ends, total = [], 0
        for count, cls in zip(counted, self.classes, strict=True):
            total += _number(count, int, cls, 0, self.label)
            ends.append(total)
        return ends

    def _entries(self, raw: bytes, split: str) -> Rows:
        table = read_table(self._held(raw, self.files[split]), delimiter=None)
        ends = self._runs(next(table, [])) if self.counts else []
        told = self._column(raw, self.label_files[split]) if self.label_files else []
        beside = {
            name: self._column(raw, wheres[split]) for name, wheres in self.beside.items()
        }
        index = -1
        for index, cells in enumerate(table):
            if len(cells) != self.width + self.onehot:
                raise ValueError(
                    f"row {index} of {self.label} holds {len(cells)} numbers, and its rows "
                    f"are {self.width + self.onehot} wide"
                )
            row: dict[str, Any] = {
                self.column: self._values(cells[: self.width], index),
                "index": index,
            }
            for name, values in beside.items():
                row[name] = _number(self._beside(values, name, index), int, name, index, self.label)
            which = self._which(cells, told, ends, index)
            row["label"] = which
            yield which, row
        for name, values in (("labels", told), *beside.items()):
            if values and len(values) != index + 1:
                raise ValueError(
                    f"{self.label} has {index + 1} rows in {split} and {len(values)} in its "
                    f"file of {name}"
                )

    def _values(self, cells: Sequence[str], index: int) -> tuple[Any, ...]:
        """One row's numbers, as whatever the column stores them as."""
        numbers = tuple(_number(cell, float, self.column, index, self.label) for cell in cells)
        return numbers if self.kind == "d" else tuple(int(number) for number in numbers)

    def _beside(self, values: Sequence[str], name: str, index: int) -> str:
        """One row's value out of a file kept beside the numbers."""
        if index >= len(values):
            raise ValueError(
                f"{self.label} has more rows than the {len(values)} in its file of {name}"
            )
        return values[index]

    def _which(
        self, cells: Sequence[str], told: Sequence[str], ends: Sequence[int], index: int
    ) -> int:
        """Which class a row belongs to, however this dataset says so."""
        if self.onehot:
            hot = [at for at, cell in enumerate(cells[self.width :]) if float(cell)]
            if len(hot) != 1:
                raise ValueError(
                    f"row {index} of {self.label} has {len(hot)} of its {self.onehot} label "
                    f"columns set, and exactly one of them is a label"
                )
            return hot[0]
        if ends:
            for at, end in enumerate(ends):
                if index < end:
                    return at
            raise ValueError(
                f"{self.label} counts {ends[-1]} rows at the top of its file and holds more "
                f"than that"
            )
        cell = self._beside(told, "labels", index)
        if cell not in self.labels:
            raise ValueError(
                f"row {index} of {self.label} is labelled {cell!r}, and its classes "
                f"are {', '.join(sorted(self.labels))}"
            )
        return self.labels[cell]


_COVER_FIELDS: tuple[tuple[str, str], ...] = (
    *(
        (name, "i")
        for name in (
            "elevation",
            "aspect",
            "slope",
            "horizontal_distance_to_hydrology",
            "vertical_distance_to_hydrology",
            "horizontal_distance_to_roadways",
            "hillshade_9am",
            "hillshade_noon",
            "hillshade_3pm",
            "horizontal_distance_to_fire_points",
            *(f"wilderness_area_{area}" for area in range(1, 5)),
            *(f"soil_type_{soil}" for soil in range(1, 41)),
        )
    ),
    ("cover_type", "label"),
)


#: The 47 balanced EMNIST classes: the ten digits, the capitals, and the
#: eleven lower-case letters that do not look like their own capital.
_EMNIST_CLASSES: tuple[str, ...] = (
    *(str(digit) for digit in range(10)),
    *(f"upper_{chr(letter)}" for letter in range(ord("a"), ord("z") + 1)),
    *(f"lower_{letter}" for letter in "abdefghnqrt"),
)

_EMNIST_FILES: dict[str, tuple[str, str]] = {
    split: (
        f"gzip/emnist-balanced-{split}-images-idx3-ubyte.gz",
        f"gzip/emnist-balanced-{split}-labels-idx1-ubyte.gz",
    )
    for split in ("train", "test")
}

#: Each mushroom field and the letters it is written with, in the order the
#: dataset's own description gives them, so the codes are the published ones.
_MUSHROOM: tuple[tuple[str, str], ...] = (
    ("cap_shape", "bcxfks"),
    ("cap_surface", "fgys"),
    ("cap_colour", "nbcgrpuewy"),
    ("bruises", "tf"),
    ("odour", "alcyfmnps"),
    ("gill_attachment", "adfn"),
    ("gill_spacing", "cwd"),
    ("gill_size", "bn"),
    ("gill_colour", "knbhgropuewy"),
    ("stalk_shape", "et"),
    ("stalk_root", "bcuezr"),
    ("stalk_surface_above_ring", "fyks"),
    ("stalk_surface_below_ring", "fyks"),
    ("stalk_colour_above_ring", "nbcgopewy"),
    ("stalk_colour_below_ring", "nbcgopewy"),
    ("veil_type", "pu"),
    ("veil_colour", "nowy"),
    ("ring_number", "not"),
    ("ring_type", "ceflnpsz"),
    ("spore_print_colour", "knbhrouwy"),
    ("population", "acnsvy"),
    ("habitat", "glmpuwd"),
)

_ADULT_CODES: dict[str, tuple[str, ...]] = {
    "workclass": (
        "Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc",
        "Self-emp-not-inc", "State-gov", "Without-pay",
    ),
    "education": (
        "10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm",
        "Assoc-voc", "Bachelors", "Doctorate", "HS-grad", "Masters", "Preschool",
        "Prof-school", "Some-college",
    ),
    "marital_status": (
        "Divorced", "Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent",
        "Never-married", "Separated", "Widowed",
    ),
    "occupation": (
        "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing",
        "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv",
        "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving",
    ),
    "relationship": (
        "Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife",
    ),
    "race": ("Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"),
    "sex": ("Female", "Male"),
    "native_country": (
        "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador",
        "El-Salvador", "England", "France", "Germany", "Greece", "Guatemala", "Haiti",
        "Holand-Netherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland",
        "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua",
        "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal",
        "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago",
        "United-States", "Vietnam", "Yugoslavia",
    ),
}

_ADULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"),
    ("workclass", "workclass"),
    ("fnlwgt", "i"),
    ("education", "education"),
    ("education_years", "i"),
    ("marital_status", "marital_status"),
    ("occupation", "occupation"),
    ("relationship", "relationship"),
    ("race", "race"),
    ("sex", "sex"),
    ("capital_gain", "i"),
    ("capital_loss", "i"),
    ("hours_per_week", "i"),
    ("native_country", "native_country"),
    ("income", "label"),
)

_LETTER_FIELDS: tuple[tuple[str, str], ...] = (
    ("letter", "label"),
    *(
        (name, "i")
        for name in (
            "x_box", "y_box", "width", "height", "on_pixels", "x_bar", "y_bar", "x2_bar",
            "y2_bar", "xy_bar", "x2y_bar", "xy2_bar", "x_edge", "x_edge_by_y", "y_edge",
            "y_edge_by_x",
        )
    ),
)

_DIGIT_FIELDS: tuple[tuple[str, str], ...] = (
    *((f"pixel_{at:02d}", "i") for at in range(64)),
    ("digit", "label"),
)

_WDBC_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"),
    ("diagnosis", "label"),
    *(
        (f"{name}_{kind}", "d")
        for kind in ("mean", "error", "worst")
        for name in (
            "radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave_points", "symmetry", "fractal_dimension",
        )
    ),
)

_WINE_FIELDS: tuple[tuple[str, str], ...] = (
    ("cultivar", "label"),
    *(
        (name, "d")
        for name in (
            "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols",
            "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "colour_intensity", "hue",
            "od280_od315", "proline",
        )
    ),
)

_SEED_FIELDS: tuple[tuple[str, str], ...] = (
    *(
        (name, "d")
        for name in (
            "area", "perimeter", "compactness", "kernel_length", "kernel_width",
            "asymmetry_coefficient", "kernel_groove_length",
        )
    ),
    ("variety", "label"),
)

_BEAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("area", "i"),
    ("perimeter", "d"),
    ("major_axis_length", "d"),
    ("minor_axis_length", "d"),
    ("aspect_ratio", "d"),
    ("eccentricity", "d"),
    ("convex_area", "i"),
    ("equivalent_diameter", "d"),
    ("extent", "d"),
    ("solidity", "d"),
    ("roundness", "d"),
    ("compactness", "d"),
    *((f"shape_factor_{at}", "d") for at in range(1, 5)),
    ("bean", "label"),
)

#: The words Spambase counts, in the order its own description lists them.
#: The six punctuation counts are named rather than spelled, because a branch
#: called ``char_freq_$`` is not a name anything downstream can use.
_SPAM_WORDS: tuple[str, ...] = (
    "make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail",
    "receive", "will", "people", "report", "addresses", "free", "business", "email", "you",
    "credit", "your", "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs",
    "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct",
    "cs", "meeting", "original", "project", "re", "edu", "table", "conference",
)

_SPAM_FIELDS: tuple[tuple[str, str], ...] = (
    *((f"word_freq_{word}", "d") for word in _SPAM_WORDS),
    *(
        (f"char_freq_{name}", "d")
        for name in ("semicolon", "bracket", "square_bracket", "exclamation", "dollar", "hash")
    ),
    ("capital_run_length_average", "d"),
    ("capital_run_length_longest", "i"),
    ("capital_run_length_total", "i"),
    ("spam", "label"),
)

_IONOSPHERE_FIELDS: tuple[tuple[str, str], ...] = (
    *(
        (f"pulse_{at // 2 + 1}_{'real' if at % 2 == 0 else 'imaginary'}", "d")
        for at in range(34)
    ),
    ("returns", "label"),
)

_GLASS_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"),
    ("refractive_index", "d"),
    *(
        (name, "d")
        for name in (
            "sodium", "magnesium", "aluminium", "silicon",
            "potassium", "calcium", "barium", "iron",
        )
    ),
    ("kind", "label"),
)

_ABALONE_FIELDS: tuple[tuple[str, str], ...] = (
    ("sex", "label"),
    *(
        (name, "d")
        for name in (
            "length", "diameter", "height", "whole_weight",
            "shucked_weight", "viscera_weight", "shell_weight",
        )
    ),
    ("rings", "i"),
)

_BANKNOTE_FIELDS: tuple[tuple[str, str], ...] = (
    ("variance", "d"),
    ("skewness", "d"),
    ("kurtosis", "d"),
    ("entropy", "d"),
    ("authenticity", "label"),
)

_WINE_QUALITY_FIELDS: tuple[tuple[str, str], ...] = (
    *(
        (name, "d")
        for name in (
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
            "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "ph", "sulphates",
            "alcohol",
        )
    ),
    ("quality", "label"),
)

#: What the six activities a phone was told to recognise are, in the order
#: the labels file numbers them from one.
_HAR_ACTIVITIES: tuple[str, ...] = (
    "walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying",
)

#: The shower of light a gamma ray leaves in the telescope, measured ten ways.
_MAGIC_FIELDS: tuple[tuple[str, str], ...] = (
    *(
        (name, "d")
        for name in (
            "length", "width", "size", "conc", "conc1", "asym", "m3_long", "m3_trans",
            "alpha", "dist",
        )
    ),
    ("shower", "label"),
)

#: Eight numbers a pulsar candidate is reduced to: four from the folded pulse
#: profile, four from the curve of signal against dispersion measure.
_HTRU_FIELDS: tuple[tuple[str, str], ...] = (
    *(
        (f"{where}_{what}", "d")
        for where in ("profile", "dmsnr")
        for what in ("mean", "stdev", "kurtosis", "skew")
    ),
    ("candidate", "label"),
)

_MPG_FIELDS: tuple[tuple[str, str], ...] = (
    ("mpg", "target"),
    ("cylinders", "i"),
    ("displacement", "d"),
    ("horsepower", "d"),
    ("weight", "d"),
    ("acceleration", "d"),
    ("model_year", "i"),
    ("origin", "i"),
    ("car_name", "text"),
)

_BIKE_FIELDS: tuple[tuple[str, str], ...] = (
    ("instant", "i"),
    ("date", "date"),
    *((name, "i") for name in ("season", "year", "month", "hour", "holiday", "weekday")),
    ("workingday", "i"),
    ("weather", "i"),
    *((name, "d") for name in ("temperature", "feels_like", "humidity", "windspeed")),
    ("casual", "i"),
    ("registered", "i"),
    ("count", "target"),
)

_ENERGY_FIELDS: tuple[tuple[str, str], ...] = (
    ("relative_compactness", "d"),
    ("surface_area", "d"),
    ("wall_area", "d"),
    ("roof_area", "d"),
    ("overall_height", "d"),
    ("orientation", "i"),
    ("glazing_area", "d"),
    ("glazing_distribution", "i"),
    ("heating_load", "target"),
    ("cooling_load", "target"),
)

_ESTATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("serial", "i"),
    ("transaction_date", "d"),
    ("house_age", "d"),
    ("distance_to_station", "d"),
    ("convenience_stores", "i"),
    ("latitude", "d"),
    ("longitude", "d"),
    ("price_per_unit_area", "target"),
)

#: The fourteen of the seventy-six columns that everyone uses, in the order
#: the processed files write them.
_HEART_FIELDS: tuple[tuple[str, str], ...] = (
    *(
        (name, "d")
        for name in (
            "age", "sex", "chest_pain", "rest_blood_pressure", "cholesterol",
            "fasting_sugar", "rest_ecg", "max_heart_rate", "exercise_angina",
            "st_depression", "st_slope", "vessels", "thallium",
        )
    ),
    ("diagnosis", "label"),
)

_CAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("buying", "price"),
    ("maintenance", "price"),
    ("doors", "doors"),
    ("persons", "persons"),
    ("luggage_boot", "boot"),
    ("safety", "safety"),
    ("acceptability", "label"),
)

#: The categories the car set is written in, worst to best in each field.
_CAR_CODES: dict[str, Mapping[str, int] | Sequence[str]] = {
    "price": ("low", "med", "high", "vhigh"),
    "doors": ("2", "3", "4", "5more"),
    "persons": ("2", "4", "more"),
    "boot": ("small", "med", "big"),
    "safety": ("low", "med", "high"),
}

_YEAST_FIELDS: tuple[tuple[str, str], ...] = (
    ("protein", "text"),
    *((name, "d") for name in ("mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc")),
    ("site", "label"),
)

#: Where in the cell a yeast protein ends up, by the abbreviation the file uses.
_YEAST_SITES: dict[str, int] = {
    name: at
    for at, name in enumerate(
        ("CYT", "NUC", "MIT", "ME3", "ME2", "ME1", "EXC", "VAC", "POX", "ERL")
    )
}

_YES_NO: tuple[str, ...] = ("no", "yes")

#: The thirty-three answers each pupil's row holds, the last of them the mark
#: to predict. The names are the ones the questionnaire uses.
_STUDENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("school", "school"),
    ("sex", "sex"),
    ("age", "i"),
    ("address", "address"),
    ("famsize", "famsize"),
    ("parents_status", "pstatus"),
    ("mother_education", "i"),
    ("father_education", "i"),
    ("mother_job", "job"),
    ("father_job", "job"),
    ("reason", "reason"),
    ("guardian", "guardian"),
    ("travel_time", "i"),
    ("study_time", "i"),
    ("failures", "i"),
    *(
        (name, "yesno")
        for name in (
            "school_support", "family_support", "paid_classes", "activities", "nursery",
            "wants_higher", "internet", "romantic",
        )
    ),
    *(
        (name, "i")
        for name in (
            "family_relations", "free_time", "going_out", "workday_alcohol",
            "weekend_alcohol", "health", "absences", "first_period", "second_period",
        )
    ),
    ("final_grade", "target"),
)

_STUDENT_CODES: dict[str, Mapping[str, int] | Sequence[str]] = {
    "school": ("GP", "MS"),
    "sex": ("F", "M"),
    "address": ("R", "U"),
    "famsize": ("LE3", "GT3"),
    "pstatus": ("A", "T"),
    "job": ("at_home", "health", "other", "services", "teacher"),
    "reason": ("course", "home", "other", "reputation"),
    "guardian": ("father", "mother", "other"),
    "yesno": _YES_NO,
}


#: Every dataset this module knows, by the name :func:`convert` takes.
#: The months a table spells with the first three letters of their name, and
#: the days a week is written the same way. A ``"date"`` column would be
#: better, but these files hold the month without the year to put it in.
_MONTHS: dict[str, int] = {
    name: at
    for at, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}
_DAYS: dict[str, int] = {
    name: at for at, name in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun"), 1)
}

#: The days a table spells out in full, Monday first, the way ISO numbers them.
_WEEKDAYS: dict[str, int] = {
    name: at
    for at, name in enumerate(
        ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"), 1
    )
}

#: How the medical sets write down a patient's sex, and how the two of them
#: that ask a yes-or-no question with a capital letter write the answer.
_SEXES: tuple[str, ...] = ("Female", "Male")
_YES_NO_CAPS: tuple[str, ...] = ("No", "Yes")

#: The aerofoil in a wind tunnel, and the noise it made.
_AIRFOIL_FIELDS: tuple[tuple[str, str], ...] = (
    ("frequency", "i"), ("angle", "d"), ("chord", "d"), ("velocity", "d"),
    ("thickness", "d"), ("sound_pressure", "target"),
)

#: How a 1985 import was described in the trade press, and what it cost. The
#: make is text because there are twenty-two of them and no code for any.
_AUTOMOBILE_FIELDS: tuple[tuple[str, str], ...] = (
    ("symboling", "i"), ("normalised_losses", "i"), ("make", "text"), ("fuel_type", "fuel"),
    ("aspiration", "aspiration"), ("doors", "doors"), ("body_style", "body"),
    ("drive_wheels", "drive"), ("engine_location", "where"), ("wheel_base", "d"),
    ("length", "d"), ("width", "d"), ("height", "d"), ("curb_weight", "i"),
    ("engine_type", "engine"), ("cylinders", "cylinders"), ("engine_size", "i"),
    ("fuel_system", "fuel_system"), ("bore", "d"), ("stroke", "d"),
    ("compression_ratio", "d"), ("horsepower", "i"), ("peak_rpm", "i"),
    ("city_mpg", "i"), ("highway_mpg", "i"), ("price", "target"),
)
_AUTOMOBILE_CODES: dict[str, tuple[str, ...] | dict[str, int]] = {
    "fuel": ("diesel", "gas"),
    "aspiration": ("std", "turbo"),
    "doors": {"two": 2, "four": 4},
    "body": ("convertible", "hardtop", "hatchback", "sedan", "wagon"),
    "drive": ("4wd", "fwd", "rwd"),
    "where": ("front", "rear"),
    "engine": ("dohc", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"),
    "cylinders": {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12},
    "fuel_system": ("1bbl", "2bbl", "4bbl", "idi", "mfi", "mpfi", "spdi", "spfi"),
}

#: Weights on the two arms of a balance, and which way it tipped.
_BALANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("label", "label"), ("left_weight", "i"), ("left_distance", "i"),
    ("right_weight", "i"), ("right_distance", "i"),
)

#: One telephone call of a Portuguese bank's campaign, and whether it sold
#: anything. ``pdays`` is -1 when nobody had called before.
_BANK_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"), ("job", "job"), ("marital", "marital"), ("education", "education"),
    ("credit_default", "yesno"), ("balance", "i"), ("housing", "yesno"), ("loan", "yesno"),
    ("contact", "contact"), ("day", "i"), ("month", "month"), ("duration", "i"),
    ("campaign", "i"), ("pdays", "i"), ("previous", "i"), ("outcome", "outcome"),
    ("label", "label"),
)
_BANK_CODES: dict[str, tuple[str, ...] | dict[str, int]] = {
    "job": (
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
        "self-employed", "services", "student", "technician", "unemployed", "unknown",
    ),
    "marital": ("divorced", "married", "single"),
    "education": ("primary", "secondary", "tertiary", "unknown"),
    "yesno": _YES_NO,
    "contact": ("cellular", "telephone", "unknown"),
    "month": _MONTHS,
    "outcome": ("failure", "other", "success", "unknown"),
}

#: How often a donor gave blood, and whether they gave again in March 2007.
_BLOOD_FIELDS: tuple[tuple[str, str], ...] = (
    ("recency", "i"), ("frequency", "i"), ("volume", "i"), ("time", "i"), ("label", "label"),
)

#: The eighteen knobs on an ocean model, and whether the run finished. Two
#: columns say which of the Latin hypercube studies the run belongs to.
_CLIMATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("study", "i"), ("run", "i"),
    *_plain(
        (
            "vconst_corr", "vconst_2", "vconst_3", "vconst_4", "vconst_5", "vconst_7",
            "ah_corr", "ah_bolus", "slm_corr", "efficiency_factor", "tidal_mix_max",
            "vertical_decay_scale", "convect_corr", "bckgrnd_vdc1", "bckgrnd_vdc_ban",
            "bckgrnd_vdc_eq", "bckgrnd_vdc_psim", "prandtl",
        ),
        "d",
    ),
    ("label", "label"),
)

#: A 1987 mainframe, its published relative performance to predict, and the
#: estimate the paper that came with the data published beside it.
_HARDWARE_FIELDS: tuple[tuple[str, str], ...] = (
    ("vendor", "text"), ("model", "text"), ("cycle_time", "i"), ("min_memory", "i"),
    ("max_memory", "i"), ("cache", "i"), ("min_channels", "i"), ("max_channels", "i"),
    ("performance", "target"), ("estimated_performance", "d"),
)

#: What went into a batch of concrete, and the three things measured of it.
_SLUMP_FIELDS: tuple[tuple[str, str], ...] = (
    ("number", "i"), ("cement", "d"), ("slag", "d"), ("fly_ash", "d"), ("water", "d"),
    ("superplasticiser", "d"), ("coarse_aggregate", "d"), ("fine_aggregate", "d"),
    ("slump", "target"), ("flow", "target"), ("compressive_strength", "target"),
)

#: A 1987 Indonesian survey. Every answer is already a number, so the codes
#: are the file's own: education and standard of living run low to high.
_CMC_FIELDS: tuple[tuple[str, str], ...] = (
    ("wife_age", "i"), ("wife_education", "i"), ("husband_education", "i"),
    ("children", "i"), ("wife_islam", "i"), ("wife_working", "i"),
    ("husband_occupation", "i"), ("living_standard", "i"), ("media_exposure", "i"),
    ("label", "label"),
)

#: Thirty-three signs a dermatologist or a microscope found, then the age.
_DERM_FIELDS: tuple[tuple[str, str], ...] = (
    *_plain(
        (
            "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
            "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
            "knee_and_elbow_involvement", "scalp_involvement", "family_history",
            "melanin_incontinence", "eosinophils_in_infiltrate", "pnl_infiltrate",
            "fibrosis_of_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
            "parakeratosis", "clubbing_of_rete_ridges", "elongation_of_rete_ridges",
            "thinning_of_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess",
            "focal_hypergranulosis", "disappearance_of_granular_layer",
            "vacuolisation_and_damage_of_basal_layer", "spongiosis",
            "saw_tooth_appearance_of_retes", "follicular_horn_plug",
            "perifollicular_parakeratosis", "inflammatory_mononuclear_infiltrate",
            "band_like_infiltrate", "age",
        )
    ),
    ("label", "label"),
)

#: Sixteen symptoms a patient was asked about, and the two they were not.
_DIABETES_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"), ("sex", "sex"),
    *_plain(
        (
            "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
            "genital_thrush", "visual_blurring", "itching", "irritability", "delayed_healing",
            "partial_paresis", "muscle_stiffness", "alopecia", "obesity",
        ),
        "yesno",
    ),
    ("label", "label"),
)

#: Seven scores a program gave a protein, and the accession number it has in
#: SWISS-PROT, which is text because it names the protein rather than measures
#: anything about it.
_ECOLI_FIELDS: tuple[tuple[str, str], ...] = (
    ("sequence", "text"), ("mcg", "d"), ("gvh", "d"), ("lip", "d"), ("chg", "d"),
    ("aac", "d"), ("alm1", "d"), ("alm2", "d"), ("label", "label"),
)

#: Nine answers about a man's health and habits, each already normalised to
#: run from -1 to 1, and whether his semen analysis came back normal.
_FERTILITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("season", "d"), ("age", "d"), ("childhood_diseases", "i"), ("accident", "i"),
    ("surgery", "i"), ("high_fevers", "i"), ("alcohol", "d"), ("smoking", "i"),
    ("hours_sitting", "d"), ("label", "label"),
)

#: Where in the park a fire started, when, what the Canadian fire weather
#: indices said that day, and how many hectares went up.
_FIRE_FIELDS: tuple[tuple[str, str], ...] = (
    ("x", "i"), ("y", "i"), ("month", "month"), ("day", "day"), ("ffmc", "d"),
    ("dmc", "d"), ("dc", "d"), ("isi", "d"), ("temperature", "d"), ("humidity", "i"),
    ("wind", "d"), ("rain", "d"), ("burnt_area", "target"),
)

#: A day's work by one team in a Bangladeshi clothing factory. ``wip`` is
#: blank for the finishing teams, who have no work in progress to count.
_GARMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("when", "date"), ("quarter", "quarter"), ("department", "department"),
    ("day", "weekday"), ("team", "i"), ("targeted_productivity", "d"), ("smv", "d"),
    ("work_in_progress", "d"), ("over_time", "i"), ("incentive", "d"), ("idle_time", "d"),
    ("idle_men", "i"), ("style_changes", "i"), ("workers", "d"), ("productivity", "target"),
)

#: Twenty things a German bank knew about an applicant in 1994. The
#: categorical ones are written as the file's own A-codes, and each is
#: numbered in the order the documentation lists it.
_GERMAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("checking_account", "checking"), ("duration", "i"), ("credit_history", "history"),
    ("purpose", "purpose"), ("amount", "i"), ("savings", "savings"),
    ("employment", "employment"), ("instalment_rate", "i"), ("personal_status", "personal"),
    ("other_debtors", "debtors"), ("residence_since", "i"), ("property", "property"),
    ("age", "i"), ("other_instalments", "instalments"), ("housing", "housing"),
    ("existing_credits", "i"), ("job", "job"), ("dependents", "i"),
    ("telephone", "telephone"), ("foreign_worker", "foreign"), ("label", "label"),
)
_GERMAN_CODES: dict[str, tuple[str, ...]] = {
    "checking": ("A11", "A12", "A13", "A14"),
    "history": ("A30", "A31", "A32", "A33", "A34"),
    "purpose": (
        "A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410",
    ),
    "savings": ("A61", "A62", "A63", "A64", "A65"),
    "employment": ("A71", "A72", "A73", "A74", "A75"),
    "personal": ("A91", "A92", "A93", "A94", "A95"),
    "debtors": ("A101", "A102", "A103"),
    "property": ("A121", "A122", "A123", "A124"),
    "instalments": ("A141", "A142", "A143"),
    "housing": ("A151", "A152", "A153"),
    "job": ("A171", "A172", "A173", "A174"),
    "telephone": ("A191", "A192"),
    "foreign": ("A201", "A202"),
}

#: A breast cancer operation at Billings Hospital, and whether the patient
#: was still alive five years later.
_HABERMAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"), ("year", "i"), ("nodes", "i"), ("label", "label"),
)

#: Blood work from a heart failure clinic in Faisalabad, and ``time``, the
#: days the patient was followed for before the study ended.
_FAILURE_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "d"), ("anaemia", "i"), ("creatinine_phosphokinase", "i"), ("diabetes", "i"),
    ("ejection_fraction", "i"), ("high_blood_pressure", "i"), ("platelets", "d"),
    ("serum_creatinine", "d"), ("serum_sodium", "i"), ("sex", "i"), ("smoking", "i"),
    ("time", "i"), ("label", "label"),
)

#: Nineteen things asked of a hepatitis patient. The yes-or-no answers are
#: written 1 for no and 2 for yes, which is the file's own numbering.
_HEPATITIS_FIELDS: tuple[tuple[str, str], ...] = (
    ("label", "label"), ("age", "i"),
    *_plain(
        (
            "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "liver_big",
            "liver_firm", "spleen_palpable", "spiders", "ascites", "varices",
        )
    ),
    ("bilirubin", "d"), ("alkaline_phosphate", "i"), ("sgot", "i"), ("albumin", "d"),
    ("prothrombin_time", "i"), ("histology", "i"),
)

#: Blood work from 583 patients in Andhra Pradesh. Four rows never had the
#: albumin to globulin ratio worked out and leave it blank.
_ILPD_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"), ("sex", "sex"), ("total_bilirubin", "d"), ("direct_bilirubin", "d"),
    ("alkaline_phosphotase", "i"), ("alamine_aminotransferase", "i"),
    ("aspartate_aminotransferase", "i"), ("total_proteins", "d"), ("albumin", "d"),
    ("albumin_globulin_ratio", "d"), ("label", "label"),
)

#: Nineteen numbers describing a 3x3 patch cut out of an outdoor photograph.
_SEGMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("label", "label"),
    *_plain(
        (
            "region_centroid_col", "region_centroid_row", "region_pixel_count",
            "short_line_density_5", "short_line_density_2", "vedge_mean", "vedge_sd",
            "hedge_mean", "hedge_sd", "intensity_mean", "rawred_mean", "rawblue_mean",
            "rawgreen_mean", "exred_mean", "exblue_mean", "exgreen_mean", "value_mean",
            "saturation_mean", "hue_mean",
        ),
        "d",
    ),
)

#: Five blood tests and the drinking to predict from them. The seventh column
#: is not a class: whoever made the file used it to split the rows in two, and
#: reading it as a diagnosis is the mistake this dataset is known for.
_BUPA_FIELDS: tuple[tuple[str, str], ...] = (
    ("mcv", "i"), ("alkaline_phosphotase", "i"), ("sgpt", "i"), ("sgot", "i"),
    ("gamma_glutamyl_transpeptidase", "i"), ("drinks", "target"), ("selector", "i"),
)

#: Eighteen findings from a lymph node X-ray, each already numbered by where
#: it comes in the list the documentation gives for that field.
_LYMPH_FIELDS: tuple[tuple[str, str], ...] = (
    ("label", "label"),
    *_plain(
        (
            "lymphatics", "block_of_afferent", "block_of_lymph_c", "block_of_lymph_s",
            "bypass", "extravasates", "regeneration", "early_uptake",
            "lymph_nodes_diminished", "lymph_nodes_enlarged", "changes_in_lymph",
            "defect_in_node", "changes_in_node", "changes_in_structure", "special_forms",
            "dislocation", "exclusion_of_node", "number_of_nodes",
        )
    ),
)

#: What a radiologist wrote down about a lump on a mammogram.
_MAMMOGRAM_FIELDS: tuple[tuple[str, str], ...] = (
    ("birads", "i"), ("age", "i"), ("shape", "i"), ("margin", "i"), ("density", "i"),
    ("label", "label"),
)

#: Six readings taken in rural Bangladeshi clinics, and the risk given.
_MATERNAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"), ("systolic_bp", "i"), ("diastolic_bp", "i"), ("blood_sugar", "d"),
    ("body_temperature", "d"), ("heart_rate", "i"), ("label", "label"),
)

#: Eight things a Ljubljana nursery asked about a family, and the ranking the
#: application got. Every combination of the eight is in the file once.
_NURSERY_FIELDS: tuple[tuple[str, str], ...] = (
    ("parents", "parents"), ("has_nursery", "nursery"), ("form", "form"),
    ("children", "children"), ("housing", "housing"), ("finance", "finance"),
    ("social", "social"), ("health", "health"), ("label", "label"),
)
_NURSERY_CODES: dict[str, tuple[str, ...] | dict[str, int]] = {
    "parents": ("usual", "pretentious", "great_pret"),
    "nursery": ("proper", "less_proper", "improper", "critical", "very_crit"),
    "form": ("complete", "completed", "incomplete", "foster"),
    "children": {"1": 1, "2": 2, "3": 3, "more": 4},
    "housing": ("convenient", "less_conv", "critical"),
    "finance": ("convenient", "inconv"),
    "social": ("nonprob", "slightly_prob", "problematic"),
    "health": ("recommended", "priority", "not_recom"),
}

#: A minute in an office, and whether anyone was in it. The first column is
#: the row's number in the file, which the file itself writes out.
_OCCUPANCY_FIELDS: tuple[tuple[str, str], ...] = (
    ("number", "i"), ("when", "time"), ("temperature", "d"), ("humidity", "d"),
    ("light", "d"), ("carbon_dioxide", "d"), ("humidity_ratio", "d"), ("label", "label"),
)

#: A session on a shopping site, and whether it ended in a sale.
_SHOPPER_FIELDS: tuple[tuple[str, str], ...] = (
    ("administrative", "i"), ("administrative_duration", "d"), ("informational", "i"),
    ("informational_duration", "d"), ("product_related", "i"),
    ("product_related_duration", "d"), ("bounce_rate", "d"), ("exit_rate", "d"),
    ("page_value", "d"), ("special_day", "d"), ("month", "month"),
    ("operating_system", "i"), ("browser", "i"), ("region", "i"), ("traffic_type", "i"),
    ("visitor_type", "visitor"), ("weekend", "truefalse"), ("label", "label"),
)
_SHOPPER_CODES: dict[str, tuple[str, ...] | dict[str, int]] = {
    "month": {
        "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jul": 7,
        "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    },
    "visitor": ("New_Visitor", "Other", "Returning_Visitor"),
    "truefalse": {"FALSE": 0, "TRUE": 1},
}

#: Twenty-two measurements of one sustained vowel, and whose voice it was.
#: The recording is named rather than measured, so it is text.
_PARKINSONS_FIELDS: tuple[tuple[str, str], ...] = (
    ("recording", "text"), ("fo", "d"), ("fhi", "d"), ("flo", "d"),
    ("jitter_percent", "d"), ("jitter_absolute", "d"), ("rap", "d"), ("ppq", "d"),
    ("jitter_ddp", "d"), ("shimmer", "d"), ("shimmer_db", "d"), ("apq3", "d"),
    ("apq5", "d"), ("apq", "d"), ("shimmer_dda", "d"), ("nhr", "d"), ("hnr", "d"),
    ("label", "label"), ("rpde", "d"), ("dfa", "d"), ("spread1", "d"), ("spread2", "d"),
    ("d2", "d"), ("ppe", "d"),
)

#: The same voice measurements taken at home over six months, with the two
#: halves of the clinician's score to predict from them.
_TELEMONITORING_FIELDS: tuple[tuple[str, str], ...] = (
    ("subject", "i"), ("age", "i"), ("sex", "i"), ("test_time", "d"),
    ("motor_updrs", "target"), ("total_updrs", "target"), ("jitter_percent", "d"),
    ("jitter_absolute", "d"), ("rap", "d"), ("ppq5", "d"), ("jitter_ddp", "d"),
    ("shimmer", "d"), ("shimmer_db", "d"), ("apq3", "d"), ("apq5", "d"), ("apq11", "d"),
    ("shimmer_dda", "d"), ("nhr", "d"), ("hnr", "d"), ("rpde", "d"), ("dfa", "d"),
    ("ppe", "d"),
)

#: Eight points off a pen's path across a tablet, resampled so that every
#: digit is written with the same number of them.
_PENDIGIT_FIELDS: tuple[tuple[str, str], ...] = (
    *tuple(
        field for at in range(1, 9) for field in ((f"x{at}", "i"), (f"y{at}", "i"))
    ),
    ("label", "label"),
)

#: Thirty rules of thumb for spotting a phishing site. Each is -1, 0 or 1:
#: suspicious, in between, or fine.
_PHISHING_FIELDS: tuple[tuple[str, str], ...] = (
    *_plain(
        (
            "having_ip_address", "url_length", "shortening_service", "having_at_symbol",
            "double_slash_redirecting", "prefix_suffix", "having_sub_domain",
            "ssl_final_state", "domain_registration_length", "favicon", "port", "https_token",
            "request_url", "url_of_anchor", "links_in_tags", "server_form_handler",
            "submitting_to_email", "abnormal_url", "redirect", "on_mouseover", "right_click",
            "popup_window", "iframe", "age_of_domain", "dns_record", "web_traffic",
            "page_rank", "google_index", "links_pointing_to_page", "statistical_report",
        )
    ),
    ("label", "label"),
)

#: The ambient conditions a gas turbine ran in, and the megawatts it made.
_POWER_FIELDS: tuple[tuple[str, str], ...] = (
    ("temperature", "d"), ("exhaust_vacuum", "d"), ("ambient_pressure", "d"),
    ("relative_humidity", "d"), ("power_output", "target"),
)

#: Molecular descriptors, and the concentration that killed half of a
#: population in 48 hours - Daphnia magna here, fathead minnow below.
_AQUATIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("tpsa", "d"), ("saacc", "d"), ("h_050", "i"), ("mlogp", "d"), ("rdchi", "d"),
    ("gats1p", "d"), ("nitrogen_atoms", "i"), ("c_040", "i"), ("lc50", "target"),
)
_FISH_FIELDS: tuple[tuple[str, str], ...] = (
    ("cic0", "d"), ("sm1_dz", "d"), ("gats1i", "d"), ("ndsch", "i"), ("ndssc", "i"),
    ("mlogp", "d"), ("lc50", "target"),
)

#: The shape of one grain, measured off a photograph of it.
_RAISIN_FIELDS: tuple[tuple[str, str], ...] = (
    ("area", "i"), ("major_axis", "d"), ("minor_axis", "d"), ("eccentricity", "d"),
    ("convex_area", "i"), ("extent", "d"), ("perimeter", "d"), ("label", "label"),
)
_RICE_FIELDS: tuple[tuple[str, str], ...] = (
    ("area", "i"), ("perimeter", "d"), ("major_axis", "d"), ("minor_axis", "d"),
    ("eccentricity", "d"), ("convex_area", "i"), ("extent", "d"), ("label", "label"),
)

#: Four Landsat bands for each of the nine pixels around one, read out left
#: to right and top to bottom, so the middle pixel is the fifth of the nine.
_SATELLITE_FIELDS: tuple[tuple[str, str], ...] = (
    *tuple(
        (f"pixel{at}_band{band}", "i") for at in range(1, 10) for band in range(1, 5)
    ),
    ("label", "label"),
)

#: A shift in a Polish coal mine, and whether the seismometers went off.
_SEISMIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("seismic", "hazard"), ("seismoacoustic", "hazard"), ("shift", "shift"),
    ("genergy", "d"), ("gpuls", "d"), ("gdenergy", "d"), ("gdpuls", "d"),
    ("ghazard", "hazard"),
    *_plain(("bumps", "bumps2", "bumps3", "bumps4", "bumps5", "bumps6", "bumps7", "bumps89")),
    ("energy", "d"), ("max_energy", "d"), ("label", "label"),
)

#: How a servomechanism was set up, and how long it took to settle.
_SERVO_FIELDS: tuple[tuple[str, str], ...] = (
    ("motor", "part"), ("screw", "part"), ("pgain", "i"), ("vgain", "i"),
    ("rise_time", "target"),
)

#: An active region on the sun, and how many flares of each size came out of
#: it in the next 24 hours.
_FLARE_FIELDS: tuple[tuple[str, str], ...] = (
    ("label", "label"), ("spot_size", "spots"), ("spot_distribution", "spread"),
    ("activity", "i"), ("evolution", "i"), ("previous_activity", "i"),
    ("historically_complex", "i"), ("became_complex", "i"), ("area", "i"),
    ("largest_spot_area", "i"), ("c_flares", "i"), ("m_flares", "i"), ("x_flares", "i"),
)

#: Sixty energies off a sonar chirp, and what it bounced off.
_SONAR_FIELDS: tuple[tuple[str, str], ...] = (
    *_plain(
        tuple(f"band_{at}" for at in range(1, 61)), "d"
    ),
    ("label", "label"),
)

#: Thirty-five things a farmer could see, already numbered by the file.
_SOYBEAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("label", "label"),
    *_plain(
        (
            "date", "plant_stand", "precipitation", "temperature", "hail", "crop_history",
            "area_damaged", "severity", "seed_treatment", "germination", "plant_growth",
            "leaves", "leafspots_halo", "leafspots_margin", "leafspot_size", "leaf_shread",
            "leaf_malformation", "leaf_mildew", "stem", "lodging", "stem_cankers",
            "canker_lesion", "fruiting_bodies", "external_decay", "mycelium",
            "internal_discolouration", "sclerotia", "fruit_pods", "fruit_spots", "seed",
            "mould_growth", "seed_discolouration", "seed_size", "shrivelling", "roots",
        )
    ),
)
_SOYBEAN_DISEASES: tuple[str, ...] = (
    "diaporthe-stem-canker", "charcoal-rot", "rhizoctonia-root-rot", "phytophthora-rot",
    "brown-stem-rot", "powdery-mildew", "downy-mildew", "brown-spot", "bacterial-blight",
    "bacterial-pustule", "purple-seed-stain", "anthracnose", "phyllosticta-leaf-spot",
    "alternarialeaf-spot", "frog-eye-leaf-spot", "diaporthe-pod-&-stem-blight",
    "cyst-nematode", "2-4-d-injury", "herbicide-injury",
)

#: The thirteen columns Statlog kept of the Cleveland heart study, all of
#: them written as decimals however whole the number is.
_STATLOG_HEART_FIELDS: tuple[tuple[str, str], ...] = (
    *_plain(
        (
            "age", "sex", "chest_pain", "resting_bp", "cholesterol", "fasting_sugar",
            "resting_ecg", "max_heart_rate", "exercise_angina", "st_depression", "st_slope",
            "vessels", "thal",
        ),
        "d",
    ),
    ("label", "label"),
)

#: A quarter of an hour in a South Korean steel plant. ``seconds_from_midnight``
#: is the file's own NSM column, and says the same thing as the time of day in
#: ``when``.
_STEEL_FIELDS: tuple[tuple[str, str], ...] = (
    ("when", "time"), ("usage", "d"), ("lagging_reactive_power", "d"),
    ("leading_reactive_power", "d"), ("carbon_dioxide", "d"),
    ("lagging_power_factor", "d"), ("leading_power_factor", "d"),
    ("seconds_from_midnight", "i"), ("week_status", "week"), ("day", "weekday"),
    ("label", "label"),
)

#: The nine squares of a finished game, x first. A square is 1 for x, -1 for
#: o and 0 for blank, so that a board adds up the way a player would read it.
_TICTACTOE_FIELDS: tuple[tuple[str, str], ...] = (
    *tuple(
        (f"{row}_{column}", "square")
        for row in ("top", "middle", "bottom")
        for column in ("left", "middle", "right")
    ),
    ("label", "label"),
)

#: Six angles and distances measured off a lower spine.
_VERTEBRAL_FIELDS: tuple[tuple[str, str], ...] = (
    *_plain(
        (
            "pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope",
            "pelvic_radius", "spondylolisthesis_grade",
        ),
        "d",
    ),
    ("label", "label"),
)

#: How strong each of seven wifi points was, and which room that was in.
_WIFI_FIELDS: tuple[tuple[str, str], ...] = (
    *_plain(
        tuple(f"signal_{at}" for at in range(1, 8))
    ),
    ("label", "label"),
)

#: The hull of a sailing yacht, and the resistance the tank measured.
_YACHT_FIELDS: tuple[tuple[str, str], ...] = (
    ("longitudinal_position", "d"), ("prismatic_coefficient", "d"),
    ("length_displacement_ratio", "d"), ("beam_draught_ratio", "d"),
    ("length_beam_ratio", "d"), ("froude_number", "d"), ("residuary_resistance", "target"),
)

#: Sixteen things an animal either does or has, then how many legs.
_ZOO_FIELDS: tuple[tuple[str, str], ...] = (
    ("animal", "text"),
    *_plain(
        (
            "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
            "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail",
            "domestic", "catsize",
        )
    ),
    ("label", "label"),
)


#: The hundred sets below are read from the archive's own ``data.csv``, which
#: is the file its Python package downloads: one header row naming the columns
#: the way the dataset's own documentation names them, then a row per example.
#: The fields keep those names, a categorical column keeps the categories the
#: file actually writes, and a column the archive marks as the thing to predict
#: becomes the label or the target depending on whether it names a class or
#: measures a number.
_ABSENTEEISM_AT_WORK_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"), ("reason_for_absence", "i"), ("month_of_absence", "i"), ("day_of_the_week", "i"),
    ("seasons", "i"), ("transportation_expense", "i"), ("distance_from_residence_to_work", "i"),
    ("service_time", "i"), ("age", "i"),
    ("work_load_average_day", "d"),
    ("hit_target", "i"), ("disciplinary_failure", "i"), ("education", "i"), ("son", "i"),
    ("social_drinker", "i"), ("social_smoker", "i"), ("pet", "i"), ("weight", "i"),
    ("height", "i"), ("body_mass_index", "i"),
    ("absenteeism_time_in_hours", "target"),
)

_ACUTE_INFLAMMATIONS_FIELDS: tuple[tuple[str, str], ...] = (
    ("temperature", "d"),
    ("nausea", "nausea"), ("lumbar_pain", "nausea"), ("urine_pushing", "nausea"),
    ("micturition_pains", "nausea"), ("burning_urethra", "nausea"),
    ("bladder_inflammation", "label"),
    ("nephritis", "nausea"),
)

_AIDS_CLINICAL_TRIALS_FIELDS: tuple[tuple[str, str], ...] = (
    ("pidnum", "i"),
    ("cid", "label"),
    ("time", "i"), ("trt", "i"), ("age", "i"),
    ("wtkg", "d"),
    ("hemo", "i"), ("homo", "i"), ("drugs", "i"), ("karnof", "i"), ("oprior", "i"), ("z30", "i"),
    ("zprior", "i"), ("preanti", "i"), ("race", "i"), ("gender", "i"), ("str2", "i"),
    ("strat", "i"), ("symptom", "i"), ("treat", "i"), ("offtrt", "i"), ("cd40", "i"),
    ("cd420", "i"), ("cd80", "i"), ("cd820", "i"),
)

_ANDROID_PERMISSIONS_FIELDS: tuple[tuple[str, str], ...] = (
    ("android_permission_get_accounts", "i"),
    ("com_sonyericsson_home_permission_broadcast_badge", "i"),
    ("android_permission_read_profile", "i"), ("android_permission_manage_accounts", "i"),
    ("android_permission_write_sync_settings", "i"),
    ("android_permission_read_external_storage", "i"), ("android_permission_receive_sms", "i"),
    ("com_android_launcher_permission_read_settings", "i"),
    ("android_permission_write_settings", "i"),
    ("com_google_android_providers_gsf_permission_read_gservices", "i"),
    ("android_permission_download_without_notification", "i"),
    ("android_permission_get_tasks", "i"), ("android_permission_write_external_storage", "i"),
    ("android_permission_record_audio", "i"),
    ("com_huawei_android_launcher_permission_change_badge", "i"),
    ("com_oppo_launcher_permission_read_settings", "i"),
    ("android_permission_change_network_state", "i"),
    ("com_android_launcher_permission_install_shortcut", "i"),
    ("android_permission_android_permission_read_phone_state", "i"),
    ("android_permission_call_phone", "i"), ("android_permission_write_contacts", "i"),
    ("android_permission_read_phone_state", "i"),
    ("com_samsung_android_providers_context_permission_write_use_app_feature_survey", "i"),
    ("android_permission_modify_audio_settings", "i"),
    ("android_permission_access_location_extra_commands", "i"),
    ("android_permission_internet", "i"), ("android_permission_mount_unmount_filesystems", "i"),
    ("com_majeur_launcher_permission_update_badge", "i"),
    ("android_permission_authenticate_accounts", "i"),
    ("com_htc_launcher_permission_read_settings", "i"),
    ("android_permission_access_wifi_state", "i"), ("android_permission_flashlight", "i"),
    ("android_permission_read_app_badge", "i"), ("android_permission_use_credentials", "i"),
    ("android_permission_change_configuration", "i"),
    ("android_permission_read_sync_settings", "i"), ("android_permission_broadcast_sticky", "i"),
    ("com_anddoes_launcher_permission_update_count", "i"),
    ("com_android_alarm_permission_set_alarm", "i"),
    ("com_google_android_c2dm_permission_receive", "i"),
    ("android_permission_kill_background_processes", "i"),
    ("com_sonymobile_home_permission_provider_insert_badge", "i"),
    ("com_sec_android_provider_badge_permission_read", "i"),
    ("android_permission_write_calendar", "i"), ("android_permission_send_sms", "i"),
    ("com_huawei_android_launcher_permission_write_settings", "i"),
    ("android_permission_request_install_packages", "i"),
    ("android_permission_set_wallpaper_hints", "i"), ("android_permission_set_wallpaper", "i"),
    ("com_oppo_launcher_permission_write_settings", "i"),
    ("android_permission_restart_packages", "i"),
    ("me_everything_badger_permission_badge_count_write", "i"),
    ("android_permission_access_mock_location", "i"),
    ("android_permission_access_coarse_location", "i"), ("android_permission_read_logs", "i"),
    ("com_google_android_gms_permission_activity_recognition", "i"),
    ("com_amazon_device_messaging_permission_receive", "i"),
    ("android_permission_system_alert_window", "i"), ("android_permission_disable_keyguard", "i"),
    ("android_permission_use_fingerprint", "i"),
    ("me_everything_badger_permission_badge_count_read", "i"),
    ("android_permission_change_wifi_state", "i"), ("android_permission_read_contacts", "i"),
    ("com_android_vending_billing", "i"), ("android_permission_read_calendar", "i"),
    ("android_permission_receive_boot_completed", "i"), ("android_permission_wake_lock", "i"),
    ("android_permission_access_fine_location", "i"), ("android_permission_bluetooth", "i"),
    ("android_permission_camera", "i"), ("com_android_vending_check_license", "i"),
    ("android_permission_foreground_service", "i"), ("android_permission_bluetooth_admin", "i"),
    ("android_permission_vibrate", "i"), ("android_permission_nfc", "i"),
    ("android_permission_receive_user_present", "i"), ("android_permission_clear_app_cache", "i"),
    ("com_android_launcher_permission_uninstall_shortcut", "i"),
    ("com_sec_android_iap_permission_billing", "i"),
    ("com_htc_launcher_permission_update_shortcut", "i"),
    ("com_sec_android_provider_badge_permission_write", "i"),
    ("android_permission_access_network_state", "i"),
    ("com_google_android_finsky_permission_bind_get_install_referrer_service", "i"),
    ("com_huawei_android_launcher_permission_read_settings", "i"),
    ("android_permission_read_sms", "i"), ("android_permission_process_incoming_calls", "i"),
    ("result", "label"),
)

_ANNEALING_FIELDS: tuple[tuple[str, str], ...] = (
    ("famiily", "famiily"),
    ("product_type", "product_type"),
    ("steel", "steel"),
    ("carbon", "i"), ("hardness", "i"),
    ("temper_rolling", "temper_rolling"),
    ("condition", "condition"),
    ("formability", "i"), ("strength", "i"),
    ("non_ageing", "non_ageing"),
    ("surface_finish", "surface_finish"),
    ("surface_quality", "surface_quality"),
    ("enamelability", "i"),
    ("bc", "bc"), ("bf", "bc"), ("bt", "bc"),
    ("bw_me", "bw_me"),
    ("bl", "bc"),
    ("m", "i"),
    ("chrom", "product_type"),
    ("phos", "surface_finish"),
    ("cbond", "bc"),
    ("marvi", "i"),
    ("exptl", "bc"), ("ferro", "bc"),
    ("corr", "i"),
    ("blue_bright_varn_clean", "blue_bright_varn_clean"),
    ("lustre", "bc"),
    ("jurofm", "i"), ("s", "i"), ("p", "i"),
    ("shape", "shape"),
    ("thick", "d"), ("width", "d"),
    ("len", "i"),
    ("oil", "oil"),
    ("bore", "i"), ("packing", "i"),
    ("class", "label"),
)

_APPLIANCES_ENERGY_FIELDS: tuple[tuple[str, str], ...] = (
    ("date", "text"),
    ("appliances", "target"),
    ("lights", "i"),
    ("t1", "d"), ("rh_1", "d"), ("t2", "d"), ("rh_2", "d"), ("t3", "d"), ("rh_3", "d"),
    ("t4", "d"), ("rh_4", "d"), ("t5", "d"), ("rh_5", "d"), ("t6", "d"), ("rh_6", "d"),
    ("t7", "d"), ("rh_7", "d"), ("t8", "d"), ("rh_8", "d"), ("t9", "d"), ("rh_9", "d"),
    ("t_out", "d"), ("press_mm_hg", "d"), ("rh_out", "d"), ("windspeed", "d"), ("visibility", "d"),
    ("tdewpoint", "d"), ("rv1", "d"), ("rv2", "d"),
)

_AUCTION_VERIFICATION_FIELDS: tuple[tuple[str, str], ...] = (
    ("process_b1_capacity", "i"), ("process_b2_capacity", "i"), ("process_b3_capacity", "i"),
    ("process_b4_capacity", "i"), ("property_price", "i"), ("property_product", "i"),
    ("property_winner", "i"),
    ("verification_result", "label"),
    ("verification_time", "d"),
)

_AUDIOLOGY_FIELDS: tuple[tuple[str, str], ...] = (
    ("age_gt_60", "age_gt_60"),
    ("air", "air"),
    ("airbonegap", "age_gt_60"),
    ("ar_c", "ar_c"), ("ar_u", "ar_c"),
    ("bone", "bone"),
    ("boneabnormal", "age_gt_60"),
    ("bser", "bser"),
    ("history_buzzing", "age_gt_60"), ("history_dizziness", "age_gt_60"),
    ("history_fluctuating", "age_gt_60"), ("history_fullness", "age_gt_60"),
    ("history_heredity", "age_gt_60"), ("history_nausea", "age_gt_60"),
    ("history_noise", "age_gt_60"), ("history_recruitment", "age_gt_60"),
    ("history_ringing", "age_gt_60"), ("history_roaring", "age_gt_60"),
    ("history_vomiting", "age_gt_60"), ("late_wave_poor", "age_gt_60"), ("m_at_2k", "age_gt_60"),
    ("m_cond_lt_1k", "m_cond_lt_1k"),
    ("m_gt_1k", "age_gt_60"), ("m_m_gt_2k", "age_gt_60"), ("m_m_sn", "age_gt_60"),
    ("m_m_sn_gt_1k", "age_gt_60"), ("m_m_sn_gt_2k", "age_gt_60"), ("m_m_sn_gt_500", "age_gt_60"),
    ("m_p_sn_gt_2k", "m_cond_lt_1k"),
    ("m_s_gt_500", "age_gt_60"), ("m_s_sn", "age_gt_60"),
    ("m_s_sn_gt_1k", "m_cond_lt_1k"),
    ("m_s_sn_gt_2k", "age_gt_60"), ("m_s_sn_3k", "age_gt_60"), ("m_s_sn_4k", "age_gt_60"),
    ("m_sn_2_3k", "age_gt_60"), ("m_sn_gt_1k", "age_gt_60"), ("m_sn_gt_2k", "age_gt_60"),
    ("m_sn_gt_3k", "age_gt_60"), ("m_sn_gt_4k", "age_gt_60"), ("m_sn_gt_500", "age_gt_60"),
    ("m_sn_gt_6k", "m_cond_lt_1k"),
    ("m_sn_lt_1k", "age_gt_60"), ("m_sn_lt_2k", "age_gt_60"), ("m_sn_lt_3k", "age_gt_60"),
    ("middle_wave_poor", "age_gt_60"), ("mod_gt_4k", "age_gt_60"),
    ("mod_mixed", "m_cond_lt_1k"), ("mod_s_mixed", "m_cond_lt_1k"),
    ("mod_s_sn_gt_500", "age_gt_60"),
    ("mod_sn", "m_cond_lt_1k"),
    ("mod_sn_gt_1k", "age_gt_60"), ("mod_sn_gt_2k", "age_gt_60"), ("mod_sn_gt_3k", "age_gt_60"),
    ("mod_sn_gt_4k", "age_gt_60"), ("mod_sn_gt_500", "age_gt_60"), ("notch_4k", "age_gt_60"),
    ("notch_at_4k", "age_gt_60"),
    ("o_ar_c", "ar_c"), ("o_ar_u", "ar_c"),
    ("s_sn_gt_1k", "age_gt_60"), ("s_sn_gt_2k", "age_gt_60"), ("s_sn_gt_4k", "age_gt_60"),
    ("speech", "speech"),
    ("static_normal", "age_gt_60"),
    ("tymp", "tymp"),
    ("viith_nerve_signs", "age_gt_60"), ("wave_v_delayed", "age_gt_60"),
    ("waveform_itov_prolonged", "age_gt_60"),
    ("identifier", "text"),
    ("class", "label"),
)

_AUTISM_SCREENING_ADULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("a1_score", "i"), ("a2_score", "i"), ("a3_score", "i"), ("a4_score", "i"), ("a5_score", "i"),
    ("a6_score", "i"), ("a7_score", "i"), ("a8_score", "i"), ("a9_score", "i"), ("a10_score", "i"),
    ("age", "i"),
    ("gender", "gender"),
    ("ethnicity", "ethnicity"),
    ("jaundice", "jaundice"), ("family_pdd", "jaundice"),
    ("country_of_res", "text"),
    ("used_app_before", "jaundice"),
    ("result", "i"),
    ("age_desc", "age_desc"),
    ("relation", "relation"),
    ("class", "label"),
)

_AUTISM_SCREENING_CHILD_FIELDS: tuple[tuple[str, str], ...] = (
    ("a1_score", "i"), ("a2_score", "i"), ("a3_score", "i"), ("a4_score", "i"), ("a5_score", "i"),
    ("a6_score", "i"), ("a7_score", "i"), ("a8_score", "i"), ("a9_score", "i"), ("a10_score", "i"),
    ("age", "i"),
    ("gender", "gender"),
    ("ethnicity", "ethnicity"),
    ("jaundice", "jaundice"), ("autism", "jaundice"),
    ("country_of_res", "text"),
    ("used_app_before", "jaundice"),
    ("result", "i"),
    ("age_desc", "age_desc"),
    ("relation", "relation"),
    ("class", "label"),
)

_BEIJING_PM25_FIELDS: tuple[tuple[str, str], ...] = (
    ("no", "i"), ("year", "i"), ("month", "i"), ("day", "i"), ("hour", "i"),
    ("pm2_5", "target"),
    ("dewp", "i"),
    ("temp", "d"), ("pres", "d"),
    ("cbwd", "cbwd"),
    ("iws", "d"),
    ("is", "i"), ("ir", "i"),
)

_BONE_MARROW_TRANSPLANT_FIELDS: tuple[tuple[str, str], ...] = (
    ("recipientgender", "i"), ("stemcellsource", "i"),
    ("donorage", "d"),
    ("donorage35", "i"), ("iiiv", "i"), ("gendermatch", "i"), ("donorabo", "i"),
    ("recipientabo", "i"), ("recipientrh", "i"), ("abomatch", "i"), ("cmvstatus", "i"),
    ("donorcmv", "i"), ("recipientcmv", "i"),
    ("disease", "disease"),
    ("riskgroup", "i"), ("txpostrelapse", "i"), ("diseasegroup", "i"), ("hlamatch", "i"),
    ("hlamismatch", "i"), ("antigen", "i"), ("allele", "i"), ("hlagri", "i"),
    ("recipientage", "d"),
    ("recipientage10", "i"), ("recipientageint", "i"), ("relapse", "i"), ("agvhdiiiiv", "i"),
    ("extcgvhd", "i"),
    ("cd34kgx10d6", "d"), ("cd3dcd34", "d"), ("cd3dkgx10d8", "d"), ("rbodymass", "d"),
    ("ancrecovery", "i"), ("pltrecovery", "i"), ("time_to_agvhd_iii_iv", "i"),
    ("survival_time", "i"),
    ("survival_status", "label"),
)

_BREAST_CANCER_COIMBRA_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"),
    ("bmi", "d"),
    ("glucose", "i"),
    ("insulin", "d"), ("homa", "d"), ("leptin", "d"), ("adiponectin", "d"), ("resistin", "d"),
    ("mcp_1", "d"),
    ("classification", "label"),
)

_BREAST_CANCER_ORIGINAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("sample_code_number", "i"), ("clump_thickness", "i"), ("uniformity_of_cell_size", "i"),
    ("uniformity_of_cell_shape", "i"), ("marginal_adhesion", "i"),
    ("single_epithelial_cell_size", "i"), ("bare_nuclei", "i"), ("bland_chromatin", "i"),
    ("normal_nucleoli", "i"), ("mitoses", "i"),
    ("class", "label"),
)

_BREAST_CANCER_PROGNOSTIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"), ("time", "i"),
    ("radius1", "d"), ("texture1", "d"), ("perimeter1", "d"), ("area1", "d"), ("smoothness1", "d"),
    ("compactness1", "d"), ("concavity1", "d"), ("concave_points1", "d"), ("symmetry1", "d"),
    ("fractal_dimension1", "d"), ("radius2", "d"), ("texture2", "d"), ("perimeter2", "d"),
    ("area2", "d"), ("smoothness2", "d"), ("compactness2", "d"), ("concavity2", "d"),
    ("concave_points2", "d"), ("symmetry2", "d"), ("fractal_dimension2", "d"), ("radius3", "d"),
    ("texture3", "d"), ("perimeter3", "d"), ("area3", "d"), ("smoothness3", "d"),
    ("compactness3", "d"), ("concavity3", "d"), ("concave_points3", "d"), ("symmetry3", "d"),
    ("fractal_dimension3", "d"), ("tumor_size", "d"),
    ("lymph_node_status", "i"),
    ("outcome", "label"),
)

_BREAST_CANCER_RECURRENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "age"),
    ("menopause", "menopause"),
    ("tumor_size", "tumor_size"),
    ("inv_nodes", "inv_nodes"),
    ("node_caps", "node_caps"),
    ("deg_malig", "i"),
    ("breast", "breast"),
    ("breast_quad", "breast_quad"),
    ("irradiat", "node_caps"),
    ("class", "label"),
)

_CARDIOTOCOGRAPHY_FIELDS: tuple[tuple[str, str], ...] = (
    ("lb", "i"),
    ("ac", "d"), ("fm", "d"), ("uc", "d"), ("dl", "d"), ("ds", "d"), ("dp", "d"),
    ("astv", "i"),
    ("mstv", "d"),
    ("altv", "i"),
    ("mltv", "d"),
    ("width", "i"), ("min", "i"), ("max", "i"), ("nmax", "i"), ("nzeros", "i"), ("mode", "i"),
    ("mean", "i"), ("median", "i"), ("variance", "i"), ("tendency", "i"), ("class", "i"),
    ("nsp", "label"),
)

_CDC_DIABETES_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"),
    ("diabetes_binary", "label"),
    ("highbp", "i"), ("highchol", "i"), ("cholcheck", "i"), ("bmi", "i"), ("smoker", "i"),
    ("stroke", "i"), ("heartdiseaseorattack", "i"), ("physactivity", "i"), ("fruits", "i"),
    ("veggies", "i"), ("hvyalcoholconsump", "i"), ("anyhealthcare", "i"), ("nodocbccost", "i"),
    ("genhlth", "i"), ("menthlth", "i"), ("physhlth", "i"), ("diffwalk", "i"), ("sex", "i"),
    ("age", "i"), ("education", "i"), ("income", "i"),
)

_CENSUS_INCOME_KDD_FIELDS: tuple[tuple[str, str], ...] = (
    ("aage", "i"),
    ("aclswkr", "aclswkr"),
    ("adtink", "i"), ("adtocc", "i"),
    ("ahga", "text"),
    ("ahrspay", "i"),
    ("ahscol", "ahscol"),
    ("amaritl", "amaritl"),
    ("amjind", "text"), ("amjocc", "text"),
    ("arace", "arace"),
    ("areorgn", "areorgn"),
    ("asex", "asex"),
    ("aunmem", "aunmem"),
    ("auntype", "auntype"),
    ("awkstat", "text"),
    ("capgain", "i"), ("gaploss", "i"), ("divval", "i"),
    ("filestat", "filestat"),
    ("grinreg", "grinreg"),
    ("grinst", "text"), ("hhdfmx", "text"), ("hhdrel", "text"),
    ("marsupwrt", "d"),
    ("migmtr1", "migmtr1"),
    ("migmtr3", "migmtr3"),
    ("migmtr4", "migmtr4"),
    ("migsame", "migsame"),
    ("migsun", "aunmem"),
    ("noemp", "i"),
    ("parent", "parent"),
    ("pefntvty", "text"), ("pemntvty", "text"), ("penatvty", "text"), ("prcitshp", "text"),
    ("seotr", "i"),
    ("vetqva", "aunmem"),
    ("vetyn", "i"), ("wkswork", "i"), ("year", "i"),
    ("income", "label"),
)

_CERVICAL_CANCER_BEHAVIOUR_FIELDS: tuple[tuple[str, str], ...] = (
    ("behavior_sexualrisk", "i"), ("behavior_eating", "i"), ("behavior_personalhygiene", "i"),
    ("intention_aggregation", "i"), ("intention_commitment", "i"), ("attitude_consistency", "i"),
    ("attitude_spontaneity", "i"), ("norm_significantperson", "i"), ("norm_fulfillment", "i"),
    ("perception_vulnerability", "i"), ("perception_severity", "i"), ("motivation_strength", "i"),
    ("motivation_willingness", "i"), ("socialsupport_emotionality", "i"),
    ("socialsupport_appreciation", "i"), ("socialsupport_instrumental", "i"),
    ("empowerment_knowledge", "i"), ("empowerment_abilities", "i"), ("empowerment_desires", "i"),
    ("ca_cervix", "label"),
)

_CERVICAL_CANCER_RISK_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"),
    ("number_of_sexual_partners", "d"), ("first_sexual_intercourse", "d"),
    ("num_of_pregnancies", "d"), ("smokes", "d"), ("smokes_years", "d"),
    ("smokes_packs_year", "d"), ("hormonal_contraceptives", "d"),
    ("hormonal_contraceptives_years", "d"), ("iud", "d"), ("iud_years", "d"), ("stds", "d"),
    ("stds_number", "d"), ("stds_condylomatosis", "d"), ("stds_cervical_condylomatosis", "d"),
    ("stds_vaginal_condylomatosis", "d"), ("stds_vulvo_perineal_condylomatosis", "d"),
    ("stds_syphilis", "d"), ("stds_pelvic_inflammatory_disease", "d"),
    ("stds_genital_herpes", "d"), ("stds_molluscum_contagiosum", "d"), ("stds_aids", "d"),
    ("stds_hiv", "d"), ("stds_hepatitis_b", "d"), ("stds_hpv", "d"),
    ("stds_number_of_diagnosis", "i"),
    ("stds_time_since_first_diagnosis", "d"), ("stds_time_since_last_diagnosis", "d"),
    ("dx_cancer", "i"), ("dx_cin", "i"), ("dx_hpv", "i"), ("dx", "i"), ("hinselmann", "i"),
    ("schiller", "i"), ("citology", "i"),
    ("biopsy", "label"),
)

_CHALLENGER_O_RINGS_FIELDS: tuple[tuple[str, str], ...] = (
    ("num_o_rings", "target"), ("num_thermal_distress", "target"),
    ("launch_temp", "i"), ("leak_check_pressure", "i"), ("temporal_order", "i"),
)

_CHESS_ENDGAME_FIELDS: tuple[tuple[str, str], ...] = (
    ("white_king_file", "white_king_file"),
    ("white_king_rank", "i"),
    ("white_rook_file", "white_rook_file"),
    ("white_rook_rank", "i"),
    ("black_king_file", "white_rook_file"),
    ("black_king_rank", "i"),
    ("white_depth_of_win", "label"),
)

_CHRONIC_KIDNEY_DISEASE_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"), ("bp", "i"),
    ("sg", "d"),
    ("al", "i"), ("su", "i"),
    ("rbc", "rbc"), ("pc", "rbc"),
    ("pcc", "pcc"), ("ba", "pcc"),
    ("bgr", "i"),
    ("bu", "d"), ("sc", "d"), ("sod", "d"), ("pot", "d"), ("hemo", "d"),
    ("pcv", "i"), ("wbcc", "i"),
    ("rbcc", "d"),
    ("htn", "htn"), ("dm", "htn"), ("cad", "htn"),
    ("appet", "appet"),
    ("pe", "htn"), ("ane", "htn"),
    ("class", "label"),
)

_COMMUNITIES_CRIME_FIELDS: tuple[tuple[str, str], ...] = (
    ("state", "i"), ("county", "i"), ("community", "i"),
    ("communityname", "text"),
    ("fold", "i"),
    ("population", "d"), ("householdsize", "d"), ("racepctblack", "d"), ("racepctwhite", "d"),
    ("racepctasian", "d"), ("racepcthisp", "d"), ("agepct12t21", "d"), ("agepct12t29", "d"),
    ("agepct16t24", "d"), ("agepct65up", "d"), ("numburban", "d"), ("pcturban", "d"),
    ("medincome", "d"), ("pctwwage", "d"), ("pctwfarmself", "d"), ("pctwinvinc", "d"),
    ("pctwsocsec", "d"), ("pctwpubasst", "d"), ("pctwretire", "d"), ("medfaminc", "d"),
    ("percapinc", "d"), ("whitepercap", "d"), ("blackpercap", "d"), ("indianpercap", "d"),
    ("asianpercap", "d"), ("otherpercap", "d"), ("hisppercap", "d"), ("numunderpov", "d"),
    ("pctpopunderpov", "d"), ("pctless9thgrade", "d"), ("pctnothsgrad", "d"), ("pctbsormore", "d"),
    ("pctunemployed", "d"), ("pctemploy", "d"), ("pctemplmanu", "d"), ("pctemplprofserv", "d"),
    ("pctoccupmanu", "d"), ("pctoccupmgmtprof", "d"), ("malepctdivorce", "d"),
    ("malepctnevmarr", "d"), ("femalepctdiv", "d"), ("totalpctdiv", "d"), ("persperfam", "d"),
    ("pctfam2par", "d"), ("pctkids2par", "d"), ("pctyoungkids2par", "d"), ("pctteen2par", "d"),
    ("pctworkmomyoungkids", "d"), ("pctworkmom", "d"), ("numilleg", "d"), ("pctilleg", "d"),
    ("numimmig", "d"), ("pctimmigrecent", "d"), ("pctimmigrec5", "d"), ("pctimmigrec8", "d"),
    ("pctimmigrec10", "d"), ("pctrecentimmig", "d"), ("pctrecimmig5", "d"), ("pctrecimmig8", "d"),
    ("pctrecimmig10", "d"), ("pctspeakenglonly", "d"), ("pctnotspeakenglwell", "d"),
    ("pctlarghousefam", "d"), ("pctlarghouseoccup", "d"), ("persperoccuphous", "d"),
    ("persperownocchous", "d"), ("persperrentocchous", "d"), ("pctpersownoccup", "d"),
    ("pctpersdensehous", "d"), ("pcthousless3br", "d"), ("mednumbr", "d"), ("housvacant", "d"),
    ("pcthousoccup", "d"), ("pcthousownocc", "d"), ("pctvacantboarded", "d"),
    ("pctvacmore6mos", "d"), ("medyrhousbuilt", "d"), ("pcthousnophone", "d"),
    ("pctwofullplumb", "d"), ("ownocclowquart", "d"), ("ownoccmedval", "d"),
    ("ownocchiquart", "d"), ("rentlowq", "d"), ("rentmedian", "d"), ("renthighq", "d"),
    ("medrent", "d"), ("medrentpcthousinc", "d"), ("medowncostpctinc", "d"),
    ("medowncostpctincnomtg", "d"), ("numinshelters", "d"), ("numstreet", "d"),
    ("pctforeignborn", "d"), ("pctbornsamestate", "d"), ("pctsamehouse85", "d"),
    ("pctsamecity85", "d"), ("pctsamestate85", "d"), ("lemasswornft", "d"),
    ("lemasswftperpop", "d"), ("lemasswftfieldops", "d"), ("lemasswftfieldperpop", "d"),
    ("lemastotalreq", "d"), ("lemastotreqperpop", "d"), ("policreqperoffic", "d"),
    ("policperpop", "d"), ("racialmatchcommpol", "d"), ("pctpolicwhite", "d"),
    ("pctpolicblack", "d"), ("pctpolichisp", "d"), ("pctpolicasian", "d"), ("pctpolicminor", "d"),
    ("officassgndrugunits", "d"), ("numkindsdrugsseiz", "d"), ("policaveotworked", "d"),
    ("landarea", "d"), ("popdens", "d"), ("pctusepubtrans", "d"), ("policcars", "d"),
    ("policoperbudg", "d"), ("lemaspctpoliconpatr", "d"), ("lemasgangunitdeploy", "d"),
    ("lemaspctofficdrugun", "d"), ("policbudgperpop", "d"),
    ("violentcrimesperpop", "target"),
)

_CONCRETE_STRENGTH_FIELDS: tuple[tuple[str, str], ...] = (
    ("cement", "d"), ("blast_furnace_slag", "d"), ("fly_ash", "d"), ("water", "d"),
    ("superplasticizer", "d"), ("coarse_aggregate", "d"), ("fine_aggregate", "d"),
    ("age", "i"),
    ("concrete_compressive_strength", "target"),
)

_CONGRESSIONAL_VOTING_FIELDS: tuple[tuple[str, str], ...] = (
    ("class", "label"),
    ("handicapped_infants", "handicapped_infants"),
    ("water_project_cost_sharing", "handicapped_infants"),
    ("adoption_of_the_budget_resolution", "handicapped_infants"),
    ("physician_fee_freeze", "handicapped_infants"), ("el_salvador_aid", "handicapped_infants"),
    ("religious_groups_in_schools", "handicapped_infants"),
    ("anti_satellite_test_ban", "handicapped_infants"),
    ("aid_to_nicaraguan_contras", "handicapped_infants"), ("mx_missile", "handicapped_infants"),
    ("immigration", "handicapped_infants"),
    ("synfuels_corporation_cutback", "handicapped_infants"),
    ("education_spending", "handicapped_infants"),
    ("superfund_right_to_sue", "handicapped_infants"), ("crime", "handicapped_infants"),
    ("duty_free_exports", "handicapped_infants"),
    ("export_administration_act_south_africa", "handicapped_infants"),
)

_CONNECT_FOUR_FIELDS: tuple[tuple[str, str], ...] = (
    ("a1", "a1"), ("a2", "a1"), ("a3", "a1"), ("a4", "a1"), ("a5", "a1"), ("a6", "a1"),
    ("b1", "a1"), ("b2", "a1"), ("b3", "a1"), ("b4", "a1"), ("b5", "a1"), ("b6", "a1"),
    ("c1", "a1"), ("c2", "a1"), ("c3", "a1"), ("c4", "a1"), ("c5", "a1"), ("c6", "a1"),
    ("d1", "a1"), ("d2", "a1"), ("d3", "a1"), ("d4", "a1"), ("d5", "a1"), ("d6", "a1"),
    ("e1", "a1"), ("e2", "a1"), ("e3", "a1"), ("e4", "a1"), ("e5", "a1"), ("e6", "a1"),
    ("f1", "a1"), ("f2", "a1"), ("f3", "a1"), ("f4", "a1"), ("f5", "a1"), ("f6", "a1"),
    ("g1", "a1"), ("g2", "a1"), ("g3", "a1"), ("g4", "a1"), ("g5", "a1"), ("g6", "a1"),
    ("class", "label"),
)

_CREDIT_CARD_DEFAULT_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"), ("x1", "i"), ("x2", "i"), ("x3", "i"), ("x4", "i"), ("x5", "i"), ("x6", "i"),
    ("x7", "i"), ("x8", "i"), ("x9", "i"), ("x10", "i"), ("x11", "i"), ("x12", "i"), ("x13", "i"),
    ("x14", "i"), ("x15", "i"), ("x16", "i"), ("x17", "i"), ("x18", "i"), ("x19", "i"),
    ("x20", "i"), ("x21", "i"), ("x22", "i"), ("x23", "i"),
    ("y", "label"),
)

_CREDIT_SCREENING_FIELDS: tuple[tuple[str, str], ...] = (
    ("a1", "a1"),
    ("a2", "d"), ("a3", "d"),
    ("a4", "a4"),
    ("a5", "a5"),
    ("a6", "a6"),
    ("a7", "a7"),
    ("a8", "d"),
    ("a9", "a9"), ("a10", "a9"),
    ("a11", "i"),
    ("a12", "a9"),
    ("a13", "a13"),
    ("a14", "i"), ("a15", "i"),
    ("a16", "label"),
)

_DAILY_DEMAND_FIELDS: tuple[tuple[str, str], ...] = (
    ("week_of_the_month", "i"), ("day_of_the_week", "i"),
    ("non_urgent_order", "d"), ("urgent_order", "d"), ("order_type_a", "d"), ("order_type_b", "d"),
    ("order_type_c", "d"), ("fiscal_sector_orders", "d"),
    ("orders_from_the_traffic_controller_sector", "i"), ("banking_orders_1", "i"),
    ("banking_orders_2", "i"), ("banking_orders_3", "i"),
    ("total_orders", "target"),
)

_DARWIN_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "text"),
    ("air_time1", "i"),
    ("disp_index1", "d"), ("gmrt_in_air1", "d"), ("gmrt_on_paper1", "d"),
    ("max_x_extension1", "i"), ("max_y_extension1", "i"),
    ("mean_acc_in_air1", "d"), ("mean_acc_on_paper1", "d"), ("mean_gmrt1", "d"),
    ("mean_jerk_in_air1", "d"), ("mean_jerk_on_paper1", "d"), ("mean_speed_in_air1", "d"),
    ("mean_speed_on_paper1", "d"),
    ("num_of_pendown1", "i"), ("paper_time1", "i"),
    ("pressure_mean1", "d"), ("pressure_var1", "d"),
    ("total_time1", "i"), ("air_time2", "i"),
    ("disp_index2", "d"), ("gmrt_in_air2", "d"), ("gmrt_on_paper2", "d"),
    ("max_x_extension2", "i"), ("max_y_extension2", "i"),
    ("mean_acc_in_air2", "d"), ("mean_acc_on_paper2", "d"), ("mean_gmrt2", "d"),
    ("mean_jerk_in_air2", "d"), ("mean_jerk_on_paper2", "d"), ("mean_speed_in_air2", "d"),
    ("mean_speed_on_paper2", "d"),
    ("num_of_pendown2", "i"), ("paper_time2", "i"),
    ("pressure_mean2", "d"), ("pressure_var2", "d"),
    ("total_time2", "i"), ("air_time3", "i"),
    ("disp_index3", "d"), ("gmrt_in_air3", "d"), ("gmrt_on_paper3", "d"),
    ("max_x_extension3", "i"), ("max_y_extension3", "i"),
    ("mean_acc_in_air3", "d"), ("mean_acc_on_paper3", "d"), ("mean_gmrt3", "d"),
    ("mean_jerk_in_air3", "d"), ("mean_jerk_on_paper3", "d"), ("mean_speed_in_air3", "d"),
    ("mean_speed_on_paper3", "d"),
    ("num_of_pendown3", "i"), ("paper_time3", "i"),
    ("pressure_mean3", "d"), ("pressure_var3", "d"),
    ("total_time3", "i"), ("air_time4", "i"),
    ("disp_index4", "d"), ("gmrt_in_air4", "d"), ("gmrt_on_paper4", "d"),
    ("max_x_extension4", "i"), ("max_y_extension4", "i"),
    ("mean_acc_in_air4", "d"), ("mean_acc_on_paper4", "d"), ("mean_gmrt4", "d"),
    ("mean_jerk_in_air4", "d"), ("mean_jerk_on_paper4", "d"), ("mean_speed_in_air4", "d"),
    ("mean_speed_on_paper4", "d"),
    ("num_of_pendown4", "i"), ("paper_time4", "i"),
    ("pressure_mean4", "d"), ("pressure_var4", "d"),
    ("total_time4", "i"), ("air_time5", "i"),
    ("disp_index5", "d"), ("gmrt_in_air5", "d"), ("gmrt_on_paper5", "d"),
    ("max_x_extension5", "i"), ("max_y_extension5", "i"),
    ("mean_acc_in_air5", "d"), ("mean_acc_on_paper5", "d"), ("mean_gmrt5", "d"),
    ("mean_jerk_in_air5", "d"), ("mean_jerk_on_paper5", "d"), ("mean_speed_in_air5", "d"),
    ("mean_speed_on_paper5", "d"),
    ("num_of_pendown5", "i"), ("paper_time5", "i"),
    ("pressure_mean5", "d"), ("pressure_var5", "d"),
    ("total_time5", "i"), ("air_time6", "i"),
    ("disp_index6", "d"), ("gmrt_in_air6", "d"), ("gmrt_on_paper6", "d"),
    ("max_x_extension6", "i"), ("max_y_extension6", "i"),
    ("mean_acc_in_air6", "d"), ("mean_acc_on_paper6", "d"), ("mean_gmrt6", "d"),
    ("mean_jerk_in_air6", "d"), ("mean_jerk_on_paper6", "d"), ("mean_speed_in_air6", "d"),
    ("mean_speed_on_paper6", "d"),
    ("num_of_pendown6", "i"), ("paper_time6", "i"),
    ("pressure_mean6", "d"), ("pressure_var6", "d"),
    ("total_time6", "i"), ("air_time7", "i"),
    ("disp_index7", "d"), ("gmrt_in_air7", "d"), ("gmrt_on_paper7", "d"),
    ("max_x_extension7", "i"), ("max_y_extension7", "i"),
    ("mean_acc_in_air7", "d"), ("mean_acc_on_paper7", "d"), ("mean_gmrt7", "d"),
    ("mean_jerk_in_air7", "d"), ("mean_jerk_on_paper7", "d"), ("mean_speed_in_air7", "d"),
    ("mean_speed_on_paper7", "d"),
    ("num_of_pendown7", "i"), ("paper_time7", "i"),
    ("pressure_mean7", "d"), ("pressure_var7", "d"),
    ("total_time7", "i"), ("air_time8", "i"),
    ("disp_index8", "d"), ("gmrt_in_air8", "d"), ("gmrt_on_paper8", "d"),
    ("max_x_extension8", "i"), ("max_y_extension8", "i"),
    ("mean_acc_in_air8", "d"), ("mean_acc_on_paper8", "d"), ("mean_gmrt8", "d"),
    ("mean_jerk_in_air8", "d"), ("mean_jerk_on_paper8", "d"), ("mean_speed_in_air8", "d"),
    ("mean_speed_on_paper8", "d"),
    ("num_of_pendown8", "i"), ("paper_time8", "i"),
    ("pressure_mean8", "d"), ("pressure_var8", "d"),
    ("total_time8", "i"), ("air_time9", "i"),
    ("disp_index9", "d"), ("gmrt_in_air9", "d"), ("gmrt_on_paper9", "d"),
    ("max_x_extension9", "i"), ("max_y_extension9", "i"),
    ("mean_acc_in_air9", "d"), ("mean_acc_on_paper9", "d"), ("mean_gmrt9", "d"),
    ("mean_jerk_in_air9", "d"), ("mean_jerk_on_paper9", "d"), ("mean_speed_in_air9", "d"),
    ("mean_speed_on_paper9", "d"),
    ("num_of_pendown9", "i"), ("paper_time9", "i"),
    ("pressure_mean9", "d"), ("pressure_var9", "d"),
    ("total_time9", "i"), ("air_time10", "i"),
    ("disp_index10", "d"), ("gmrt_in_air10", "d"), ("gmrt_on_paper10", "d"),
    ("max_x_extension10", "i"), ("max_y_extension10", "i"),
    ("mean_acc_in_air10", "d"), ("mean_acc_on_paper10", "d"), ("mean_gmrt10", "d"),
    ("mean_jerk_in_air10", "d"), ("mean_jerk_on_paper10", "d"), ("mean_speed_in_air10", "d"),
    ("mean_speed_on_paper10", "d"),
    ("num_of_pendown10", "i"), ("paper_time10", "i"),
    ("pressure_mean10", "d"), ("pressure_var10", "d"),
    ("total_time10", "i"), ("air_time11", "i"),
    ("disp_index11", "d"), ("gmrt_in_air11", "d"), ("gmrt_on_paper11", "d"),
    ("max_x_extension11", "i"), ("max_y_extension11", "i"),
    ("mean_acc_in_air11", "d"), ("mean_acc_on_paper11", "d"), ("mean_gmrt11", "d"),
    ("mean_jerk_in_air11", "d"), ("mean_jerk_on_paper11", "d"), ("mean_speed_in_air11", "d"),
    ("mean_speed_on_paper11", "d"),
    ("num_of_pendown11", "i"), ("paper_time11", "i"),
    ("pressure_mean11", "d"), ("pressure_var11", "d"),
    ("total_time11", "i"), ("air_time12", "i"),
    ("disp_index12", "d"), ("gmrt_in_air12", "d"), ("gmrt_on_paper12", "d"),
    ("max_x_extension12", "i"), ("max_y_extension12", "i"),
    ("mean_acc_in_air12", "d"), ("mean_acc_on_paper12", "d"), ("mean_gmrt12", "d"),
    ("mean_jerk_in_air12", "d"), ("mean_jerk_on_paper12", "d"), ("mean_speed_in_air12", "d"),
    ("mean_speed_on_paper12", "d"),
    ("num_of_pendown12", "i"), ("paper_time12", "i"),
    ("pressure_mean12", "d"), ("pressure_var12", "d"),
    ("total_time12", "i"), ("air_time13", "i"),
    ("disp_index13", "d"), ("gmrt_in_air13", "d"), ("gmrt_on_paper13", "d"),
    ("max_x_extension13", "i"), ("max_y_extension13", "i"),
    ("mean_acc_in_air13", "d"), ("mean_acc_on_paper13", "d"), ("mean_gmrt13", "d"),
    ("mean_jerk_in_air13", "d"), ("mean_jerk_on_paper13", "d"), ("mean_speed_in_air13", "d"),
    ("mean_speed_on_paper13", "d"),
    ("num_of_pendown13", "i"), ("paper_time13", "i"),
    ("pressure_mean13", "d"), ("pressure_var13", "d"),
    ("total_time13", "i"), ("air_time14", "i"),
    ("disp_index14", "d"), ("gmrt_in_air14", "d"), ("gmrt_on_paper14", "d"),
    ("max_x_extension14", "i"), ("max_y_extension14", "i"),
    ("mean_acc_in_air14", "d"), ("mean_acc_on_paper14", "d"), ("mean_gmrt14", "d"),
    ("mean_jerk_in_air14", "d"), ("mean_jerk_on_paper14", "d"), ("mean_speed_in_air14", "d"),
    ("mean_speed_on_paper14", "d"),
    ("num_of_pendown14", "i"), ("paper_time14", "i"),
    ("pressure_mean14", "d"), ("pressure_var14", "d"),
    ("total_time14", "i"), ("air_time15", "i"),
    ("disp_index15", "d"), ("gmrt_in_air15", "d"), ("gmrt_on_paper15", "d"),
    ("max_x_extension15", "i"), ("max_y_extension15", "i"),
    ("mean_acc_in_air15", "d"), ("mean_acc_on_paper15", "d"), ("mean_gmrt15", "d"),
    ("mean_jerk_in_air15", "d"), ("mean_jerk_on_paper15", "d"), ("mean_speed_in_air15", "d"),
    ("mean_speed_on_paper15", "d"),
    ("num_of_pendown15", "i"), ("paper_time15", "i"),
    ("pressure_mean15", "d"), ("pressure_var15", "d"),
    ("total_time15", "i"), ("air_time16", "i"),
    ("disp_index16", "d"), ("gmrt_in_air16", "d"), ("gmrt_on_paper16", "d"),
    ("max_x_extension16", "i"), ("max_y_extension16", "i"),
    ("mean_acc_in_air16", "d"), ("mean_acc_on_paper16", "d"), ("mean_gmrt16", "d"),
    ("mean_jerk_in_air16", "d"), ("mean_jerk_on_paper16", "d"), ("mean_speed_in_air16", "d"),
    ("mean_speed_on_paper16", "d"),
    ("num_of_pendown16", "i"), ("paper_time16", "i"),
    ("pressure_mean16", "d"), ("pressure_var16", "d"),
    ("total_time16", "i"), ("air_time17", "i"),
    ("disp_index17", "d"), ("gmrt_in_air17", "d"), ("gmrt_on_paper17", "d"),
    ("max_x_extension17", "i"), ("max_y_extension17", "i"),
    ("mean_acc_in_air17", "d"), ("mean_acc_on_paper17", "d"), ("mean_gmrt17", "d"),
    ("mean_jerk_in_air17", "d"), ("mean_jerk_on_paper17", "d"), ("mean_speed_in_air17", "d"),
    ("mean_speed_on_paper17", "d"),
    ("num_of_pendown17", "i"), ("paper_time17", "i"),
    ("pressure_mean17", "d"), ("pressure_var17", "d"),
    ("total_time17", "i"), ("air_time18", "i"),
    ("disp_index18", "d"), ("gmrt_in_air18", "d"), ("gmrt_on_paper18", "d"),
    ("max_x_extension18", "i"), ("max_y_extension18", "i"),
    ("mean_acc_in_air18", "d"), ("mean_acc_on_paper18", "d"), ("mean_gmrt18", "d"),
    ("mean_jerk_in_air18", "d"), ("mean_jerk_on_paper18", "d"), ("mean_speed_in_air18", "d"),
    ("mean_speed_on_paper18", "d"),
    ("num_of_pendown18", "i"), ("paper_time18", "i"),
    ("pressure_mean18", "d"), ("pressure_var18", "d"),
    ("total_time18", "i"), ("air_time19", "i"),
    ("disp_index19", "d"), ("gmrt_in_air19", "d"), ("gmrt_on_paper19", "d"),
    ("max_x_extension19", "i"), ("max_y_extension19", "i"),
    ("mean_acc_in_air19", "d"), ("mean_acc_on_paper19", "d"), ("mean_gmrt19", "d"),
    ("mean_jerk_in_air19", "d"), ("mean_jerk_on_paper19", "d"), ("mean_speed_in_air19", "d"),
    ("mean_speed_on_paper19", "d"),
    ("num_of_pendown19", "i"), ("paper_time19", "i"),
    ("pressure_mean19", "d"), ("pressure_var19", "d"),
    ("total_time19", "i"), ("air_time20", "i"),
    ("disp_index20", "d"), ("gmrt_in_air20", "d"), ("gmrt_on_paper20", "d"),
    ("max_x_extension20", "i"), ("max_y_extension20", "i"),
    ("mean_acc_in_air20", "d"), ("mean_acc_on_paper20", "d"), ("mean_gmrt20", "d"),
    ("mean_jerk_in_air20", "d"), ("mean_jerk_on_paper20", "d"), ("mean_speed_in_air20", "d"),
    ("mean_speed_on_paper20", "d"),
    ("num_of_pendown20", "i"), ("paper_time20", "i"),
    ("pressure_mean20", "d"), ("pressure_var20", "d"),
    ("total_time20", "i"), ("air_time21", "i"),
    ("disp_index21", "d"), ("gmrt_in_air21", "d"), ("gmrt_on_paper21", "d"),
    ("max_x_extension21", "i"), ("max_y_extension21", "i"),
    ("mean_acc_in_air21", "d"), ("mean_acc_on_paper21", "d"), ("mean_gmrt21", "d"),
    ("mean_jerk_in_air21", "d"), ("mean_jerk_on_paper21", "d"), ("mean_speed_in_air21", "d"),
    ("mean_speed_on_paper21", "d"),
    ("num_of_pendown21", "i"), ("paper_time21", "i"),
    ("pressure_mean21", "d"), ("pressure_var21", "d"),
    ("total_time21", "i"), ("air_time22", "i"),
    ("disp_index22", "d"), ("gmrt_in_air22", "d"), ("gmrt_on_paper22", "d"),
    ("max_x_extension22", "i"), ("max_y_extension22", "i"),
    ("mean_acc_in_air22", "d"), ("mean_acc_on_paper22", "d"), ("mean_gmrt22", "d"),
    ("mean_jerk_in_air22", "d"), ("mean_jerk_on_paper22", "d"), ("mean_speed_in_air22", "d"),
    ("mean_speed_on_paper22", "d"),
    ("num_of_pendown22", "i"), ("paper_time22", "i"),
    ("pressure_mean22", "d"), ("pressure_var22", "d"),
    ("total_time22", "i"), ("air_time23", "i"),
    ("disp_index23", "d"), ("gmrt_in_air23", "d"), ("gmrt_on_paper23", "d"),
    ("max_x_extension23", "i"), ("max_y_extension23", "i"),
    ("mean_acc_in_air23", "d"), ("mean_acc_on_paper23", "d"), ("mean_gmrt23", "d"),
    ("mean_jerk_in_air23", "d"), ("mean_jerk_on_paper23", "d"), ("mean_speed_in_air23", "d"),
    ("mean_speed_on_paper23", "d"),
    ("num_of_pendown23", "i"), ("paper_time23", "i"),
    ("pressure_mean23", "d"), ("pressure_var23", "d"),
    ("total_time23", "i"), ("air_time24", "i"),
    ("disp_index24", "d"), ("gmrt_in_air24", "d"), ("gmrt_on_paper24", "d"),
    ("max_x_extension24", "i"), ("max_y_extension24", "i"),
    ("mean_acc_in_air24", "d"), ("mean_acc_on_paper24", "d"), ("mean_gmrt24", "d"),
    ("mean_jerk_in_air24", "d"), ("mean_jerk_on_paper24", "d"), ("mean_speed_in_air24", "d"),
    ("mean_speed_on_paper24", "d"),
    ("num_of_pendown24", "i"), ("paper_time24", "i"),
    ("pressure_mean24", "d"), ("pressure_var24", "d"),
    ("total_time24", "i"), ("air_time25", "i"),
    ("disp_index25", "d"), ("gmrt_in_air25", "d"), ("gmrt_on_paper25", "d"),
    ("max_x_extension25", "i"), ("max_y_extension25", "i"),
    ("mean_acc_in_air25", "d"), ("mean_acc_on_paper25", "d"), ("mean_gmrt25", "d"),
    ("mean_jerk_in_air25", "d"), ("mean_jerk_on_paper25", "d"), ("mean_speed_in_air25", "d"),
    ("mean_speed_on_paper25", "d"),
    ("num_of_pendown25", "i"), ("paper_time25", "i"),
    ("pressure_mean25", "d"), ("pressure_var25", "d"),
    ("total_time25", "i"),
    ("class", "label"),
)

_DIABETES_HOSPITALS_FIELDS: tuple[tuple[str, str], ...] = (
    ("encounter_id", "i"), ("patient_nbr", "i"),
    ("race", "race"),
    ("gender", "gender"),
    ("age", "age"),
    ("weight", "weight"),
    ("admission_type_id", "i"), ("discharge_disposition_id", "i"), ("admission_source_id", "i"),
    ("time_in_hospital", "i"),
    ("payer_code", "payer_code"),
    ("medical_specialty", "text"),
    ("num_lab_procedures", "i"), ("num_procedures", "i"), ("num_medications", "i"),
    ("number_outpatient", "i"), ("number_emergency", "i"), ("number_inpatient", "i"),
    ("diag_1", "text"), ("diag_2", "text"), ("diag_3", "text"),
    ("number_diagnoses", "i"),
    ("max_glu_serum", "max_glu_serum"),
    ("a1cresult", "a1cresult"),
    ("metformin", "metformin"), ("repaglinide", "metformin"), ("nateglinide", "metformin"),
    ("chlorpropamide", "metformin"), ("glimepiride", "metformin"),
    ("acetohexamide", "acetohexamide"),
    ("glipizide", "metformin"), ("glyburide", "metformin"),
    ("tolbutamide", "acetohexamide"),
    ("pioglitazone", "metformin"), ("rosiglitazone", "metformin"), ("acarbose", "metformin"),
    ("miglitol", "metformin"),
    ("troglitazone", "acetohexamide"),
    ("tolazamide", "tolazamide"),
    ("examide", "examide"), ("citoglipton", "examide"),
    ("insulin", "metformin"), ("glyburide_metformin", "metformin"),
    ("glipizide_metformin", "acetohexamide"), ("glimepiride_pioglitazone", "acetohexamide"),
    ("metformin_rosiglitazone", "acetohexamide"), ("metformin_pioglitazone", "acetohexamide"),
    ("change", "change"),
    ("diabetesmed", "diabetesmed"),
    ("readmitted", "label"),
)

_DIABETIC_RETINOPATHY_FIELDS: tuple[tuple[str, str], ...] = (
    ("quality", "i"), ("pre_screening", "i"), ("ma1", "i"), ("ma2", "i"), ("ma3", "i"),
    ("ma4", "i"), ("ma5", "i"), ("ma6", "i"),
    ("exudate1", "d"), ("exudate2", "d"), ("exudate3", "d"), ("exudate32", "d"), ("exudate5", "d"),
    ("exudate6", "d"), ("exudate7", "d"), ("exudate8", "d"), ("macula_opticdisc_distance", "d"),
    ("opticdisc_diameter", "d"),
    ("am_fm_classification", "i"),
    ("class", "label"),
)

_DOTA2_GAMES_FIELDS: tuple[tuple[str, str], ...] = (
    ("win", "label"),
    ("clusterid", "i"), ("gamemode", "i"), ("gametype", "i"), ("hero1", "i"), ("hero2", "i"),
    ("hero3", "i"), ("hero4", "i"), ("hero5", "i"), ("hero6", "i"), ("hero7", "i"), ("hero8", "i"),
    ("hero9", "i"), ("hero10", "i"), ("hero11", "i"), ("hero12", "i"), ("hero13", "i"),
    ("hero14", "i"), ("hero15", "i"), ("hero16", "i"), ("hero17", "i"), ("hero18", "i"),
    ("hero19", "i"), ("hero20", "i"), ("hero21", "i"), ("hero22", "i"), ("hero23", "i"),
    ("hero24", "i"), ("hero25", "i"), ("hero26", "i"), ("hero27", "i"), ("hero28", "i"),
    ("hero29", "i"), ("hero30", "i"), ("hero31", "i"), ("hero32", "i"), ("hero33", "i"),
    ("hero34", "i"), ("hero35", "i"), ("hero36", "i"), ("hero37", "i"), ("hero38", "i"),
    ("hero39", "i"), ("hero40", "i"), ("hero41", "i"), ("hero42", "i"), ("hero43", "i"),
    ("hero44", "i"), ("hero45", "i"), ("hero46", "i"), ("hero47", "i"), ("hero48", "i"),
    ("hero49", "i"), ("hero50", "i"), ("hero51", "i"), ("hero52", "i"), ("hero53", "i"),
    ("hero54", "i"), ("hero55", "i"), ("hero56", "i"), ("hero57", "i"), ("hero58", "i"),
    ("hero59", "i"), ("hero60", "i"), ("hero61", "i"), ("hero62", "i"), ("hero63", "i"),
    ("hero64", "i"), ("hero65", "i"), ("hero66", "i"), ("hero67", "i"), ("hero68", "i"),
    ("hero69", "i"), ("hero70", "i"), ("hero71", "i"), ("hero72", "i"), ("hero73", "i"),
    ("hero74", "i"), ("hero75", "i"), ("hero76", "i"), ("hero77", "i"), ("hero78", "i"),
    ("hero79", "i"), ("hero80", "i"), ("hero81", "i"), ("hero82", "i"), ("hero83", "i"),
    ("hero84", "i"), ("hero85", "i"), ("hero86", "i"), ("hero87", "i"), ("hero88", "i"),
    ("hero89", "i"), ("hero90", "i"), ("hero91", "i"), ("hero92", "i"), ("hero93", "i"),
    ("hero94", "i"), ("hero95", "i"), ("hero96", "i"), ("hero97", "i"), ("hero98", "i"),
    ("hero99", "i"), ("hero100", "i"), ("hero101", "i"), ("hero102", "i"), ("hero103", "i"),
    ("hero104", "i"), ("hero105", "i"), ("hero106", "i"), ("hero107", "i"), ("hero108", "i"),
    ("hero109", "i"), ("hero110", "i"), ("hero111", "i"), ("hero112", "i"), ("hero113", "i"),
)

_DRUG_CONSUMPTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"),
    ("age", "d"), ("gender", "d"), ("education", "d"), ("country", "d"), ("ethnicity", "d"),
    ("nscore", "d"), ("escore", "d"), ("oscore", "d"), ("ascore", "d"), ("cscore", "d"),
    ("impuslive", "d"), ("ss", "d"),
    ("alcohol", "alcohol"), ("amphet", "alcohol"), ("amyl", "alcohol"), ("benzos", "alcohol"),
    ("caff", "alcohol"),
    ("cannabis", "label"),
    ("choc", "alcohol"), ("coke", "alcohol"), ("crack", "alcohol"), ("ecstasy", "alcohol"),
    ("heroin", "alcohol"), ("ketamine", "alcohol"), ("legalh", "alcohol"), ("lsd", "alcohol"),
    ("meth", "alcohol"), ("mushrooms", "alcohol"), ("nicotine", "alcohol"),
    ("semer", "semer"),
    ("vsa", "alcohol"),
)

_EEG_EYE_STATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("af3", "d"), ("f7", "d"), ("f3", "d"), ("fc5", "d"), ("t7", "d"), ("p7", "d"), ("o1", "d"),
    ("o2", "d"), ("p8", "d"), ("t8", "d"), ("fc6", "d"), ("f4", "d"), ("f8", "d"), ("af4", "d"),
    ("eyedetection", "label"),
)

_EL_NINO_FIELDS: tuple[tuple[str, str], ...] = (
    ("obs", "i"), ("year", "i"), ("month", "i"), ("day", "i"), ("date", "i"),
    ("latitude", "d"), ("longitude", "d"), ("zon_winds", "d"), ("mer_winds", "d"),
    ("humidity", "d"),
    ("air_temp", "target"),
    ("ss_temp", "d"),
)

_ENTRANCE_EXAM_FIELDS: tuple[tuple[str, str], ...] = (
    ("performance", "label"),
    ("gender", "gender"),
    ("caste", "caste"),
    ("coaching", "coaching"),
    ("time", "time_code"),
    ("class_ten_education", "class_ten_education"),
    ("twelve_education", "twelve_education"),
    ("medium", "medium"),
    ("class_x_percentage", "class_x_percentage"), ("class_xii_percentage", "class_x_percentage"),
    ("father_occupation", "father_occupation"),
    ("mother_occupation", "mother_occupation"),
)

_FACEBOOK_LIVE_SELLERS_FIELDS: tuple[tuple[str, str], ...] = (
    ("status_id", "i"),
    ("status_type", "label"),
    ("status_published", "time"),
    ("num_reactions", "i"), ("num_comments", "i"), ("num_shares", "i"), ("num_likes", "i"),
    ("num_loves", "i"), ("num_wows", "i"), ("num_hahas", "i"), ("num_sads", "i"),
    ("num_angrys", "i"),
)

_FACEBOOK_METRICS_FIELDS: tuple[tuple[str, str], ...] = (
    ("page_total_likes", "i"),
    ("type", "type"),
    ("category", "i"), ("post_month", "i"), ("post_weekday", "i"), ("post_hour", "i"),
    ("paid", "i"), ("lifetime_post_total_reach", "i"), ("lifetime_post_total_impressions", "i"),
    ("lifetime_engaged_users", "i"), ("lifetime_post_consumers", "i"),
    ("lifetime_post_consumptions", "i"),
    ("lifetime_post_impressions_by_people_who_have_liked_your_page", "i"),
    ("lifetime_post_reach_by_people_who_like_your_page", "i"),
    ("lifetime_people_who_have_liked_your_page_and_engaged_with_your_post", "i"), ("comment", "i"),
    ("like", "i"), ("share", "i"),
    ("total_interactions", "target"),
)

_FLAGS_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "text"),
    ("landmass", "i"), ("zone", "i"), ("area", "i"), ("population", "i"), ("language", "i"),
    ("religion", "label"),
    ("bars", "i"), ("stripes", "i"), ("colours", "i"), ("red", "i"), ("green", "i"), ("blue", "i"),
    ("gold", "i"), ("white", "i"), ("black", "i"), ("orange", "i"),
    ("mainhue", "mainhue"),
    ("circles", "i"), ("crosses", "i"), ("saltries", "i"), ("quarters", "i"), ("sunstars", "i"),
    ("crescent", "i"), ("triangle", "i"), ("icon", "i"), ("animate", "i"), ("text", "i"),
    ("topleft", "topleft"),
    ("botright", "mainhue"),
)

_GAS_TURBINE_EMISSIONS_FIELDS: tuple[tuple[str, str], ...] = (
    ("year", "i"),
    ("at", "d"), ("ap", "d"), ("ah", "d"), ("afdp", "d"), ("gtep", "d"), ("tit", "d"),
    ("tat", "d"), ("tey", "d"), ("cdp", "d"),
    ("co", "target"), ("nox", "target"),
)

_GENDER_BY_NAME_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "text"),
    ("gender", "label"),
    ("count", "i"),
    ("probability", "d"),
)

_GLIOMA_GRADING_FIELDS: tuple[tuple[str, str], ...] = (
    ("case_id", "text"),
    ("gender", "i"),
    ("age_at_diagnosis", "d"),
    ("race", "race"),
    ("idh1", "i"), ("tp53", "i"), ("atrx", "i"), ("pten", "i"), ("egfr", "i"), ("cic", "i"),
    ("muc16", "i"), ("pik3ca", "i"), ("nf1", "i"), ("pik3r1", "i"), ("fubp1", "i"), ("rb1", "i"),
    ("notch1", "i"), ("bcor", "i"), ("csmd3", "i"), ("smarca4", "i"), ("grin2a", "i"),
    ("idh2", "i"), ("fat4", "i"), ("pdgfra", "i"),
    ("grade", "label"),
)

_GRID_STABILITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("tau1", "d"), ("tau2", "d"), ("tau3", "d"), ("tau4", "d"), ("p1", "d"), ("p2", "d"),
    ("p3", "d"), ("p4", "d"), ("g1", "d"), ("g2", "d"), ("g3", "d"), ("g4", "d"), ("stab", "d"),
    ("stabf", "label"),
)

_HCV_BLOOD_DONORS_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"), ("age", "i"),
    ("sex", "sex"),
    ("alb", "d"), ("alp", "d"), ("alt", "d"), ("ast", "d"), ("bil", "d"), ("che", "d"),
    ("chol", "d"), ("crea", "d"), ("cgt", "d"), ("prot", "d"),
    ("category", "label"),
)

_HEALTHY_AGING_POLL_FIELDS: tuple[tuple[str, str], ...] = (
    ("number_of_doctors_visited", "label"),
    ("age", "i"), ("physical_health", "i"), ("mental_health", "i"), ("dental_health", "i"),
    ("employment", "i"), ("stress_keeps_patient_from_sleeping", "i"),
    ("medication_keeps_patient_from_sleeping", "i"), ("pain_keeps_patient_from_sleeping", "i"),
    ("bathroom_needs_keeps_patient_from_sleeping", "i"),
    ("uknown_keeps_patient_from_sleeping", "i"), ("trouble_sleeping", "i"),
    ("prescription_sleep_medication", "i"), ("race", "i"), ("gender", "i"),
)

_HEPATITIS_C_EGYPT_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"), ("gender", "i"), ("bmi", "i"), ("fever", "i"), ("nausea_vomting", "i"),
    ("headache", "i"), ("diarrhea", "i"), ("fatigue_generalized_bone_ache", "i"),
    ("jaundice", "i"), ("epigastric_pain", "i"), ("wbc", "i"),
    ("rbc", "d"),
    ("hgb", "i"),
    ("plat", "d"),
    ("ast_1", "i"), ("alt_1", "i"),
    ("alt4", "d"),
    ("alt_12", "i"), ("alt_24", "i"), ("alt_36", "i"), ("alt_48", "i"), ("alt_after_24_w", "i"),
    ("rna_base", "i"), ("rna_4", "i"), ("rna_12", "i"), ("rna_eot", "i"), ("rna_ef", "i"),
    ("baseline_histological_grading", "i"),
    ("baselinehistological_staging", "label"),
)

_HIGHER_EDUCATION_STUDENTS_FIELDS: tuple[tuple[str, str], ...] = (
    ("student_id", "text"),
    ("student_age", "i"), ("sex", "i"), ("graduated_high_school_type", "i"),
    ("scholarship_type", "i"), ("additional_work", "i"),
    ("regular_artistic_or_sports_activity", "i"), ("do_you_have_a_partner", "i"),
    ("total_salary_if_available", "i"), ("transportation_to_the_university", "i"),
    ("accomodation_type_in_cyprus", "i"), ("mother_s_education", "i"), ("father_s_education", "i"),
    ("number_of_sisters_brothers_if_available", "i"), ("parental_status", "i"),
    ("mother_s_occupation", "i"), ("father_s_occupation", "i"), ("weekly_study_hours", "i"),
    ("reading_frequency_non_scientific_books_journals", "i"),
    ("reading_frequency_scientific_books_journals", "i"),
    ("attendance_to_the_seminars_conferences_related_to_the_department", "i"),
    ("impact_of_your_projects_activities_on_your_success", "i"), ("attendance_to_classes", "i"),
    ("preparation_to_midterm_exams_1", "i"), ("preparation_to_midterm_exams_2", "i"),
    ("taking_notes_in_classes", "i"), ("listening_in_classes", "i"),
    ("discussion_improves_my_interest_and_success_in_the_course", "i"), ("flip_classroom", "i"),
    ("cumulative_grade_point_average_in_the_last_semester_4_00", "i"),
    ("expected_cumulative_grade_point_average_in_the_graduation_4_00", "i"), ("course_id", "i"),
    ("output_grade", "label"),
)

_HORSE_COLIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("surgery", "i"), ("age", "i"), ("hospital_number", "i"),
    ("rectal_temperature", "d"),
    ("pulse", "i"), ("respiratory_rate", "i"), ("temperature_of_extremities", "i"),
    ("peripheral_pulse", "i"), ("mucous_membranes", "i"), ("capillary_refill_time", "i"),
    ("pain", "i"), ("peristalsis", "i"), ("abdominal_distension", "i"), ("nasogastric_tube", "i"),
    ("nasogastric_reflux", "i"),
    ("nasogastric_reflux_ph", "d"),
    ("rectal_examination_feces", "i"), ("abdomen", "i"),
    ("packed_cell_volume", "d"), ("total_protein", "d"),
    ("abdominocentesis_appearance", "i"),
    ("abdominocentesis_total_protein", "d"),
    ("outcome", "i"),
    ("surgical_lesion", "label"),
    ("lesion_site", "i"), ("lesion_type", "i"), ("lesion_subtype", "i"), ("cp_data", "i"),
)

_IN_VEHICLE_COUPON_FIELDS: tuple[tuple[str, str], ...] = (
    ("destination", "destination"),
    ("passenger", "passenger"),
    ("weather", "weather"),
    ("temperature", "i"),
    ("time", "time_code"),
    ("coupon", "coupon"),
    ("expiration", "expiration"),
    ("gender", "gender"),
    ("age", "age"),
    ("maritalstatus", "maritalstatus"),
    ("has_children", "i"),
    ("education", "text"), ("occupation", "text"),
    ("income", "income"),
    ("car", "text"),
    ("bar", "bar"), ("coffeehouse", "bar"), ("carryaway", "bar"), ("restaurantlessthan20", "bar"),
    ("restaurant20to50", "bar"),
    ("tocoupon_geq5min", "i"), ("tocoupon_geq15min", "i"), ("tocoupon_geq25min", "i"),
    ("direction_same", "i"), ("direction_opp", "i"),
    ("y", "label"),
)

_INFRARED_THERMOGRAPHY_FIELDS: tuple[tuple[str, str], ...] = (
    ("subjectid", "text"),
    ("aveoralf", "target"), ("aveoralm", "target"),
    ("gender", "gender"),
    ("age", "age"),
    ("ethnicity", "text"),
    ("t_atm", "d"), ("humidity", "d"), ("distance", "d"), ("t_offset1", "d"), ("max1r13_1", "d"),
    ("max1l13_1", "d"), ("aveallr13_1", "d"), ("avealll13_1", "d"), ("t_rc1", "d"),
    ("t_rc_dry1", "d"), ("t_rc_wet1", "d"), ("t_rc_max1", "d"), ("t_lc1", "d"), ("t_lc_dry1", "d"),
    ("t_lc_wet1", "d"), ("t_lc_max1", "d"), ("rcc1", "d"), ("lcc1", "d"), ("canthimax1", "d"),
    ("canthi4max1", "d"), ("t_fhcc1", "d"), ("t_fhrc1", "d"), ("t_fhlc1", "d"), ("t_fhbc1", "d"),
    ("t_fhtc1", "d"), ("t_fh_max1", "d"), ("t_fhc_max1", "d"), ("t_max1", "d"), ("t_or1", "d"),
    ("t_or_max1", "d"),
)

_IOT_INTRUSION_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"), ("id_orig_p", "i"), ("id_resp_p", "i"),
    ("proto", "proto"),
    ("service", "service"),
    ("flow_duration", "d"),
    ("fwd_pkts_tot", "i"), ("bwd_pkts_tot", "i"), ("fwd_data_pkts_tot", "i"),
    ("bwd_data_pkts_tot", "i"),
    ("fwd_pkts_per_sec", "d"), ("bwd_pkts_per_sec", "d"), ("flow_pkts_per_sec", "d"),
    ("down_up_ratio", "d"),
    ("fwd_header_size_tot", "i"), ("fwd_header_size_min", "i"), ("fwd_header_size_max", "i"),
    ("bwd_header_size_tot", "i"), ("bwd_header_size_min", "i"), ("bwd_header_size_max", "i"),
    ("flow_fin_flag_count", "i"), ("flow_syn_flag_count", "i"), ("flow_rst_flag_count", "i"),
    ("fwd_psh_flag_count", "i"), ("bwd_psh_flag_count", "i"), ("flow_ack_flag_count", "i"),
    ("fwd_urg_flag_count", "i"), ("bwd_urg_flag_count", "i"), ("flow_cwr_flag_count", "i"),
    ("flow_ece_flag_count", "i"), ("fwd_pkts_payload_min", "i"), ("fwd_pkts_payload_max", "i"),
    ("fwd_pkts_payload_tot", "i"),
    ("fwd_pkts_payload_avg", "d"), ("fwd_pkts_payload_std", "d"),
    ("bwd_pkts_payload_min", "i"), ("bwd_pkts_payload_max", "i"), ("bwd_pkts_payload_tot", "i"),
    ("bwd_pkts_payload_avg", "d"), ("bwd_pkts_payload_std", "d"),
    ("flow_pkts_payload_min", "i"), ("flow_pkts_payload_max", "i"), ("flow_pkts_payload_tot", "i"),
    ("flow_pkts_payload_avg", "d"), ("flow_pkts_payload_std", "d"), ("fwd_iat_min", "d"),
    ("fwd_iat_max", "d"), ("fwd_iat_tot", "d"), ("fwd_iat_avg", "d"), ("fwd_iat_std", "d"),
    ("bwd_iat_min", "d"), ("bwd_iat_max", "d"), ("bwd_iat_tot", "d"), ("bwd_iat_avg", "d"),
    ("bwd_iat_std", "d"), ("flow_iat_min", "d"), ("flow_iat_max", "d"), ("flow_iat_tot", "d"),
    ("flow_iat_avg", "d"), ("flow_iat_std", "d"), ("payload_bytes_per_second", "d"),
    ("fwd_subflow_pkts", "d"), ("bwd_subflow_pkts", "d"), ("fwd_subflow_bytes", "d"),
    ("bwd_subflow_bytes", "d"), ("fwd_bulk_bytes", "d"), ("bwd_bulk_bytes", "d"),
    ("fwd_bulk_packets", "d"), ("bwd_bulk_packets", "d"), ("fwd_bulk_rate", "d"),
    ("bwd_bulk_rate", "d"), ("active_min", "d"), ("active_max", "d"), ("active_tot", "d"),
    ("active_avg", "d"), ("active_std", "d"), ("idle_min", "d"), ("idle_max", "d"),
    ("idle_tot", "d"), ("idle_avg", "d"), ("idle_std", "d"),
    ("fwd_init_window_size", "i"), ("bwd_init_window_size", "i"), ("fwd_last_window_size", "i"),
    ("attack_type", "label"),
)

_IRANIAN_CHURN_FIELDS: tuple[tuple[str, str], ...] = (
    ("call_failure", "i"), ("complains", "i"), ("subscription_length", "i"),
    ("charge_amount", "i"), ("seconds_of_use", "i"), ("frequency_of_use", "i"),
    ("frequency_of_sms", "i"), ("distinct_called_numbers", "i"), ("age_group", "i"),
    ("tariff_plan", "i"), ("status", "i"), ("age", "i"),
    ("customer_value", "d"),
    ("churn", "label"),
)

_ISOLET_FIELDS: tuple[tuple[str, str], ...] = (
    *_plain(tuple(f"attribute{at}" for at in range(1, 618)), "d"),
    ("class", "label"),
)

_ISTANBUL_EXCHANGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("date", "text"),
    ("ise", "target"),
    ("ise2", "d"), ("sp", "d"), ("dax", "d"), ("ftse", "d"), ("nikkei", "d"), ("bovespa", "d"),
    ("eu", "d"), ("em", "d"),
)

_KIDNEY_RISK_FACTORS_FIELDS: tuple[tuple[str, str], ...] = (
    ("bp_diastolic", "i"), ("bp_limit", "i"),
    ("sg", "sg"),
    ("al", "al"),
    ("class", "label"),
    ("rbc", "i"),
    ("su", "su"),
    ("pc", "i"), ("pcc", "i"), ("ba", "i"),
    ("bgr", "bgr"),
    ("bu", "bu"),
    ("sod", "sod"),
    ("sc", "sc"),
    ("pot", "pot"),
    ("hemo", "hemo"),
    ("pcv", "pcv"),
    ("rbcc", "rbcc"),
    ("wbcc", "wbcc"),
    ("htn", "i"), ("dm", "i"), ("cad", "i"), ("appet", "i"), ("pe", "i"), ("ane", "i"),
    ("grf", "grf"),
    ("stage", "stage"),
    ("affected", "i"),
    ("age", "age"),
)

_LAND_MINES_FIELDS: tuple[tuple[str, str], ...] = (
    ("v", "d"), ("h", "d"), ("s", "d"),
    ("m", "label"),
)

_METRO_TRAFFIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("holiday", "holiday"),
    ("temp", "d"), ("rain_1h", "d"), ("snow_1h", "d"),
    ("clouds_all", "i"),
    ("weather_main", "weather_main"),
    ("weather_description", "text"),
    ("date_time", "time"),
    ("traffic_volume", "target"),
)

_MICE_PROTEIN_FIELDS: tuple[tuple[str, str], ...] = (
    ("mouseid", "text"),
    ("dyrk1a_n", "d"), ("itsn1_n", "d"), ("bdnf_n", "d"), ("nr1_n", "d"), ("nr2a_n", "d"),
    ("pakt_n", "d"), ("pbraf_n", "d"), ("pcamkii_n", "d"), ("pcreb_n", "d"), ("pelk_n", "d"),
    ("perk_n", "d"), ("pjnk_n", "d"), ("pkca_n", "d"), ("pmek_n", "d"), ("pnr1_n", "d"),
    ("pnr2a_n", "d"), ("pnr2b_n", "d"), ("ppkcab_n", "d"), ("prsk_n", "d"), ("akt_n", "d"),
    ("braf_n", "d"), ("camkii_n", "d"), ("creb_n", "d"), ("elk_n", "d"), ("erk_n", "d"),
    ("gsk3b_n", "d"), ("jnk_n", "d"), ("mek_n", "d"), ("trka_n", "d"), ("rsk_n", "d"),
    ("app_n", "d"), ("bcatenin_n", "d"), ("sod1_n", "d"), ("mtor_n", "d"), ("p38_n", "d"),
    ("pmtor_n", "d"), ("dscr1_n", "d"), ("ampka_n", "d"), ("nr2b_n", "d"), ("pnumb_n", "d"),
    ("raptor_n", "d"), ("tiam1_n", "d"), ("pp70s6_n", "d"), ("numb_n", "d"), ("p70s6_n", "d"),
    ("pgsk3b_n", "d"), ("ppkcg_n", "d"), ("cdk5_n", "d"), ("s6_n", "d"), ("adarb1_n", "d"),
    ("acetylh3k9_n", "d"), ("rrp1_n", "d"), ("bax_n", "d"), ("arc_n", "d"), ("erbb4_n", "d"),
    ("nnos_n", "d"), ("tau_n", "d"), ("gfap_n", "d"), ("glur3_n", "d"), ("glur4_n", "d"),
    ("il1b_n", "d"), ("p3525_n", "d"), ("pcasp9_n", "d"), ("psd95_n", "d"), ("snca_n", "d"),
    ("ubiquitin_n", "d"), ("pgsk3b_tyr216_n", "d"), ("shh_n", "d"), ("bad_n", "d"),
    ("bcl2_n", "d"), ("ps6_n", "d"), ("pcfos_n", "d"), ("syp_n", "d"), ("h3ack18_n", "d"),
    ("egr1_n", "d"), ("h3mek4_n", "d"), ("cana_n", "d"),
    ("genotype", "genotype"),
    ("treatment", "treatment"),
    ("behavior", "behavior"),
    ("class", "label"),
)

_MONKS_PROBLEMS_FIELDS: tuple[tuple[str, str], ...] = (
    ("class", "label"),
    *_plain(tuple(f"a{at}" for at in range(1, 7)), "i"),
    ("id", "text"),
)

_MULTIVARIATE_GAIT_FIELDS: tuple[tuple[str, str], ...] = (
    ("subject", "i"), ("condition", "i"), ("replication", "i"), ("leg", "i"), ("joint", "i"),
    ("time", "i"),
    ("angle", "target"),
)

_MUSK_VERSION1_FIELDS: tuple[tuple[str, str], ...] = (
    ("molecule_name", "text"), ("conformation_name", "text"),
    *_plain(tuple(f"f{at}" for at in range(1, 167)), "i"),
    ("class", "label"),
)

_MUSK_VERSION2_FIELDS: tuple[tuple[str, str], ...] = (
    ("molecule_name", "text"), ("conformation_name", "text"),
    *_plain(tuple(f"f{at}" for at in range(1, 167)), "i"),
    ("class", "label"),
)

_NEWS_POPULARITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("url", "text"),
    ("timedelta", "d"), ("n_tokens_title", "d"), ("n_tokens_content", "d"),
    ("n_unique_tokens", "d"), ("n_non_stop_words", "d"), ("n_non_stop_unique_tokens", "d"),
    ("num_hrefs", "d"), ("num_self_hrefs", "d"), ("num_imgs", "d"), ("num_videos", "d"),
    ("average_token_length", "d"), ("num_keywords", "d"), ("data_channel_is_lifestyle", "d"),
    ("data_channel_is_entertainment", "d"), ("data_channel_is_bus", "d"),
    ("data_channel_is_socmed", "d"), ("data_channel_is_tech", "d"), ("data_channel_is_world", "d"),
    ("kw_min_min", "d"), ("kw_max_min", "d"), ("kw_avg_min", "d"), ("kw_min_max", "d"),
    ("kw_max_max", "d"), ("kw_avg_max", "d"), ("kw_min_avg", "d"), ("kw_max_avg", "d"),
    ("kw_avg_avg", "d"), ("self_reference_min_shares", "d"), ("self_reference_max_shares", "d"),
    ("self_reference_avg_sharess", "d"), ("weekday_is_monday", "d"), ("weekday_is_tuesday", "d"),
    ("weekday_is_wednesday", "d"), ("weekday_is_thursday", "d"), ("weekday_is_friday", "d"),
    ("weekday_is_saturday", "d"), ("weekday_is_sunday", "d"), ("is_weekend", "d"), ("lda_00", "d"),
    ("lda_01", "d"), ("lda_02", "d"), ("lda_03", "d"), ("lda_04", "d"),
    ("global_subjectivity", "d"), ("global_sentiment_polarity", "d"),
    ("global_rate_positive_words", "d"), ("global_rate_negative_words", "d"),
    ("rate_positive_words", "d"), ("rate_negative_words", "d"), ("avg_positive_polarity", "d"),
    ("min_positive_polarity", "d"), ("max_positive_polarity", "d"), ("avg_negative_polarity", "d"),
    ("min_negative_polarity", "d"), ("max_negative_polarity", "d"), ("title_subjectivity", "d"),
    ("title_sentiment_polarity", "d"), ("abs_title_subjectivity", "d"),
    ("abs_title_sentiment_polarity", "d"),
    ("shares", "target"),
)

_NHANES_AGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("seqn", "d"),
    ("age_group", "label"),
    ("ridageyr", "d"), ("riagendr", "d"), ("paq605", "d"), ("bmxbmi", "d"), ("lbxglu", "d"),
    ("diq010", "d"), ("lbxglt", "d"), ("lbxin", "d"),
)

_OBESITY_LEVELS_FIELDS: tuple[tuple[str, str], ...] = (
    ("gender", "gender"),
    ("age", "d"), ("height", "d"), ("weight", "d"),
    ("family_history_with_overweight", "family_history_with_overweight"),
    ("favc", "family_history_with_overweight"),
    ("fcvc", "d"), ("ncp", "d"),
    ("caec", "caec"),
    ("smoke", "family_history_with_overweight"),
    ("ch2o", "d"),
    ("scc", "family_history_with_overweight"),
    ("faf", "d"), ("tue", "d"),
    ("calc", "caec"),
    ("mtrans", "mtrans"),
    ("nobeyesdad", "label"),
)

_OZONE_LEVEL_FIELDS: tuple[tuple[str, str], ...] = (
    ("dataset", "dataset"),
    ("date", "date"),
    ("wsr0", "d"), ("wsr1", "d"), ("wsr2", "d"), ("wsr3", "d"), ("wsr4", "d"), ("wsr5", "d"),
    ("wsr6", "d"), ("wsr7", "d"), ("wsr8", "d"), ("wsr9", "d"), ("wsr10", "d"), ("wsr11", "d"),
    ("wsr12", "d"), ("wsr13", "d"), ("wsr14", "d"), ("wsr15", "d"), ("wsr16", "d"), ("wsr17", "d"),
    ("wsr18", "d"), ("wsr19", "d"), ("wsr20", "d"), ("wsr21", "d"), ("wsr22", "d"), ("wsr23", "d"),
    ("wsr_pk", "d"), ("wsr_av", "d"), ("t0", "d"), ("t1", "d"), ("t2", "d"), ("t3", "d"),
    ("t4", "d"), ("t5", "d"), ("t6", "d"), ("t7", "d"), ("t8", "d"), ("t9", "d"), ("t10", "d"),
    ("t11", "d"), ("t12", "d"), ("t13", "d"), ("t14", "d"), ("t15", "d"), ("t16", "d"),
    ("t17", "d"), ("t18", "d"), ("t19", "d"), ("t20", "d"), ("t21", "d"), ("t22", "d"),
    ("t23", "d"), ("t_pk", "d"), ("t_av", "d"), ("t85", "d"), ("rh85", "d"), ("u85", "d"),
    ("v85", "d"), ("ht85", "d"), ("t70", "d"), ("rh70", "d"), ("u70", "d"), ("v70", "d"),
    ("ht70", "d"), ("t50", "d"), ("rh50", "d"), ("u50", "d"), ("v50", "d"),
    ("ht50", "i"),
    ("ki", "d"), ("tt", "d"),
    ("slp", "i"), ("slp2", "i"),
    ("precp", "d"),
    ("class", "label"),
)

_PAGE_BLOCKS_FIELDS: tuple[tuple[str, str], ...] = (
    ("height", "i"), ("length", "i"), ("area", "i"),
    ("eccen", "d"), ("p_black", "d"), ("p_and", "d"), ("mean_tr", "d"),
    ("blackpix", "i"), ("blackand", "i"), ("wb_trans", "i"),
    ("class", "label"),
)

_PITTSBURGH_BRIDGES_FIELDS: tuple[tuple[str, str], ...] = (
    ("identif", "text"),
    ("river", "river"),
    ("location", "d"),
    ("erected", "label"),
    ("purpose", "purpose"),
    ("length", "length"),
    ("lanes", "i"),
    ("clear_g", "clear_g"),
    ("t_or_d", "t_or_d"),
    ("material", "material"),
    ("span", "length"),
    ("rel_l", "rel_l"),
    ("type", "type"),
)

_POKER_HAND_FIELDS: tuple[tuple[str, str], ...] = (
    ("s1", "i"), ("c1", "i"), ("s2", "i"), ("c2", "i"), ("s3", "i"), ("c3", "i"), ("s4", "i"),
    ("c4", "i"), ("s5", "i"), ("c5", "i"),
    ("class", "label"),
)

_POLISH_BANKRUPTCY_FIELDS: tuple[tuple[str, str], ...] = (
    ("year", "i"),
    *_plain(tuple(f"a{at}" for at in range(1, 65)), "d"),
    ("class", "label"),
)

_POST_OPERATIVE_PATIENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("l_core", "l_core"), ("l_surf", "l_core"),
    ("l_o2", "l_o2"),
    ("l_bp", "l_core"),
    ("surf_stbl", "surf_stbl"),
    ("core_stbl", "core_stbl"), ("bp_stbl", "core_stbl"),
    ("comfort", "i"),
    ("adm_decs", "label"),
)

_PREDICTIVE_MAINTENANCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("uid", "i"),
    ("product_id", "text"),
    ("type", "type"),
    ("air_temperature", "d"), ("process_temperature", "d"),
    ("rotational_speed", "i"),
    ("torque", "d"),
    ("tool_wear", "i"),
    ("machine_failure", "label"),
    ("twf", "i"), ("hdf", "i"), ("pwf", "i"), ("osf", "i"), ("rnf", "i"),
)

_ROOM_OCCUPANCY_COUNT_FIELDS: tuple[tuple[str, str], ...] = (
    ("date", "date_code"),
    ("time", "time"),
    ("s1_temp", "d"), ("s2_temp", "d"), ("s3_temp", "d"), ("s4_temp", "d"),
    ("s1_light", "i"), ("s2_light", "i"), ("s3_light", "i"), ("s4_light", "i"),
    ("s1_sound", "d"), ("s2_sound", "d"), ("s3_sound", "d"), ("s4_sound", "d"),
    ("s5_co2", "i"),
    ("s5_co2_slope", "d"),
    ("s6_pir", "i"), ("s7_pir", "i"),
    ("room_occupancy_count", "target"),
)

_SECONDARY_MUSHROOM_FIELDS: tuple[tuple[str, str], ...] = (
    ("class", "label"),
    ("cap_diameter", "d"),
    ("cap_shape", "cap_shape"),
    ("cap_surface", "cap_surface"),
    ("cap_color", "cap_color"),
    ("does_bruise_or_bleed", "does_bruise_or_bleed"),
    ("gill_attachment", "gill_attachment"),
    ("gill_spacing", "gill_spacing"),
    ("gill_color", "gill_color"),
    ("stem_height", "d"), ("stem_width", "d"),
    ("stem_root", "stem_root"),
    ("stem_surface", "stem_surface"),
    ("stem_color", "stem_color"),
    ("veil_type", "veil_type"),
    ("veil_color", "veil_color"),
    ("has_ring", "does_bruise_or_bleed"),
    ("ring_type", "ring_type"),
    ("spore_print_color", "spore_print_color"),
    ("habitat", "habitat"),
    ("season", "season"),
)

_SEOUL_BIKE_SHARING_FIELDS: tuple[tuple[str, str], ...] = (
    ("date", "date"),
    ("rented_bike_count", "target"),
    ("hour", "i"),
    ("temperature", "d"),
    ("humidity", "i"),
    ("wind_speed", "d"),
    ("visibility", "i"),
    ("dew_point_temperature", "d"), ("solar_radiation", "d"), ("rainfall", "d"), ("snowfall", "d"),
    ("seasons", "seasons"),
    ("holiday", "holiday"),
    ("functioning_day", "functioning_day"),
)

_SEPSIS_SURVIVAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("age_years", "i"), ("sex_0male_1female", "i"), ("episode_number", "i"),
    ("hospital_outcome_1alive_0dead", "label"),
)

_SKIN_SEGMENTATION_FIELDS: tuple[tuple[str, str], ...] = (
    ("b", "i"), ("g", "i"), ("r", "i"),
    ("y", "label"),
)

_SOYBEAN_CULTIVARS_FIELDS: tuple[tuple[str, str], ...] = (
    ("season", "i"),
    ("cultivar", "cultivar"),
    ("repetition", "i"),
    ("ph", "d"), ("ifp", "d"), ("nlp", "d"), ("ngp", "d"), ("ngl", "d"), ("ns", "d"), ("mhg", "d"),
    ("gy", "target"),
)

_SOYBEAN_SMALL_FIELDS: tuple[tuple[str, str], ...] = (
    ("date", "i"), ("plant_stand", "i"), ("precip", "i"), ("temp", "i"), ("hail", "i"),
    ("crop_hist", "i"), ("area_damaged", "i"), ("severity", "i"), ("seed_tmt", "i"),
    ("germination", "i"), ("plant_growth", "i"), ("leaves", "i"), ("leafspots_halo", "i"),
    ("leafspots_marg", "i"), ("leafspot_size", "i"), ("leaf_shread", "i"), ("leaf_malf", "i"),
    ("leaf_mild", "i"), ("stem", "i"), ("lodging", "i"), ("stem_cankers", "i"),
    ("canker_lesion", "i"), ("fruiting_bodies", "i"), ("external_decay", "i"), ("mycelium", "i"),
    ("int_discolor", "i"), ("sclerotia", "i"), ("fruit_pods", "i"), ("fruit_spots", "i"),
    ("seed", "i"), ("mold_growth", "i"), ("seed_discolor", "i"), ("seed_size", "i"),
    ("shriveling", "i"), ("roots", "i"),
    ("class", "label"),
)

_SPECT_HEART_FIELDS: tuple[tuple[str, str], ...] = (
    ("overall_diagnosis", "label"),
    *_plain(tuple(f"f{at}" for at in range(1, 23)), "i"),
)

_SPECTF_HEART_FIELDS: tuple[tuple[str, str], ...] = (
    ("diagnosis", "label"),
    ("f1r", "i"), ("f1s", "i"), ("f2r", "i"), ("f2s", "i"), ("f3r", "i"), ("f3s", "i"),
    ("f4r", "i"), ("f4s", "i"), ("f5r", "i"), ("f5s", "i"), ("f6r", "i"), ("f6s", "i"),
    ("f7r", "i"), ("f7s", "i"), ("f8r", "i"), ("f8s", "i"), ("f9r", "i"), ("f9s", "i"),
    ("f10r", "i"), ("f10s", "i"), ("f11r", "i"), ("f11s", "i"), ("f12r", "i"), ("f12s", "i"),
    ("f13r", "i"), ("f13s", "i"), ("f14r", "i"), ("f14s", "i"), ("f15r", "i"), ("f15s", "i"),
    ("f16r", "i"), ("f16s", "i"), ("f17r", "i"), ("f17s", "i"), ("f18r", "i"), ("f18s", "i"),
    ("f19r", "i"), ("f19s", "i"), ("f20r", "i"), ("f20s", "i"), ("f21r", "i"), ("f21s", "i"),
    ("f22r", "i"), ("f22s", "i"),
)

_SPLICE_JUNCTIONS_FIELDS: tuple[tuple[str, str], ...] = (
    ("class", "label"),
    ("instancename", "text"),
    ("base1", "base1"), ("base2", "base1"),
    *_plain(tuple(f"base{at}" for at in range(3, 14)), "base3"),
    ("base14", "base14"),
    ("base15", "base3"), ("base16", "base3"), ("base17", "base3"), ("base18", "base3"),
    *_plain(tuple(f"base{at}" for at in range(19, 35)), "base14"),
    ("base35", "base35"),
    ("base36", "base36"),
    *_plain(tuple(f"base{at}" for at in range(37, 61)), "base14"),
)

_STEEL_PLATES_FIELDS: tuple[tuple[str, str], ...] = (
    ("x_minimum", "i"), ("x_maximum", "i"), ("y_minimum", "i"), ("y_maximum", "i"),
    ("pixels_areas", "i"), ("x_perimeter", "i"), ("y_perimeter", "i"), ("sum_of_luminosity", "i"),
    ("minimum_of_luminosity", "i"), ("maximum_of_luminosity", "i"), ("length_of_conveyer", "i"),
    ("typeofsteel_a300", "i"), ("typeofsteel_a400", "i"), ("steel_plate_thickness", "i"),
    ("edges_index", "d"), ("empty_index", "d"), ("square_index", "d"), ("outside_x_index", "d"),
    ("edges_x_index", "d"), ("edges_y_index", "d"), ("outside_global_index", "d"),
    ("logofareas", "d"), ("log_x_index", "d"), ("log_y_index", "d"), ("orientation_index", "d"),
    ("luminosity_index", "d"), ("sigmoidofareas", "d"),
    ("pastry", "target"), ("z_scratch", "target"), ("k_scratch", "target"), ("stains", "target"),
    ("dirtiness", "target"), ("bumps", "target"), ("other_faults", "target"),
)

_STUDENT_ACADEMICS_FIELDS: tuple[tuple[str, str], ...] = (
    ("ge", "ge"),
    ("cst", "cst"),
    ("tnp", "tnp"), ("twp", "tnp"), ("iap", "tnp"),
    ("esp", "label"),
    ("arr", "arr"),
    ("ms", "ms"),
    ("ls", "ls"),
    ("as", "as"),
    ("fmi", "fmi"),
    ("fs", "fs"),
    ("fq", "fq"), ("mq", "fq"),
    ("fo", "fo"),
    ("mo", "mo"),
    ("nf", "fs"),
    ("sh", "sh"),
    ("ss", "ss"),
    ("me", "me"),
    ("tt", "fs"),
    ("atd", "sh"),
)

_STUDENT_DROPOUT_FIELDS: tuple[tuple[str, str], ...] = (
    ("marital_status", "i"), ("application_mode", "i"), ("application_order", "i"),
    ("course", "i"), ("daytime_evening_attendance", "i"), ("previous_qualification", "i"),
    ("previous_qualification_grade", "d"),
    ("nacionality", "i"), ("mother_s_qualification", "i"), ("father_s_qualification", "i"),
    ("mother_s_occupation", "i"), ("father_s_occupation", "i"),
    ("admission_grade", "d"),
    ("displaced", "i"), ("educational_special_needs", "i"), ("debtor", "i"),
    ("tuition_fees_up_to_date", "i"), ("gender", "i"), ("scholarship_holder", "i"),
    ("age_at_enrollment", "i"), ("international", "i"), ("curricular_units_1st_sem_credited", "i"),
    ("curricular_units_1st_sem_enrolled", "i"), ("curricular_units_1st_sem_evaluations", "i"),
    ("curricular_units_1st_sem_approved", "i"),
    ("curricular_units_1st_sem_grade", "d"),
    ("curricular_units_1st_sem_without_evaluations", "i"),
    ("curricular_units_2nd_sem_credited", "i"), ("curricular_units_2nd_sem_enrolled", "i"),
    ("curricular_units_2nd_sem_evaluations", "i"), ("curricular_units_2nd_sem_approved", "i"),
    ("curricular_units_2nd_sem_grade", "d"),
    ("curricular_units_2nd_sem_without_evaluations", "i"),
    ("unemployment_rate", "d"), ("inflation_rate", "d"), ("gdp", "d"),
    ("target", "label"),
)

_SUPERCONDUCTIVITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("number_of_elements", "i"),
    ("mean_atomic_mass", "d"), ("wtd_mean_atomic_mass", "d"), ("gmean_atomic_mass", "d"),
    ("wtd_gmean_atomic_mass", "d"), ("entropy_atomic_mass", "d"), ("wtd_entropy_atomic_mass", "d"),
    ("range_atomic_mass", "d"), ("wtd_range_atomic_mass", "d"), ("std_atomic_mass", "d"),
    ("wtd_std_atomic_mass", "d"), ("mean_fie", "d"), ("wtd_mean_fie", "d"), ("gmean_fie", "d"),
    ("wtd_gmean_fie", "d"), ("entropy_fie", "d"), ("wtd_entropy_fie", "d"), ("range_fie", "d"),
    ("wtd_range_fie", "d"), ("std_fie", "d"), ("wtd_std_fie", "d"), ("mean_atomic_radius", "d"),
    ("wtd_mean_atomic_radius", "d"), ("gmean_atomic_radius", "d"),
    ("wtd_gmean_atomic_radius", "d"), ("entropy_atomic_radius", "d"),
    ("wtd_entropy_atomic_radius", "d"),
    ("range_atomic_radius", "i"),
    ("wtd_range_atomic_radius", "d"), ("std_atomic_radius", "d"), ("wtd_std_atomic_radius", "d"),
    ("mean_density", "d"), ("wtd_mean_density", "d"), ("gmean_density", "d"),
    ("wtd_gmean_density", "d"), ("entropy_density", "d"), ("wtd_entropy_density", "d"),
    ("range_density", "d"), ("wtd_range_density", "d"), ("std_density", "d"),
    ("wtd_std_density", "d"), ("mean_electronaffinity", "d"), ("wtd_mean_electronaffinity", "d"),
    ("gmean_electronaffinity", "d"), ("wtd_gmean_electronaffinity", "d"),
    ("entropy_electronaffinity", "d"), ("wtd_entropy_electronaffinity", "d"),
    ("range_electronaffinity", "d"), ("wtd_range_electronaffinity", "d"),
    ("std_electronaffinity", "d"), ("wtd_std_electronaffinity", "d"), ("mean_fusionheat", "d"),
    ("wtd_mean_fusionheat", "d"), ("gmean_fusionheat", "d"), ("wtd_gmean_fusionheat", "d"),
    ("entropy_fusionheat", "d"), ("wtd_entropy_fusionheat", "d"), ("range_fusionheat", "d"),
    ("wtd_range_fusionheat", "d"), ("std_fusionheat", "d"), ("wtd_std_fusionheat", "d"),
    ("mean_thermalconductivity", "d"), ("wtd_mean_thermalconductivity", "d"),
    ("gmean_thermalconductivity", "d"), ("wtd_gmean_thermalconductivity", "d"),
    ("entropy_thermalconductivity", "d"), ("wtd_entropy_thermalconductivity", "d"),
    ("range_thermalconductivity", "d"), ("wtd_range_thermalconductivity", "d"),
    ("std_thermalconductivity", "d"), ("wtd_std_thermalconductivity", "d"), ("mean_valence", "d"),
    ("wtd_mean_valence", "d"), ("gmean_valence", "d"), ("wtd_gmean_valence", "d"),
    ("entropy_valence", "d"), ("wtd_entropy_valence", "d"),
    ("range_valence", "i"),
    ("wtd_range_valence", "d"), ("std_valence", "d"), ("wtd_std_valence", "d"),
    ("critical_temp", "target"),
)

_SUPPORT2_FIELDS: tuple[tuple[str, str], ...] = (
    ("id", "i"),
    ("age", "d"),
    ("death", "i"),
    ("sex", "sex"),
    ("hospdead", "label"),
    ("slos", "i"), ("d_time", "i"),
    ("dzgroup", "dzgroup"),
    ("dzclass", "dzclass"),
    ("num_co", "i"), ("edu", "i"),
    ("income", "income"),
    ("scoma", "i"),
    ("charges", "d"), ("totcst", "d"), ("totmcst", "d"), ("avtisst", "d"),
    ("race", "race"),
    ("sps", "d"),
    ("aps", "i"),
    ("surv2m", "d"), ("surv6m", "d"),
    ("hday", "i"), ("diabetes", "i"), ("dementia", "i"),
    ("ca", "ca"),
    ("prg2m", "d"), ("prg6m", "d"),
    ("dnr", "dnr"),
    ("dnrday", "i"),
    ("meanbp", "d"), ("wblc", "d"), ("hrt", "d"),
    ("resp", "i"),
    ("temp", "d"), ("pafi", "d"), ("alb", "d"), ("bili", "d"), ("crea", "d"),
    ("sod", "i"),
    ("ph", "d"), ("glucose", "d"), ("bun", "d"), ("urine", "d"),
    ("adlp", "i"), ("adls", "i"),
    ("sfdm2", "sfdm2"),
    ("adlsc", "d"),
)

_TAIWANESE_BANKRUPTCY_FIELDS: tuple[tuple[str, str], ...] = (
    ("bankrupt", "label"),
    ("roa_c_before_interest_and_depreciation_before_interest", "d"),
    ("roa_a_before_interest_and_after_tax", "d"),
    ("roa_b_before_interest_and_depreciation_after_tax", "d"), ("operating_gross_margin", "d"),
    ("realized_sales_gross_margin", "d"), ("operating_profit_rate", "d"),
    ("pre_tax_net_interest_rate", "d"), ("after_tax_net_interest_rate", "d"),
    ("non_industry_income_and_expenditure_revenue", "d"),
    ("continuous_interest_rate_after_tax", "d"), ("operating_expense_rate", "d"),
    ("research_and_development_expense_rate", "d"), ("cash_flow_rate", "d"),
    ("interest_bearing_debt_interest_rate", "d"), ("tax_rate_a", "d"),
    ("net_value_per_share_b", "d"), ("net_value_per_share_a", "d"), ("net_value_per_share_c", "d"),
    ("persistent_eps_in_the_last_four_seasons", "d"), ("cash_flow_per_share", "d"),
    ("revenue_per_share_yuan", "d"), ("operating_profit_per_share_yuan", "d"),
    ("per_share_net_profit_before_tax_yuan", "d"),
    ("realized_sales_gross_profit_growth_rate", "d"), ("operating_profit_growth_rate", "d"),
    ("after_tax_net_profit_growth_rate", "d"), ("regular_net_profit_growth_rate", "d"),
    ("continuous_net_profit_growth_rate", "d"), ("total_asset_growth_rate", "d"),
    ("net_value_growth_rate", "d"), ("total_asset_return_growth_rate_ratio", "d"),
    ("cash_reinvestment", "d"), ("current_ratio", "d"), ("quick_ratio", "d"),
    ("interest_expense_ratio", "d"), ("total_debt_total_net_worth", "d"), ("debt_ratio", "d"),
    ("net_worth_assets", "d"), ("long_term_fund_suitability_ratio_a", "d"),
    ("borrowing_dependency", "d"), ("contingent_liabilities_net_worth", "d"),
    ("operating_profit_paid_in_capital", "d"), ("net_profit_before_tax_paid_in_capital", "d"),
    ("inventory_and_accounts_receivable_net_value", "d"), ("total_asset_turnover", "d"),
    ("accounts_receivable_turnover", "d"), ("average_collection_days", "d"),
    ("inventory_turnover_rate_times", "d"), ("fixed_assets_turnover_frequency", "d"),
    ("net_worth_turnover_rate_times", "d"), ("revenue_per_person", "d"),
    ("operating_profit_per_person", "d"), ("allocation_rate_per_person", "d"),
    ("working_capital_to_total_assets", "d"), ("quick_assets_total_assets", "d"),
    ("current_assets_total_assets", "d"), ("cash_total_assets", "d"),
    ("quick_assets_current_liability", "d"), ("cash_current_liability", "d"),
    ("current_liability_to_assets", "d"), ("operating_funds_to_liability", "d"),
    ("inventory_working_capital", "d"), ("inventory_current_liability", "d"),
    ("current_liabilities_liability", "d"), ("working_capital_equity", "d"),
    ("current_liabilities_equity", "d"), ("long_term_liability_to_current_assets", "d"),
    ("retained_earnings_to_total_assets", "d"), ("total_income_total_expense", "d"),
    ("total_expense_assets", "d"), ("current_asset_turnover_rate", "d"),
    ("quick_asset_turnover_rate", "d"), ("working_capitcal_turnover_rate", "d"),
    ("cash_turnover_rate", "d"), ("cash_flow_to_sales", "d"), ("fixed_assets_to_assets", "d"),
    ("current_liability_to_liability", "d"), ("current_liability_to_equity", "d"),
    ("equity_to_long_term_liability", "d"), ("cash_flow_to_total_assets", "d"),
    ("cash_flow_to_liability", "d"), ("cfo_to_assets", "d"), ("cash_flow_to_equity", "d"),
    ("current_liability_to_current_assets", "d"),
    ("liability_assets_flag", "i"),
    ("net_income_to_total_assets", "d"), ("total_assets_to_gnp_price", "d"),
    ("no_credit_interval", "d"), ("gross_profit_to_sales", "d"),
    ("net_income_to_stockholder_s_equity", "d"), ("liability_to_equity", "d"),
    ("degree_of_financial_leverage_dfl", "d"),
    ("interest_coverage_ratio_interest_expense_to_ebit", "d"),
    ("net_income_flag", "i"),
    ("equity_to_liability", "d"),
)

_TENNIS_MAJORS_FIELDS: tuple[tuple[str, str], ...] = (
    ("tournament", "tournament"),
    ("player1", "text"), ("player2", "text"),
    ("round", "i"),
    ("result", "label"),
    ("fnl1", "i"), ("fnl2", "i"), ("fsp_1", "i"), ("fsw_1", "i"), ("ssp_1", "i"), ("ssw_1", "i"),
    ("ace_1", "i"), ("dbf_1", "i"), ("wnr_1", "i"), ("ufe_1", "i"), ("bpc_1", "i"), ("bpw_1", "i"),
    ("npa_1", "i"), ("npw_1", "i"), ("tpw_1", "i"), ("st1_1", "i"), ("st2_1", "i"), ("st3_1", "i"),
    ("st4_1", "i"), ("st5_1", "i"), ("fsp_2", "i"), ("fsw_2", "i"), ("ssp_2", "i"), ("ssw_2", "i"),
    ("ace_2", "i"), ("dbf_2", "i"), ("wnr_2", "i"), ("ufe_2", "i"), ("bpc_2", "i"), ("bpw_2", "i"),
    ("npa_2", "i"), ("npw_2", "i"), ("tpw_2", "i"), ("st1_2", "i"), ("st2_2", "i"), ("st3_2", "i"),
    ("st4_2", "i"), ("st5_2", "i"),
)

_TETOUAN_POWER_FIELDS: tuple[tuple[str, str], ...] = (
    ("datetime", "time"),
    ("temperature", "d"), ("humidity", "d"), ("wind_speed", "d"), ("general_diffuse_flows", "d"),
    ("diffuse_flows", "d"),
    ("zone_1_power_consumption", "target"), ("zone_2_power_consumption", "target"),
    ("zone_3_power_consumption", "target"),
)

_THORACIC_SURGERY_FIELDS: tuple[tuple[str, str], ...] = (
    ("dgn", "dgn"),
    ("pre4", "d"), ("pre5", "d"),
    ("pre6", "pre6"),
    ("pre7", "pre7"), ("pre8", "pre7"), ("pre9", "pre7"), ("pre10", "pre7"), ("pre11", "pre7"),
    ("pre14", "pre14"),
    ("pre17", "pre7"), ("pre19", "pre7"), ("pre25", "pre7"), ("pre30", "pre7"), ("pre32", "pre7"),
    ("age", "i"),
    ("risk1yr", "label"),
)

_THYROID_RECURRENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("age", "i"),
    ("gender", "gender"),
    ("smoking", "smoking"), ("hx_smoking", "smoking"), ("hx_radiothreapy", "smoking"),
    ("thyroid_function", "thyroid_function"),
    ("physical_examination", "physical_examination"),
    ("adenopathy", "adenopathy"),
    ("pathology", "pathology"),
    ("focality", "focality"),
    ("risk", "risk"),
    ("t", "t"),
    ("n", "n"),
    ("m", "m"),
    ("stage", "stage"),
    ("response", "response"),
    ("recurred", "label"),
)

_USER_KNOWLEDGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("stg", "d"), ("scg", "d"), ("str", "d"), ("lpr", "d"), ("peg", "d"),
    ("uns", "label"),
)

_WAVEFORM_FIELDS: tuple[tuple[str, str], ...] = (
    *_plain(tuple(f"attribute{at}" for at in range(1, 22)), "d"),
    ("class", "label"),
)

_WEBSITE_PHISHING_FIELDS: tuple[tuple[str, str], ...] = (
    ("sfh", "i"), ("popupwindow", "i"), ("sslfinal_state", "i"), ("request_url", "i"),
    ("url_of_anchor", "i"), ("web_traffic", "i"), ("url_length", "i"), ("age_of_domain", "i"),
    ("having_ip_address", "i"),
    ("result", "label"),
)

_WHOLESALE_CUSTOMERS_FIELDS: tuple[tuple[str, str], ...] = (
    ("channel", "label"),
    ("region", "i"), ("fresh", "i"), ("milk", "i"), ("grocery", "i"), ("frozen", "i"),
    ("detergents_paper", "i"), ("delicassen", "i"),
)

_YOUTUBE_SPAM_FIELDS: tuple[tuple[str, str], ...] = (
    ("video", "video"),
    ("comment_id", "text"), ("author", "text"), ("date", "text"), ("content", "text"),
    ("class", "label"),
)

DATASETS: dict[str, Images | CIFAR | Audio | Matrix | Table] = {
    "mnist": Images(
        name="mnist",
        label="MNIST",
        title="70,000 handwritten digits, 28x28 greyscale, 10 classes",
        licence="CC BY-SA 3.0",
        source="https://yann.lecun.com/exdb/mnist/",
        classes=tuple(str(digit) for digit in range(10)),
        splits=("train", "test"),
        basket_size=IMAGE_BASKET,
        base=MNIST_MIRROR,
    ),
    "fashion_mnist": Images(
        name="fashion_mnist",
        label="Fashion-MNIST",
        title="70,000 clothing photographs, 28x28 greyscale, 10 classes",
        licence="MIT",
        source="https://github.com/zalandoresearch/fashion-mnist",
        classes=(
            "t_shirt_top", "trouser", "pullover", "dress", "coat",
            "sandal", "shirt", "sneaker", "bag", "ankle_boot",
        ),
        splits=("train", "test"),
        basket_size=IMAGE_BASKET,
        base="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    ),
    "kmnist": Images(
        name="kmnist",
        label="KMNIST",
        title="70,000 handwritten classical Japanese characters, 28x28 greyscale, 10 classes",
        licence="CC BY-SA 4.0",
        source="https://github.com/rois-codh/kmnist",
        classes=("o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"),
        splits=("train", "test"),
        basket_size=IMAGE_BASKET,
        base="http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
    ),
    "cifar10": CIFAR(
        name="cifar10",
        label="CIFAR-10",
        title="60,000 photographs, 32x32 colour, 10 classes",
        licence="no formal licence; Krizhevsky asks that the tech report be cited",
        source="https://www.cs.toronto.edu/~kriz/cifar.html",
        classes=(
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ),
        splits=("train", "test"),
        basket_size=IMAGE_BASKET,
        archive="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        files={
            "train": tuple(f"data_batch_{batch}.bin" for batch in range(1, 6)),
            "test": ("test_batch.bin",),
        },
        meta="batches.meta.txt",
    ),
    "cifar100": CIFAR(
        name="cifar100",
        label="CIFAR-100",
        title="60,000 photographs, 32x32 colour, 100 classes in 20 superclasses",
        licence="no formal licence; Krizhevsky asks that the tech report be cited",
        source="https://www.cs.toronto.edu/~kriz/cifar.html",
        classes=(),
        splits=("train", "test"),
        basket_size=IMAGE_BASKET,
        archive="https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
        files={"train": ("train.bin",), "test": ("test.bin",)},
        meta="fine_label_names.txt",
        coarse="coarse_label_names.txt",
    ),
    "iris": Table(
        name="iris",
        label="Iris",
        title="150 iris flowers measured four ways, 3 species",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/53/iris",
        classes=("setosa", "versicolor", "virginica"),
        url="https://archive.ics.uci.edu/static/public/53/iris.zip",
        member="iris.data",
        fields=(
            ("sepal_length", "d"),
            ("sepal_width", "d"),
            ("petal_length", "d"),
            ("petal_width", "d"),
            ("species", "label"),
        ),
        labels={"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2},
    ),
    "penguins": Table(
        name="penguins",
        label="Palmer penguins",
        title="344 penguins measured at Palmer Station, 3 species, with gaps",
        licence="CC0",
        source="https://allisonhorst.github.io/palmerpenguins/",
        classes=("adelie", "chinstrap", "gentoo"),
        url=(
            "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
            "main/inst/extdata/penguins.csv"
        ),
        header=True,
        fields=(
            ("species", "label"),
            ("island", "island"),
            ("bill_length_mm", "d"),
            ("bill_depth_mm", "d"),
            ("flipper_length_mm", "d"),
            ("body_mass_g", "d"),
            ("sex", "sex"),
            ("year", "i"),
        ),
        labels={"Adelie": 0, "Chinstrap": 1, "Gentoo": 2},
        codes={
            "island": {"Biscoe": 0, "Dream": 1, "Torgersen": 2},
            "sex": {"female": 0, "male": 1},
        },
    ),
    "covertype": Table(
        name="covertype",
        label="Covertype",
        title="581,012 patches of Colorado forest, 54 features, 7 cover types",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/31/covertype",
        classes=(
            "spruce_fir", "lodgepole_pine", "ponderosa_pine", "cottonwood_willow",
            "aspen", "douglas_fir", "krummholz",
        ),
        basket_size=IMAGE_BASKET,
        url="https://archive.ics.uci.edu/static/public/31/covertype.zip",
        member="covtype.data.gz",
        fields=_COVER_FIELDS,
        labels={str(kind): kind - 1 for kind in range(1, 8)},
    ),
    "emnist": Images(
        name="emnist",
        label="EMNIST",
        title="131,600 handwritten characters, 28x28 greyscale, 47 balanced classes",
        licence=(
            "NIST Special Database 19, a work of the US federal government; "
            "the EMNIST paper asks to be cited"
        ),
        source="https://www.nist.gov/itl/products-and-services/emnist-dataset",
        classes=_EMNIST_CLASSES,
        splits=("train", "test"),
        basket_size=IMAGE_BASKET,
        archive="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
        files=_EMNIST_FILES,
        shade="greyscale, laid out as NIST wrote it with rows and columns swapped",
    ),
    "fsdd": Audio(
        name="fsdd",
        label="Free Spoken Digit Dataset",
        title="3,000 recordings of spoken digits, 8 kHz mono, 6 speakers, 10 classes",
        licence="CC BY-SA 4.0",
        source="https://github.com/Jakobovski/free-spoken-digit-dataset",
        classes=tuple(str(digit) for digit in range(10)),
        basket_size=IMAGE_BASKET,
        archive=(
            "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/"
            "refs/heads/master.zip"
        ),
        folder="recordings",
        labels={str(digit): digit for digit in range(10)},
        speakers=("george", "jackson", "lucas", "nicolas", "theo", "yweweler"),
        rate=8000,
        samples=20000,
    ),
    "adult": Table(
        name="adult",
        label="Adult",
        title="48,842 census records, 14 features, 2 income classes, with gaps",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/2/adult",
        classes=("under_50k", "over_50k"),
        splits=("train", "test"),
        url="https://archive.ics.uci.edu/static/public/2/adult.zip",
        files={"train": "adult.data", "test": "adult.test"},
        comment="|",
        fields=_ADULT_FIELDS,
        labels={"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1},
        codes=_ADULT_CODES,
    ),
    "mushroom": Table(
        name="mushroom",
        label="Mushroom",
        title="8,124 mushrooms described 22 ways, edible or poisonous, with gaps",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/73/mushroom",
        classes=("edible", "poisonous"),
        url="https://archive.ics.uci.edu/static/public/73/mushroom.zip",
        member="agaricus-lepiota.data",
        fields=(("edibility", "label"), *((name, name) for name, _ in _MUSHROOM)),
        labels={"e": 0, "p": 1},
        codes=dict(_MUSHROOM),
    ),
    "letter": Table(
        name="letter",
        label="Letter Recognition",
        title="20,000 printed capitals measured 16 ways, 26 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/59/letter+recognition",
        classes=tuple(chr(letter) for letter in range(ord("a"), ord("z") + 1)),
        url="https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
        member="letter-recognition.data",
        fields=_LETTER_FIELDS,
        labels={chr(ord("A") + at): at for at in range(26)},
    ),
    "digits": Table(
        name="digits",
        label="Optical Digits",
        title="5,620 handwritten digits as 8x8 counts of ink, 10 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        classes=tuple(str(digit) for digit in range(10)),
        splits=("train", "test"),
        url=(
            "https://archive.ics.uci.edu/static/public/80/"
            "optical+recognition+of+handwritten+digits.zip"
        ),
        files={"train": "optdigits.tra", "test": "optdigits.tes"},
        fields=_DIGIT_FIELDS,
        labels={str(digit): digit for digit in range(10)},
    ),
    "wine": Table(
        name="wine",
        label="Wine",
        title="178 wines analysed 13 ways, 3 cultivars",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/109/wine",
        classes=("cultivar_1", "cultivar_2", "cultivar_3"),
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        fields=_WINE_FIELDS,
        labels={"1": 0, "2": 1, "3": 2},
    ),
    "breast_cancer": Table(
        name="breast_cancer",
        label="Breast Cancer Wisconsin",
        title="569 cell-nucleus images measured 30 ways, benign or malignant",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        classes=("benign", "malignant"),
        url="https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        member="wdbc.data",
        fields=_WDBC_FIELDS,
        labels={"B": 0, "M": 1},
    ),
    "dry_bean": Table(
        name="dry_bean",
        label="Dry Bean",
        title="13,611 beans measured 16 ways from photographs, 7 varieties",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
        classes=("seker", "barbunya", "bombay", "cali", "horoz", "sira", "dermason"),
        url="https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
        member="DryBeanDataset/Dry_Bean_Dataset.arff",
        arff=True,
        fields=_BEAN_FIELDS,
        labels={
            "SEKER": 0, "BARBUNYA": 1, "BOMBAY": 2, "CALI": 3,
            "HOROZ": 4, "SIRA": 5, "DERMASON": 6,
        },
    ),
    "miniboone": Matrix(
        name="miniboone",
        label="MiniBooNE",
        title="130,064 particle-identification events measured 50 ways, signal or background",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification",
        classes=("electron_neutrino", "muon_neutrino"),
        basket_size=IMAGE_BASKET,
        url="https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip",
        files={"all": "MiniBooNE_PID.txt"},
        width=50,
        counts=True,
    ),
    "har": Matrix(
        name="har",
        label="Human Activity Recognition",
        title="10,299 windows of phone accelerometer and gyroscope, 561 features, 6 activities",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        classes=_HAR_ACTIVITIES,
        splits=("train", "test"),
        basket_size=IMAGE_BASKET,
        url=(
            "https://archive.ics.uci.edu/static/public/240/"
            "human+activity+recognition+using+smartphones.zip"
        ),
        inner="UCI HAR Dataset.zip",
        files={split: f"UCI HAR Dataset/{split}/X_{split}.txt" for split in ("train", "test")},
        label_files={
            split: f"UCI HAR Dataset/{split}/y_{split}.txt" for split in ("train", "test")
        },
        labels={str(at + 1): at for at in range(6)},
        beside={
            "subject": {
                split: f"UCI HAR Dataset/{split}/subject_{split}.txt"
                for split in ("train", "test")
            }
        },
        width=561,
    ),
    "semeion": Matrix(
        name="semeion",
        label="Semeion",
        title="1,593 handwritten digits as 16x16 black and white, 10 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/235/semeion+handwritten+digit",
        classes=tuple(str(digit) for digit in range(10)),
        basket_size=IMAGE_BASKET,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data",
        files={"all": ""},
        width=256,
        column="image",
        kind="B",
        onehot=10,
    ),
    "sms_spam": Table(
        name="sms_spam",
        label="SMS Spam Collection",
        title="5,574 text messages, ham or spam",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        classes=("ham", "spam"),
        url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        member="SMSSpamCollection",
        delimiter="\t",
        quoted=False,
        text_size=1024,
        fields=(("kind", "label"), ("message", "text")),
        labels={"ham": 0, "spam": 1},
    ),
    "wine_quality": Table(
        name="wine_quality",
        label="Wine Quality",
        title="6,497 Portuguese wines analysed 11 ways and scored out of ten",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/186/wine+quality",
        classes=tuple(f"quality_{score}" for score in range(3, 10)),
        splits=("red", "white"),
        url="https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        files={colour: f"winequality-{colour}.csv" for colour in ("red", "white")},
        delimiter=";",
        header=True,
        fields=_WINE_QUALITY_FIELDS,
        labels={str(score): score - 3 for score in range(3, 10)},
    ),
    "spambase": Table(
        name="spambase",
        label="Spambase",
        title="4,601 e-mails counted 57 ways, spam or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/94/spambase",
        classes=("not_spam", "spam"),
        url="https://archive.ics.uci.edu/static/public/94/spambase.zip",
        member="spambase.data",
        fields=_SPAM_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "ionosphere": Table(
        name="ionosphere",
        label="Ionosphere",
        title="351 radar returns from the ionosphere measured 34 ways, good or bad",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/52/ionosphere",
        classes=("good", "bad"),
        url="https://archive.ics.uci.edu/static/public/52/ionosphere.zip",
        member="ionosphere.data",
        fields=_IONOSPHERE_FIELDS,
        labels={"g": 0, "b": 1},
    ),
    "glass": Table(
        name="glass",
        label="Glass Identification",
        title="214 fragments of glass measured 9 ways, 6 kinds",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/42/glass+identification",
        classes=(
            "building_windows_float", "building_windows_nonfloat", "vehicle_windows_float",
            "containers", "tableware", "headlamps",
        ),
        url="https://archive.ics.uci.edu/static/public/42/glass+identification.zip",
        member="glass.data",
        fields=_GLASS_FIELDS,
        labels={"1": 0, "2": 1, "3": 2, "5": 3, "6": 4, "7": 5},
    ),
    "abalone": Table(
        name="abalone",
        label="Abalone",
        title="4,177 abalone measured 8 ways and counted for rings, 3 sexes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/1/abalone",
        classes=("male", "female", "infant"),
        url="https://archive.ics.uci.edu/static/public/1/abalone.zip",
        member="abalone.data",
        fields=_ABALONE_FIELDS,
        labels={"M": 0, "F": 1, "I": 2},
    ),
    "banknote": Table(
        name="banknote",
        label="Banknote Authentication",
        title="1,372 photographed banknotes measured 4 ways, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/267/banknote+authentication",
        classes=("class_0", "class_1"),
        url="https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        member="data_banknote_authentication.txt",
        fields=_BANKNOTE_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "seeds": Table(
        name="seeds",
        label="Seeds",
        title="210 wheat kernels measured 7 ways, 3 varieties",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/236/seeds",
        classes=("kama", "rosa", "canadian"),
        url="https://archive.ics.uci.edu/static/public/236/seeds.zip",
        member="seeds_dataset.txt",
        delimiter=None,
        fields=_SEED_FIELDS,
        labels={"1": 0, "2": 1, "3": 2},
    ),
    "magic": Table(
        name="magic",
        label="MAGIC Gamma Telescope",
        title="19,020 air showers seen by a Cherenkov telescope, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
        classes=("gamma", "hadron"),
        url="https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
        member="magic04.data",
        fields=_MAGIC_FIELDS,
        labels={"g": 0, "h": 1},
    ),
    "htru2": Table(
        name="htru2",
        label="HTRU2",
        title="17,898 pulsar candidates from a radio survey, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/372/htru2",
        classes=("not_pulsar", "pulsar"),
        url="https://archive.ics.uci.edu/static/public/372/htru2.zip",
        member="HTRU_2.csv",
        fields=_HTRU_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "auto_mpg": Table(
        name="auto_mpg",
        label="Auto MPG",
        title="398 cars of the 1970s and how far they went on a gallon",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/9/auto+mpg",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/9/auto+mpg.zip",
        member="auto-mpg.data",
        delimiter=None,
        tail=9,
        text_size=48,
        fields=_MPG_FIELDS,
    ),
    "bike_sharing": Table(
        name="bike_sharing",
        label="Bike Sharing",
        title="17,379 hours of a bicycle hire scheme, and how many were taken out",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        member="hour.csv",
        header=True,
        fields=_BIKE_FIELDS,
    ),
    "energy_efficiency": Table(
        name="energy_efficiency",
        label="Energy Efficiency",
        title="768 simulated buildings and the heating and cooling they need",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/242/energy+efficiency",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
        member="ENB2012_data.xlsx",
        xlsx=True,
        header=True,
        fields=_ENERGY_FIELDS,
    ),
    "real_estate": Table(
        name="real_estate",
        label="Real Estate Valuation",
        title="414 flats sold in Taipei and what a unit of floor cost",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip",
        member="Real estate valuation data set.xlsx",
        xlsx=True,
        header=True,
        fields=_ESTATE_FIELDS,
    ),
    "student": Table(
        name="student",
        label="Student Performance",
        title="1,044 pupils, 32 answers each, and the mark they finished on",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/320/student+performance",
        classes=(),
        splits=("maths", "portuguese"),
        url="https://archive.ics.uci.edu/static/public/320/student+performance.zip",
        inner="student.zip",
        files={"maths": "student-mat.csv", "portuguese": "student-por.csv"},
        delimiter=";",
        header=True,
        fields=_STUDENT_FIELDS,
        codes=_STUDENT_CODES,
    ),
    "heart_disease": Table(
        name="heart_disease",
        label="Heart Disease",
        title="920 patients from four hospitals, 5 degrees of narrowed arteries",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/45/heart+disease",
        classes=("none", "stage_1", "stage_2", "stage_3", "stage_4"),
        splits=("cleveland", "hungary", "switzerland", "long_beach"),
        url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        files={
            "cleveland": "processed.cleveland.data",
            "hungary": "processed.hungarian.data",
            "switzerland": "processed.switzerland.data",
            "long_beach": "processed.va.data",
        },
        fields=_HEART_FIELDS,
        labels={str(stage): stage for stage in range(5)},
    ),
    "car_evaluation": Table(
        name="car_evaluation",
        label="Car Evaluation",
        title="1,728 cars described six ways and judged acceptable or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/19/car+evaluation",
        classes=("unacceptable", "acceptable", "good", "very_good"),
        url="https://archive.ics.uci.edu/static/public/19/car+evaluation.zip",
        member="car.data",
        fields=_CAR_FIELDS,
        labels={"unacc": 0, "acc": 1, "good": 2, "vgood": 3},
        codes=_CAR_CODES,
    ),
    "yeast": Table(
        name="yeast",
        label="Yeast",
        title="1,484 yeast proteins measured 8 ways, 10 places in the cell",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/110/yeast",
        classes=(
            "cytosol", "nucleus", "mitochondria", "membrane_no_signal",
            "membrane_uncleaved_signal", "membrane_cleaved_signal", "extracellular",
            "vacuole", "peroxisome", "endoplasmic_reticulum",
        ),
        url="https://archive.ics.uci.edu/static/public/110/yeast.zip",
        member="yeast.data",
        delimiter=None,
        text_size=16,
        fields=_YEAST_FIELDS,
        labels=_YEAST_SITES,
    ),
    "airfoil": Table(
        name="airfoil",
        label="Airfoil Self-Noise",
        title="1,503 wind tunnel runs and how loud the aerofoil was",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/291/airfoil+self+noise",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip",
        member="airfoil_self_noise.dat",
        delimiter=None,
        fields=_AIRFOIL_FIELDS,
    ),
    "automobile": Table(
        name="automobile",
        label="Automobile",
        title="205 cars imported into America in 1985 and what they cost",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/10/automobile",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/10/automobile.zip",
        member="automobile/imports-85.data",
        text_size=16,
        fields=_AUTOMOBILE_FIELDS,
        codes=_AUTOMOBILE_CODES,
    ),
    "balance_scale": Table(
        name="balance_scale",
        label="Balance Scale",
        title="625 balances and which way each one tips, 3 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/12/balance+scale",
        classes=("balanced", "left", "right"),
        url="https://archive.ics.uci.edu/static/public/12/balance+scale.zip",
        member="balance-scale.data",
        fields=_BALANCE_FIELDS,
        labels={"B": 0, "L": 1, "R": 2},
    ),
    "bank_marketing": Table(
        name="bank_marketing",
        label="Bank Marketing",
        title="45,211 sales calls and whether the customer took the deposit",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/222/bank+marketing",
        classes=("no_deposit", "deposit"),
        url="https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        inner="bank.zip",
        member="bank-full.csv",
        delimiter=";",
        header=True,
        fields=_BANK_FIELDS,
        labels={"no": 0, "yes": 1},
        codes=_BANK_CODES,
    ),
    "blood_transfusion": Table(
        name="blood_transfusion",
        label="Blood Transfusion",
        title="748 blood donors and whether each gave again, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center",
        classes=("did_not_give", "gave"),
        url="https://archive.ics.uci.edu/static/public/176/blood+transfusion+service+center.zip",
        member="transfusion.data",
        header=True,
        fields=_BLOOD_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "climate_crashes": Table(
        name="climate_crashes",
        label="Climate Model Crashes",
        title="540 runs of an ocean model and whether each one finished",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/252/climate+model+simulation+crashes",
        classes=("crashed", "finished"),
        url="https://archive.ics.uci.edu/static/public/252/climate+model+simulation+crashes.zip",
        member="pop_failures.dat",
        delimiter=None,
        header=True,
        fields=_CLIMATE_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "computer_hardware": Table(
        name="computer_hardware",
        label="Computer Hardware",
        title="209 mainframes of the 1980s and how fast each one was",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/29/computer+hardware",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/29/computer+hardware.zip",
        member="machine.data",
        text_size=24,
        fields=_HARDWARE_FIELDS,
    ),
    "concrete_slump": Table(
        name="concrete_slump",
        label="Concrete Slump Test",
        title="103 concrete mixes and how each one flowed and held",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/182/concrete+slump+test",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/182/concrete+slump+test.zip",
        member="slump_test.data",
        header=True,
        fields=_SLUMP_FIELDS,
    ),
    "contraceptive": Table(
        name="contraceptive",
        label="Contraceptive Method Choice",
        title="1,473 Indonesian couples and what they used, 3 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice",
        classes=("none", "long_term", "short_term"),
        url="https://archive.ics.uci.edu/static/public/30/contraceptive+method+choice.zip",
        member="cmc.data",
        fields=_CMC_FIELDS,
        labels={"1": 0, "2": 1, "3": 2},
    ),
    "dermatology": Table(
        name="dermatology",
        label="Dermatology",
        title="366 patients with one of 6 red scaly skin diseases",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/33/dermatology",
        classes=(
            "psoriasis", "seboreic_dermatitis", "lichen_planus", "pityriasis_rosea",
            "chronic_dermatitis", "pityriasis_rubra_pilaris",
        ),
        url="https://archive.ics.uci.edu/static/public/33/dermatology.zip",
        member="dermatology.data",
        fields=_DERM_FIELDS,
        labels={str(which): which - 1 for which in range(1, 7)},
    ),
    "diabetes_risk": Table(
        name="diabetes_risk",
        label="Early Stage Diabetes Risk",
        title="520 patients asked about 14 symptoms, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset",
        classes=("negative", "positive"),
        url="https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip",
        member="diabetes_data_upload.csv",
        header=True,
        fields=_DIABETES_FIELDS,
        labels={"Negative": 0, "Positive": 1},
        codes={"sex": _SEXES, "yesno": _YES_NO_CAPS},
    ),
    "ecoli": Table(
        name="ecoli",
        label="Ecoli",
        title="336 E. coli proteins measured 7 ways, 8 places in the cell",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/39/ecoli",
        classes=(
            "cytoplasm", "inner_membrane_no_signal", "periplasm",
            "inner_membrane_uncleaved_signal", "outer_membrane",
            "outer_membrane_lipoprotein", "inner_membrane_lipoprotein",
            "inner_membrane_cleaved_signal",
        ),
        url="https://archive.ics.uci.edu/static/public/39/ecoli.zip",
        member="ecoli.data",
        delimiter=None,
        text_size=16,
        fields=_ECOLI_FIELDS,
        labels={
            name: at
            for at, name in enumerate(("cp", "im", "pp", "imU", "om", "omL", "imL", "imS"))
        },
    ),
    "fertility": Table(
        name="fertility",
        label="Fertility",
        title="100 men, 9 questions each, and their semen analysis",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/244/fertility",
        classes=("normal", "altered"),
        url="https://archive.ics.uci.edu/static/public/244/fertility.zip",
        member="fertility_Diagnosis.txt",
        fields=_FERTILITY_FIELDS,
        labels={"N": 0, "O": 1},
    ),
    "forest_fires": Table(
        name="forest_fires",
        label="Forest Fires",
        title="517 fires in a Portuguese park and how far each one spread",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/162/forest+fires",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/162/forest+fires.zip",
        member="forestfires.csv",
        header=True,
        fields=_FIRE_FIELDS,
        codes={"month": _MONTHS, "day": _DAYS},
    ),
    "garment_productivity": Table(
        name="garment_productivity",
        label="Garment Worker Productivity",
        title="1,197 team-days in a clothing factory and what each got done",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/597/productivity+prediction+of+garment+employees.zip",
        member="garments_worker_productivity.csv",
        header=True,
        dates="%m/%d/%Y",
        fields=_GARMENT_FIELDS,
        codes={
            "quarter": ("Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"),
            "department": ("finishing", "sweing"),
            "weekday": _WEEKDAYS,
        },
    ),
    "german_credit": Table(
        name="german_credit",
        label="Statlog German Credit",
        title="1,000 loan applications judged good or bad, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        classes=("good", "bad"),
        url="https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
        member="german.data",
        delimiter=None,
        fields=_GERMAN_FIELDS,
        labels={"1": 0, "2": 1},
        codes=_GERMAN_CODES,
    ),
    "haberman": Table(
        name="haberman",
        label="Haberman's Survival",
        title="306 breast cancer operations and who was alive 5 years on",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/43/haberman+s+survival",
        classes=("survived", "died"),
        url="https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip",
        member="haberman.data",
        fields=_HABERMAN_FIELDS,
        labels={"1": 0, "2": 1},
    ),
    "heart_failure": Table(
        name="heart_failure",
        label="Heart Failure Clinical Records",
        title="299 heart failure patients and who survived the follow-up",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records",
        classes=("survived", "died"),
        url="https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip",
        member="heart_failure_clinical_records_dataset.csv",
        header=True,
        fields=_FAILURE_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "hepatitis": Table(
        name="hepatitis",
        label="Hepatitis",
        title="155 hepatitis patients, 19 findings each, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/46/hepatitis",
        classes=("died", "lived"),
        url="https://archive.ics.uci.edu/static/public/46/hepatitis.zip",
        member="hepatitis.data",
        fields=_HEPATITIS_FIELDS,
        labels={"1": 0, "2": 1},
    ),
    "image_segmentation": Table(
        name="image_segmentation",
        label="Image Segmentation",
        title="2,310 patches of outdoor photographs, 7 things they are of",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/50/image+segmentation",
        classes=("brickface", "sky", "foliage", "cement", "window", "path", "grass"),
        splits=("train", "test"),
        url="https://archive.ics.uci.edu/static/public/50/image+segmentation.zip",
        files={"train": "segmentation.data", "test": "segmentation.test"},
        comment=";",
        header=True,
        fields=_SEGMENT_FIELDS,
        labels={
            name.upper(): at
            for at, name in enumerate(
                ("brickface", "sky", "foliage", "cement", "window", "path", "grass")
            )
        },
    ),
    "indian_liver": Table(
        name="indian_liver",
        label="Indian Liver Patient",
        title="583 patients from Andhra Pradesh, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset",
        classes=("liver_patient", "not_a_patient"),
        url="https://archive.ics.uci.edu/static/public/225/ilpd+indian+liver+patient+dataset.zip",
        member="Indian Liver Patient Dataset (ILPD).csv",
        fields=_ILPD_FIELDS,
        labels={"1": 0, "2": 1},
        codes={"sex": _SEXES},
    ),
    "liver_disorders": Table(
        name="liver_disorders",
        label="Liver Disorders",
        title="345 blood tests and the drinking to predict from them",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/60/liver+disorders",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/60/liver+disorders.zip",
        member="bupa.data",
        fields=_BUPA_FIELDS,
    ),
    "lymphography": Table(
        name="lymphography",
        label="Lymphography",
        title="148 lymph node X-rays read 18 ways, 4 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/63/lymphography",
        classes=("normal", "metastases", "malignant_lymphoma", "fibrosis"),
        url="https://archive.ics.uci.edu/static/public/63/lymphography.zip",
        member="lymphography.data",
        fields=_LYMPH_FIELDS,
        labels={str(which): which - 1 for which in range(1, 5)},
    ),
    "mammographic_mass": Table(
        name="mammographic_mass",
        label="Mammographic Mass",
        title="961 lumps seen on a mammogram, benign or malignant",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/161/mammographic+mass",
        classes=("benign", "malignant"),
        url="https://archive.ics.uci.edu/static/public/161/mammographic+mass.zip",
        member="mammographic_masses.data",
        fields=_MAMMOGRAM_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "maternal_health": Table(
        name="maternal_health",
        label="Maternal Health Risk",
        title="1,014 pregnancies seen in rural clinics, 3 degrees of risk",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/863/maternal+health+risk",
        classes=("low_risk", "mid_risk", "high_risk"),
        url="https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip",
        member="Maternal Health Risk Data Set.csv",
        header=True,
        fields=_MATERNAL_FIELDS,
        labels={"low risk": 0, "mid risk": 1, "high risk": 2},
    ),
    "nursery": Table(
        name="nursery",
        label="Nursery",
        title="12,960 nursery applications ranked 5 ways",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/76/nursery",
        classes=("not_recom", "recommend", "very_recom", "priority", "spec_prior"),
        url="https://archive.ics.uci.edu/static/public/76/nursery.zip",
        member="nursery.data",
        fields=_NURSERY_FIELDS,
        labels={
            name: at
            for at, name in enumerate(
                ("not_recom", "recommend", "very_recom", "priority", "spec_prior")
            )
        },
        codes=_NURSERY_CODES,
    ),
    "occupancy": Table(
        name="occupancy",
        label="Occupancy Detection",
        title="20,560 minutes in an office and whether anyone was in it",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/357/occupancy+detection",
        classes=("empty", "occupied"),
        splits=("train", "test", "second_test"),
        url="https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip",
        files={
            "train": "datatraining.txt",
            "test": "datatest.txt",
            "second_test": "datatest2.txt",
        },
        header=True,
        dates="%Y-%m-%d %H:%M:%S",
        fields=_OCCUPANCY_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "online_shoppers": Table(
        name="online_shoppers",
        label="Online Shoppers Intention",
        title="12,330 shopping sessions and which of them ended in a sale",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
        classes=("no_purchase", "purchase"),
        url="https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        member="online_shoppers_intention.csv",
        header=True,
        fields=_SHOPPER_FIELDS,
        labels={"FALSE": 0, "TRUE": 1},
        codes=_SHOPPER_CODES,
    ),
    "parkinsons": Table(
        name="parkinsons",
        label="Parkinsons",
        title="195 voice recordings, 22 measurements each, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/174/parkinsons",
        classes=("healthy", "parkinsons"),
        url="https://archive.ics.uci.edu/static/public/174/parkinsons.zip",
        member="parkinsons.data",
        header=True,
        text_size=24,
        fields=_PARKINSONS_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "parkinsons_telemonitoring": Table(
        name="parkinsons_telemonitoring",
        label="Parkinsons Telemonitoring",
        title="5,875 recordings made at home and the two scores to predict",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip",
        member="parkinsons_updrs.data",
        header=True,
        fields=_TELEMONITORING_FIELDS,
    ),
    "pendigits": Table(
        name="pendigits",
        label="Pen-Based Digits",
        title="10,992 digits written on a tablet, 8 points each, 10 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits",
        classes=tuple(str(digit) for digit in range(10)),
        splits=("train", "test"),
        url="https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip",
        files={"train": "pendigits.tra", "test": "pendigits.tes"},
        fields=_PENDIGIT_FIELDS,
        labels={str(digit): digit for digit in range(10)},
    ),
    "phishing": Table(
        name="phishing",
        label="Phishing Websites",
        title="11,055 websites scored 30 ways, phishing or legitimate",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/327/phishing+websites",
        classes=("phishing", "legitimate"),
        url="https://archive.ics.uci.edu/static/public/327/phishing+websites.zip",
        member="Training Dataset.arff",
        arff=True,
        fields=_PHISHING_FIELDS,
        labels={"-1": 0, "1": 1},
    ),
    "power_plant": Table(
        name="power_plant",
        label="Combined Cycle Power Plant",
        title="9,568 hours of a gas turbine and the power it put out",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip",
        member="CCPP/Folds5x2_pp.xlsx",
        xlsx=True,
        header=True,
        fields=_POWER_FIELDS,
    ),
    "qsar_aquatic": Table(
        name="qsar_aquatic",
        label="QSAR Aquatic Toxicity",
        title="546 chemicals and the dose that kills half the water fleas",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/505/qsar+aquatic+toxicity",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/505/qsar+aquatic+toxicity.zip",
        member="qsar_aquatic_toxicity.csv",
        delimiter=";",
        fields=_AQUATIC_FIELDS,
    ),
    "qsar_fish": Table(
        name="qsar_fish",
        label="QSAR Fish Toxicity",
        title="908 chemicals and the dose that kills half the minnows",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/504/qsar+fish+toxicity",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/504/qsar+fish+toxicity.zip",
        member="qsar_fish_toxicity.csv",
        delimiter=";",
        fields=_FISH_FIELDS,
    ),
    "raisin": Table(
        name="raisin",
        label="Raisin",
        title="900 raisins measured off a photograph, 2 varieties",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/850/raisin",
        classes=("kecimen", "besni"),
        url="https://archive.ics.uci.edu/static/public/850/raisin.zip",
        inner="Raisin_Dataset.zip",
        member="Raisin_Dataset/Raisin_Dataset.arff",
        arff=True,
        fields=_RAISIN_FIELDS,
        labels={"Kecimen": 0, "Besni": 1},
    ),
    "rice": Table(
        name="rice",
        label="Rice",
        title="3,810 grains of rice measured off a photograph, 2 varieties",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik",
        classes=("cammeo", "osmancik"),
        url="https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip",
        member="Rice_Cammeo_Osmancik.arff",
        arff=True,
        fields=_RICE_FIELDS,
        labels={"Cammeo": 0, "Osmancik": 1},
    ),
    "satellite": Table(
        name="satellite",
        label="Statlog Landsat Satellite",
        title="6,435 Landsat neighbourhoods of 9 pixels, 6 kinds of ground",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite",
        classes=(
            "red_soil", "cotton_crop", "grey_soil", "damp_grey_soil",
            "vegetation_stubble", "very_damp_grey_soil",
        ),
        splits=("train", "test"),
        url="https://archive.ics.uci.edu/static/public/146/statlog+landsat+satellite.zip",
        files={"train": "sat.trn", "test": "sat.tst"},
        delimiter=None,
        fields=_SATELLITE_FIELDS,
        labels={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "7": 5},
    ),
    "seismic": Table(
        name="seismic",
        label="Seismic Bumps",
        title="2,584 shifts in a Polish coal mine, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/266/seismic+bumps",
        classes=("no_bump", "bump"),
        url="https://archive.ics.uci.edu/static/public/266/seismic+bumps.zip",
        member="seismic-bumps.arff",
        arff=True,
        fields=_SEISMIC_FIELDS,
        labels={"0": 0, "1": 1},
        codes={"hazard": ("a", "b", "c", "d"), "shift": ("W", "N")},
    ),
    "servo": Table(
        name="servo",
        label="Servo",
        title="167 servomechanisms and how long each took to settle",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/87/servo",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/87/servo.zip",
        member="servo.data",
        fields=_SERVO_FIELDS,
        codes={"part": ("A", "B", "C", "D", "E")},
    ),
    "solar_flare": Table(
        name="solar_flare",
        label="Solar Flare",
        title="1,066 active regions on the sun, 7 modified Zurich classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/89/solar+flare",
        classes=tuple(f"class_{letter}" for letter in "abcdefh"),
        url="https://archive.ics.uci.edu/static/public/89/solar+flare.zip",
        member="flare.data2",
        delimiter=None,
        comment="*",
        fields=_FLARE_FIELDS,
        labels={letter.upper(): at for at, letter in enumerate("abcdefh")},
        codes={"spots": ("X", "R", "S", "A", "H", "K"), "spread": ("X", "O", "I", "C")},
    ),
    "sonar": Table(
        name="sonar",
        label="Sonar",
        title="208 sonar returns off a rock or a mine, 60 bands each",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks",
        classes=("rock", "mine"),
        url="https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip",
        member="sonar.all-data",
        fields=_SONAR_FIELDS,
        labels={"R": 0, "M": 1},
    ),
    "soybean": Table(
        name="soybean",
        label="Soybean (Large)",
        title="683 diseased soybean plants, 19 diseases",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/90/soybean+large",
        classes=tuple(
            name.replace("&", "and").replace("2-4-d", "two_four_d").replace("-", "_")
            for name in _SOYBEAN_DISEASES
        ),
        splits=("train", "test"),
        url="https://archive.ics.uci.edu/static/public/90/soybean+large.zip",
        files={"train": "soybean-large.data", "test": "soybean-large.test"},
        fields=_SOYBEAN_FIELDS,
        labels={name: at for at, name in enumerate(_SOYBEAN_DISEASES)},
    ),
    "statlog_heart": Table(
        name="statlog_heart",
        label="Statlog Heart",
        title="270 patients from the Cleveland heart study, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/145/statlog+heart",
        classes=("absent", "present"),
        url="https://archive.ics.uci.edu/static/public/145/statlog+heart.zip",
        member="heart.dat",
        delimiter=None,
        fields=_STATLOG_HEART_FIELDS,
        labels={"1": 0, "2": 1},
    ),
    "steel_industry": Table(
        name="steel_industry",
        label="Steel Industry Energy",
        title="35,040 quarter hours in a steel plant, 3 kinds of load",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption",
        classes=("light_load", "medium_load", "maximum_load"),
        url="https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip",
        member="Steel_industry_data.csv",
        header=True,
        dates="%d/%m/%Y %H:%M",
        fields=_STEEL_FIELDS,
        labels={"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2},
        codes={"week": ("Weekday", "Weekend"), "weekday": _WEEKDAYS},
    ),
    "tic_tac_toe": Table(
        name="tic_tac_toe",
        label="Tic-Tac-Toe Endgame",
        title="958 finished games and whether x had won, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame",
        classes=("x_lost", "x_won"),
        url="https://archive.ics.uci.edu/static/public/101/tic+tac+toe+endgame.zip",
        member="tic-tac-toe.data",
        fields=_TICTACTOE_FIELDS,
        labels={"negative": 0, "positive": 1},
        codes={"square": {"x": 1, "o": -1, "b": 0}},
    ),
    "vertebral_column": Table(
        name="vertebral_column",
        label="Vertebral Column",
        title="310 lower spines measured 6 ways, 3 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/212/vertebral+column",
        classes=("disc_hernia", "spondylolisthesis", "normal"),
        url="https://archive.ics.uci.edu/static/public/212/vertebral+column.zip",
        member="column_3C.dat",
        delimiter=None,
        fields=_VERTEBRAL_FIELDS,
        labels={"DH": 0, "SL": 1, "NO": 2},
    ),
    "wifi_localisation": Table(
        name="wifi_localisation",
        label="Wireless Indoor Localisation",
        title="2,000 readings of 7 wifi points, taken in 4 rooms",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization",
        classes=tuple(f"room_{at}" for at in range(1, 5)),
        url="https://archive.ics.uci.edu/static/public/422/wireless+indoor+localization.zip",
        member="wifi_localization.txt",
        delimiter=None,
        fields=_WIFI_FIELDS,
        labels={str(at): at - 1 for at in range(1, 5)},
    ),
    "yacht": Table(
        name="yacht",
        label="Yacht Hydrodynamics",
        title="308 towing tank runs and the resistance each hull made",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip",
        member="yacht_hydrodynamics.data",
        delimiter=None,
        fields=_YACHT_FIELDS,
    ),
    "zoo": Table(
        name="zoo",
        label="Zoo",
        title="101 animals described 16 ways, 7 kinds of animal",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/111/zoo",
        classes=(
            "mammal", "bird", "reptile", "fish", "amphibian", "insect", "invertebrate",
        ),
        url="https://archive.ics.uci.edu/static/public/111/zoo.zip",
        member="zoo.data",
        text_size=16,
        fields=_ZOO_FIELDS,
        labels={str(which): which - 1 for which in range(1, 8)},
    ),
    "absenteeism_at_work": Table(
        name="absenteeism_at_work",
        label="Absenteeism at work",
        title="740 absences at a Brazilian courier firm and the hours each one lost",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/445/absenteeism+at+work",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/445/data.csv",
        header=True,
        fields=_ABSENTEEISM_AT_WORK_FIELDS,
    ),
    "acute_inflammations": Table(
        name="acute_inflammations",
        label="Acute Inflammations",
        title="120 patients with urinary symptoms, judged for inflammation of the bladder",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/184/acute+inflammations",
        classes=("no", "yes"),
        url="https://archive.ics.uci.edu/static/public/184/data.csv",
        header=True,
        fields=_ACUTE_INFLAMMATIONS_FIELDS,
        labels={"no": 0, "yes": 1},
        codes={
            "nausea": ("no", "yes"),
        },
    ),
    "aids_clinical_trials": Table(
        name="aids_clinical_trials",
        label="AIDS Clinical Trials Group Study 175",
        title="2,139 people on HIV treatment and whose illness went on progressing",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175",
        classes=("no_event", "progressed"),
        url="https://archive.ics.uci.edu/static/public/890/data.csv",
        header=True,
        fields=_AIDS_CLINICAL_TRIALS_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "android_permissions": Table(
        name="android_permissions",
        label="NATICUSdroid (Android Permissions)",
        title="29,332 Android apps described by the permissions they ask for, benign or malware",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/722/naticusdroid+android+permissions+dataset",
        classes=("benign", "malware"),
        url="https://archive.ics.uci.edu/static/public/722/data.csv",
        header=True,
        fields=_ANDROID_PERMISSIONS_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "annealing": Table(
        name="annealing",
        label="Annealing",
        title="898 steel coils, their chemistry, and the annealing class each took",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/3/annealing",
        classes=("class_1", "class_2", "class_3", "class_5", "class_u"),
        url="https://archive.ics.uci.edu/static/public/3/data.csv",
        header=True,
        fields=_ANNEALING_FIELDS,
        labels={"1": 0, "2": 1, "3": 2, "5": 3, "U": 4},
        codes={
            "famiily": ("TN", "ZS"),
            "product_type": ("C",),
            "steel": ("A", "K", "M", "R", "S", "V", "W"),
            "temper_rolling": ("T",),
            "condition": ("A", "S"),
            "non_ageing": ("N",),
            "surface_finish": ("P",),
            "surface_quality": ("D", "E", "F", "G"),
            "bc": ("Y",),
            "bw_me": ("B", "M"),
            "blue_bright_varn_clean": ("B", "C", "V"),
            "shape": ("COIL", "SHEET"),
            "oil": ("N", "Y"),
        },
    ),
    "appliances_energy": Table(
        name="appliances_energy",
        label="Appliances Energy Prediction",
        title="19,735 ten-minute readings of a low-energy house and the watt-hours it drew",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/374/data.csv",
        header=True,
        text_size=18,
        fields=_APPLIANCES_ENERGY_FIELDS,
    ),
    "auction_verification": Table(
        name="auction_verification",
        label="Auction Verification",
        title="2,043 runs of a model checker over simulated spectrum auctions, verified or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/713/auction+verification",
        classes=("false", "true"),
        url="https://archive.ics.uci.edu/static/public/713/data.csv",
        header=True,
        fields=_AUCTION_VERIFICATION_FIELDS,
        labels={"False": 0, "True": 1},
    ),
    "audiology": Table(
        name="audiology",
        label="Audiology (Standardized)",
        title="200 hearing tests and the diagnosis each led to, 24 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/8/audiology+standardized",
        classes=(
            "acoustic_neuroma", "bells_palsy", "cochlear_age", "cochlear_age_and_noise",
            "cochlear_age_plus_poss_menieres", "cochlear_noise_and_heredity",
            "cochlear_poss_noise", "cochlear_unknown", "conductive_discontinuity",
            "conductive_fixation", "mixed_cochlear_age_fixation",
            "mixed_cochlear_age_otitis_media", "mixed_cochlear_age_s_om",
            "mixed_cochlear_unk_discontinuity", "mixed_cochlear_unk_fixation",
            "mixed_cochlear_unk_ser_om", "mixed_poss_central_om", "mixed_poss_noise_om",
            "normal_ear", "otitis_media", "poss_central", "possible_brainstem_disorder",
            "possible_menieres", "retrocochlear_unknown",
        ),
        url="https://archive.ics.uci.edu/static/public/8/data.csv",
        header=True,
        text_size=4,
        fields=_AUDIOLOGY_FIELDS,
        labels={
            "acoustic_neuroma": 0, "bells_palsy": 1, "cochlear_age": 2,
            "cochlear_age_and_noise": 3, "cochlear_age_plus_poss_menieres": 4,
            "cochlear_noise_and_heredity": 5, "cochlear_poss_noise": 6, "cochlear_unknown": 7,
            "conductive_discontinuity": 8, "conductive_fixation": 9,
            "mixed_cochlear_age_fixation": 10, "mixed_cochlear_age_otitis_media": 11,
            "mixed_cochlear_age_s_om": 12, "mixed_cochlear_unk_discontinuity": 13,
            "mixed_cochlear_unk_fixation": 14, "mixed_cochlear_unk_ser_om": 15,
            "mixed_poss_central_om": 16, "mixed_poss_noise_om": 17, "normal_ear": 18,
            "otitis_media": 19, "poss_central": 20, "possible_brainstem_disorder": 21,
            "possible_menieres": 22, "retrocochlear_unknown": 23,
        },
        codes={
            "age_gt_60": ("f", "t"),
            "air": ("mild", "moderate", "normal", "profound", "severe"),
            "ar_c": ("absent", "elevated", "normal"),
            "bone": ("mild", "moderate", "normal", "unmeasured"),
            "bser": ("degraded", "normal"),
            "m_cond_lt_1k": ("f",),
            "speech": ("good", "normal", "poor", "unmeasured", "very_good", "very_poor"),
            "tymp": ("a", "ad", "as", "b", "c"),
        },
    ),
    "autism_screening_adult": Table(
        name="autism_screening_adult",
        label="Autism Screening Adult",
        title="704 adults put through the autism screening questionnaire, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/426/autism+screening+adult",
        classes=("no", "yes"),
        url="https://archive.ics.uci.edu/static/public/426/data.csv",
        header=True,
        text_size=22,
        fields=_AUTISM_SCREENING_ADULT_FIELDS,
        labels={"NO": 0, "YES": 1},
        codes={
            "gender": ("f", "m"),
            "ethnicity": (
                "'Middle Eastern '", "'South Asian'", "Asian", "Black", "Hispanic", "Latino",
                "Others", "Pasifika", "Turkish", "White-European", "others",
            ),
            "jaundice": ("no", "yes"),
            "age_desc": ("'18 and more'",),
            "relation": ("'Health care professional'", "Others", "Parent", "Relative", "Self"),
        },
    ),
    "autism_screening_child": Table(
        name="autism_screening_child",
        label="Autistic Spectrum Disorder Screening Data for Children",
        title="292 children put through the autism screening questionnaire, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children",
        classes=("no", "yes"),
        url="https://archive.ics.uci.edu/static/public/419/data.csv",
        header=True,
        text_size=23,
        fields=_AUTISM_SCREENING_CHILD_FIELDS,
        labels={"NO": 0, "YES": 1},
        codes={
            "gender": ("f", "m"),
            "ethnicity": (
                "'Middle Eastern '", "'South Asian'", "Asian", "Black", "Hispanic", "Latino",
                "Others", "Pasifika", "Turkish", "White-European",
            ),
            "jaundice": ("no", "yes"),
            "age_desc": ("'4-11 years'",),
            "relation": ("'Health care professional'", "Parent", "Relative", "Self", "self"),
        },
    ),
    "beijing_pm25": Table(
        name="beijing_pm25",
        label="Beijing PM2.5",
        title="43,824 hours of Beijing weather and the fine particles in the air",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/381/data.csv",
        header=True,
        fields=_BEIJING_PM25_FIELDS,
        codes={
            "cbwd": ("NE", "NW", "SE", "cv"),
        },
    ),
    "bone_marrow_transplant": Table(
        name="bone_marrow_transplant",
        label="Bone marrow transplant: children",
        title="187 children given a bone marrow transplant, and who survived it",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children",
        classes=("alive", "died"),
        url="https://archive.ics.uci.edu/static/public/565/data.csv",
        header=True,
        fields=_BONE_MARROW_TRANSPLANT_FIELDS,
        labels={"0": 0, "1": 1},
        codes={
            "disease": ("ALL", "AML", "chronic", "lymphoma", "nonmalignant"),
        },
    ),
    "breast_cancer_coimbra": Table(
        name="breast_cancer_coimbra",
        label="Breast Cancer Coimbra",
        title="116 blood samples from Coimbra, healthy women and cancer patients",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra",
        classes=("healthy", "patient"),
        url="https://archive.ics.uci.edu/static/public/451/data.csv",
        header=True,
        fields=_BREAST_CANCER_COIMBRA_FIELDS,
        labels={"1": 0, "2": 1},
    ),
    "breast_cancer_original": Table(
        name="breast_cancer_original",
        label="Breast Cancer Wisconsin (Original)",
        title="699 breast tissue samples scored nine ways, benign or malignant",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original",
        classes=("benign", "malignant"),
        url="https://archive.ics.uci.edu/static/public/15/data.csv",
        header=True,
        fields=_BREAST_CANCER_ORIGINAL_FIELDS,
        labels={"2": 0, "4": 1},
    ),
    "breast_cancer_prognostic": Table(
        name="breast_cancer_prognostic",
        label="Breast Cancer Wisconsin (Prognostic)",
        title="198 breast cancer operations and whether the disease came back",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic",
        classes=("no_recurrence", "recurrence"),
        url="https://archive.ics.uci.edu/static/public/16/data.csv",
        header=True,
        fields=_BREAST_CANCER_PROGNOSTIC_FIELDS,
        labels={"N": 0, "R": 1},
    ),
    "breast_cancer_recurrence": Table(
        name="breast_cancer_recurrence",
        label="Breast Cancer",
        title="286 breast cancer cases from Ljubljana and which of them came back",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/14/breast+cancer",
        classes=("no_recurrence_events", "recurrence_events"),
        url="https://archive.ics.uci.edu/static/public/14/data.csv",
        header=True,
        fields=_BREAST_CANCER_RECURRENCE_FIELDS,
        labels={"no-recurrence-events": 0, "recurrence-events": 1},
        codes={
            "age": ("20-29", "30-39", "40-49", "50-59", "60-69", "70-79"),
            "menopause": ("ge40", "lt40", "premeno"),
            "tumor_size": (
                "0-4", "14-Oct", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                "50-54", "9-May",
            ),
            "inv_nodes": ("0-2", "11-Sep", "14-Dec", "15-17", "24-26", "5-Mar", "8-Jun"),
            "node_caps": ("no", "yes"),
            "breast": ("left", "right"),
            "breast_quad": ("central", "left_low", "left_up", "right_low", "right_up"),
        },
    ),
    "cardiotocography": Table(
        name="cardiotocography",
        label="Cardiotocography",
        title="2,126 foetal heart traces read by three obstetricians, 3 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/193/cardiotocography",
        classes=("normal", "suspect", "pathologic"),
        url="https://archive.ics.uci.edu/static/public/193/data.csv",
        header=True,
        fields=_CARDIOTOCOGRAPHY_FIELDS,
        labels={"1": 0, "2": 1, "3": 2},
    ),
    "cdc_diabetes": Table(
        name="cdc_diabetes",
        label="CDC Diabetes Health Indicators",
        title="253,680 answers to a CDC health survey, and who among them had diabetes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
        classes=("no_diabetes", "diabetes"),
        url="https://archive.ics.uci.edu/static/public/891/data.csv",
        header=True,
        fields=_CDC_DIABETES_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "census_income_kdd": Table(
        name="census_income_kdd",
        label="Census-Income (KDD)",
        title="199,523 census records from the 1990s and who earned over $50,000",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/117/census+income+kdd",
        classes=("under_50k", "over_50k"),
        url="https://archive.ics.uci.edu/static/public/117/data.csv",
        header=True,
        text_size=47,
        fields=_CENSUS_INCOME_KDD_FIELDS,
        labels={"-50000": 0, "50000+.": 1},
        codes={
            "aclswkr": (
                "Federal government", "Local government", "Never worked", "Not in universe",
                "Private", "Self-employed-incorporated", "Self-employed-not incorporated",
                "State government", "Without pay",
            ),
            "ahscol": ("College or university", "High school", "Not in universe"),
            "amaritl": (
                "Divorced", "Married-A F spouse present", "Married-civilian spouse present",
                "Married-spouse absent", "Never married", "Separated", "Widowed",
            ),
            "arace": (
                "Amer Indian Aleut or Eskimo", "Asian or Pacific Islander", "Black", "Other",
                "White",
            ),
            "areorgn": (
                "All other", "Central or South American", "Chicano", "Cuban", "Do not know",
                "Mexican (Mexicano)", "Mexican-American", "Other Spanish", "Puerto Rican",
            ),
            "asex": ("Female", "Male"),
            "aunmem": ("No", "Not in universe", "Yes"),
            "auntype": (
                "Job leaver", "Job loser - on layoff", "New entrant", "Not in universe",
                "Other job loser", "Re-entrant",
            ),
            "filestat": (
                "Head of household", "Joint both 65+", "Joint both under 65",
                "Joint one under 65 & one 65+", "Nonfiler", "Single",
            ),
            "grinreg": ("Abroad", "Midwest", "Northeast", "Not in universe", "South", "West"),
            "migmtr1": (
                "Abroad to MSA", "Abroad to nonMSA", "MSA to MSA", "MSA to nonMSA",
                "NonMSA to MSA", "NonMSA to nonMSA", "Nonmover", "Not identifiable",
                "Not in universe",
            ),
            "migmtr3": (
                "Abroad", "Different county same state", "Different division same region",
                "Different region", "Different state same division", "Nonmover", "Not in universe",
                "Same county",
            ),
            "migmtr4": (
                "Abroad", "Different county same state", "Different state in Midwest",
                "Different state in Northeast", "Different state in South",
                "Different state in West", "Nonmover", "Not in universe", "Same county",
            ),
            "migsame": ("No", "Not in universe under 1 year old", "Yes"),
            "parent": (
                "Both parents present", "Father only present", "Mother only present",
                "Neither parent present", "Not in universe",
            ),
        },
    ),
    "cervical_cancer_behaviour": Table(
        name="cervical_cancer_behaviour",
        label="Cervical Cancer Behavior Risk",
        title="72 women answering a behaviour questionnaire, with and without cervical cancer",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/537/cervical+cancer+behavior+risk",
        classes=("no_cancer", "cancer"),
        url="https://archive.ics.uci.edu/static/public/537/data.csv",
        header=True,
        fields=_CERVICAL_CANCER_BEHAVIOUR_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "cervical_cancer_risk": Table(
        name="cervical_cancer_risk",
        label="Cervical Cancer (Risk Factors)",
        title="858 cervical cancer screenings and what the biopsy found",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors",
        classes=("negative", "positive"),
        url="https://archive.ics.uci.edu/static/public/383/data.csv",
        header=True,
        fields=_CERVICAL_CANCER_RISK_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "challenger_o_rings": Table(
        name="challenger_o_rings",
        label="Challenger USA Space Shuttle O-Ring",
        title="23 shuttle launches, the temperature at each, and the O-rings that failed",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/92/challenger+usa+space+shuttle+o+ring",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/92/data.csv",
        header=True,
        fields=_CHALLENGER_O_RINGS_FIELDS,
    ),
    "chess_endgame": Table(
        name="chess_endgame",
        label="Chess (King-Rook vs. King)",
        title="28,056 king-and-rook against king-and-king positions, by the moves white needs",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/23/chess+king+rook+vs+king",
        classes=(
            "draw", "eight", "eleven", "fifteen", "five", "four", "fourteen", "nine", "one",
            "seven", "six", "sixteen", "ten", "thirteen", "three", "twelve", "two", "zero",
        ),
        url="https://archive.ics.uci.edu/static/public/23/data.csv",
        header=True,
        fields=_CHESS_ENDGAME_FIELDS,
        labels={
            "draw": 0, "eight": 1, "eleven": 2, "fifteen": 3, "five": 4, "four": 5, "fourteen": 6,
            "nine": 7, "one": 8, "seven": 9, "six": 10, "sixteen": 11, "ten": 12, "thirteen": 13,
            "three": 14, "twelve": 15, "two": 16, "zero": 17,
        },
        codes={
            "white_king_file": ("a", "b", "c", "d"),
            "white_rook_file": ("a", "b", "c", "d", "e", "f", "g", "h"),
        },
    ),
    "chronic_kidney_disease": Table(
        name="chronic_kidney_disease",
        label="Chronic Kidney Disease",
        title="400 patients tested for chronic kidney disease, 2 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease",
        classes=("ckd", "notckd"),
        url="https://archive.ics.uci.edu/static/public/336/data.csv",
        header=True,
        fields=_CHRONIC_KIDNEY_DISEASE_FIELDS,
        labels={"ckd": 0, "notckd": 1},
        codes={
            "rbc": ("abnormal", "normal"),
            "pcc": ("notpresent", "present"),
            "htn": ("no", "yes"),
            "appet": ("good", "poor"),
        },
    ),
    "communities_crime": Table(
        name="communities_crime",
        label="Communities and Crime",
        title="1,994 American communities described by the census, and the violent crime in each",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/183/communities+and+crime",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/183/data.csv",
        header=True,
        text_size=28,
        fields=_COMMUNITIES_CRIME_FIELDS,
    ),
    "concrete_strength": Table(
        name="concrete_strength",
        label="Concrete Compressive Strength",
        title="1,030 concrete mixes, how long each cured, and the strength it reached",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/165/data.csv",
        header=True,
        fields=_CONCRETE_STRENGTH_FIELDS,
    ),
    "congressional_voting": Table(
        name="congressional_voting",
        label="Congressional Voting Records",
        title="435 congressmen and the sixteen votes that give away the party",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/105/congressional+voting+records",
        classes=("democrat", "republican"),
        url="https://archive.ics.uci.edu/static/public/105/data.csv",
        header=True,
        fields=_CONGRESSIONAL_VOTING_FIELDS,
        labels={"democrat": 0, "republican": 1},
        codes={
            "handicapped_infants": ("n", "y"),
        },
    ),
    "connect_four": Table(
        name="connect_four",
        label="Connect-4",
        title="67,557 Connect Four positions eight moves in, won, lost or drawn",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/26/connect+4",
        classes=("draw", "loss", "win"),
        url="https://archive.ics.uci.edu/static/public/26/data.csv",
        header=True,
        fields=_CONNECT_FOUR_FIELDS,
        labels={"draw": 0, "loss": 1, "win": 2},
        codes={
            "a1": ("b", "o", "x"),
        },
    ),
    "credit_card_default": Table(
        name="credit_card_default",
        label="Default of Credit Card Clients",
        title="30,000 Taiwanese credit cards and which of them defaulted the next month",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        classes=("paid", "defaulted"),
        url="https://archive.ics.uci.edu/static/public/350/data.csv",
        header=True,
        fields=_CREDIT_CARD_DEFAULT_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "credit_screening": Table(
        name="credit_screening",
        label="Japanese Credit Screening",
        title="690 credit card applications with every field anonymised, approved or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/28/japanese+credit+screening",
        classes=("approved", "declined"),
        url="https://archive.ics.uci.edu/static/public/28/data.csv",
        header=True,
        fields=_CREDIT_SCREENING_FIELDS,
        labels={"+": 0, "-": 1},
        codes={
            "a1": ("a", "b"),
            "a4": ("l", "u", "y"),
            "a5": ("g", "gg", "p"),
            "a6": ("aa", "c", "cc", "d", "e", "ff", "i", "j", "k", "m", "q", "r", "w", "x"),
            "a7": ("bb", "dd", "ff", "h", "j", "n", "o", "v", "z"),
            "a9": ("f", "t"),
            "a13": ("g", "p", "s"),
        },
    ),
    "daily_demand": Table(
        name="daily_demand",
        label="Daily Demand Forecasting Orders",
        title="60 days at a Brazilian logistics firm and the orders each brought in",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/409/data.csv",
        header=True,
        fields=_DAILY_DEMAND_FIELDS,
    ),
    "darwin": Table(
        name="darwin",
        label="DARWIN",
        title="174 people writing on a graphics tablet, with and without Alzheimer's",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/732/darwin",
        classes=("healthy", "patient"),
        url="https://archive.ics.uci.edu/static/public/732/data.csv",
        header=True,
        text_size=6,
        fields=_DARWIN_FIELDS,
        labels={"H": 0, "P": 1},
    ),
    "diabetes_hospitals": Table(
        name="diabetes_hospitals",
        label="Diabetes 130-US Hospitals for Years 1999-2008",
        title="101,766 hospital stays for diabetes, and who came back",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        classes=("readmitted_within_30_days", "readmitted_later", "not_readmitted"),
        url="https://archive.ics.uci.edu/static/public/296/data.csv",
        header=True,
        text_size=36,
        fields=_DIABETES_HOSPITALS_FIELDS,
        labels={"<30": 0, ">30": 1, "NO": 2},
        codes={
            "race": ("AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"),
            "gender": ("Female", "Male", "Unknown/Invalid"),
            "age": (
                "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)",
                "[70-80)", "[80-90)", "[90-100)",
            ),
            "weight": (
                ">200", "[0-25)", "[100-125)", "[125-150)", "[150-175)", "[175-200)", "[25-50)",
                "[50-75)", "[75-100)",
            ),
            "payer_code": (
                "BC", "CH", "CM", "CP", "DM", "FR", "HM", "MC", "MD", "MP", "OG", "OT", "PO", "SI",
                "SP", "UN", "WC",
            ),
            "max_glu_serum": (">200", ">300", "None", "Norm"),
            "a1cresult": (">7", ">8", "None", "Norm"),
            "metformin": ("Down", "No", "Steady", "Up"),
            "acetohexamide": ("No", "Steady"),
            "tolazamide": ("No", "Steady", "Up"),
            "examide": ("No",),
            "change": ("Ch", "No"),
            "diabetesmed": ("No", "Yes"),
        },
    ),
    "diabetic_retinopathy": Table(
        name="diabetic_retinopathy",
        label="Diabetic Retinopathy Debrecen",
        title="1,151 eye images scored by a lesion detector, with and without retinopathy",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen",
        classes=("no_retinopathy", "retinopathy"),
        url="https://archive.ics.uci.edu/static/public/329/data.csv",
        header=True,
        fields=_DIABETIC_RETINOPATHY_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "dota2_games": Table(
        name="dota2_games",
        label="Dota2 Games Results",
        title="102,944 games of Dota 2, the heroes each side picked, and who won",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/367/dota2+games+results",
        classes=("radiant_lost", "radiant_won"),
        url="https://archive.ics.uci.edu/static/public/367/data.csv",
        header=True,
        fields=_DOTA2_GAMES_FIELDS,
        labels={"-1": 0, "1": 1},
    ),
    "drug_consumption": Table(
        name="drug_consumption",
        label="Drug Consumption (Quantified)",
        title="1,885 personality profiles, and how recently each person last used cannabis",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified",
        classes=("cl0", "cl1", "cl2", "cl3", "cl4", "cl5", "cl6"),
        url="https://archive.ics.uci.edu/static/public/373/data.csv",
        header=True,
        fields=_DRUG_CONSUMPTION_FIELDS,
        labels={"CL0": 0, "CL1": 1, "CL2": 2, "CL3": 3, "CL4": 4, "CL5": 5, "CL6": 6},
        codes={
            "alcohol": ("CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"),
            "semer": ("CL0", "CL1", "CL2", "CL3", "CL4"),
        },
    ),
    "eeg_eye_state": Table(
        name="eeg_eye_state",
        label="EEG Eye State",
        title="14,980 moments of EEG from fourteen electrodes, eyes open or shut",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/264/eeg+eye+state",
        classes=("eye_open", "eye_closed"),
        url="https://archive.ics.uci.edu/static/public/264/data.csv",
        header=True,
        fields=_EEG_EYE_STATE_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "el_nino": Table(
        name="el_nino",
        label="El Nino",
        title="178,080 readings from the Pacific buoy array and the air temperature at each",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/122/el+nino",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/122/data.csv",
        header=True,
        fields=_EL_NINO_FIELDS,
    ),
    "entrance_exam": Table(
        name="entrance_exam",
        label="Student Performance on an Entrance Examination",
        title="666 students sitting an engineering entrance exam, and how well each did",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/582/student+performance+on+an+entrance+examination",
        classes=("excellent", "very_good", "good", "average"),
        url="https://archive.ics.uci.edu/static/public/582/data.csv",
        header=True,
        fields=_ENTRANCE_EXAM_FIELDS,
        labels={"Excellent": 0, "Vg": 1, "Good": 2, "Average": 3},
        codes={
            "gender": ("female", "male"),
            "caste": ("General", "OBC", "SC", "ST"),
            "coaching": ("NO", "OA", "WA"),
            "time_code": ("FIVE", "FOUR", "ONE", "SEVEN", "THREE", "TWO"),
            "class_ten_education": ("CBSE", "OTHERS", "SEBA"),
            "twelve_education": ("AHSEC", "CBSE", "OTHERS"),
            "medium": ("ASSAMESE", "ENGLISH", "OTHERS"),
            "class_x_percentage": ("Average", "Excellent", "Good", "Vg"),
            "father_occupation": (
                "BANK_OFFICIAL", "BUSINESS", "COLLEGE_TEACHER", "CULTIVATOR", "DOCTOR", "ENGINEER",
                "OTHERS", "SCHOOL_TEACHER",
            ),
            "mother_occupation": (
                "BANK_OFFICIAL", "BUSINESS", "COLLEGE_TEACHER", "CULTIVATOR", "DOCTOR", "ENGINEER",
                "HOUSE_WIFE", "OTHERS", "SCHOOL_TEACHER",
            ),
        },
    ),
    "facebook_live_sellers": Table(
        name="facebook_live_sellers",
        label="Facebook Live Sellers in Thailand",
        title="7,050 posts by Thai fashion sellers, by what kind of post each one was",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand",
        classes=("link", "photo", "status", "video"),
        url="https://archive.ics.uci.edu/static/public/488/data.csv",
        header=True,
        dates="%m/%d/%Y %H:%M",
        fields=_FACEBOOK_LIVE_SELLERS_FIELDS,
        labels={"link": 0, "photo": 1, "status": 2, "video": 3},
    ),
    "facebook_metrics": Table(
        name="facebook_metrics",
        label="Facebook Metrics",
        title="500 posts by a cosmetics brand and the reaction each of them drew",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/368/facebook+metrics",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/368/data.csv",
        header=True,
        fields=_FACEBOOK_METRICS_FIELDS,
        codes={
            "type": ("Link", "Photo", "Status", "Video"),
        },
    ),
    "flags": Table(
        name="flags",
        label="Flags",
        title="194 national flags, their colours and shapes, and the country's religion",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/40/flags",
        classes=(
            "catholic", "other_christian", "muslim", "buddhist", "hindu", "ethnic", "marxist",
            "other_religion",
        ),
        url="https://archive.ics.uci.edu/static/public/40/data.csv",
        header=True,
        text_size=24,
        fields=_FLAGS_FIELDS,
        labels={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7},
        codes={
            "mainhue": ("black", "blue", "brown", "gold", "green", "orange", "red", "white"),
            "topleft": ("black", "blue", "gold", "green", "orange", "red", "white"),
        },
    ),
    "gas_turbine_emissions": Table(
        name="gas_turbine_emissions",
        label="Gas Turbine CO and NOx Emission Data Set",
        title="36,733 hours of a gas turbine and the carbon monoxide and nitrogen oxides it made",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/551/data.csv",
        header=True,
        fields=_GAS_TURBINE_EMISSIONS_FIELDS,
    ),
    "gender_by_name": Table(
        name="gender_by_name",
        label="Gender by Name",
        title="147,269 first names and the sex of the babies given them",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/591/gender+by+name",
        classes=("female", "male"),
        url="https://archive.ics.uci.edu/static/public/591/data.csv",
        header=True,
        text_size=25,
        fields=_GENDER_BY_NAME_FIELDS,
        labels={"F": 0, "M": 1},
    ),
    "glioma_grading": Table(
        name="glioma_grading",
        label="Glioma Grading Clinical and Mutation Features",
        title="839 glioma patients, the genes mutated in each, and the grade of the tumour",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset",
        classes=("lower_grade_glioma", "glioblastoma"),
        url="https://archive.ics.uci.edu/static/public/759/data.csv",
        header=True,
        text_size=12,
        fields=_GLIOMA_GRADING_FIELDS,
        labels={"0": 0, "1": 1},
        codes={
            "race": (
                "american indian or alaska native", "asian", "black or african american", "white",
            ),
        },
    ),
    "grid_stability": Table(
        name="grid_stability",
        label="Electrical Grid Stability Simulated Data",
        title="10,000 simulated four-node power grids, stable or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data",
        classes=("stable", "unstable"),
        url="https://archive.ics.uci.edu/static/public/471/data.csv",
        header=True,
        fields=_GRID_STABILITY_FIELDS,
        labels={"stable": 0, "unstable": 1},
    ),
    "hcv_blood_donors": Table(
        name="hcv_blood_donors",
        label="HCV data",
        title="615 blood samples from donors and hepatitis C patients, 5 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/571/hcv+data",
        classes=("blood_donor", "suspect_blood_donor", "hepatitis", "fibrosis", "cirrhosis"),
        url="https://archive.ics.uci.edu/static/public/571/data.csv",
        header=True,
        fields=_HCV_BLOOD_DONORS_FIELDS,
        labels={
            "0=Blood Donor": 0, "0s=suspect Blood Donor": 1, "1=Hepatitis": 2, "2=Fibrosis": 3,
            "3=Cirrhosis": 4,
        },
        codes={
            "sex": ("f", "m"),
        },
    ),
    "healthy_aging_poll": Table(
        name="healthy_aging_poll",
        label="National Poll on Healthy Aging (NPHA)",
        title="714 older Americans polled on their health, by how many doctors each sees",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha)",
        classes=("doctors_0_1", "doctors_2_3", "doctors_4_or_more"),
        url="https://archive.ics.uci.edu/static/public/936/data.csv",
        header=True,
        fields=_HEALTHY_AGING_POLL_FIELDS,
        labels={"1": 0, "2": 1, "3": 2},
    ),
    "hepatitis_c_egypt": Table(
        name="hepatitis_c_egypt",
        label="Hepatitis C Virus (HCV) for Egyptian patients",
        title="1,385 Egyptian hepatitis C patients and how far the fibrosis had gone",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/503/hepatitis+c+virus+hcv+for+egyptian+patients",
        classes=("portal_fibrosis", "few_septa", "many_septa", "cirrhosis"),
        url="https://archive.ics.uci.edu/static/public/503/data.csv",
        header=True,
        fields=_HEPATITIS_C_EGYPT_FIELDS,
        labels={"1": 0, "2": 1, "3": 2, "4": 3},
    ),
    "higher_education_students": Table(
        name="higher_education_students",
        label="Higher Education Students Performance Evaluation",
        title="145 students answering how they live and study, and the grade each got",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation",
        classes=("fail", "dd", "dc", "cc", "cb", "bb", "ba", "aa"),
        url="https://archive.ics.uci.edu/static/public/856/data.csv",
        header=True,
        text_size=10,
        fields=_HIGHER_EDUCATION_STUDENTS_FIELDS,
        labels={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7},
    ),
    "horse_colic": Table(
        name="horse_colic",
        label="Horse Colic",
        title="368 horses with colic and whether the lesion turned out to need surgery",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/47/horse+colic",
        classes=("surgical", "not_surgical"),
        url="https://archive.ics.uci.edu/static/public/47/data.csv",
        header=True,
        fields=_HORSE_COLIC_FIELDS,
        labels={"1": 0, "2": 1},
    ),
    "in_vehicle_coupon": Table(
        name="in_vehicle_coupon",
        label="In-Vehicle Coupon Recommendation",
        title="12,684 drivers offered a coupon at the wheel, and who took one",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation",
        classes=("declined", "accepted"),
        url="https://archive.ics.uci.edu/static/public/603/data.csv",
        header=True,
        text_size=41,
        fields=_IN_VEHICLE_COUPON_FIELDS,
        labels={"0": 0, "1": 1},
        codes={
            "destination": ("Home", "No Urgent Place", "Work"),
            "passenger": ("Alone", "Friend(s)", "Kid(s)", "Partner"),
            "weather": ("Rainy", "Snowy", "Sunny"),
            "time_code": ("10AM", "10PM", "2PM", "6PM", "7AM"),
            "coupon": (
                "Bar", "Carry out & Take away", "Coffee House", "Restaurant(20-50)",
                "Restaurant(<20)",
            ),
            "expiration": ("1d", "2h"),
            "gender": ("Female", "Male"),
            "age": ("21", "26", "31", "36", "41", "46", "50plus", "below21"),
            "maritalstatus": (
                "Divorced", "Married partner", "Single", "Unmarried partner", "Widowed",
            ),
            "income": (
                "$100000 or More", "$12500 - $24999", "$25000 - $37499", "$37500 - $49999",
                "$50000 - $62499", "$62500 - $74999", "$75000 - $87499", "$87500 - $99999",
                "Less than $12500",
            ),
            "bar": ("1~3", "4~8", "gt8", "less1", "never"),
        },
    ),
    "infrared_thermography": Table(
        name="infrared_thermography",
        label="Infrared Thermography Temperature",
        title="1,020 thermal images of a face and the oral temperature measured after each",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/925/data.csv",
        header=True,
        text_size=33,
        fields=_INFRARED_THERMOGRAPHY_FIELDS,
        codes={
            "gender": ("Female", "Male"),
            "age": ("18-20", "21-25", "21-30", "26-30", "31-40", "41-50", "51-60", ">60"),
        },
    ),
    "iot_intrusion": Table(
        name="iot_intrusion",
        label="RT-IoT2022",
        title="123,117 network flows from an IoT testbed, ordinary traffic and twelve attacks",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/942/rt-iot2022",
        classes=(
            "arp_poisioning", "ddos_slowloris", "dos_syn_hping", "mqtt_publish",
            "metasploit_brute_force_ssh", "nmap_fin_scan", "nmap_os_detection", "nmap_tcp_scan",
            "nmap_udp_scan", "nmap_xmas_tree_scan", "thing_speak", "wipro_bulb",
        ),
        url="https://archive.ics.uci.edu/static/public/942/data.csv",
        header=True,
        fields=_IOT_INTRUSION_FIELDS,
        labels={
            "ARP_poisioning": 0, "DDOS_Slowloris": 1, "DOS_SYN_Hping": 2, "MQTT_Publish": 3,
            "Metasploit_Brute_Force_SSH": 4, "NMAP_FIN_SCAN": 5, "NMAP_OS_DETECTION": 6,
            "NMAP_TCP_scan": 7, "NMAP_UDP_SCAN": 8, "NMAP_XMAS_TREE_SCAN": 9, "Thing_Speak": 10,
            "Wipro_bulb": 11,
        },
        codes={
            "proto": ("icmp", "tcp", "udp"),
            "service": ("-", "dhcp", "dns", "http", "irc", "mqtt", "ntp", "radius", "ssh", "ssl"),
        },
    ),
    "iranian_churn": Table(
        name="iranian_churn",
        label="Iranian Churn",
        title="3,150 customers of an Iranian telecom and which of them left",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        classes=("stayed", "left"),
        url="https://archive.ics.uci.edu/static/public/563/data.csv",
        header=True,
        fields=_IRANIAN_CHURN_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "isolet": Table(
        name="isolet",
        label="ISOLET",
        title="7,797 spoken letters described 617 ways, 26 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/54/isolet",
        classes=(
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
            "r", "s", "t", "u", "v", "w", "x", "y", "z",
        ),
        url="https://archive.ics.uci.edu/static/public/54/data.csv",
        header=True,
        fields=_ISOLET_FIELDS,
        labels={
            "1.": 0, "2.": 1, "3.": 2, "4.": 3, "5.": 4, "6.": 5, "7.": 6, "8.": 7, "9.": 8,
            "10.": 9, "11.": 10, "12.": 11, "13.": 12, "14.": 13, "15.": 14, "16.": 15, "17.": 16,
            "18.": 17, "19.": 18, "20.": 19, "21.": 20, "22.": 21, "23.": 22, "24.": 23, "25.": 24,
            "26.": 25,
        },
    ),
    "istanbul_exchange": Table(
        name="istanbul_exchange",
        label="ISTANBUL STOCK EXCHANGE",
        title="536 trading days of the Istanbul exchange beside eight other indices",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/247/istanbul+stock+exchange",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/247/data.csv",
        header=True,
        dates="%d-%b-%y",
        text_size=9,
        fields=_ISTANBUL_EXCHANGE_FIELDS,
    ),
    "kidney_risk_factors": Table(
        name="kidney_risk_factors",
        label="Risk Factor Prediction of Chronic Kidney Disease",
        title="200 villagers in India screened for chronic kidney disease",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/857/risk+factor+prediction+of+chronic+kidney+disease",
        classes=("ckd", "notckd"),
        url="https://archive.ics.uci.edu/static/public/857/data.csv",
        header=True,
        fields=_KIDNEY_RISK_FACTORS_FIELDS,
        labels={"ckd": 0, "notckd": 1},
        codes={
            "sg": ("1.009 - 1.011", "1.015 - 1.017", "1.019 - 1.021", "< 1.007", "≥ 1.023"),
            "al": ("1-Jan", "2-Feb", "3-Mar", "< 0", "≥ 4"),
            "su": ("2-Feb", "2-Jan", "4-Apr", "4-Mar", "< 0", "≥ 4"),
            "bgr": (
                "112 - 154", "154 - 196", "196 - 238", "238 - 280", "280 - 322", "322 - 364",
                "364 - 406", "406 - 448", "< 112", "≥ 448",
            ),
            "bu": (
                "124.3 - 162.4", "162.4 - 200.5", "200.5 - 238.6", "238.6 - 276.7", "48.1 - 86.2",
                "86.2 - 124.3", "< 48.1", "≥ 352.9",
            ),
            "sod": (
                "118 - 123", "123 - 128", "128 - 133", "133 - 138", "138 - 143", "143 - 148",
                "148 - 153", "< 118", "≥ 158",
            ),
            "sc": (
                "13.1 - 16.25", "16.25 - 19.4", "3.65 - 6.8", "6.8 - 9.95", "9.95 - 13.1",
                "< 3.65", "≥ 28.85",
            ),
            "pot": ("38.18 - 42.59", "7.31 - 11.72", "< 7.31", "≥ 42.59"),
            "hemo": (
                "10 - 11.3", "11.3 - 12.6", "12.6 - 13.9", "13.9 - 15.2", "15.2 - 16.5",
                "6.1 - 7.4", "7.4 - 8.7", "8.7 - 10", "< 6.1", "≥ 16.5",
            ),
            "pcv": (
                "17.9 - 21.8", "21.8 - 25.7", "25.7 - 29.6", "29.6 - 33.5", "33.5 - 37.4",
                "37.4 - 41.3", "41.3 - 45.2", "45.2 - 49.1", "< 17.9", "≥ 49.1",
            ),
            "rbcc": (
                "2.69 - 3.28", "3.28 - 3.87", "3.87 - 4.46", "4.46 - 5.05", "5.05 - 5.64",
                "5.64 - 6.23", "6.23 - 6.82", "< 2.69", "≥ 7.41",
            ),
            "wbcc": (
                "12120 - 14500", "14500 - 16880", "16880 - 19260", "19260 - 21640", "4980 - 7360",
                "7360 - 9740", "9740 - 12120", "< 4980", "≥ 24020",
            ),
            "grf": (
                "102.115 - 127.281", "127.281 - 152.446", "152.446 - 177.612", "177.612 - 202.778",
                "202.778 - 227.944", "26.6175 - 51.7832", "51.7832 - 76.949", "76.949 - 102.115",
                "< 26.6175", "p", "≥ 227.944",
            ),
            "stage": ("s1", "s2", "s3", "s4", "s5"),
            "age": (
                "20 - 27", "20-Dec", "27 - 35", "35 - 43", "43 - 51", "51 - 59", "59 - 66",
                "66 - 74", "< 12", "≥ 74",
            ),
        },
    ),
    "land_mines": Table(
        name="land_mines",
        label="Land Mines",
        title="338 passes of a metal detector and the kind of mine underneath",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/763/land+mines-1",
        classes=(
            "no_mine", "anti_personnel", "anti_tank", "booby_trapped_anti_personnel",
            "m14_anti_personnel",
        ),
        url="https://archive.ics.uci.edu/static/public/763/data.csv",
        header=True,
        fields=_LAND_MINES_FIELDS,
        labels={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
    ),
    "metro_traffic": Table(
        name="metro_traffic",
        label="Metro Interstate Traffic Volume",
        title="48,204 hours on an interstate near Minneapolis and the cars that went by",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/492/data.csv",
        header=True,
        dates="%Y-%m-%d %H:%M:%S",
        text_size=35,
        fields=_METRO_TRAFFIC_FIELDS,
        codes={
            "holiday": (
                "Christmas Day", "Columbus Day", "Independence Day", "Labor Day",
                "Martin Luther King Jr Day", "Memorial Day", "New Years Day", "None", "State Fair",
                "Thanksgiving Day", "Veterans Day", "Washingtons Birthday",
            ),
            "weather_main": (
                "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow",
                "Squall", "Thunderstorm",
            ),
        },
    ),
    "mice_protein": Table(
        name="mice_protein",
        label="Mice Protein Expression",
        title="1,080 measurements of 77 proteins in the cortex of mice, 8 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/342/mice+protein+expression",
        classes=("c_cs_m", "c_cs_s", "c_sc_m", "c_sc_s", "t_cs_m", "t_cs_s", "t_sc_m", "t_sc_s"),
        url="https://archive.ics.uci.edu/static/public/342/data.csv",
        header=True,
        text_size=9,
        fields=_MICE_PROTEIN_FIELDS,
        labels={
            "c-CS-m": 0, "c-CS-s": 1, "c-SC-m": 2, "c-SC-s": 3, "t-CS-m": 4, "t-CS-s": 5,
            "t-SC-m": 6, "t-SC-s": 7,
        },
        codes={
            "genotype": ("Control", "Ts65Dn"),
            "treatment": ("Memantine", "Saline"),
            "behavior": ("C/S", "S/C"),
        },
    ),
    "monks_problems": Table(
        name="monks_problems",
        label="MONK's Problems",
        title="432 toy robots described six ways, and whether the rule holds of each",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/70/monk+s+problems",
        classes=("false", "true"),
        url="https://archive.ics.uci.edu/static/public/70/data.csv",
        header=True,
        text_size=8,
        fields=_MONKS_PROBLEMS_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "multivariate_gait": Table(
        name="multivariate_gait",
        label="Multivariate Gait Data",
        title="181,800 joint angles measured as ten people walked at three speeds",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/760/multivariate+gait+data",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/760/data.csv",
        header=True,
        fields=_MULTIVARIATE_GAIT_FIELDS,
    ),
    "musk_version1": Table(
        name="musk_version1",
        label="Musk (Version 1)",
        title="476 molecular conformations described 166 ways, musk or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/74/musk+version+1",
        classes=("not_musk", "musk"),
        url="https://archive.ics.uci.edu/static/public/74/data.csv",
        header=True,
        text_size=13,
        fields=_MUSK_VERSION1_FIELDS,
        labels={"0.": 0, "1.": 1},
    ),
    "musk_version2": Table(
        name="musk_version2",
        label="Musk (Version 2)",
        title="6,598 conformations of 102 molecules, musk or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/75/musk+version+2",
        classes=("not_musk", "musk"),
        url="https://archive.ics.uci.edu/static/public/75/data.csv",
        header=True,
        text_size=13,
        fields=_MUSK_VERSION2_FIELDS,
        labels={"0.": 0, "1.": 1},
    ),
    "news_popularity": Table(
        name="news_popularity",
        label="Online News Popularity",
        title="39,644 Mashable articles described 58 ways, and the times each was shared",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/332/online+news+popularity",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/332/data.csv",
        header=True,
        text_size=192,
        fields=_NEWS_POPULARITY_FIELDS,
    ),
    "nhanes_age": Table(
        name="nhanes_age",
        label="NHANES 2013-2014 Age Prediction Subset",
        title="2,278 people in an American health survey, adult or senior",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset",
        classes=("adult", "senior"),
        url="https://archive.ics.uci.edu/static/public/887/data.csv",
        header=True,
        fields=_NHANES_AGE_FIELDS,
        labels={"Adult": 0, "Senior": 1},
    ),
    "obesity_levels": Table(
        name="obesity_levels",
        label="Estimation of Obesity Levels Based On Eating Habits and Physical Condition",
        title="2,111 people's eating and moving habits, and the weight class each fell in",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition",
        classes=(
            "insufficient_weight", "normal_weight", "obesity_type_i", "obesity_type_ii",
            "obesity_type_iii", "overweight_level_i", "overweight_level_ii",
        ),
        url="https://archive.ics.uci.edu/static/public/544/data.csv",
        header=True,
        fields=_OBESITY_LEVELS_FIELDS,
        labels={
            "Insufficient_Weight": 0, "Normal_Weight": 1, "Obesity_Type_I": 2,
            "Obesity_Type_II": 3, "Obesity_Type_III": 4, "Overweight_Level_I": 5,
            "Overweight_Level_II": 6,
        },
        codes={
            "gender": ("Female", "Male"),
            "family_history_with_overweight": ("no", "yes"),
            "caec": ("Always", "Frequently", "Sometimes", "no"),
            "mtrans": ("Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"),
        },
    ),
    "ozone_level": Table(
        name="ozone_level",
        label="Ozone Level Detection",
        title="5,070 days of Houston weather, and which of them broke the ozone limit",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/172/ozone+level+detection",
        classes=("normal_day", "ozone_day"),
        url="https://archive.ics.uci.edu/static/public/172/data.csv",
        header=True,
        dates="%m/%d/%Y",
        fields=_OZONE_LEVEL_FIELDS,
        labels={"0": 0, "1": 1},
        codes={
            "dataset": ("1hr", "8hr"),
        },
    ),
    "page_blocks": Table(
        name="page_blocks",
        label="Page Blocks Classification",
        title="5,473 blocks of a scanned document page, 5 classes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/78/page+blocks+classification",
        classes=("text", "horizontal_line", "picture", "vertical_line", "graphic"),
        url="https://archive.ics.uci.edu/static/public/78/data.csv",
        header=True,
        fields=_PAGE_BLOCKS_FIELDS,
        labels={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
    ),
    "pittsburgh_bridges": Table(
        name="pittsburgh_bridges",
        label="Pittsburgh Bridges",
        title="108 bridges over the three rivers of Pittsburgh, by the era each was built in",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/18/pittsburgh+bridges",
        classes=("crafts", "emerging", "mature", "modern"),
        url="https://archive.ics.uci.edu/static/public/18/data.csv",
        header=True,
        text_size=5,
        fields=_PITTSBURGH_BRIDGES_FIELDS,
        labels={"CRAFTS": 0, "EMERGING": 1, "MATURE": 2, "MODERN": 3},
        codes={
            "river": ("A", "M", "O", "Y"),
            "purpose": ("AQUEDUCT", "HIGHWAY", "RR", "WALK"),
            "length": ("LONG", "MEDIUM", "SHORT"),
            "clear_g": ("G", "N"),
            "t_or_d": ("DECK", "THROUGH"),
            "material": ("IRON", "STEEL", "WOOD"),
            "rel_l": ("F", "S", "S-F"),
            "type": ("ARCH", "CANTILEV", "CONT-T", "NIL", "SIMPLE-T", "SUSPEN", "WOOD"),
        },
    ),
    "poker_hand": Table(
        name="poker_hand",
        label="Poker Hand",
        title="1,025,010 poker hands of five cards, by what each is worth",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/158/poker+hand",
        classes=(
            "nothing", "one_pair", "two_pairs", "three_of_a_kind", "straight", "flush",
            "full_house", "four_of_a_kind", "straight_flush", "royal_flush",
        ),
        url="https://archive.ics.uci.edu/static/public/158/data.csv",
        header=True,
        fields=_POKER_HAND_FIELDS,
        labels={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9},
    ),
    "polish_bankruptcy": Table(
        name="polish_bankruptcy",
        label="Polish Companies Bankruptcy",
        title="43,405 years of Polish company accounts and which firms went bankrupt",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data",
        classes=("solvent", "bankrupt"),
        url="https://archive.ics.uci.edu/static/public/365/data.csv",
        header=True,
        fields=_POLISH_BANKRUPTCY_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "post_operative_patient": Table(
        name="post_operative_patient",
        label="Post-Operative Patient",
        title="90 patients leaving the recovery room, and where each was sent next",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/82/post+operative+patient",
        classes=("general_floor", "intensive_care", "sent_home"),
        url="https://archive.ics.uci.edu/static/public/82/data.csv",
        header=True,
        fields=_POST_OPERATIVE_PATIENT_FIELDS,
        labels={"A": 0, "I": 1, "S": 2},
        codes={
            "l_core": ("high", "low", "mid"),
            "l_o2": ("excellent", "good"),
            "surf_stbl": ("stable", "unstable"),
            "core_stbl": ("mod-stable", "stable", "unstable"),
        },
    ),
    "predictive_maintenance": Table(
        name="predictive_maintenance",
        label="AI4I 2020 Predictive Maintenance Dataset",
        title="10,000 simulated hours of a milling machine, and the tools that broke",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
        classes=("kept_going", "failed"),
        url="https://archive.ics.uci.edu/static/public/601/data.csv",
        header=True,
        text_size=6,
        fields=_PREDICTIVE_MAINTENANCE_FIELDS,
        labels={"0": 0, "1": 1},
        codes={
            "type": ("H", "L", "M"),
        },
    ),
    "room_occupancy_count": Table(
        name="room_occupancy_count",
        label="Room Occupancy Estimation",
        title="10,129 minutes of light, sound, temperature and CO2, and the people in the room",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/864/data.csv",
        header=True,
        dates="%H:%M:%S",
        fields=_ROOM_OCCUPANCY_COUNT_FIELDS,
        codes={
            "date_code": (
                "2017/12/22", "2017/12/23", "2017/12/24", "2017/12/25", "2017/12/26", "2018/01/10",
                "2018/01/11",
            ),
        },
    ),
    "secondary_mushroom": Table(
        name="secondary_mushroom",
        label="Secondary Mushroom",
        title="61,069 mushrooms grown from a field guide's own descriptions, edible or poisonous",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset",
        classes=("edible", "poisonous"),
        url="https://archive.ics.uci.edu/static/public/848/data.csv",
        header=True,
        fields=_SECONDARY_MUSHROOM_FIELDS,
        labels={"e": 0, "p": 1},
        codes={
            "cap_shape": ("b", "c", "f", "o", "p", "s", "x"),
            "cap_surface": ("d", "e", "g", "h", "i", "k", "l", "s", "t", "w", "y"),
            "cap_color": ("b", "e", "g", "k", "l", "n", "o", "p", "r", "u", "w", "y"),
            "does_bruise_or_bleed": ("f", "t"),
            "gill_attachment": ("a", "d", "e", "f", "p", "s", "x"),
            "gill_spacing": ("c", "d", "f"),
            "gill_color": ("b", "e", "f", "g", "k", "n", "o", "p", "r", "u", "w", "y"),
            "stem_root": ("b", "c", "f", "r", "s"),
            "stem_surface": ("f", "g", "h", "i", "k", "s", "t", "y"),
            "stem_color": ("b", "e", "f", "g", "k", "l", "n", "o", "p", "r", "u", "w", "y"),
            "veil_type": ("u",),
            "veil_color": ("e", "k", "n", "u", "w", "y"),
            "ring_type": ("e", "f", "g", "l", "m", "p", "r", "z"),
            "spore_print_color": ("g", "k", "n", "p", "r", "u", "w"),
            "habitat": ("d", "g", "h", "l", "m", "p", "u", "w"),
            "season": ("a", "s", "u", "w"),
        },
    ),
    "seoul_bike_sharing": Table(
        name="seoul_bike_sharing",
        label="Seoul Bike Sharing Demand",
        title="8,760 hours in Seoul, the weather at each, and the bikes rented",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/560/data.csv",
        header=True,
        dates="%d/%m/%Y",
        fields=_SEOUL_BIKE_SHARING_FIELDS,
        codes={
            "seasons": ("Autumn", "Spring", "Summer", "Winter"),
            "holiday": ("Holiday", "No Holiday"),
            "functioning_day": ("No", "Yes"),
        },
    ),
    "sepsis_survival": Table(
        name="sepsis_survival",
        label="Sepsis Survival Minimal Clinical Records",
        title="110,341 hospital admissions for sepsis in Norway, and who lived",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records",
        classes=("died", "survived"),
        url="https://archive.ics.uci.edu/static/public/827/data.csv",
        header=True,
        fields=_SEPSIS_SURVIVAL_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "skin_segmentation": Table(
        name="skin_segmentation",
        label="Skin Segmentation",
        title="245,057 pixels of face and background, as three colour channels",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/229/skin+segmentation",
        classes=("skin", "not_skin"),
        url="https://archive.ics.uci.edu/static/public/229/data.csv",
        header=True,
        fields=_SKIN_SEGMENTATION_FIELDS,
        labels={"1": 0, "2": 1},
    ),
    "soybean_cultivars": Table(
        name="soybean_cultivars",
        label="Forty Soybean Cultivars from Subsequent Harvests",
        title="320 soybean plants of forty cultivars, and the grain each yielded",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/913/forty+soybean+cultivars+from+subsequent+harvests",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/913/data.csv",
        header=True,
        fields=_SOYBEAN_CULTIVARS_FIELDS,
        codes={
            "cultivar": (
                "74K75RSF CE", "77HO111I2X - GUAPORÉ", "79I81RSF IPRO",
                "82HO111 IPRO - HO COXIM IPRO", "82I78RSF IPRO", "83IX84RSF I2X", "96R29 IPRO",
                "97Y97 IPRO", "98R30 CE", "ADAPTA LTT 8402 IPRO", "ATAQUE I2X",
                "BRASMAX BÔNUS IPRO", "BRASMAX OLIMPO IPRO", "ELISA IPRO", "EXPANDE LTT 8301 IPRO",
                "FORTALECE L090183 RR", "FORTALEZA IPRO", "FTR 3179 IPRO", "FTR 3190 IPRO",
                "FTR 3868 IPRO", "FTR 4280 IPRO", "FTR 4288 IPRO", "GNS7700 IPRO",
                "GNS7900 IPRO - AMPLA", "LAT 1330BT", "LTT 7901 IPRO", "LYNDA IPRO", "M 8644 IPRO",
                "MANU IPRO", "MONSOY 8330I2X", "MONSOY M8606I2X", "NEO 760 CE", "NEO 790 IPRO",
                "NK 7777 IPRO", "NK 8100 IPRO", "NK 8770 IPRO", "PAULA IPRO", "SUZY IPRO",
                "SYN2282IPRO", "TMG 22X83I2X",
            ),
        },
    ),
    "soybean_small": Table(
        name="soybean_small",
        label="Soybean (Small)",
        title="47 soybean plants with one of four diseases, described 35 ways",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/91/soybean+small",
        classes=(
            "diaporthe_stem_canker", "charcoal_rot", "rhizoctonia_root_rot", "phytophthora_rot",
        ),
        url="https://archive.ics.uci.edu/static/public/91/data.csv",
        header=True,
        fields=_SOYBEAN_SMALL_FIELDS,
        labels={"D1": 0, "D2": 1, "D3": 2, "D4": 3},
    ),
    "spect_heart": Table(
        name="spect_heart",
        label="SPECT Heart",
        title="267 cardiac SPECT images reduced to 22 binary patterns, normal or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/95/spect+heart",
        classes=("normal", "abnormal"),
        url="https://archive.ics.uci.edu/static/public/95/data.csv",
        header=True,
        fields=_SPECT_HEART_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "spectf_heart": Table(
        name="spectf_heart",
        label="SPECTF Heart",
        title="267 cardiac SPECT images described by 44 counts, normal or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/96/spectf+heart",
        classes=("normal", "abnormal"),
        url="https://archive.ics.uci.edu/static/public/96/data.csv",
        header=True,
        fields=_SPECTF_HEART_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "splice_junctions": Table(
        name="splice_junctions",
        label="Molecular Biology (Splice-junction Gene Sequences)",
        title="3,190 stretches of DNA and the splice junction, if any, in the middle",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences",
        classes=("exon_intron", "intron_exon", "neither"),
        url="https://archive.ics.uci.edu/static/public/69/data.csv",
        header=True,
        text_size=24,
        fields=_SPLICE_JUNCTIONS_FIELDS,
        labels={"EI": 0, "IE": 1, "N": 2},
        codes={
            "base1": ("A", "C", "D", "G", "T"),
            "base3": ("A", "C", "G", "T"),
            "base14": ("A", "C", "G", "N", "T"),
            "base35": ("A", "C", "G", "N", "R", "T"),
            "base36": ("A", "C", "G", "N", "S", "T"),
        },
    ),
    "steel_plates": Table(
        name="steel_plates",
        label="Steel Plates Faults",
        title="1,941 steel plates and the seven kinds of surface fault found on them",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/198/steel+plates+faults",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/198/data.csv",
        header=True,
        fields=_STEEL_PLATES_FIELDS,
    ),
    "student_academics": Table(
        name="student_academics",
        label="Student Academics Performance",
        title="131 students in Kalyani and the band each ended the semester in",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/467/student+academics+performance",
        classes=("best", "very_good", "good", "pass"),
        url="https://archive.ics.uci.edu/static/public/467/data.csv",
        header=True,
        fields=_STUDENT_ACADEMICS_FIELDS,
        labels={"Best": 0, "Vg": 1, "Good": 2, "Pass": 3},
        codes={
            "ge": ("F", "M"),
            "cst": ("G", "MOBC", "OBC", "SC", "ST"),
            "tnp": ("Best", "Good", "Pass", "Vg"),
            "arr": ("N", "Y"),
            "ms": ("Unmarried",),
            "ls": ("T", "V"),
            "as": ("Free", "Paid"),
            "fmi": ("Am", "High", "Low", "Medium", "Vh"),
            "fs": ("Average", "Large", "Small"),
            "fq": ("10", "12", "Degree", "Il", "Pg", "Um"),
            "fo": ("Business", "Farmer", "Others", "Retired", "Service"),
            "mo": ("Business", "Housewife", "Others", "Retired", "Service"),
            "sh": ("Average", "Good", "Poor"),
            "ss": ("Govt", "Private"),
            "me": ("Asm", "Ben", "Eng", "Hin"),
        },
    ),
    "student_dropout": Table(
        name="student_dropout",
        label="Predict Students' Dropout and Academic Success",
        title="4,424 university students, and whether each dropped out, stayed on or graduated",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
        classes=("dropout", "enrolled", "graduate"),
        url="https://archive.ics.uci.edu/static/public/697/data.csv",
        header=True,
        fields=_STUDENT_DROPOUT_FIELDS,
        labels={"Dropout": 0, "Enrolled": 1, "Graduate": 2},
    ),
    "superconductivity": Table(
        name="superconductivity",
        label="Superconductivty Data",
        title="21,263 superconductors described by their chemistry, and the critical temperature",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/464/superconductivty+data",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/464/data.csv",
        header=True,
        fields=_SUPERCONDUCTIVITY_FIELDS,
    ),
    "support2": Table(
        name="support2",
        label="SUPPORT2",
        title="9,105 seriously ill hospital patients, and who died before going home",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/880/support2",
        classes=("left_hospital", "died_in_hospital"),
        url="https://archive.ics.uci.edu/static/public/880/data.csv",
        header=True,
        fields=_SUPPORT2_FIELDS,
        labels={"0": 0, "1": 1},
        codes={
            "sex": ("female", "male"),
            "dzgroup": (
                "ARF/MOSF w/Sepsis", "CHF", "COPD", "Cirrhosis", "Colon Cancer", "Coma",
                "Lung Cancer", "MOSF w/Malig",
            ),
            "dzclass": ("ARF/MOSF", "COPD/CHF/Cirrhosis", "Cancer", "Coma"),
            "income": ("$11-$25k", "$25-$50k", ">$50k", "under $11k"),
            "race": ("asian", "black", "hispanic", "other", "white"),
            "ca": ("metastatic", "no", "yes"),
            "dnr": ("dnr after sadm", "dnr before sadm", "no dnr"),
            "sfdm2": (
                "<2 mo. follow-up", "Coma or Intub", "SIP>=30", "adl>=4 (>=5 if sur)",
                "no(M2 and SIP pres)",
            ),
        },
    ),
    "taiwanese_bankruptcy": Table(
        name="taiwanese_bankruptcy",
        label="Taiwanese Bankruptcy Prediction",
        title="6,819 Taiwanese companies described by 95 financial ratios, solvent or bankrupt",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction",
        classes=("solvent", "bankrupt"),
        url="https://archive.ics.uci.edu/static/public/572/data.csv",
        header=True,
        fields=_TAIWANESE_BANKRUPTCY_FIELDS,
        labels={"0": 0, "1": 1},
    ),
    "tennis_majors": Table(
        name="tennis_majors",
        label="Tennis Major Tournament Match Statistics",
        title="943 matches at the 2013 grand slams, point by point, and who won",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/300/tennis+major+tournament+match+statistics",
        classes=("player_1_lost", "player_1_won"),
        url="https://archive.ics.uci.edu/static/public/300/data.csv",
        header=True,
        text_size=26,
        fields=_TENNIS_MAJORS_FIELDS,
        labels={"0": 0, "1": 1},
        codes={
            "tournament": (
                "AusOpen-men", "AusOpen-women", "FrenchOpen-men", "FrenchOpen-women", "USOpen-men",
                "USOpen-women",
            ),
        },
    ),
    "tetouan_power": Table(
        name="tetouan_power",
        label="Power Consumption of Tetouan City",
        title="52,416 ten-minute readings of Tetouan's weather and the power its three zones drew",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city",
        classes=(),
        url="https://archive.ics.uci.edu/static/public/849/data.csv",
        header=True,
        dates="%m/%d/%Y %H:%M",
        fields=_TETOUAN_POWER_FIELDS,
    ),
    "thoracic_surgery": Table(
        name="thoracic_surgery",
        label="Thoracic Surgery Data",
        title="470 lung cancer operations and who was still alive a year later",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data",
        classes=("survived", "died"),
        url="https://archive.ics.uci.edu/static/public/277/data.csv",
        header=True,
        fields=_THORACIC_SURGERY_FIELDS,
        labels={"F": 0, "T": 1},
        codes={
            "dgn": ("DGN1", "DGN2", "DGN3", "DGN4", "DGN5", "DGN6", "DGN8"),
            "pre6": ("PRZ0", "PRZ1", "PRZ2"),
            "pre7": ("F", "T"),
            "pre14": ("OC11", "OC12", "OC13", "OC14"),
        },
    ),
    "thyroid_recurrence": Table(
        name="thyroid_recurrence",
        label="Differentiated Thyroid Cancer Recurrence",
        title="383 thyroid cancer patients followed fifteen years, and whose cancer came back",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence",
        classes=("no", "yes"),
        url="https://archive.ics.uci.edu/static/public/915/data.csv",
        header=True,
        fields=_THYROID_RECURRENCE_FIELDS,
        labels={"No": 0, "Yes": 1},
        codes={
            "gender": ("F", "M"),
            "smoking": ("No", "Yes"),
            "thyroid_function": (
                "Clinical Hyperthyroidism", "Clinical Hypothyroidism", "Euthyroid",
                "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism",
            ),
            "physical_examination": (
                "Diffuse goiter", "Multinodular goiter", "Normal", "Single nodular goiter-left",
                "Single nodular goiter-right",
            ),
            "adenopathy": ("Bilateral", "Extensive", "Left", "No", "Posterior", "Right"),
            "pathology": ("Follicular", "Hurthel cell", "Micropapillary", "Papillary"),
            "focality": ("Multi-Focal", "Uni-Focal"),
            "risk": ("High", "Intermediate", "Low"),
            "t": ("T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"),
            "n": ("N0", "N1a", "N1b"),
            "m": ("M0", "M1"),
            "stage": ("I", "II", "III", "IVA", "IVB"),
            "response": (
                "Biochemical Incomplete", "Excellent", "Indeterminate", "Structural Incomplete",
            ),
        },
    ),
    "user_knowledge": Table(
        name="user_knowledge",
        label="User Knowledge Modeling",
        title="403 students studying DC machines, and how well each knew the subject",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/257/user+knowledge+modeling",
        classes=("very_low", "low", "middle", "high"),
        url="https://archive.ics.uci.edu/static/public/257/data.csv",
        header=True,
        fields=_USER_KNOWLEDGE_FIELDS,
        labels={"very_low": 0, "Very Low": 0, "Low": 1, "Middle": 2, "High": 3},
    ),
    "waveform": Table(
        name="waveform",
        label="Waveform Database Generator (Version 1)",
        title="5,000 generated waveforms of three kinds, described by 21 noisy attributes",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/107/waveform+database+generator+version+1",
        classes=("wave_0", "wave_1", "wave_2"),
        url="https://archive.ics.uci.edu/static/public/107/data.csv",
        header=True,
        fields=_WAVEFORM_FIELDS,
        labels={"0": 0, "1": 1, "2": 2},
    ),
    "website_phishing": Table(
        name="website_phishing",
        label="Website Phishing",
        title="1,353 web pages judged phishing, suspicious or legitimate on nine signs",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/379/website+phishing",
        classes=("phishing", "suspicious", "legitimate"),
        url="https://archive.ics.uci.edu/static/public/379/data.csv",
        header=True,
        fields=_WEBSITE_PHISHING_FIELDS,
        labels={"-1": 0, "0": 1, "1": 2},
    ),
    "wholesale_customers": Table(
        name="wholesale_customers",
        label="Wholesale customers",
        title="440 wholesale customers and what each spent on six kinds of goods",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/292/wholesale+customers",
        classes=("horeca", "retail"),
        url="https://archive.ics.uci.edu/static/public/292/data.csv",
        header=True,
        fields=_WHOLESALE_CUSTOMERS_FIELDS,
        labels={"1": 0, "2": 1},
    ),
    "youtube_spam": Table(
        name="youtube_spam",
        label="YouTube Spam Collection",
        title="1,956 comments under five pop videos, spam or not",
        licence="CC BY 4.0",
        source="https://archive.ics.uci.edu/dataset/380/youtube+spam+collection",
        classes=("not_spam", "spam"),
        url="https://archive.ics.uci.edu/static/public/380/data.csv",
        header=True,
        text_size=1202,
        fields=_YOUTUBE_SPAM_FIELDS,
        labels={"0": 0, "1": 1},
        codes={
            "video": ("Eminem", "Katy Perry", "LMFAO", "Psy", "Shakira"),
        },
    ),
}


def _spec(name: str) -> Images | CIFAR | Audio | Matrix | Table:
    """The dataset called ``name``, or a refusal naming the ones there are."""
    spec = DATASETS.get(name)
    if spec is None:
        raise ValueError(f"the datasets here are {', '.join(DATASETS)}, not {name!r}")
    return spec


def describe(name: str | None = None) -> str:
    """What each dataset is, what its licence says, and where it comes from.

        >>> print(describe("iris"))
        iris           150 iris flowers measured four ways, 3 species
                       3 classes, CC BY 4.0
                       https://archive.ics.uci.edu/dataset/53/iris
    """
    chosen = DATASETS.values() if name is None else [_spec(name)]
    return "\n".join(
        f"{spec.name:<14} {spec.title}\n"
        f"{'':14} {spec.sorting()}, {spec.licence}\n"
        f"{'':14} {spec.source}"
        for spec in chosen
    )


def convert(
    name: str,
    target: Any,
    *,
    split: str | None = None,
    parts: Mapping[str, Any] | None = None,
    base: str | None = None,
    prefix: str | None = None,
    compression: str | None = "zlib",
    basket_size: int | None = None,
    config: Any = None,
) -> dict[str, int]:
    """Write one dataset into a ROOT file, a tree per class; say what went where.

        >>> from xrd.root import datasets                    # doctest: +SKIP
        >>> datasets.convert("iris", "iris.root")            # doctest: +SKIP
        {'setosa': 50, 'versicolor': 50, 'virginica': 50}

    ``target`` is where the file goes - a path, a URL, an open binary file, or
    a :class:`~.writer.WritableFile` already open, which is how several splits
    end up in one file. The data is downloaded from wherever the dataset is
    published unless ``parts`` supplies it: a mapping of the roles in
    :meth:`Images.urls` and its siblings to bytes, a path or a URL. ``base``
    redirects the downloads at a mirror of your own.

    Each tree is named for its split and its class - ``train_frog`` - with the
    split left off when the dataset has only one. Beside them goes an
    ``about`` key holding the licence and the source, so the file keeps saying
    where it came from.
    """
    spec = _spec(name)
    if split is None:
        split = spec.splits[0]
    if split not in spec.splits:
        raise ValueError(
            f"the splits {spec.label} comes in are {' and '.join(spec.splits)}, not {split!r}"
        )
    urls = spec.urls(split)
    if base is not None:
        urls = {role: base + url.rsplit("/", 1)[-1] for role, url in urls.items()}
    supplied = dict(parts or {})
    raw = {role: fetch(supplied.get(role, url), config=config) for role, url in urls.items()}
    classes, columns, rows = spec.rows(raw, split)
    if prefix is None:
        prefix = split if len(spec.splits) > 1 else ""

    given = isinstance(target, WritableFile)
    out = target if given else create(target, compression=compression, config=config)
    try:
        trees = {
            at: out.tree(
                f"{prefix}_{cls}" if prefix else cls,
                columns,
                title=spec.entry_title(split, cls),
                basket_size=spec.basket_size if basket_size is None else basket_size,
            )
            for at, cls in enumerate(classes)
        }
        for at, row in rows:
            tree = trees.get(at)
            if tree is None:
                raise ValueError(
                    f"row {row['index']} of {spec.label} is labelled {at}, and it has "
                    f"{len(classes)} classes"
                )
            tree.fill(**row)
        out.write(f"{prefix}_about" if prefix else "about", spec.about(split))
        written = {tree.name: len(tree) for tree in trees.values()}
    finally:
        if not given:
            out.close()
    return written
