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
