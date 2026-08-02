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
zip, a zip inside a zip, gzip, WAV, ARFF and CSV - and what comes out is
pictures, sound, sentences, tables and plain blocks of numbers, because a
training loop should not have to care which of those it is reading. The CIFAR
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
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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
) -> Iterator[list[str]]:
    """Every row of a delimited text file, as stripped strings, blanks dropped.

    ``delimiter`` is what separates the fields, or ``None`` for any run of
    whitespace, which is how the older sets are written. ``header`` drops the
    first line, and ``comment`` drops every line that starts with it.
    ``quoted`` is whether a double quote groups a field, as it does in a CSV;
    a file of English sentences means its quotes literally, and setting this
    ``False`` keeps them rather than reading them as punctuation.
    """
    text = io.StringIO(raw.decode("utf-8-sig"))
    rows: Iterator[list[str]] = (
        (line.split() for line in text)
        if delimiter is None
        else csv.reader(
            text,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL if quoted else csv.QUOTE_NONE,
        )
    )
    if header:
        next(rows, None)
    for row in rows:
        if any(cell.strip() for cell in row) and not (comment and row[0].startswith(comment)):
            yield [cell.strip() for cell in row]


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
            f"source: {self.source}\nconverted by xrd.root.datasets, one tree per class"
        )

    def entry_title(self, split: str, cls: str) -> str:
        """The title of the tree holding one class."""
        where = f"{split} " if len(self.splits) > 1 else ""
        return f"{self.label} {where}rows labelled {cls}"


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
    #: Whether a double quote groups a field. A table of sentences means its
    #: quotes literally and sets this ``False``.
    quoted: bool = True
    #: How many bytes a ``"text"`` field is given. A tree column is a fixed
    #: size, so the text is written into one this wide with a ``_length``
    #: column beside it; anything longer is refused rather than cut short.
    text_size: int = 0
    #: The fields in the order the file writes them: a name, and either a
    #: typecode, ``"label"``, ``"text"``, or the name of an entry in
    #: :attr:`codes`.
    fields: tuple[tuple[str, str], ...]
    #: What each label in the file means, as an index into :attr:`classes`.
    labels: Mapping[str, int]
    #: How the categorical fields are numbered, either as a mapping or as the
    #: categories in code order - a string of them where each is one letter.
    #: A category nobody wrote down is refused rather than guessed at.
    codes: Mapping[str, Mapping[str, int] | Sequence[str]] = field(default_factory=dict)

    def urls(self, split: str) -> dict[str, str]:
        """The one file it comes in."""
        return {"table": self.url}

    def rows(self, raw: Mapping[str, bytes], split: str) -> Loaded:
        """The columns the fields become, then the rows themselves."""
        columns: dict[str, Any] = {}
        for name, role in self.fields:
            if role == "label":
                continue
            if role == "text":
                columns[name] = ("B", self.text_size)
                columns[f"{name}_length"] = "i"
            else:
                columns[name] = role if role == "d" else "i"
        columns["label"] = "i"
        columns["index"] = "i"
        return self.classes, columns, self._entries(raw["table"], split)

    def _text(self, cell: str, name: str, index: int) -> bytes:
        """One field as the bytes its column holds, padded out to the width of it."""
        written = cell.encode()
        if len(written) > self.text_size:
            raise ValueError(
                f"row {index} of {self.label} has {len(written)} bytes in {name}, and the "
                f"column holds {self.text_size}"
            )
        return written + bytes(self.text_size - len(written))

    def _cell(
        self, cell: str, name: str, role: str, index: int, coded: Mapping[str, Mapping[str, int]]
    ) -> Any:
        """One field as the number its column holds, or what a gap means there."""
        if role == "d":
            return nan if cell in MISSING else _number(cell, float, name, index, self.label)
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

    def _entries(self, raw: bytes, split: str) -> Rows:
        held = _member(raw, self.files.get(split, self.member))
        table = (
            read_arff(held)
            if self.arff
            else read_table(
                held,
                delimiter=self.delimiter,
                header=self.header,
                comment=self.comment,
                quoted=self.quoted,
            )
        )
        coded = {role: _numbered(kinds) for role, kinds in self.codes.items()}
        for index, cells in enumerate(table):
            if len(cells) != len(self.fields):
                raise ValueError(
                    f"row {index} of {self.label} has {len(cells)} fields, and its columns "
                    f"are {len(self.fields)}"
                )
            which = -1
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


#: Every dataset this module knows, by the name :func:`convert` takes.
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
        f"{'':14} {len(spec.classes) or 100} classes, {spec.licence}\n"
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
