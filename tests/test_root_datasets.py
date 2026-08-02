"""The teaching datasets, out of their own archives and into trees.

Every archive here is built in the test rather than downloaded, so the whole
file runs without a network: a tar the shape CIFAR ships, a zip the shape UCI
ships, and IDX bytes the shape MNIST ships. The registry entries are checked
against what the real archives hold, and the readers against bytes laid out
the way the real ones are.
"""

from __future__ import annotations

import gzip
import io
import struct
import tarfile
import wave
import zipfile
from collections.abc import Sequence
from dataclasses import replace

import pytest

from xrd.root import open_root
from xrd.root.datasets import (
    CIFAR,
    DATASETS,
    IDX_FILES,
    MISSING,
    Audio,
    Dataset,
    Images,
    Matrix,
    Table,
    convert,
    describe,
    read_arff,
    read_table,
    read_xlsx,
)
from xrd.root.writer import create


def tarred(members: dict[str, bytes], *, folders: tuple[str, ...] = ()) -> bytes:
    """A gzipped tar laid out the way the CIFAR archives are, one folder deep."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        for name in folders:
            entry = tarfile.TarInfo(f"batches/{name}")
            entry.type = tarfile.DIRTYPE
            tar.addfile(entry)
        for name, data in members.items():
            entry = tarfile.TarInfo(f"batches/{name}")
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    return raw.getvalue()


def zipped(members: dict[str, bytes]) -> bytes:
    """A zip the shape UCI serves."""
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return raw.getvalue()


#: A CIFAR-shaped archive small enough to write out by hand: two by two
#: pictures, three colour planes, one label byte in front.
TINY = CIFAR(
    name="tiny",
    label="Tiny",
    title="four small pictures",
    licence="CC0",
    source="https://example.invalid/tiny",
    classes=("cat", "dog"),
    splits=("train", "test"),
    archive="https://example.invalid/tiny.tar.gz",
    files={"train": ("one.bin",), "test": ("test.bin",)},
    meta="names.txt",
    side=2,
)

#: The same again with a coarse label in front of the fine one, as CIFAR-100
#: writes it.
TINY_COARSE = CIFAR(
    name="tiny_coarse",
    label="Tiny-100",
    title="four small pictures in a hierarchy",
    licence="CC0",
    source="https://example.invalid/tiny",
    classes=(),
    splits=("train",),
    archive="https://example.invalid/tiny.tar.gz",
    files={"train": ("one.bin",)},
    meta="names.txt",
    coarse="coarse.txt",
    side=2,
)

#: A table with one of everything a table can hold: a float, an int, a
#: category and the label.
FLOWERS = Table(
    name="flowers",
    label="Flowers",
    title="a few flowers",
    licence="CC0",
    source="https://example.invalid/flowers",
    classes=("red", "blue"),
    url="https://example.invalid/flowers.csv",
    fields=(("width", "d"), ("count", "i"), ("where", "where"), ("kind", "label")),
    labels={"Red": 0, "Blue": 1},
    codes={"where": {"north": 0, "south": 1}},
)

FLOWER_ROWS = b"1.5,3,north,Red\n2.5,4,south,Blue\n3.5,5,north,Red\n"


#: An archive of recordings the shape a spoken-word set ships: a folder of
#: WAV files named for what was said and who said it.
SPOKEN = Audio(
    name="spoken",
    label="Spoken",
    title="a few short recordings",
    licence="CC0",
    source="https://example.invalid/spoken",
    classes=("zero", "one"),
    archive="https://example.invalid/spoken.zip",
    folder="clips",
    labels={"0": 0, "1": 1},
    speakers=("ann", "bob"),
    rate=8000,
    samples=6,
)

#: An image set that arrives in one archive rather than four files, the way
#: EMNIST does.
SCRIBBLES = Images(
    name="scribbles",
    label="Scribbles",
    title="a few small scribbles",
    licence="CC0",
    source="https://example.invalid/scribbles",
    classes=("up", "down"),
    splits=("train", "test"),
    archive="https://example.invalid/scribbles.zip",
    files={
        split: (f"in/{split}-images.gz", f"in/{split}-labels.gz")
        for split in ("train", "test")
    },
    side=2,
)


#: A table whose second field is a sentence, which means its quotes literally
#: and needs a column wide enough to hold it.
MESSAGES = Table(
    name="messages",
    label="Messages",
    title="a few short messages",
    licence="CC0",
    source="https://example.invalid/messages",
    classes=("plain", "shouted"),
    url="https://example.invalid/messages.zip",
    member="messages.txt",
    delimiter="\t",
    quoted=False,
    text_size=16,
    fields=(("kind", "label"), ("message", "text")),
    labels={"plain": 0, "shouted": 1},
)

MESSAGE_ROWS = b'plain\the said "no"\nshouted\tOI\n'

#: A block of numbers with its labels in a file beside it, inside an archive
#: inside the archive, which is how the phone-sensor sets arrive.
SENSORS = Matrix(
    name="sensors",
    label="Sensors",
    title="a few windows of sensor readings",
    licence="CC0",
    source="https://example.invalid/sensors",
    classes=("still", "moving"),
    splits=("train", "test"),
    url="https://example.invalid/sensors.zip",
    inner="inner.zip",
    files={split: f"{split}/X.txt" for split in ("train", "test")},
    label_files={split: f"{split}/y.txt" for split in ("train", "test")},
    labels={"1": 0, "2": 1},
    beside={"subject": {split: f"{split}/subject.txt" for split in ("train", "test")}},
    width=3,
)

#: A block of numbers whose label is the last columns of the row itself.
HOT = Matrix(
    name="hot",
    label="Hot",
    title="a few rows labelled where they lie",
    licence="CC0",
    source="https://example.invalid/hot",
    classes=("left", "right"),
    url="https://example.invalid/hot.data",
    files={"all": ""},
    width=2,
    column="image",
    kind="B",
    onehot=2,
)

#: A block of numbers with no labels in it at all, only a line at the top
#: saying how many rows each class has in turn.
RUNS = Matrix(
    name="runs",
    label="Runs",
    title="a few rows counted at the top",
    licence="CC0",
    source="https://example.invalid/runs",
    classes=("signal", "background"),
    url="https://example.invalid/runs.txt",
    files={"all": ""},
    width=2,
    counts=True,
)


#: A table with no classes at all: a date and a number to predict from it.
PRICES = Table(
    name="prices",
    label="Prices",
    title="a few days and what a thing cost",
    licence="CC0",
    source="https://example.invalid/prices",
    classes=(),
    splits=("north", "south"),
    url="https://example.invalid/prices.zip",
    files={"north": "north.csv", "south": "south.csv"},
    fields=(("when", "date"), ("price", "target")),
)

#: A table of numbers that ends in free text with spaces in it, the way the
#: older whitespace-separated files do.
CARS = Table(
    name="cars",
    label="Cars",
    title="a few cars and how far they went",
    licence="CC0",
    source="https://example.invalid/cars",
    classes=(),
    url="https://example.invalid/cars.data",
    delimiter=None,
    tail=3,
    text_size=16,
    fields=(("mpg", "target"), ("cylinders", "i"), ("car_name", "text")),
)

#: A table that arrives as a spreadsheet rather than as text.
SHEET = Table(
    name="sheet",
    label="Sheet",
    title="a few rows somebody typed into a spreadsheet",
    licence="CC0",
    source="https://example.invalid/sheet",
    classes=(),
    url="https://example.invalid/sheet.zip",
    member="book.xlsx",
    xlsx=True,
    header=True,
    fields=(("width", "d"), ("height", "target")),
)

#: The namespace a spreadsheet writes everything it holds in.
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def celled(ref: str, value: str) -> str:
    """One cell: a number, an ``s:`` shared string, an ``i:`` inline one, or nothing."""
    if value.startswith("s:"):
        return f'<c r="{ref}" t="s"><v>{value[2:]}</v></c>'
    if value.startswith("i:"):
        return f'<c r="{ref}" t="inlineStr"><is><t>{value[2:]}</t></is></c>'
    return f'<c r="{ref}"><v>{value}</v></c>' if value else f'<c r="{ref}"/>'


def spreadsheet(
    rows: Sequence[dict[str, str]], *, shared: Sequence[str] = (), sheets: int = 1
) -> bytes:
    """A spreadsheet the shape Excel writes one: a zip of XML, text kept once."""
    members = {
        f"xl/worksheets/sheet{at}.xml": f'<worksheet xmlns="{SHEET_NS}"><sheetData/></worksheet>'
        for at in range(2, sheets + 1)
    }
    if shared:
        members["xl/sharedStrings.xml"] = (
            f'<sst xmlns="{SHEET_NS}">'
            + "".join(f"<si><t>{word}</t></si>" for word in shared)
            + "</sst>"
        )
    body = "".join(
        f'<row r="{at}">' + "".join(celled(ref, value) for ref, value in row.items()) + "</row>"
        for at, row in enumerate(rows, 1)
    )
    members["xl/worksheets/sheet1.xml"] = (
        f'<worksheet xmlns="{SHEET_NS}"><sheetData>{body}</sheetData></worksheet>'
    )
    return zipped({name: data.encode() for name, data in members.items()})


def sensed(rows: str, labels: str, subjects: str, *, split: str = "train") -> bytes:
    """The archive SENSORS describes, which is a zip wrapped in another zip."""
    inner = zipped(
        {
            f"{split}/X.txt": rows.encode(),
            f"{split}/y.txt": labels.encode(),
            f"{split}/subject.txt": subjects.encode(),
        }
    )
    return zipped({"inner.zip": inner})


def waved(samples: Sequence[int], *, rate: int = 8000, channels: int = 1, width: int = 2) -> bytes:
    """One WAV file, laid out the way a recording arrives in an archive."""
    raw = io.BytesIO()
    with wave.open(raw, "wb") as clip:
        clip.setnchannels(channels)
        clip.setsampwidth(width)
        clip.setframerate(rate)
        clip.writeframes(
            struct.pack(f"<{len(samples)}h", *samples) if width == 2 else bytes(samples)
        )
    return raw.getvalue()


def clips(**names: bytes) -> bytes:
    """A zip of recordings, one folder deep, the way a repository downloads."""
    return zipped({f"spoken-master/clips/{name}.wav": data for name, data in names.items()})


def idx(*shape: int, values: bytes) -> bytes:
    """IDX bytes: the magic, the shape, and the values, gzipped as they ship."""
    head = b"\x00\x00\x08" + bytes([len(shape)]) + struct.pack(f">{len(shape)}i", *shape)
    return gzip.compress(head + values)


def scribbled(labels: bytes) -> bytes:
    """The archive SCRIBBLES describes: two by two pictures and their labels."""
    members = {}
    for split in ("train", "test"):
        members[f"in/{split}-images.gz"] = idx(
            len(labels), 2, 2, values=bytes(range(4 * len(labels)))
        )
        members[f"in/{split}-labels.gz"] = idx(len(labels), values=labels)
    return zipped(members)


def pictures(labels: bytes, *, side: int = 2, coarse: bytes = b"") -> bytes:
    """Records of a label and three colour planes, each pixel its own number."""
    width = 3 * side * side
    out = bytearray()
    for index, label in enumerate(labels):
        if coarse:
            out.append(coarse[index])
        out.append(label)
        out += bytes((index * width + pixel) % 256 for pixel in range(width))
    return bytes(out)


def tiny_archive(**names: bytes) -> bytes:
    """The tiny archive with its label list and whichever members were asked for."""
    return tarred({"names.txt": b"cat\ndog\n", **names})


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test's own datasets, alongside the real ones, for the length of a test."""
    for spec in (TINY, TINY_COARSE, FLOWERS, SPOKEN, SCRIBBLES, MESSAGES, SENSORS, HOT, RUNS,
                 PRICES, CARS, SHEET):
        monkeypatch.setitem(DATASETS, spec.name, spec)


# --- the registry itself ----------------------------------------------------


def test_every_dataset_is_named_after_the_key_it_is_filed_under():
    assert all(name == spec.name for name, spec in DATASETS.items())


def test_every_dataset_says_what_its_licence_is_and_where_it_came_from():
    for spec in DATASETS.values():
        assert spec.licence
        assert spec.source.startswith("http")
        assert spec.title


def test_the_datasets_asked_for_are_all_there():
    assert set(DATASETS) == {
        "mnist", "fashion_mnist", "kmnist", "cifar10", "cifar100",
        "iris", "penguins", "covertype", "emnist", "fsdd", "adult", "mushroom",
        "letter", "digits", "wine", "breast_cancer", "dry_bean", "seeds",
        "miniboone", "har", "semeion", "sms_spam", "wine_quality", "spambase",
        "ionosphere", "glass", "abalone", "banknote",
        "magic", "htru2", "auto_mpg", "bike_sharing", "energy_efficiency",
        "real_estate", "student", "heart_disease", "car_evaluation", "yeast",
    }


def test_class_names_can_all_be_written_as_tree_names():
    for spec in DATASETS.values():
        for cls in spec.classes:
            assert cls.replace("_", "").isalnum(), (spec.name, cls)


def test_the_image_sets_agree_on_the_four_files_mnist_named():
    for name in ("mnist", "fashion_mnist", "kmnist"):
        spec = DATASETS[name]
        assert isinstance(spec, Images)
        assert spec.files == IDX_FILES
        assert spec.urls("train")["images"].endswith("train-images-idx3-ubyte.gz")
        assert spec.urls("test")["labels"].endswith("t10k-labels-idx1-ubyte.gz")
        assert len(spec.classes) == 10


def test_the_cifar_sets_take_the_binary_distribution_and_not_the_pickled_one():
    for name in ("cifar10", "cifar100"):
        spec = DATASETS[name]
        assert isinstance(spec, CIFAR)
        assert spec.archive.endswith("-binary.tar.gz")
        assert spec.urls("train") == spec.urls("test") == {"archive": spec.archive}


def test_cifar100_carries_a_coarse_label_and_cifar10_does_not():
    ten, hundred = DATASETS["cifar10"], DATASETS["cifar100"]
    assert isinstance(ten, CIFAR) and isinstance(hundred, CIFAR)
    assert not ten.coarse
    assert hundred.coarse == "coarse_label_names.txt"


def test_covertype_has_the_fifty_four_features_and_the_label_the_paper_describes():
    spec = DATASETS["covertype"]
    assert isinstance(spec, Table)
    assert len(spec.fields) == 55
    assert sum(role == "label" for _, role in spec.fields) == 1
    assert spec.fields[0] == ("elevation", "i")
    assert ("soil_type_40", "i") in spec.fields
    assert spec.urls("all") == {"table": spec.url}


def test_every_table_names_a_code_for_every_category_it_will_meet():
    for spec in DATASETS.values():
        if isinstance(spec, Table):
            plain = {"d", "i", "label", "text", "date", "target"}
            assert {role for _, role in spec.fields} - plain == set(spec.codes), spec.name


def test_a_set_with_no_classes_has_a_number_to_predict_instead():
    for spec in DATASETS.values():
        if isinstance(spec, Table):
            roles = [role for _, role in spec.fields]
            assert bool(spec.classes) == ("label" in roles), spec.name
            assert bool(spec.classes) != ("target" in roles), spec.name


def test_no_dataset_labels_a_class_it_does_not_have():
    for spec in DATASETS.values():
        if isinstance(spec, (Table, Audio, Matrix)) and spec.classes and spec.labels:
            assert max(spec.labels.values()) == len(spec.classes) - 1, spec.name
            assert min(spec.labels.values()) == 0, spec.name


# --- reading the shapes the archives come in --------------------------------


def test_a_table_is_read_row_by_row_with_the_blank_lines_dropped():
    assert list(read_table(b"1,2\n\n3,4\n")) == [["1", "2"], ["3", "4"]]


def test_a_header_is_skipped_when_the_file_has_one():
    raw = b"a,b\n1,2\n"
    assert list(read_table(raw)) == [["a", "b"], ["1", "2"]]
    assert list(read_table(raw, header=True)) == [["1", "2"]]


def test_a_table_can_be_separated_by_something_other_than_a_comma():
    assert list(read_table(b'"a";"b"\n1;2\n', delimiter=";", header=True)) == [["1", "2"]]


def test_quoted_fields_and_a_byte_order_mark_are_both_handled():
    assert list(read_table('﻿"one, two",3\n'.encode())) == [["one, two", "3"]]


def test_whitespace_around_a_field_is_not_part_of_it():
    assert list(read_table(b" 1 , 2 \n")) == [["1", "2"]]


def test_a_dataset_says_what_it_is_in_a_string_the_file_keeps():
    about = DATASETS["cifar10"].about("train")
    assert "CIFAR-10" in about
    assert "split: train" in about
    assert "licence: no formal licence" in about
    assert "https://www.cs.toronto.edu/~kriz/cifar.html" in about


def test_a_single_split_dataset_does_not_talk_about_its_split():
    assert "split:" not in DATASETS["iris"].about("all")


def test_a_tree_title_says_the_class_and_the_split_when_there_is_one():
    assert DATASETS["mnist"].entry_title("train", "7") == (
        "MNIST train images of class 7, 28x28 greyscale"
    )
    assert DATASETS["iris"].entry_title("all", "setosa") == "Iris rows labelled setosa"
    assert DATASETS["cifar10"].entry_title("test", "frog") == (
        "CIFAR-10 test images of class frog, 32x32 colour, three planes"
    )


def test_a_dataset_with_no_reader_of_its_own_still_describes_itself():
    plain = Dataset(
        name="plain", label="Plain", title="nothing much", licence="CC0",
        source="https://example.invalid/plain", classes=("one",),
    )
    assert plain.entry_title("all", "one") == "Plain rows labelled one"
    assert "licence: CC0" in plain.about("all")


def test_describe_lists_every_dataset_with_its_licence():
    listing = describe()
    for spec in DATASETS.values():
        assert spec.name in listing
        assert spec.licence in listing
    assert "100 classes" in listing


def test_describe_can_be_asked_about_one_dataset():
    assert describe("iris").splitlines()[0].startswith("iris ")
    assert "CC BY 4.0" in describe("iris")
    assert "mnist" not in describe("iris")


def test_a_dataset_nobody_has_is_refused_by_name():
    with pytest.raises(ValueError, match=r"the datasets here are mnist.*not 'imagenet'"):
        describe("imagenet")


# --- CIFAR ------------------------------------------------------------------


def test_a_cifar_archive_is_read_a_record_at_a_time(registry):
    raw = tiny_archive(**{"one.bin": pictures(bytes([0, 1, 1]))})
    classes, columns, rows = TINY.rows({"archive": raw}, "train")
    assert classes == ("cat", "dog")
    assert columns == {"image": ("B", 12), "label": "i", "index": "i"}
    got = list(rows)
    assert [label for label, _ in got] == [0, 1, 1]
    assert [row["index"] for _, row in got] == [0, 1, 2]
    assert bytes(got[0][1]["image"]) == bytes(range(12))
    assert bytes(got[1][1]["image"]) == bytes(range(12, 24))


def test_the_pictures_of_a_split_run_on_across_the_files_it_is_in():
    spec = replace(TINY, files={"train": ("one.bin", "two.bin"), "test": ("test.bin",)})
    raw = tiny_archive(**{"one.bin": pictures(b"\0\0"), "two.bin": pictures(b"\1")})
    _, _, rows = spec.rows({"archive": raw}, "train")
    assert [row["index"] for _, row in rows] == [0, 1, 2]


def test_a_coarse_label_is_read_in_front_of_the_fine_one(registry):
    raw = tarred({"names.txt": b"tulip\nrose\n", "coarse.txt": b"flower\n",
                  "one.bin": pictures(bytes([1, 0]), coarse=bytes([0, 0]))})
    classes, columns, rows = TINY_COARSE.rows({"archive": raw}, "train")
    assert classes == ("tulip", "rose")
    assert list(columns) == ["image", "label", "coarse", "index"]
    got = list(rows)
    assert [label for label, _ in got] == [1, 0]
    assert [row["coarse"] for _, row in got] == [0, 0]
    assert bytes(got[0][1]["image"]) == bytes(range(12))


def test_the_class_names_come_out_of_the_archive_and_are_made_safe_to_use():
    spec = CIFAR(
        name="odd", label="Odd", title="odd names", licence="CC0",
        source="https://example.invalid/odd", classes=(), splits=("train",),
        archive="", files={"train": ("one.bin",)}, meta="names.txt", side=2,
    )
    raw = tarred({"names.txt": b"aquarium fish\n\nmaple-tree\n", "one.bin": pictures(b"\0")})
    classes, _, rows = spec.rows({"archive": raw}, "train")
    assert classes == ("aquarium_fish", "maple_tree")
    list(rows)


def test_an_archive_whose_classes_are_not_the_expected_ones_is_refused():
    raw = tarred({"names.txt": b"cat\nferret\n", "one.bin": pictures(b"\0")})
    with pytest.raises(ValueError, match="says its classes are cat, ferret, and Tiny has cat, dog"):
        TINY.rows({"archive": raw}, "train")


def test_a_member_the_archive_does_not_hold_is_refused_by_name():
    raw = tarred({"names.txt": b"cat\ndog\n"})
    _, _, rows = TINY.rows({"archive": raw}, "train")
    with pytest.raises(ValueError, match=r"this archive holds .*names.txt.*and not 'one.bin'"):
        list(rows)


def test_a_meta_member_the_archive_does_not_hold_is_refused_too():
    with pytest.raises(ValueError, match=r"and not 'names.txt'"):
        TINY.rows({"archive": tarred({"one.bin": pictures(b"\0")})}, "train")


def test_a_folder_where_a_file_should_be_is_not_read_as_one():
    raw = tarred({"names.txt": b"cat\ndog\n"}, folders=("one.bin",))
    _, _, rows = TINY.rows({"archive": raw}, "train")
    with pytest.raises(ValueError, match=r"and not 'one.bin'"):
        list(rows)


def test_a_file_that_does_not_divide_into_records_is_refused():
    raw = tiny_archive(**{"one.bin": pictures(b"\0")[:-1]})
    _, _, rows = TINY.rows({"archive": raw}, "train")
    with pytest.raises(ValueError, match=r"one.bin is 12 bytes, and Tiny records are 13 bytes"):
        list(rows)


def test_the_archive_is_let_go_of_once_its_records_have_been_read():
    _, _, rows = TINY.rows({"archive": tiny_archive(**{"test.bin": pictures(b"\0")})}, "test")
    assert len(list(rows)) == 1
    with pytest.raises(StopIteration):
        next(rows)


# --- tables -----------------------------------------------------------------


def test_a_table_becomes_one_column_per_field_with_the_label_beside_them():
    classes, columns, rows = FLOWERS.rows({"table": FLOWER_ROWS}, "all")
    assert classes == ("red", "blue")
    assert columns == {"width": "d", "count": "i", "where": "i", "label": "i", "index": "i"}
    got = list(rows)
    assert [label for label, _ in got] == [0, 1, 0]
    assert got[0][1] == {"width": 1.5, "count": 3, "where": 0, "label": 0, "index": 0}
    assert got[1][1]["where"] == 1


def test_a_table_can_arrive_gzipped_or_as_it_is():
    rows, gz = FLOWER_ROWS, gzip.compress(FLOWER_ROWS)
    assert [row["index"] for _, row in FLOWERS._entries(rows, "all")] == [0, 1, 2]
    assert [row["index"] for _, row in FLOWERS._entries(gz, "all")] == [0, 1, 2]


def test_a_member_is_taken_out_of_a_zip_and_unzipped_again_if_it_is_gzipped():
    inner = zipped({"rows.csv.gz": gzip.compress(FLOWER_ROWS)})
    spec = Table(
        name="z", label="Z", title="zipped", licence="CC0", source="https://example.invalid/z",
        classes=FLOWERS.classes, url="", member="rows.csv.gz", fields=FLOWERS.fields,
        labels=FLOWERS.labels, codes=FLOWERS.codes,
    )
    assert len(list(spec._entries(inner, "all"))) == 3


def test_a_zip_without_the_member_wanted_names_what_it_does_hold():
    spec = Table(
        name="z", label="Z", title="zipped", licence="CC0", source="https://example.invalid/z",
        classes=FLOWERS.classes, url="", member="rows.data", fields=FLOWERS.fields,
        labels=FLOWERS.labels, codes=FLOWERS.codes,
    )
    with pytest.raises(ValueError, match=r"this zip holds other.csv, and not 'rows.data'"):
        list(spec._entries(zipped({"other.csv": FLOWER_ROWS}), "all"))


def test_a_gap_in_a_number_becomes_a_nan_and_a_gap_in_a_category_a_minus_one():
    for blank in sorted(MISSING - {""}):
        _, _, rows = FLOWERS.rows({"table": f"{blank},{blank},{blank},Red\n".encode()}, "all")
        (_, row), = rows
        assert row["width"] != row["width"]
        assert row["count"] == -1
        assert row["where"] == -1


def test_a_row_with_the_wrong_number_of_fields_is_refused():
    _, _, rows = FLOWERS.rows({"table": b"1.5,3,north\n"}, "all")
    with pytest.raises(ValueError, match="row 0 of Flowers has 3 fields, and its columns are 4"):
        list(rows)


def test_a_label_nobody_declared_is_refused_with_the_ones_that_were():
    _, _, rows = FLOWERS.rows({"table": b"1.5,3,north,Green\n"}, "all")
    with pytest.raises(ValueError, match=r"row 0 of Flowers is labelled 'Green'.*are Blue, Red"):
        list(rows)


def test_a_category_nobody_declared_is_refused_rather_than_guessed_at():
    _, _, rows = FLOWERS.rows({"table": b"1.5,3,east,Red\n"}, "all")
    with pytest.raises(ValueError, match="'east' in where, and the ones it knows are north, south"):
        list(rows)


@pytest.mark.parametrize(
    "cell,column", [("wide,3,north,Red", "width"), ("1.5,many,north,Red", "count")]
)
def test_a_field_that_is_not_a_number_says_which_row_and_column_it_was_in(cell, column):
    _, _, rows = FLOWERS.rows({"table": cell.encode() + b"\n"}, "all")
    with pytest.raises(ValueError, match=rf"row 0 of Flowers has .* in {column}, which is not"):
        list(rows)


# --- converting -------------------------------------------------------------


def written(name: str, source: bytes, role: str, **kwargs) -> tuple[dict[str, int], bytes]:
    """One dataset converted into memory, and the bytes it came to."""
    buf = io.BytesIO()
    counts = convert(name, buf, parts={role: source}, **kwargs)
    return counts, buf.getvalue()


def test_a_dataset_becomes_one_tree_per_class_named_for_the_split(registry):
    counts, raw = written("tiny", tiny_archive(**{"test.bin": pictures(b"\0\1\1")}), "archive",
                          split="test")
    assert counts == {"test_cat": 1, "test_dog": 2}
    with open_root(io.BytesIO(raw)) as back:
        assert sorted(back.keys()) == ["test_about", "test_cat", "test_dog"]
        tree = back["test_dog"]
        assert tree.num_entries == 2
        assert tree.title == "Tiny test images of class dog, 2x2 colour, three planes"
        assert list(tree["image"].array()) == list(range(12, 36))


def test_a_class_with_no_rows_still_gets_a_tree_of_its_own(registry):
    counts, raw = written("tiny", tiny_archive(**{"test.bin": pictures(b"\0")}), "archive",
                          split="test")
    assert counts == {"test_cat": 1, "test_dog": 0}
    with open_root(io.BytesIO(raw)) as back:
        assert back["test_dog"].num_entries == 0


def test_a_dataset_with_one_split_leaves_the_split_out_of_the_tree_names(registry):
    counts, raw = written("flowers", FLOWER_ROWS, "table")
    assert counts == {"red": 2, "blue": 1}
    with open_root(io.BytesIO(raw)) as back:
        assert sorted(back.keys()) == ["about", "blue", "red"]
        assert back["red"].title == "Flowers rows labelled red"


def test_the_file_says_what_it_holds_and_what_its_licence_is(registry):
    _, raw = written("flowers", FLOWER_ROWS, "table")
    with open_root(io.BytesIO(raw)) as back:
        about = back["about"]
    assert "Flowers: a few flowers" in about
    assert "licence: CC0" in about
    assert "one tree per class" in about


def test_the_prefix_can_be_chosen_rather_than_taken_from_the_split(registry):
    counts, raw = written("flowers", FLOWER_ROWS, "table", prefix="v1")
    assert counts == {"v1_red": 2, "v1_blue": 1}
    with open_root(io.BytesIO(raw)) as back:
        assert "v1_about" in back.keys()


def test_two_splits_can_be_written_into_one_file(registry):
    archive = tiny_archive(**{"one.bin": pictures(b"\0"), "test.bin": pictures(b"\1")})
    buf = io.BytesIO()
    with create(buf) as out:
        convert("tiny", out, split="train", parts={"archive": archive})
        convert("tiny", out, split="test", parts={"archive": archive})
    with open_root(io.BytesIO(buf.getvalue())) as back:
        assert sorted(back.keys()) == [
            "test_about", "test_cat", "test_dog", "train_about", "train_cat", "train_dog",
        ]
        assert back["train_cat"].num_entries == 1
        assert back["test_dog"].num_entries == 1


def test_the_split_defaults_to_the_first_one_the_dataset_has(registry):
    counts, _ = written("tiny", tiny_archive(**{"one.bin": pictures(b"\0")}), "archive")
    assert set(counts) == {"train_cat", "train_dog"}


def test_a_split_the_dataset_does_not_come_in_is_refused(registry):
    with pytest.raises(ValueError, match="the splits Tiny comes in are train and test, not 'val'"):
        convert("tiny", io.BytesIO(), split="val")


def test_a_dataset_nobody_has_is_refused_before_anything_is_downloaded():
    with pytest.raises(ValueError, match=r"the datasets here are mnist.*not 'imagenet'"):
        convert("imagenet", io.BytesIO())


def test_a_label_past_the_end_of_the_class_list_is_refused(registry):
    raw = tarred({"names.txt": b"cat\ndog\n", "one.bin": pictures(bytes([0, 7]))})
    with pytest.raises(ValueError, match="row 1 of Tiny is labelled 7, and it has 2 classes"):
        convert("tiny", io.BytesIO(), split="train", parts={"archive": raw})


def test_the_downloads_can_be_pointed_at_a_mirror(registry, tmp_path):
    archive = tmp_path / "tiny.tar.gz"
    archive.write_bytes(tiny_archive(**{"one.bin": pictures(b"\0")}))
    counts = convert("tiny", io.BytesIO(), split="train", base=f"{tmp_path}/")
    assert counts == {"train_cat": 1, "train_dog": 0}


def test_the_basket_size_and_the_compression_can_both_be_chosen(registry):
    archive = tiny_archive(**{"one.bin": pictures(b"\0\0\0\0")})
    small, plain = io.BytesIO(), io.BytesIO()
    convert("tiny", small, split="train", basket_size=16, parts={"archive": archive})
    convert("tiny", plain, split="train", compression=None, parts={"archive": archive})
    for raw in (small.getvalue(), plain.getvalue()):
        with open_root(io.BytesIO(raw)) as back:
            assert back["train_cat"].num_entries == 4
    with open_root(io.BytesIO(small.getvalue())) as back:
        assert back["train_cat"]["image"].num_baskets == 2
    with open_root(io.BytesIO(plain.getvalue())) as back:
        assert back["train_cat"]["image"].num_baskets == 1


def test_a_table_that_needs_no_archive_reads_straight_from_a_path(registry, tmp_path):
    rows = tmp_path / "flowers.csv"
    rows.write_bytes(FLOWER_ROWS)
    assert convert("flowers", io.BytesIO(), parts={"table": str(rows)}) == {"red": 2, "blue": 1}


# --- the ten that came later ------------------------------------------------


def test_emnist_takes_the_balanced_split_out_of_the_one_archive_it_ships_in():
    spec = DATASETS["emnist"]
    assert isinstance(spec, Images)
    assert spec.urls("train") == spec.urls("test") == {"archive": spec.archive}
    assert len(spec.classes) == 47
    assert spec.classes[:11] == ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "upper_a")
    assert spec.classes[-1] == "lower_t"
    for split in ("train", "test"):
        images, labels = spec.files[split]
        assert images == f"gzip/emnist-balanced-{split}-images-idx3-ubyte.gz"
        assert labels == f"gzip/emnist-balanced-{split}-labels-idx1-ubyte.gz"


def test_the_spoken_digits_name_each_of_their_speakers_once():
    spec = DATASETS["fsdd"]
    assert isinstance(spec, Audio)
    assert len(set(spec.speakers)) == len(spec.speakers) == 6
    assert spec.urls("all") == {"archive": spec.archive}
    assert spec.rate == 8000
    assert spec.samples > 18262  # the longest recording in the set


def test_adult_knows_both_ways_its_two_classes_are_spelled():
    spec = DATASETS["adult"]
    assert isinstance(spec, Table)
    assert set(spec.labels) == {"<=50K", ">50K", "<=50K.", ">50K."}
    assert spec.files == {"train": "adult.data", "test": "adult.test"}
    assert spec.comment == "|"


def test_mushroom_spells_its_categories_as_the_letters_the_file_uses():
    spec = DATASETS["mushroom"]
    assert isinstance(spec, Table)
    assert len(spec.fields) == 23
    assert spec.codes["odour"] == "alcyfmnps"
    assert set(spec.codes) == {name for name, _ in spec.fields[1:]}


def test_the_optical_digits_are_sixty_four_counts_of_ink_and_a_label():
    spec = DATASETS["digits"]
    assert isinstance(spec, Table)
    assert len(spec.fields) == 65
    assert spec.fields[0] == ("pixel_00", "i")
    assert spec.fields[-1] == ("digit", "label")
    assert spec.files == {"train": "optdigits.tra", "test": "optdigits.tes"}


def test_the_dry_beans_are_read_from_the_arff_and_the_seeds_from_whitespace():
    beans, seeds = DATASETS["dry_bean"], DATASETS["seeds"]
    assert isinstance(beans, Table) and isinstance(seeds, Table)
    assert beans.arff and beans.member.endswith(".arff")
    assert seeds.delimiter is None
    assert not seeds.arff


# --- reading the shapes the later ten come in -------------------------------


def test_a_table_can_be_split_on_any_run_of_whitespace():
    assert list(read_table(b"1\t2\t\t3\n 4  5\t6 \n", delimiter=None)) == [
        ["1", "2", "3"],
        ["4", "5", "6"],
    ]


def test_a_line_the_file_marks_as_not_data_is_dropped():
    raw = b"|a note\n1,2\n"
    assert list(read_table(raw)) == [["|a note"], ["1", "2"]]
    assert list(read_table(raw, comment="|")) == [["1", "2"]]


def test_an_arff_file_is_read_from_its_data_line_onwards():
    raw = b"% who wrote it\n@RELATION beans\n@ATTRIBUTE area INTEGER\n@DATA\n% a note\n1,x\n2,y\n"
    assert list(read_arff(raw)) == [["1", "x"], ["2", "y"]]


def test_the_data_line_of_an_arff_file_is_found_whatever_its_case():
    assert list(read_arff(b"@relation r\n  @data  \n1,2\n")) == [["1", "2"]]


def test_something_with_no_data_line_is_not_an_arff_file():
    with pytest.raises(ValueError, match="no @data line at all"):
        read_arff(b"@relation r\n@attribute a REAL\n")


def test_an_image_set_can_arrive_in_one_archive_rather_than_four_files():
    assert SCRIBBLES.urls("train") == {"archive": SCRIBBLES.archive}
    classes, columns, rows = SCRIBBLES.rows({"archive": scribbled(b"\0\1")}, "test")
    assert classes == ("up", "down")
    assert columns == {"image": ("B", 4), "label": "i", "index": "i"}
    got = list(rows)
    assert [label for label, _ in got] == [0, 1]
    assert list(got[1][1]["image"]) == [4, 5, 6, 7]


def test_a_split_can_name_its_own_member_of_the_archive():
    spec = replace(
        FLOWERS,
        splits=("train", "test"),
        files={"train": "a.csv", "test": "b.csv"},
    )
    archive = zipped({"a.csv": FLOWER_ROWS, "b.csv": b"9.5,1,south,Blue\n"})
    assert [row["width"] for _, row in spec._entries(archive, "train")] == [1.5, 2.5, 3.5]
    assert [row["width"] for _, row in spec._entries(archive, "test")] == [9.5]


def test_categories_can_be_spelled_as_the_letters_or_the_names_in_code_order():
    letters = replace(FLOWERS, codes={"where": "ns"})
    names = replace(FLOWERS, codes={"where": ("north", "south")})
    for spec in (letters, names):
        cells = b"1.5,3,north,Red\n2.5,4,south,Blue\n"
        if spec is letters:
            cells = b"1.5,3,n,Red\n2.5,4,s,Blue\n"
        assert [row["where"] for _, row in spec._entries(cells, "all")] == [0, 1]


def test_a_category_nobody_declared_is_refused_however_the_codes_were_spelled():
    spec = replace(FLOWERS, codes={"where": "ns"})
    with pytest.raises(ValueError, match="has 'e' in where, and the ones it knows are n, s"):
        list(spec._entries(b"1.5,3,e,Red\n", "all"))


# --- recordings -------------------------------------------------------------


def test_a_recording_is_padded_out_to_the_width_of_the_column():
    archive = clips(**{"0_ann_0": waved([1, -2, 3])})
    classes, columns, rows = SPOKEN.rows({"archive": archive}, "all")
    assert classes == ("zero", "one")
    assert columns == {
        "audio": ("h", 6), "length": "i", "label": "i", "speaker": "i", "index": "i",
    }
    (label, row), = list(rows)
    assert label == 0
    assert row == {"audio": (1, -2, 3, 0, 0, 0), "length": 3, "label": 0, "speaker": 0, "index": 0}


def test_the_recordings_are_read_in_name_order_with_who_spoke_them():
    archive = clips(**{"1_bob_1": waved([4]), "0_ann_2": waved([5]), "1_ann_0": waved([6])})
    _, _, rows = SPOKEN.rows({"archive": archive}, "all")
    got = [(label, row["speaker"], row["index"], row["audio"][0]) for label, row in rows]
    assert got == [(0, 0, 0, 5), (1, 0, 1, 6), (1, 1, 2, 4)]


def test_a_set_that_names_no_speakers_writes_no_speaker_column():
    spec = replace(SPOKEN, speakers=())
    _, columns, rows = spec.rows({"archive": clips(**{"0_ann_0": waved([1])})}, "all")
    assert "speaker" not in columns
    assert "speaker" not in next(iter(rows))[1]


def test_an_archive_with_no_recordings_where_they_should_be_is_refused():
    with pytest.raises(ValueError, match=r"no \.wav files in a 'clips' folder"):
        SPOKEN.rows({"archive": zipped({"spoken-master/notes/read.me": b"hello"})}, "all")


@pytest.mark.parametrize(
    ("kwargs", "said"),
    [
        ({"rate": 16000}, "1-channel 16-bit at 16000 Hz"),
        ({"channels": 2}, "2-channel 16-bit at 8000 Hz"),
        ({"width": 1}, "1-channel 8-bit at 8000 Hz"),
    ],
)
def test_a_recording_that_is_not_what_the_set_says_is_refused_by_name(kwargs, said):
    archive = clips(**{"0_ann_0": waved([1, 2], **kwargs)})
    _, _, rows = SPOKEN.rows({"archive": archive}, "all")
    with pytest.raises(ValueError, match=rf"0_ann_0 of Spoken is {said}"):
        list(rows)


def test_a_recording_longer_than_the_column_is_refused_rather_than_cut_down():
    _, _, rows = SPOKEN.rows({"archive": clips(**{"0_ann_0": waved(list(range(9)))})}, "all")
    with pytest.raises(ValueError, match="is 9 samples long, and the column holds 6"):
        list(rows)


def test_a_recording_of_something_nobody_declared_is_refused():
    _, _, rows = SPOKEN.rows({"archive": clips(**{"7_ann_0": waved([1])})}, "all")
    with pytest.raises(ValueError, match="is labelled '7', and its classes are 0, 1"):
        list(rows)


@pytest.mark.parametrize("named", ["0_zoe_0", "0"])
def test_a_speaker_nobody_declared_is_refused_rather_than_numbered(named):
    _, _, rows = SPOKEN.rows({"archive": clips(**{named: waved([1])})}, "all")
    with pytest.raises(ValueError, match=r"is spoken by '(zoe|)', and the speakers it knows"):
        list(rows)


def test_the_archive_is_let_go_of_once_the_recordings_have_been_read():
    _, _, rows = SPOKEN.rows({"archive": clips(**{"0_ann_0": waved([1])})}, "all")
    assert len(list(rows)) == 1
    with pytest.raises(StopIteration):
        next(rows)


def test_recordings_become_one_tree_per_class_with_the_silence_written_out(registry):
    archive = clips(**{"0_ann_0": waved([1, -2]), "1_bob_0": waved([3, 4, 5])})
    counts, raw = written("spoken", archive, "archive")
    assert counts == {"zero": 1, "one": 1}
    with open_root(io.BytesIO(raw)) as back:
        assert sorted(back.keys()) == ["about", "one", "zero"]
        tree = back["one"]
        assert tree.title == "Spoken recordings of class one, 8000 Hz mono, 6 samples an entry"
        assert list(tree["audio"].array()) == [3, 4, 5, 0, 0, 0]
        assert list(tree["length"].array()) == [3]
        assert list(tree["speaker"].array()) == [1]


def test_an_image_set_in_an_archive_converts_like_any_other(registry):
    counts, raw = written("scribbles", scribbled(b"\1\1\0"), "archive", split="train")
    assert counts == {"train_up": 1, "train_down": 2}
    with open_root(io.BytesIO(raw)) as back:
        assert list(back["train_up"]["image"].array()) == [8, 9, 10, 11]
        assert back["train_down"].num_entries == 2


# --- tables of sentences ----------------------------------------------------


def test_a_table_of_sentences_keeps_the_quotes_it_was_written_with():
    raw = b'a\t"no" he said\n'
    assert list(read_table(raw, delimiter="\t")) == [["a", "no he said"]]
    assert list(read_table(raw, delimiter="\t", quoted=False)) == [["a", '"no" he said']]


def test_a_quote_that_is_never_closed_does_not_swallow_the_lines_after_it():
    raw = b'a\t"two\nb\tthree\n'
    assert len(list(read_table(raw, delimiter="\t"))) == 1
    assert list(read_table(raw, delimiter="\t", quoted=False)) == [["a", '"two'], ["b", "three"]]


def test_text_is_written_into_a_column_of_its_own_with_the_length_beside_it():
    _, columns, rows = MESSAGES.rows({"table": MESSAGE_ROWS}, "all")
    assert columns == {
        "message": ("B", 16),
        "message_length": "i",
        "label": "i",
        "index": "i",
    }
    first, second = list(rows)
    assert first == (0, {"message": b'he said "no"' + bytes(4), "message_length": 12,
                         "label": 0, "index": 0})
    assert second[1]["message"] == b"OI" + bytes(14)


def test_a_message_too_long_for_its_column_is_refused_rather_than_cut_short():
    _, _, rows = MESSAGES.rows({"table": b"plain\t" + b"x" * 17 + b"\n"}, "all")
    with pytest.raises(ValueError, match="has 17 bytes in message, and the column holds 16"):
        list(rows)


def test_messages_become_trees_holding_the_text_and_how_much_of_it_is_real(registry):
    counts, raw = written("messages", zipped({"messages.txt": MESSAGE_ROWS}), "table")
    assert counts == {"plain": 1, "shouted": 1}
    with open_root(io.BytesIO(raw)) as back:
        tree = back["plain"]
        held = bytes(tree["message"].array())
        assert held[: tree["message_length"].array()[0]] == b'he said "no"'
        assert set(held[12:]) == {0}


# --- blocks of numbers ------------------------------------------------------


def test_a_matrix_takes_its_labels_and_its_subjects_from_the_files_beside_it():
    archive = sensed("1 2 3\n4 5 6\n", "2\n1\n", "7\n9\n")
    classes, columns, rows = SENSORS.rows({"archive": archive}, "train")
    assert classes == ("still", "moving")
    assert columns == {"features": ("d", 3), "label": "i", "subject": "i", "index": "i"}
    assert list(rows) == [
        (1, {"features": (1.0, 2.0, 3.0), "index": 0, "subject": 7, "label": 1}),
        (0, {"features": (4.0, 5.0, 6.0), "index": 1, "subject": 9, "label": 0}),
    ]


def test_a_label_the_matrix_was_not_told_about_is_refused_by_name():
    _, _, rows = SENSORS.rows({"archive": sensed("1 2 3\n", "9\n", "7\n")}, "train")
    with pytest.raises(ValueError, match="is labelled '9', and its classes are 1, 2"):
        list(rows)


def test_a_row_that_is_not_as_wide_as_the_others_is_refused():
    _, _, rows = SENSORS.rows({"archive": sensed("1 2 3\n4 5\n", "1\n1\n", "7\n7\n")}, "train")
    with pytest.raises(ValueError, match="row 1 of Sensors holds 2 numbers, and its rows are 3"):
        list(rows)


def test_a_number_that_is_not_one_is_refused_by_where_it_was():
    _, _, rows = SENSORS.rows({"archive": sensed("1 x 3\n", "1\n", "7\n")}, "train")
    with pytest.raises(ValueError, match="has 'x' in features, which is not a number"):
        list(rows)


def test_more_rows_than_labels_is_refused_rather_than_labelled_by_guesswork():
    _, _, rows = SENSORS.rows({"archive": sensed("1 2 3\n4 5 6\n", "1\n", "7\n8\n")}, "train")
    with pytest.raises(ValueError, match="more rows than the 1 in its file of labels"):
        list(rows)


def test_more_rows_than_subjects_is_refused_the_same_way():
    _, _, rows = SENSORS.rows({"archive": sensed("1 2 3\n4 5 6\n", "1\n1\n", "7\n")}, "train")
    with pytest.raises(ValueError, match="more rows than the 1 in its file of subject"):
        list(rows)


@pytest.mark.parametrize(
    ("labels", "subjects", "complaint"),
    [
        ("1\n1\n1\n", "7\n7\n", "2 rows in train and 3 in its file of labels"),
        ("1\n1\n", "7\n7\n7\n", "2 rows in train and 3 in its file of subject"),
    ],
)
def test_a_file_beside_the_numbers_that_is_longer_than_they_are_is_refused(
    labels, subjects, complaint
):
    archive = sensed("1 2 3\n4 5 6\n", labels, subjects)
    _, _, rows = SENSORS.rows({"archive": archive}, "train")
    with pytest.raises(ValueError, match=complaint):
        list(rows)


def test_a_one_hot_label_is_whichever_of_the_last_columns_is_set():
    _, columns, rows = HOT.rows({"archive": b"1.0000 0.0000 0 1\n0.0000 1.0000 1 0\n"}, "all")
    assert columns == {"image": ("B", 2), "label": "i", "index": "i"}
    assert list(rows) == [
        (1, {"image": (1, 0), "index": 0, "label": 1}),
        (0, {"image": (0, 1), "index": 1, "label": 0}),
    ]


@pytest.mark.parametrize("tail", ["0 0", "1 1"])
def test_a_row_with_anything_but_one_label_column_set_is_refused(tail):
    _, _, rows = HOT.rows({"archive": f"1 0 {tail}\n".encode()}, "all")
    with pytest.raises(ValueError, match="of its 2 label columns set, and exactly one"):
        list(rows)


def test_a_matrix_counted_at_the_top_labels_its_rows_by_where_they_lie():
    _, _, rows = RUNS.rows({"archive": b" 2 1\n1 2\n3 4\n5 6\n"}, "all")
    assert [which for which, _ in rows] == [0, 0, 1]


def test_a_count_line_that_names_the_wrong_number_of_classes_is_refused():
    _, _, rows = RUNS.rows({"archive": b"1 1 1\n1 2\n"}, "all")
    with pytest.raises(ValueError, match="starts by counting 3 classes, and it has 2"):
        list(rows)


def test_more_rows_than_the_counts_promised_is_refused():
    _, _, rows = RUNS.rows({"archive": b"1 1\n1 2\n3 4\n5 6\n"}, "all")
    with pytest.raises(ValueError, match="counts 2 rows at the top of its file and holds more"):
        list(rows)


def test_a_block_of_numbers_becomes_one_tree_per_class(registry):
    counts, raw = written("hot", b"1 0 1 0\n0 1 0 1\n1 1 1 0\n", "archive")
    assert counts == {"left": 2, "right": 1}
    with open_root(io.BytesIO(raw)) as back:
        assert back["left"].title == "Hot rows labelled left, 2 numbers each"
        assert list(back["left"]["image"].array()) == [1, 0, 1, 1]
        assert list(back["right"]["image"].array()) == [0, 1]


def test_a_matrix_that_comes_in_splits_says_so_in_the_tree_it_writes(registry):
    archive = sensed("1 2 3\n", "2\n", "7\n")
    counts, raw = written("sensors", archive, "archive", split="train")
    assert counts == {"train_still": 0, "train_moving": 1}
    with open_root(io.BytesIO(raw)) as back:
        assert back["train_moving"].title == "Sensors train rows labelled moving, 3 numbers each"
        assert list(back["train_moving"]["subject"].array()) == [7]


# --- what the new registry entries say --------------------------------------


def test_miniboone_counts_its_two_classes_at_the_top_of_the_one_file():
    spec = DATASETS["miniboone"]
    assert isinstance(spec, Matrix)
    assert spec.counts and spec.width == 50 and not spec.label_files
    assert spec.urls("all") == {"archive": spec.url}


def test_har_reads_the_archive_inside_the_archive_and_the_files_beside_it():
    spec = DATASETS["har"]
    assert isinstance(spec, Matrix)
    assert spec.inner.endswith(".zip")
    assert spec.width == 561
    assert spec.files["train"] == "UCI HAR Dataset/train/X_train.txt"
    assert spec.label_files["test"] == "UCI HAR Dataset/test/y_test.txt"
    assert set(spec.beside) == {"subject"}
    assert sorted(spec.labels.values()) == list(range(6))


def test_semeion_is_labelled_by_the_last_ten_columns_of_its_own_rows():
    spec = DATASETS["semeion"]
    assert isinstance(spec, Matrix)
    assert (spec.onehot, spec.width, spec.kind, spec.column) == (10, 16 * 16, "B", "image")


def test_the_wine_quality_splits_are_the_two_colours_it_is_published_in():
    spec = DATASETS["wine_quality"]
    assert isinstance(spec, Table)
    assert spec.splits == ("red", "white")
    assert spec.files == {
        "red": "winequality-red.csv",
        "white": "winequality-white.csv",
    }
    assert spec.delimiter == ";" and spec.header
    assert spec.classes[0] == "quality_3" and spec.labels["3"] == 0


def test_spambase_names_its_punctuation_counts_rather_than_spelling_them():
    spec = DATASETS["spambase"]
    assert isinstance(spec, Table)
    assert len(spec.fields) == 58
    assert ("char_freq_dollar", "d") in spec.fields
    assert all(name.replace("_", "").isalnum() for name, _ in spec.fields)


def test_glass_leaves_out_the_class_number_its_data_never_uses():
    spec = DATASETS["glass"]
    assert isinstance(spec, Table)
    assert "4" not in spec.labels
    assert sorted(spec.labels.values()) == list(range(6))


def test_the_sms_set_gives_its_text_a_column_and_keeps_the_quotes_in_it():
    spec = DATASETS["sms_spam"]
    assert isinstance(spec, Table)
    assert spec.delimiter == "\t" and not spec.quoted
    assert spec.text_size >= 910
    assert dict(spec.fields)["message"] == "text"


# --- spreadsheets -----------------------------------------------------------


def test_a_spreadsheet_is_read_out_of_the_xml_it_keeps_its_rows_in():
    raw = spreadsheet([{"A1": "1", "B1": "2.5"}, {"A2": "3", "B2": "4"}])
    assert list(read_xlsx(raw)) == [["1", "2.5"], ["3", "4"]]


def test_the_words_a_spreadsheet_keeps_once_are_put_back_where_they_were():
    raw = spreadsheet(
        [{"A1": "s:1", "B1": "s:0"}, {"A2": "i:typed here", "B2": "7"}],
        shared=("width", "height"),
    )
    assert list(read_xlsx(raw)) == [["height", "width"], ["typed here", "7"]]


def test_an_empty_row_is_not_data_and_neither_are_the_cells_after_the_last_one():
    raw = spreadsheet([{"A1": "1", "B1": "", "C1": ""}, {}, {"A3": "2"}])
    assert list(read_xlsx(raw)) == [["1"], ["2"]]


def test_a_gap_in_the_middle_of_a_row_keeps_the_fields_after_it_lined_up():
    raw = spreadsheet([{"A1": "1", "D1": "4"}, {"B2": "2", "C2": "3"}])
    assert list(read_xlsx(raw)) == [["1", "", "", "4"], ["", "2", "3"]]


def test_a_spreadsheet_with_no_words_in_it_needs_no_table_of_them():
    raw = spreadsheet([{"AA1": "1"}])
    assert "xl/sharedStrings.xml" not in zipfile.ZipFile(io.BytesIO(raw)).namelist()
    assert list(read_xlsx(raw)) == [[""] * 26 + ["1"]]


def test_a_spreadsheet_header_is_dropped_like_any_other():
    raw = spreadsheet([{"A1": "s:0"}, {"A2": "1"}], shared=("width",))
    assert list(read_xlsx(raw, header=True)) == [["1"]]


def test_the_sheet_wanted_can_be_chosen_and_one_that_is_not_there_is_refused():
    raw = spreadsheet([{"A1": "1"}], sheets=3)
    assert list(read_xlsx(raw, sheet=2)) == []
    with pytest.raises(ValueError, match="this spreadsheet has 3 sheets in it, and no sheet 4"):
        list(read_xlsx(raw, sheet=4))


def test_a_spreadsheet_becomes_a_dataset_like_any_other_table(registry):
    book = spreadsheet(
        [{"A1": "s:0", "B1": "s:1"}, {"A2": "1.5", "B2": "2.5"}], shared=("width", "height")
    )
    counts, raw = written("sheet", zipped({"book.xlsx": book}), "table")
    assert counts == {"rows": 1}
    with open_root(io.BytesIO(raw)) as back:
        assert list(back["rows"]["width"].array()) == [1.5]
        assert list(back["rows"]["height"].array()) == [2.5]


# --- rows that end in free text ---------------------------------------------


def test_a_row_that_ends_in_free_text_is_split_no_further_than_that():
    raw = b'18.0   8\t"chevrolet chevelle malibu"\n15.0 6\tford pinto\n'
    assert list(read_table(raw, delimiter=None, tail=3)) == [
        ["18.0", "8", "chevrolet chevelle malibu"],
        ["15.0", "6", "ford pinto"],
    ]


def test_free_text_keeps_its_quotes_when_the_file_means_them():
    raw = b'1 2 "so he said"\n'
    rows = read_table(raw, delimiter=None, tail=3, quoted=False)
    assert next(rows) == ["1", "2", '"so he said"']


def test_rows_divided_by_carriage_returns_alone_are_still_rows():
    assert list(read_table(b"1,2\r3,4\r")) == [["1", "2"], ["3", "4"]]


def test_the_text_at_the_end_of_a_row_arrives_in_its_own_column(registry):
    counts, raw = written("cars", b'18.0 8\t"chevelle"\n', "table")
    assert counts == {"rows": 1}
    with open_root(io.BytesIO(raw)) as back:
        tree = back["rows"]
        assert tree.title == "Cars rows"
        held = tree["car_name"].array()
        assert bytes(held[: tree["car_name_length"].array()[0]]) == b"chevelle"
        assert len(held) == 16


# --- a number to predict rather than a class to sort into -------------------


def test_a_table_with_no_classes_writes_one_tree_of_every_row(registry):
    classes, columns, rows = PRICES.rows({"table": zipped({"north.csv": b"1970-01-03,5\n"})},
                                         "north")
    assert classes == ("rows",)
    assert columns == {"when": "i", "price": "d", "index": "i"}
    assert list(rows) == [(0, {"when": 2, "price": 5.0, "index": 0})]


def test_a_set_with_a_number_to_predict_says_so_rather_than_counting_classes():
    assert PRICES.sorting() == "no classes, a number to predict"
    assert DATASETS["iris"].sorting() == "3 classes"
    assert "one tree of every row" in PRICES.about("north")
    assert PRICES.entry_title("north", "rows") == "Prices north rows"


def test_a_regression_set_that_comes_in_splits_names_its_tree_for_the_split(registry):
    counts, raw = written(
        "prices", zipped({"south.csv": b"1970-01-01,2\n1970-01-02,3\n"}), "table", split="south"
    )
    assert counts == {"south_rows": 2}
    with open_root(io.BytesIO(raw)) as back:
        assert sorted(back.keys()) == ["south_about", "south_rows"]
        assert list(back["south_rows"]["when"].array()) == [0, 1]
        assert "label" not in back["south_rows"].keys()


def test_a_date_becomes_the_days_since_1970_and_a_gap_becomes_minus_one():
    _, _, rows = PRICES.rows({"table": zipped({"north.csv": b"2011-01-01,1\n?,2\n"})}, "north")
    assert [row["when"] for _, row in rows] == [14975, -1]


def test_a_date_written_some_other_way_says_how_this_one_is_written():
    _, _, rows = PRICES.rows({"table": zipped({"north.csv": b"01/01/2011,1\n"})}, "north")
    with pytest.raises(ValueError, match=r"row 0 of Prices has '01/01/2011' in when, and the "
                                         r"dates in it are written %Y-%m-%d"):
        list(rows)


def test_dates_can_be_read_the_way_the_file_happens_to_write_them():
    spec = replace(PRICES, dates="%d/%m/%Y")
    _, _, rows = spec.rows({"table": zipped({"north.csv": b"02/01/1970,1\n"})}, "north")
    assert [row["when"] for _, row in rows] == [1]


# --- what the newest registry entries say -----------------------------------


def test_the_two_telescope_sets_are_labelled_the_way_their_papers_label_them():
    magic, htru2 = DATASETS["magic"], DATASETS["htru2"]
    assert isinstance(magic, Table) and isinstance(htru2, Table)
    assert magic.classes == ("gamma", "hadron") and magic.labels == {"g": 0, "h": 1}
    assert len(magic.fields) == 11 and magic.member == "magic04.data"
    assert htru2.classes == ("not_pulsar", "pulsar")
    assert [name for name, _ in htru2.fields][:2] == ["profile_mean", "profile_stdev"]
    assert [name for name, _ in htru2.fields][4] == "dmsnr_mean"


def test_auto_mpg_keeps_the_car_name_at_the_end_of_the_row():
    spec = DATASETS["auto_mpg"]
    assert isinstance(spec, Table)
    assert spec.delimiter is None and spec.tail == len(spec.fields) == 9
    assert spec.fields[0] == ("mpg", "target")
    assert spec.fields[-1] == ("car_name", "text")
    assert spec.text_size >= 38  # the longest name in the file, quotes and all


def test_the_bike_hires_are_counted_by_the_hour_with_the_date_beside_them():
    spec = DATASETS["bike_sharing"]
    assert isinstance(spec, Table)
    assert spec.member == "hour.csv" and spec.header
    assert dict(spec.fields)["date"] == "date"
    assert spec.fields[-1] == ("count", "target")
    assert spec.dates == "%Y-%m-%d"


def test_the_two_spreadsheet_sets_are_read_as_spreadsheets():
    energy, estate = DATASETS["energy_efficiency"], DATASETS["real_estate"]
    assert isinstance(energy, Table) and isinstance(estate, Table)
    assert energy.xlsx and estate.xlsx
    assert energy.member.endswith(".xlsx") and estate.member.endswith(".xlsx")
    assert energy.header and estate.header
    assert [name for name, role in energy.fields if role == "target"] == [
        "heating_load", "cooling_load"
    ]
    assert estate.fields[-1] == ("price_per_unit_area", "target")


def test_the_pupils_come_out_of_a_zip_inside_a_zip_in_both_their_subjects():
    spec = DATASETS["student"]
    assert isinstance(spec, Table)
    assert spec.inner == "student.zip"
    assert spec.splits == ("maths", "portuguese")
    assert spec.files == {"maths": "student-mat.csv", "portuguese": "student-por.csv"}
    assert spec.delimiter == ";" and spec.header
    assert len(spec.fields) == 33
    assert spec.codes["yesno"] == ("no", "yes")


def test_heart_disease_comes_in_the_four_hospitals_that_gathered_it():
    spec = DATASETS["heart_disease"]
    assert isinstance(spec, Table)
    assert spec.splits == ("cleveland", "hungary", "switzerland", "long_beach")
    assert set(spec.files) == set(spec.splits)
    assert all(name.startswith("processed.") for name in spec.files.values())
    assert len(spec.fields) == 14 and spec.classes[0] == "none"
    assert spec.labels == {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}


def test_the_car_categories_are_numbered_worst_to_best():
    spec = DATASETS["car_evaluation"]
    assert isinstance(spec, Table)
    assert spec.codes["price"] == ("low", "med", "high", "vhigh")
    assert spec.codes["safety"] == ("low", "med", "high")
    assert spec.codes["boot"] == ("small", "med", "big")
    assert spec.classes == ("unacceptable", "acceptable", "good", "very_good")


def test_yeast_names_the_place_in_the_cell_each_protein_ends_up():
    spec = DATASETS["yeast"]
    assert isinstance(spec, Table)
    assert spec.delimiter is None and spec.text_size >= 10
    assert spec.fields[0] == ("protein", "text")
    assert spec.labels["CYT"] == 0 and spec.classes[0] == "cytosol"
    assert sorted(spec.labels.values()) == list(range(10))


def test_a_table_can_arrive_in_a_zip_inside_a_zip(registry):
    spec = replace(PRICES, inner="inner.zip")
    held = zipped({"inner.zip": zipped({"north.csv": b"1970-01-01,4\n"})})
    _, _, rows = spec.rows({"table": held}, "north")
    assert [row["price"] for _, row in rows] == [4.0]


def test_a_table_written_as_an_arff_is_read_as_one(registry):
    spec = replace(CARS, arff=True)
    _, _, rows = spec.rows({"table": b"@relation cars\n@data\n18.0,8,chevelle\n"}, "all")
    assert [row["mpg"] for _, row in rows] == [18.0]
