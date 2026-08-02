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
import tarfile
import zipfile
from dataclasses import replace

import pytest

from xrd.root import open_root
from xrd.root.datasets import (
    CIFAR,
    DATASETS,
    IDX_FILES,
    MISSING,
    Dataset,
    Images,
    Table,
    convert,
    describe,
    read_table,
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
    for spec in (TINY, TINY_COARSE, FLOWERS):
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
        "iris", "penguins", "covertype",
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


def test_penguins_names_a_code_for_every_category_it_will_meet():
    spec = DATASETS["penguins"]
    assert isinstance(spec, Table)
    coded = {role for _, role in spec.fields} - {"d", "i", "label"}
    assert coded == set(spec.codes)


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
    assert [row["index"] for _, row in FLOWERS._entries(rows)] == [0, 1, 2]
    assert [row["index"] for _, row in FLOWERS._entries(gz)] == [0, 1, 2]


def test_a_member_is_taken_out_of_a_zip_and_unzipped_again_if_it_is_gzipped():
    inner = zipped({"rows.csv.gz": gzip.compress(FLOWER_ROWS)})
    spec = Table(
        name="z", label="Z", title="zipped", licence="CC0", source="https://example.invalid/z",
        classes=FLOWERS.classes, url="", member="rows.csv.gz", fields=FLOWERS.fields,
        labels=FLOWERS.labels, codes=FLOWERS.codes,
    )
    assert len(list(spec._entries(inner))) == 3


def test_a_zip_without_the_member_wanted_names_what_it_does_hold():
    spec = Table(
        name="z", label="Z", title="zipped", licence="CC0", source="https://example.invalid/z",
        classes=FLOWERS.classes, url="", member="rows.data", fields=FLOWERS.fields,
        labels=FLOWERS.labels, codes=FLOWERS.codes,
    )
    with pytest.raises(ValueError, match=r"this zip holds other.csv, and not 'rows.data'"):
        list(spec._entries(zipped({"other.csv": FLOWER_ROWS})))


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
