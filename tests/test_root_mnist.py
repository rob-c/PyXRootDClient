"""MNIST into a ROOT file, one tree per digit.

Nothing here touches the network: the IDX bytes are built in memory in the
same format the real files are in, served over the library's own HTTP test
server where the download path is what is under test, and the written file is
read back with this library's reader.
"""

from __future__ import annotations

import gzip
import io
import struct

import pytest

from xrd.root import create, open_root
from xrd.root.mnist import COLUMNS, FILES, MIRROR, PIXELS, SIDE, convert, fetch, read_idx
from xrd.testing.http import FakeDAVServer


def idx(shape, values: bytes) -> bytes:
    """An IDX file of unsigned bytes, as the MNIST files are."""
    return struct.pack(f">BBBB{len(shape)}i", 0, 0, 0x08, len(shape), *shape) + values


def images(count: int, *, side: int = SIDE) -> bytes:
    """``count`` pictures, each filled with a byte that says which one it is."""
    pictures = (bytes([step % 256]) * (side * side) for step in range(count))
    return idx((count, side, side), b"".join(pictures))


def labels(digits) -> bytes:
    return idx((len(digits),), bytes(digits))


def read_back(data: bytes):
    return open_root(io.BytesIO(data))


def converted(digits, **kwargs) -> tuple[bytes, dict[str, int]]:
    """A file holding ``digits``, and what the conversion said it wrote."""
    buf = io.BytesIO()
    written = convert(buf, images=images(len(digits)), labels=labels(digits), **kwargs)
    return buf.getvalue(), written


# -- reading IDX -----------------------------------------------------------


def test_an_idx_file_gives_back_its_shape_and_its_values():
    assert read_idx(idx((3,), b"abc")) == ((3,), b"abc")
    assert read_idx(idx((2, 2, 2), bytes(8))) == ((2, 2, 2), bytes(8))


def test_an_idx_file_is_unzipped_when_it_arrives_zipped():
    assert read_idx(gzip.compress(idx((3,), b"abc"))) == ((3,), b"abc")


def test_something_that_is_not_an_idx_file_is_refused():
    for raw in (b"", b"\x89PNG\r\n", b"\x00\x01\x08\x01"):
        with pytest.raises(ValueError, match="not an IDX file"):
            read_idx(raw)


def test_an_idx_file_of_something_other_than_bytes_is_refused_by_what_it_holds():
    with pytest.raises(ValueError, match="holds floats, and MNIST holds unsigned bytes"):
        read_idx(b"\x00\x00\x0d\x01" + struct.pack(">i", 1) + bytes(4))
    with pytest.raises(ValueError, match="holds values of type 0x42"):
        read_idx(b"\x00\x00\x42\x01" + struct.pack(">i", 1) + b"\x00")


def test_an_idx_file_that_stops_before_its_shape_is_refused():
    with pytest.raises(ValueError, match="3 dimensions and stops before"):
        read_idx(b"\x00\x00\x08\x03" + struct.pack(">ii", 2, 2))


def test_an_idx_file_that_does_not_hold_what_it_says_is_refused():
    with pytest.raises(ValueError, match=r"says 2x3, which is 6 values, and holds 5"):
        read_idx(idx((2, 3), b"abcde"))


# -- where the bytes come from ---------------------------------------------


def test_bytes_are_taken_as_they_are():
    assert fetch(b"abc") == b"abc"
    assert fetch(bytearray(b"abc")) == b"abc"
    assert fetch(memoryview(b"abc")) == b"abc"


def test_an_open_file_is_read_whole():
    assert fetch(io.BytesIO(b"abc")) == b"abc"


def test_a_local_path_is_read_whole(tmp_path):
    path = tmp_path / "idx"
    path.write_bytes(b"abc")
    assert fetch(str(path)) == b"abc"
    assert fetch(path) == b"abc"


def test_a_url_is_read_through_this_library():
    with FakeDAVServer(files={"/d/idx": b"abc"}) as server:
        assert fetch(f"{server.url}/d/idx") == b"abc"


# -- the conversion --------------------------------------------------------


def test_every_image_lands_in_the_tree_for_its_own_digit():
    digits = [0, 1, 2, 1, 9, 1, 0]
    data, written = converted(digits)
    assert written == {
        "train_0": 2, "train_1": 3, "train_2": 1, "train_3": 0, "train_4": 0,
        "train_5": 0, "train_6": 0, "train_7": 0, "train_8": 0, "train_9": 1,
    }
    with read_back(data) as back:
        assert sorted(back.keys()) == [f"train_{digit}" for digit in range(10)]
        for digit in range(10):
            tree = back[f"train_{digit}"]
            assert len(tree) == digits.count(digit)
            assert set(tree["label"].array()) <= {digit}
            assert list(tree["index"].array()) == [
                at for at, value in enumerate(digits) if value == digit
            ]


def test_the_pixels_of_an_entry_are_the_pixels_of_that_image():
    data, _written = converted([7, 3, 7])
    with read_back(data) as back:
        tree = back["train_7"]
        assert tree["image"].length == PIXELS
        pixels = list(tree["image"].array())
        assert pixels[:PIXELS] == [0] * PIXELS  # image 0, filled with 0
        assert pixels[PIXELS:] == [2] * PIXELS  # image 2, filled with 2
        assert list(tree["image"].array(1, 2)) == [2] * PIXELS


def test_the_columns_are_what_a_training_loop_needs_and_nothing_else():
    data, _written = converted([4])
    with read_back(data) as back:
        tree = back["train_4"]
        assert tree.keys() == list(COLUMNS)
        assert tree.typenames() == {"image": "uint8", "label": "int32", "index": "int32"}
        assert [tree[name].length for name in COLUMNS] == [PIXELS, 1, 1]
        assert tree.title.startswith("MNIST train images of the digit 4")


def test_the_test_split_is_named_for_itself_so_both_fit_in_one_file():
    buf = io.BytesIO()
    with create(buf) as out:
        first = convert(out, images=images(2), labels=labels([1, 1]))
        second = convert(out, split="test", images=images(1), labels=labels([1]))
    assert first["train_1"] == 2
    assert second["test_1"] == 1
    with read_back(buf.getvalue()) as back:
        assert len(back.keys()) == 20
        assert len(back["train_1"]) == 2
        assert len(back["test_1"]) == 1
        assert back["test_1"].title.startswith("MNIST test images")


def test_the_trees_can_be_named_something_else_entirely():
    data, written = converted([5], prefix="digit")
    assert written["digit_5"] == 1
    with read_back(data) as back:
        assert "digit_5" in back.keys()


def test_a_file_given_already_open_is_left_for_its_owner_to_close():
    buf = io.BytesIO()
    out = create(buf)
    convert(out, images=images(1), labels=labels([2]))
    assert not out.closed
    out.tree("extra", {"x": "i"}).fill(x=1)
    out.close()
    with read_back(buf.getvalue()) as back:
        assert len(back["train_2"]) == 1
        assert list(back["extra"]["x"].array()) == [1]


def test_the_conversion_can_be_told_where_the_dataset_is():
    with FakeDAVServer(
        files={
            f"/mnist/{FILES['train'][0]}": gzip.compress(images(3)),
            f"/mnist/{FILES['train'][1]}": gzip.compress(labels([6, 6, 8])),
        }
    ) as server:
        buf = io.BytesIO()
        written = convert(buf, base=f"{server.url}/mnist/")
    assert written["train_6"] == 2
    assert written["train_8"] == 1
    with read_back(buf.getvalue()) as back:
        assert list(back["train_8"]["index"].array()) == [2]


def test_the_mirror_is_the_one_the_files_are_actually_on():
    assert MIRROR.startswith("https://")
    assert MIRROR.endswith("/")
    assert set(FILES) == {"train", "test"}


def test_a_bigger_conversion_writes_more_than_one_basket_and_still_reads():
    digits = [step % 2 for step in range(40)]
    data, _written = converted(digits, basket_size=4 * PIXELS)
    with read_back(data) as back:
        tree = back["train_1"]
        assert tree["image"].num_baskets == 5
        assert len(tree) == 20
        assert list(tree["index"].array()) == list(range(1, 40, 2))
        assert list(tree["image"].array(4, 5)) == [9] * PIXELS  # entry 4 is image 9


def test_the_file_can_be_written_without_compression():
    data, _written = converted([1], compression=None)
    with read_back(data) as back:
        assert len(back["train_1"]) == 1


# -- what the conversion refuses -------------------------------------------


def test_a_split_that_is_not_a_split_is_refused_by_name():
    with pytest.raises(ValueError, match="are train and test, not 'validation'"):
        convert(io.BytesIO(), split="validation")


def test_images_that_are_not_mnist_images_are_refused():
    with pytest.raises(ValueError, match=r"are 3x4x4, and MNIST images come"):
        convert(io.BytesIO(), images=images(3, side=4), labels=labels([1, 2, 3]))
    with pytest.raises(ValueError, match=r"are 4, and MNIST images come"):
        convert(io.BytesIO(), images=idx((4,), bytes(4)), labels=labels([1]))


def test_a_label_for_every_image_is_required():
    with pytest.raises(ValueError, match="2 images and 3 labels"):
        convert(io.BytesIO(), images=images(2), labels=labels([1, 2, 3]))
    with pytest.raises(ValueError, match=r"2 images and \(2, 2\) labels"):
        convert(io.BytesIO(), images=images(2), labels=idx((2, 2), bytes(4)))


def test_a_label_that_is_not_a_digit_is_refused_by_which_image_it_is():
    with pytest.raises(ValueError, match="image 1 is labelled 10, and MNIST is the digits"):
        convert(io.BytesIO(), images=images(2), labels=labels([3, 10]))
