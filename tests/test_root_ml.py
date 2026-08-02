"""Turning a tree into tensors.

Neither PyTorch nor TensorFlow is a dependency of this library, and neither is
installed in the test environment, so the tests stand a small module in for
each: what they check is that the right calls are made with the right shapes
and types, which is the part this library is responsible for.
"""

from __future__ import annotations

import array
import pathlib
import sys
import types

import pytest

from xrd.root import Jagged, UnsupportedFeatureError, open_root
from xrd.root.ml import (
    dataset,
    iter_tensors,
    numeric,
    tf_dataset,
    to_tensor,
    to_tensors,
    to_tf_tensor,
    to_tf_tensors,
)

DATA = pathlib.Path(__file__).parent / "data"


class Tensor:
    """Enough of a tensor to see what was asked for."""

    def __init__(self, values, dtype, shape=None, device=None):
        self.values = list(values)
        self.dtype = dtype
        self.shape = shape
        self.device = device

    def __len__(self):
        return len(self.values)

    def reshape(self, rows, width):
        return Tensor(self.values, self.dtype, (rows, width), self.device)

    def to(self, device):
        return Tensor(self.values, self.dtype, self.shape, device)


@pytest.fixture
def torch(monkeypatch):
    """A stand-in for PyTorch, in the place ``import torch`` looks."""
    module = types.ModuleType("torch")
    for name in ("int8", "uint8", "int16", "int32", "int64", "float32", "float64"):
        setattr(module, name, name)
    module.frombuffer = lambda values, dtype: Tensor(values, dtype)

    data = types.ModuleType("torch.utils.data")

    class IterableDataset:
        pass

    data.IterableDataset = IterableDataset
    data.get_worker_info = lambda: None
    utils = types.ModuleType("torch.utils")
    utils.data = data
    module.utils = utils

    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data)
    return module


@pytest.fixture
def flat():
    with open_root(str(DATA / "small-flat-tree.root")) as handle:
        yield handle["tree"]


def test_without_pytorch_the_refusal_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(UnsupportedFeatureError, match="pip install torch"):
        to_tensor(array.array("i", [1]))


def test_a_column_becomes_a_tensor_of_the_matching_type(torch):
    tensor = to_tensor(array.array("f", [1.5, 2.5]))
    assert (tensor.dtype, tensor.values, tensor.shape) == ("float32", [1.5, 2.5], None)
    assert to_tensor(array.array("b", [1]), device="cuda").device == "cuda"


def test_a_fixed_size_array_column_becomes_rows(torch):
    tensor = to_tensor(array.array("i", [1, 2, 3, 4]), width=2)
    assert tensor.shape == (2, 2)


def test_a_variable_column_is_padded_into_a_rectangle(torch):
    rows = Jagged(array.array("d", [1.0, 2.0, 3.0]), array.array("q", [0, 2, 2, 3]))
    tensor = to_tensor(rows)
    assert tensor.shape == (3, 2)
    assert tensor.values == [1.0, 2.0, 0.0, 0.0, 3.0, 0.0]


def test_a_column_with_nothing_in_any_row_still_gives_a_shape(torch):
    empty = Jagged(array.array("i", []), array.array("q", [0, 0]))
    assert to_tensor(empty).shape == (0, 0)


def test_an_unsigned_column_is_widened_into_a_type_torch_takes(torch):
    tensor = to_tensor(array.array("H", [65535]))
    assert (tensor.dtype, tensor.values) == ("int32", [65535])


def test_a_value_too_large_for_any_signed_type_is_refused_not_wrapped(torch):
    with pytest.raises(UnsupportedFeatureError, match="too large for int64"):
        to_tensor(array.array("Q", [2**63]))


def test_a_column_that_is_not_numbers_is_refused(torch):
    with pytest.raises(UnsupportedFeatureError, match="strings and objects"):
        to_tensor(["uno", "dos"])


def test_a_batch_of_columns_becomes_a_batch_of_tensors(torch):
    batch = {"pt": array.array("f", [1.0, 2.0]), "hits": array.array("i", [1, 2, 3, 4])}
    tensors = to_tensors(batch, widths={"hits": 2}, device="cpu")
    assert tensors["pt"].shape is None
    assert tensors["hits"].shape == (2, 2)
    assert tensors["hits"].device == "cpu"


def test_only_the_numeric_columns_are_offered_as_tensors(flat):
    names = numeric(flat)
    assert "Str" not in names
    assert "Int32" in names and "SliceInt32" in names


def test_a_tree_iterates_straight_into_tensors(torch, flat):
    batches = list(iter_tensors(flat, ["Int32", "ArrayInt32"], step=40))
    assert len(batches) == 3
    assert batches[0]["Int32"].values[:3] == [0, 1, 2]
    assert batches[0]["ArrayInt32"].shape == (40, 10)
    assert batches[-1]["Int32"].values == [80 + n for n in range(20)]


def test_iterating_with_no_names_takes_every_numeric_column(torch, flat):
    batch = next(iter_tensors(flat, step=10, entry_start=5, entry_stop=15))
    assert set(batch) == set(numeric(flat))
    assert batch["Int32"].values == list(range(5, 15))


def test_a_dataset_hands_a_loader_one_batch_at_a_time(torch, flat):
    loader = dataset(flat, ["Int32"], step=25)
    assert len(loader) == 4
    batches = list(iter(loader))
    assert [len(b["Int32"]) for b in batches] == [25, 25, 25, 25]
    assert batches[1]["Int32"].values[0] == 25


def test_several_workers_split_the_entries_between_them(torch, flat):
    torch.utils.data.get_worker_info = lambda: types.SimpleNamespace(id=1, num_workers=3)
    batches = list(iter(dataset(flat, ["Int32"], step=10, device="cuda")))
    assert [b["Int32"].values[0] for b in batches] == [34, 44, 54, 64]
    assert batches[0]["Int32"].device == "cuda"

    torch.utils.data.get_worker_info = lambda: types.SimpleNamespace(id=9, num_workers=3)
    assert list(iter(dataset(flat, ["Int32"], step=10))) == []


class TfTensor:
    """Enough of a TensorFlow tensor to see what was asked for."""

    def __init__(self, raw, dtype, shape=None):
        self.raw = raw
        self.dtype = dtype
        self.shape = shape

    def values(self, code):
        return array.array(code, self.raw).tolist()


@pytest.fixture
def tf(monkeypatch):
    """A stand-in for TensorFlow, in the place ``import tensorflow`` looks."""
    module = types.ModuleType("tensorflow")
    for name in ("int8", "uint8", "int16", "int32", "int64", "float32", "float64"):
        setattr(module, name, name)
    module.constant = lambda raw: raw
    module.io = types.SimpleNamespace(
        decode_raw=lambda raw, dtype, little_endian: TfTensor(raw, dtype)
    )
    module.reshape = lambda tensor, shape: TfTensor(tensor.raw, tensor.dtype, shape)
    module.TensorSpec = lambda shape, dtype: (shape, dtype)

    class Dataset:
        @staticmethod
        def from_generator(batches, output_signature):
            return types.SimpleNamespace(
                signature=output_signature, batches=lambda: list(batches())
            )

    module.data = types.SimpleNamespace(Dataset=Dataset)
    monkeypatch.setitem(sys.modules, "tensorflow", module)
    return module


def test_without_tensorflow_the_refusal_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    with pytest.raises(UnsupportedFeatureError, match="pip install tensorflow"):
        to_tf_tensor(array.array("i", [1]))


def test_a_column_becomes_a_tensorflow_tensor_of_the_matching_type(tf):
    tensor = to_tf_tensor(array.array("f", [1.5, 2.5]))
    assert (tensor.dtype, tensor.shape) == ("float32", None)
    assert tensor.values("f") == [1.5, 2.5]


def test_a_fixed_size_array_column_becomes_rows_in_tensorflow(tf):
    assert to_tf_tensor(array.array("i", [1, 2, 3, 4]), width=2).shape == (2, 2)


def test_a_variable_column_is_padded_for_tensorflow_too(tf):
    rows = Jagged(array.array("d", [1.0, 2.0, 3.0]), array.array("q", [0, 2, 2, 3]))
    tensor = to_tf_tensor(rows, fill=-1.0)
    assert tensor.shape == (3, 2)
    assert tensor.values("d") == [1.0, 2.0, -1.0, -1.0, 3.0, -1.0]
    empty = Jagged(array.array("i", []), array.array("q", [0, 0]))
    assert to_tf_tensor(empty).shape == (0, 0)


def test_a_batch_becomes_a_batch_of_tensorflow_tensors(tf):
    batch = {"pt": array.array("f", [1.0, 2.0]), "hits": array.array("i", [1, 2, 3, 4])}
    tensors = to_tf_tensors(batch, widths={"hits": 2})
    assert tensors["pt"].shape is None
    assert tensors["hits"].shape == (2, 2)


def test_a_tf_dataset_declares_its_shapes_before_reading_anything(tf, flat):
    data = tf_dataset(flat, ["Int32", "ArrayInt32", "SliceInt32"], step=40)
    assert data.signature == {
        "Int32": ((None,), "int32"),
        "ArrayInt32": ((None, 10), "int32"),
        "SliceInt32": ((None, None), "int32"),
    }
    batches = data.batches()
    assert len(batches) == 3
    assert batches[0]["Int32"].values("i")[:3] == [0, 1, 2]
    assert batches[0]["ArrayInt32"].shape == (40, 10)


def test_a_tf_dataset_with_no_names_takes_every_numeric_column(tf, flat):
    assert set(tf_dataset(flat, step=100).signature) == set(numeric(flat))


def test_a_column_that_is_not_numbers_has_no_tensor_shape(tf, flat):
    with pytest.raises(UnsupportedFeatureError, match="which is not numbers"):
        tf_dataset(flat, ["Str"])


# -- a tree this library wrote, on its way into a training loop ------------


def test_a_written_mnist_tree_batches_into_the_shape_the_demo_expects(torch):
    """The images this library writes come back out as ``(entries, 784)``,
    which is what makes the PyTorch recipe in the docs a `view` and no more."""
    import io
    import struct

    from xrd.root.mnist import PIXELS, SIDE, convert

    pictures = b"".join(bytes([step]) * PIXELS for step in range(6))
    images = struct.pack(">BBBBiii", 0, 0, 8, 3, 6, SIDE, SIDE) + pictures
    labels = struct.pack(">BBBBi", 0, 0, 8, 1, 6) + bytes([4, 2, 4, 4, 2, 4])
    buf = io.BytesIO()
    convert(buf, images=images, labels=labels)

    with open_root(io.BytesIO(buf.getvalue())) as back:
        tree = back["train_4"]
        assert numeric(tree) == ["image", "label", "index"]
        batches = list(iter_tensors(tree, ["image", "label"], step=2))
        assert [batch["image"].shape for batch in batches] == [(2, PIXELS), (2, PIXELS)]
        assert [len(batch["label"]) for batch in batches] == [2, 2]
        assert batches[0]["image"].dtype == "uint8"
        assert batches[0]["label"].dtype == "int32"
        assert batches[0]["image"].values[:PIXELS] == [0] * PIXELS
        assert batches[0]["image"].values[PIXELS:] == [2] * PIXELS
        assert batches[1]["label"].values == [4, 4]
