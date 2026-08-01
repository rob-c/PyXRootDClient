"""From a tree on a storage element to tensors in a training loop.

    >>> import torch, xrd.root, xrd.root.ml                    # doctest: +SKIP
    >>> tree = xrd.root.open_root("root://eos.example.org//store/events.root").tree()
    >>> loader = torch.utils.data.DataLoader(xrd.root.ml.dataset(tree), batch_size=None)
    >>> for batch in loader:
    ...     model(batch["pt"])

Nothing here is imported until it is called, so :mod:`xrd.root` costs nothing
on a machine with no PyTorch on it, and the batches come off the wire a basket
at a time - the file never has to fit anywhere.
"""

from __future__ import annotations

import array
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from .errors import UnsupportedFeatureError
from .tree import DEFAULT_STEP, Jagged

if TYPE_CHECKING:
    from .tree import TTree

__all__ = ["to_tensor", "to_tensors", "iter_tensors", "dataset"]

#: Which torch type each :mod:`array` code becomes, and what to widen it to
#: first. Torch's unsigned types beyond a byte are too thinly supported to
#: hand a beginner, so those are widened into a signed type that holds them.
DTYPES = {
    "b": ("int8", None),
    "B": ("uint8", None),
    "h": ("int16", None),
    "H": ("int32", "i"),
    "i": ("int32", None),
    "I": ("int64", "q"),
    "q": ("int64", None),
    "Q": ("int64", "q"),
    "f": ("float32", None),
    "d": ("float64", None),
}


def _torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        raise UnsupportedFeatureError(
            "reading the tree worked, but turning it into tensors needs PyTorch: "
            "pip install torch, or use the arrays as they come (tree.arrays())"
        ) from None
    return torch


def to_tensor(
    values: Any,
    *,
    width: int | None = None,
    fill: float = 0.0,
    device: Any = None,
) -> Any:
    """One column as a tensor.

    A fixed column of single values gives a 1-D tensor; ``width`` reshapes a
    fixed-size array column into rows. A :class:`~xrd.root.Jagged` column is
    padded to a rectangle - to its longest row unless ``width`` says how wide.
    """
    torch = _torch()
    if isinstance(values, Jagged):
        values, width = values.padded(width, fill)
    if not isinstance(values, array.array):
        raise UnsupportedFeatureError(
            f"a column of {type(values).__name__} is not a tensor; strings and objects have to "
            f"be turned into numbers first, which only you know how to do"
        )

    name, widen = DTYPES[values.typecode]
    if widen is not None:
        try:
            values = array.array(widen, values)
        except OverflowError:
            raise UnsupportedFeatureError(
                f"a value in this column is too large for torch's {name}, which is the widest "
                f"signed type there is; read it with .array() and scale it yourself"
            ) from None
    tensor = torch.frombuffer(values, dtype=getattr(torch, name))
    if width is not None:
        tensor = tensor.reshape(len(tensor) // width if width else 0, width)
    return tensor.to(device) if device is not None else tensor


def to_tensors(
    batch: dict[str, Any],
    *,
    widths: dict[str, int] | None = None,
    device: Any = None,
) -> dict[str, Any]:
    """A whole batch of columns, each one a tensor."""
    widths = widths or {}
    return {
        name: to_tensor(values, width=widths.get(name), device=device)
        for name, values in batch.items()
    }


def numeric(tree: TTree) -> list[str]:
    """The columns of a tree that can become tensors: the numbers."""
    return [name for name in tree.readable() if tree[name].typename != "str"]


def iter_tensors(
    tree: TTree,
    names: Sequence[str] | None = None,
    *,
    step: int = DEFAULT_STEP,
    entry_start: int = 0,
    entry_stop: int | None = None,
    device: Any = None,
) -> Iterator[dict[str, Any]]:
    """Walk a tree in batches, each one a dictionary of tensors.

    With no names it takes every numeric column, since a string is not a
    tensor and guessing an encoding for one would be worse than leaving it.
    Fixed-size array columns come out as ``(entries, width)``, and variable
    ones are padded to the widest row in that batch.
    """
    wanted = numeric(tree) if names is None else list(names)
    widths = {name: tree[name].length for name in wanted if tree[name].length > 1}
    for batch in tree.iterate(wanted, step=step, entry_start=entry_start, entry_stop=entry_stop):
        yield to_tensors(batch, widths=widths, device=device)


def dataset(
    tree: TTree,
    names: Sequence[str] | None = None,
    *,
    step: int = DEFAULT_STEP,
    device: Any = None,
) -> Any:
    """A PyTorch ``IterableDataset`` over a tree, batches already made.

        >>> loader = DataLoader(dataset(tree, step=4096), batch_size=None)  # doctest: +SKIP

    Use it with ``batch_size=None``: each item is already a batch of ``step``
    entries, which is the point - one basket read serves thousands of rows.
    Several worker processes split the entries between them, so each reads its
    own share of the file rather than all of them reading all of it.
    """
    torch = _torch()

    class TreeDataset(torch.utils.data.IterableDataset):  # type: ignore[misc, name-defined]
        """Entries of one tree, in batches, over whatever it was opened on."""

        def __iter__(self) -> Iterator[dict[str, Any]]:
            start, stop = 0, len(tree)
            worker = torch.utils.data.get_worker_info()
            if worker is not None:
                share = -(-len(tree) // worker.num_workers)
                start = min(worker.id * share, stop)
                stop = min(start + share, stop)
            return iter_tensors(
                tree, names, step=step, entry_start=start, entry_stop=stop, device=device
            )

        def __len__(self) -> int:
            return -(-len(tree) // step)

    return TreeDataset()
