"""MNIST as a ROOT file: one tree per digit, images and labels inside.

The handwritten digits are the worked example this library's machine
learning side is documented against: small enough to convert in a few
seconds, big enough that reading it entry by entry would be the wrong idea,
and shaped like real data - a fixed-size array per entry, a label beside it.

The images come out separated by class, one tree per digit, because that is
the shape most training loops want: sample a class, read a range of that
class's entries, and the file gives them back in one read rather than ten
thousand. Every tree carries the label anyway, so concatenating them all and
shuffling works exactly as well.

Nothing here needs the dataset to be on disk first - the IDX files are read
through this library like any other URL, so converting straight from the
mirror into a file on an XRootD server is one call.
"""

from __future__ import annotations

import gzip
import struct
from typing import Any

from ..url import parse
from .writer import WritableFile, create

__all__ = [
    "SIDE", "PIXELS", "MIRROR", "FILES", "COLUMNS", "IMAGE_BASKET",
    "read_idx", "fetch", "convert",
]

#: One MNIST image is 28 by 28 greyscale pixels, one byte apiece.
SIDE = 28
PIXELS = SIDE * SIDE

#: Where the dataset is served from. The original site stopped offering it;
#: this is the mirror PyTorch's own MNIST downloader uses.
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

#: The two files each split comes in: the images, then their labels.
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}

#: What each entry holds: the image flat, the digit it is, and where it came
#: from in the original file, so a row can always be traced back.
COLUMNS: dict[str, Any] = {"image": ("B", PIXELS), "label": "i", "index": "i"}

#: How many bytes of images gather before a basket goes out. Larger than the
#: usual default on purpose: a training loop reads thousands of rows at a
#: time, and at this size that is one read rather than twenty.
IMAGE_BASKET = 512 * 1024

#: What the type byte of an IDX file says its values are. MNIST holds
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
        raise ValueError(f"this IDX file holds {what}, and MNIST holds unsigned bytes")
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


def convert(
    target: Any,
    *,
    split: str = "train",
    images: Any = None,
    labels: Any = None,
    base: str = MIRROR,
    prefix: str | None = None,
    compression: str | None = "zlib",
    basket_size: int = IMAGE_BASKET,
    config: Any = None,
) -> dict[str, int]:
    """Write MNIST into a ROOT file, one tree per digit; say what went where.

        >>> from xrd.root import mnist                       # doctest: +SKIP
        >>> mnist.convert("mnist.root")                      # doctest: +SKIP
        {'train_0': 5923, 'train_1': 6742, ...}

    ``target`` is where the file goes - a path, a URL, an open binary file,
    or a :class:`~.writer.WritableFile` already open, which is how both
    splits end up in one file. ``images`` and ``labels`` default to the
    ``split`` downloaded from ``base``, and take raw IDX bytes, a path or a
    URL when the dataset is already to hand.

    Each tree is named for its split and digit - ``train_7`` - and holds the
    image flat as 784 bytes, the label, and the entry's place in the original
    file. Reading it back needs nothing but this library.
    """
    if split not in FILES:
        raise ValueError(f"the splits MNIST comes in are {' and '.join(FILES)}, not {split!r}")
    prefix = split if prefix is None else prefix
    image_file, label_file = FILES[split]
    shape, pixels = read_idx(fetch(base + image_file if images is None else images, config=config))
    counts, digits = read_idx(fetch(base + label_file if labels is None else labels, config=config))

    if len(shape) != 3 or shape[1:] != (SIDE, SIDE):
        raise ValueError(
            f"these images are {'x'.join(str(size) for size in shape)}, and MNIST images come "
            f"as a count of {SIDE}x{SIDE} pictures"
        )
    if len(counts) != 1 or counts[0] != shape[0]:
        raise ValueError(
            f"there are {shape[0]} images and {counts[0] if len(counts) == 1 else counts} "
            f"labels, and every image needs exactly one"
        )

    given = isinstance(target, WritableFile)
    out = target if given else create(target, compression=compression, config=config)
    try:
        trees = {
            digit: out.tree(
                f"{prefix}_{digit}",
                COLUMNS,
                title=f"MNIST {split} images of the digit {digit}, {SIDE}x{SIDE} greyscale",
                basket_size=basket_size,
            )
            for digit in range(10)
        }
        view = memoryview(pixels)
        for index, digit in enumerate(digits):
            tree = trees.get(digit)
            if tree is None:
                raise ValueError(
                    f"image {index} is labelled {digit}, and MNIST is the digits 0 to 9"
                )
            at = index * PIXELS
            tree.fill(image=view[at : at + PIXELS], label=digit, index=index)
        written = {tree.name: len(tree) for tree in trees.values()}
    finally:
        if not given:
            out.close()
    return written
