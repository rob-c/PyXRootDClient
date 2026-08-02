"""A small CNN that learns Fashion-MNIST off a ``root://`` URL.

The same streaming as the other two, with the pictures given back their shape:
a row of 784 numbers becomes the 1x28x28 that a convolution wants.

    $ python -m xrd.testing datasets --port 21094 --pattern 'fashion_mnist.root'
    $ python examples/fashion_mnist_cnn.py root://127.0.0.1:21094//fashion_mnist.root
"""

import resource
import sys
import time
import tracemalloc

import torch
from torch import nn
from torch.utils.data import DataLoader

import xrd
from xrd.root import open_root
from xrd.root.ml import mixed

URL = sys.argv[1] if len(sys.argv) > 1 else "root://127.0.0.1:21094//fashion_mnist.root"
STEP, BATCH, EPOCHS = 512, 256, 5
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tracemalloc.start()


def pictures(batch):
    """A batch of rows of 784 numbers, as pictures between nought and one."""
    return batch["image"].float().view(-1, 1, 28, 28) / 255


with open_root(URL) as handle:
    trees = {split: [handle[name] for name in handle.trees() if name.startswith(split)]
             for split in ("train_", "test_")}
    train = DataLoader(
        mixed(trees["train_"], ["image", "label"], step=STEP, batch=BATCH, device=device),
        batch_size=None)
    test = DataLoader(
        mixed(trees["test_"], ["image", "label"], step=STEP, batch=1000, shuffle=False,
              device=device),
        batch_size=None)

    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    megabytes = xrd.Path(URL).stat().st_size / 1e6
    names = [name.removeprefix("train_") for name in handle.trees() if name.startswith("train_")]
    print(f"{URL}: {megabytes:,.0f} MB, {len(names)} classes {', '.join(names[:3])} ..., "
          f"on {device}")

    # From here on the only thing growing the heap is the reading, so what the
    # counter has gone up by at the end is what the data cost.
    tracemalloc.reset_peak()
    settled = tracemalloc.get_traced_memory()[0]
    for epoch in range(EPOCHS):
        start, seen, total = time.time(), 0, 0.0
        for batch in train:
            labels = batch["label"].long()
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(pictures(batch)), labels)
            loss.backward()
            optimiser.step()
            seen, total = seen + len(labels), total + loss.item() * len(labels)
        took = time.time() - start
        print(f"  epoch {epoch + 1}: {seen:,} rows in {took:4.1f}s ({seen / took:,.0f} rows/s), "
              f"loss {total / seen:.4f}")

    right = rows = 0
    with torch.no_grad():
        for batch in test:
            guessed = model(pictures(batch)).argmax(1)
            right += int((guessed == batch["label"].long()).sum())
            rows += len(guessed)
    held = (tracemalloc.get_traced_memory()[1] - settled) / 1e6
    resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  accuracy {right / rows:.4f} on {rows:,} test rows, streamed the same way")
    print(f"  never more than {held:,.1f} MB in hand at once, of a {megabytes:,.0f} MB file, "
          f"after {EPOCHS} passes over its training rows and one over the rest")
    print(f"  peak resident {resident:,.0f} MB, which is mostly PyTorch and CUDA")
