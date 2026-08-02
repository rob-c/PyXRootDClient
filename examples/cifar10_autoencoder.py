"""An MLP autoencoder that learns CIFAR-10 off a ``root://`` URL.

The file is 169 MB and every epoch reads the whole of it, in baskets. The line
at the end says how much of it was in this process at the widest moment: a
pool of rows, which would be the same figure for a terabyte.

    $ python -m xrd.testing datasets --port 21094 --pattern 'cifar10.root'
    $ python examples/cifar10_autoencoder.py root://127.0.0.1:21094//cifar10.root
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

URL = sys.argv[1] if len(sys.argv) > 1 else "root://127.0.0.1:21094//cifar10.root"
STEP, BATCH, EPOCHS, CODE = 1024, 256, 8, 64
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tracemalloc.start()

with open_root(URL) as handle:
    trees = {split: [handle[name] for name in handle.trees() if name.startswith(split)]
             for split in ("train_", "test_")}
    loaders = {
        split: DataLoader(
            mixed(these, ["image"], step=STEP, batch=BATCH, shuffle=(split == "train_"),
                  device=device),
            batch_size=None)
        for split, these in trees.items()}

    # 3072 numbers a photograph - three 32x32 planes - down to CODE and back.
    # Nothing is bent at the narrow layer: a code that can only be positive
    # throws away half of what a picture had to say about itself.
    model = nn.Sequential(
        nn.Linear(3072, 512), nn.ReLU(), nn.Linear(512, CODE),
        nn.Linear(CODE, 512), nn.ReLU(), nn.Linear(512, 3072), nn.Sigmoid()).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    megabytes = xrd.Path(URL).stat().st_size / 1e6
    print(f"{URL}: {megabytes:,.0f} MB, {len(trees['train_'])} classes, "
          f"{len(loaders['train_']):,} minibatches an epoch, on {device}")

    # From here on the only thing growing the heap is the reading, so what the
    # counter has gone up by at the end is what the data cost.
    tracemalloc.reset_peak()
    settled = tracemalloc.get_traced_memory()[0]
    for epoch in range(EPOCHS):
        start, seen, total = time.time(), 0, 0.0
        for batch in loaders["train_"]:
            pictures = batch["image"].float() / 255
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(pictures), pictures)
            loss.backward()
            optimiser.step()
            seen, total = seen + len(pictures), total + loss.item() * len(pictures)
        took = time.time() - start
        print(f"  epoch {epoch + 1}: {seen:,} pictures in {took:4.1f}s "
              f"({seen / took:,.0f} pictures/s), mse {total / seen:.5f}")

    squares = rows = 0.0
    with torch.no_grad():
        for batch in loaders["test_"]:
            pictures = batch["image"].float() / 255
            squares += float(((model(pictures) - pictures) ** 2).sum())
            rows += len(pictures)
    off = (squares / (rows * 3072)) ** 0.5 * 255
    held = (tracemalloc.get_traced_memory()[1] - settled) / 1e6
    resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  {off:.1f} levels out of 255 per pixel on {rows:,.0f} held-out photographs")
    print(f"  never more than {held:,.1f} MB in hand at once, of a {megabytes:,.0f} MB file, "
          f"after {EPOCHS} passes over its training rows and one over the rest")
    print(f"  peak resident {resident:,.0f} MB, which is mostly PyTorch and CUDA")
