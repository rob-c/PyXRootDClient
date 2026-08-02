"""An MLP that learns MNIST off a ``root://`` URL, a basket at a time.

Nothing is downloaded first and nothing is held: every minibatch is a read of
one basket out of the file on the server. The line at the end says how much of
the file was in this process at the widest moment, counted by ``tracemalloc``,
which sees the reading but not PyTorch's own arenas.

    $ python -m xrd.testing datasets --port 21094 --pattern 'mnist.root'
    $ python examples/mnist_mlp.py root://127.0.0.1:21094//mnist.root
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

URL = sys.argv[1] if len(sys.argv) > 1 else "root://127.0.0.1:21094//mnist.root"
STEP, BATCH, EPOCHS = 512, 256, 5
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tracemalloc.start()

with open_root(URL) as handle:
    # One tree per class, so a loop over them in turn would teach the model
    # nothing but zeroes and then nothing but ones: mixed() pools a read from
    # each and cuts minibatches off the pool.
    trees = {split: [handle[name] for name in handle.trees() if name.startswith(split)]
             for split in ("train_", "test_")}
    train = DataLoader(
        mixed(trees["train_"], ["image", "label"], step=STEP, batch=BATCH, device=device),
        batch_size=None)
    test = DataLoader(
        mixed(trees["test_"], ["image", "label"], step=STEP, batch=1000, shuffle=False,
              device=device),
        batch_size=None)

    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    megabytes = xrd.Path(URL).stat().st_size / 1e6
    print(f"{URL}: {megabytes:,.0f} MB, {len(train):,} minibatches an epoch, on {device}")

    # From here on the only thing growing the heap is the reading, so what the
    # counter has gone up by at the end is what the data cost.
    tracemalloc.reset_peak()
    settled = tracemalloc.get_traced_memory()[0]
    for epoch in range(EPOCHS):
        start, seen, total = time.time(), 0, 0.0
        for batch in train:
            labels = batch["label"].long()
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch["image"].float() / 255), labels)
            loss.backward()
            optimiser.step()
            seen, total = seen + len(labels), total + loss.item() * len(labels)
        took = time.time() - start
        print(f"  epoch {epoch + 1}: {seen:,} rows in {took:4.1f}s ({seen / took:,.0f} rows/s), "
              f"loss {total / seen:.4f}")

    right = rows = 0
    with torch.no_grad():
        for batch in test:
            guessed = model(batch["image"].float() / 255).argmax(1)
            right += int((guessed == batch["label"].long()).sum())
            rows += len(guessed)
    held = (tracemalloc.get_traced_memory()[1] - settled) / 1e6
    resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  accuracy {right / rows:.4f} on {rows:,} test rows, streamed the same way")
    print(f"  never more than {held:,.1f} MB in hand at once, of a {megabytes:,.0f} MB file, "
          f"after {EPOCHS} passes over its training rows and one over the rest")
    print(f"  peak resident {resident:,.0f} MB, which is mostly PyTorch and CUDA")
