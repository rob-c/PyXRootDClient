"""The same MNIST classifier as ``mnist_mlp.py``, in the words a beginner has.

There are no baskets, offsets or dtypes here: a URL goes in, minibatches of
``(images, labels)`` come out, and the file stays on the server throughout.
Everything else - which trees are the training rows, which column is the
picture and which the answer, how much to hold at once, what to divide the
bytes by - is read off the file by :mod:`xrd.ml` and can be printed.

    $ python -m xrd.testing datasets --port 21094 --pattern 'mnist.root'
    $ python examples/mnist_easy.py root://127.0.0.1:21094//mnist.root

``mnist_mlp.py`` is this same program written against :mod:`xrd.root.ml`, one
layer down, for when the reading itself is what needs changing.
"""

import sys

import torch
from torch import nn

import xrd.ml

URL = sys.argv[1] if len(sys.argv) > 1 else "root://127.0.0.1:21094//mnist.root"
EPOCHS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

with xrd.ml.load(URL) as data:
    print(data)  # what the file holds, and what a batch will be
    print(data.train.preview())  # the first picture, drawn in characters
    print(f"rows per class: {data.train.counts()}\n")

    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        seen, total = 0, 0.0
        for images, labels in data.train.batches(256, device=device):
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimiser.step()
            seen, total = seen + len(labels), total + loss.item() * len(labels)
        print(f"epoch {epoch + 1}: {seen:,} rows, loss {total / seen:.4f}")

    right = rows = 0
    with torch.no_grad():
        for images, labels in data.test.batches(1000, shuffle=False, device=device):
            right += int((model(images).argmax(1) == labels).sum())
            rows += len(labels)
    print(f"accuracy {right / rows:.4f} on {rows:,} test rows, none of which were downloaded")
