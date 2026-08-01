# Reading ROOT files

`xrd.root` opens a ROOT file and reads its trees in pure Python — over
`root://`, `https://`, WebDAV, `s3://` or a local path, with no ROOT, no
`uproot`, no `numpy` and no compiled extension anywhere in the way.

```python
import xrd.root

with xrd.root.open_root("root://eos.example.org//store/events.root") as f:
    tree = f["Events"]
    print(len(tree), tree.keys())
    pt = tree["Muon_pt"].array(0, 10_000)
```

Nothing is downloaded. A tree is a set of *baskets* — compressed blocks of
consecutive entries, one branch at a time — and asking for entries 0 to 10 000
of one branch reads the baskets that hold them and nothing else. That is what
makes it reasonable to walk a hundred-gigabyte file from a laptop: the bytes
that cross the wire are the ones asked for.

## Opening

```python
xrd.root.open_root("/local/path/f.root")                  # a path
xrd.root.open_root("davs://dav.example.org/store/f.root") # any scheme this library speaks
xrd.root.open_root(open("f.root", "rb"))                  # an open file, left open
```

A file is a mapping from name to object:

```python
f.keys()             # ['Events', 'metadata']
f.classnames()       # {'Events': 'TTree', 'metadata': 'TH1F'}
f.trees()            # ['Events']
f.tree()             # the only tree, or a KeyError naming the ones there are
f["dir/sub/Events"]  # directories nest, with a path
f["Events;1"]        # an older cycle, when a file kept several
```

`classnames()` is the first thing to try when something will not open: it says
what a name is without reading any of it.

## Columns

```python
tree.show()                 # one line per column: name, type, variable or not
tree.typenames()            # {'Muon_pt': 'float32', 'nMuon': 'int32', ...}
tree.keys()                 # every column
tree.readable()             # the ones this reader decodes
tree.unreadable             # {name: why not}, rather than quietly missing
```

A column comes back as one of three things, and which one is knowable in
advance from `tree[name].is_jagged` and `typename`:

| The column | What `array()` gives |
| --- | --- |
| a plain number (`x/F`) | an `array.array` of one value per entry |
| a fixed array (`x[10]/F`) | an `array.array`, `branch.length` values per entry |
| a variable one (`x[n]/F`) | a `Jagged` — rows of different lengths |
| a character leaf (`x/C`) | a `list[str]` |

```python
tree["nMuon"].array()             # array('i', [2, 0, 3, ...])
tree["Muon_pt"].array(0, 1000)    # <Jagged 1000 rows of 2431 f values>
jets = tree["Muon_pt"].array()
jets[7]                           # array('f', [22.5, 19.0])
jets.lengths()                    # [2, 0, 3, ...]
jets.tolist()
values, width = jets.padded()     # a flat rectangle and its width
```

Entry numbers behave like a Python slice, negatives included:
`branch.array(-1000)` is the last thousand entries.

## Reading a file that does not fit

`iterate` walks the tree in batches, holding one batch rather than one file:

```python
for batch in tree.iterate(["Muon_pt", "Muon_eta"], step=50_000):
    analyse(batch["Muon_pt"], batch["Muon_eta"])
```

`tree.arrays()` is the same for one range, and takes every readable column
when it is not told which.

## Into PyTorch

`xrd.root.ml` turns those batches into tensors, and is not imported until it
is called, so nothing here costs anything on a machine with no PyTorch:

```python
import torch, xrd.root, xrd.root.ml

tree = xrd.root.open_root("root://eos.example.org//store/events.root").tree()
loader = torch.utils.data.DataLoader(
    xrd.root.ml.dataset(tree, ["Muon_pt", "Muon_eta"], step=8192),
    batch_size=None,          # each item is already a batch
    num_workers=4,            # each worker reads its own share of the entries
)

for batch in loader:
    model(batch["Muon_pt"])
```

`batch_size=None` is the point: one basket read serves thousands of rows, so
the batching belongs here and not in the loader. Several workers split the
entry range between them, so each reads a different part of the file rather
than all of them reading all of it.

Fixed-size array columns arrive shaped `(entries, width)`. Variable ones are
padded to the widest row in the batch, or to a width you give:

```python
xrd.root.ml.to_tensor(jets, width=4, fill=0.0)      # (entries, 4)
xrd.root.ml.iter_tensors(tree, step=8192, device="cuda")
```

Unsigned columns wider than a byte are widened into the signed type that holds
them, because torch's unsigned types beyond `uint8` are too thinly supported
to hand anybody; a value too large for `int64` is refused by name rather than
wrapped around.

## What it refuses, and why by name

This reader does the plain numeric and string leaves. A column of split C++
objects needs the file's streamer information, which it does not decode, so
such a column is listed in `tree.unreadable` with the reason and raises with
the same sentence if it is asked for:

```python
>>> tree.unreadable
{'hits': "a split C++ object, which needs the file's streamer information"}
```

That is deliberate. A plausible misreading of physics data is worse than a
refusal, so anything not understood is named — the truncated `Float16_t` and
`Double32_t` leaves, `zstd` compression where neither Python 3.14 nor the
`zstandard` package is present, and trees written by ROOT 4 or older. Files
compressed with zlib, lzma or LZ4 are read as they are; the LZ4 decoder is
Python, because a physics file should not need a wheel to be read.

## Errors

`xrd.root.ROOTError` is an `XRootDError`, so it is an `OSError` like
everything else this library raises.

| Exception | Means |
| --- | --- |
| `FormatError` | the bytes are not the ROOT format they claim — a truncated download and an HTML error page both look like this |
| `UnsupportedFeatureError` | a valid file using something this reader does not do, named rather than guessed at |
