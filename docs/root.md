# ROOT files

`xrd.root` opens a ROOT file and reads its trees in pure Python — over
`root://`, `https://`, WebDAV, `s3://` or a local path, with no ROOT, no
`uproot`, no `numpy` and no compiled extension anywhere in the way. It
[writes new files](#writing) too, and histograms and graphs
[draw themselves](#drawing), onto matplotlib axes or into plain characters.

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

A name that is not a tree is read too, when the file describes the class it
holds — which is the usual case for a C++ class somebody wrote, and for most of
ROOT's own kit as well:

```python
f["tlv"]                # {'TObject': {...}, 'fP': {'fX': 10.0, ...}, 'fE': 40.0}
f["FileSummaryRecord"]  # a std::string key is a str
f["written"]            # a TDatime is a datetime.datetime
f["h1d"]                # a histogram is a Histogram, and a graph a Graph
```

Objects held by pointer are followed, whether the class promises they are there
or writes the name of the class in front of them; a `TList`, `TObjArray`,
`TClonesArray` or `TArray` member is read the way it streams itself rather than
by the members it declares — a `TClonesArray` names the one class it holds at
the front, and a slot never filled comes back as `None` rather than shifting
the rest along; and a container of objects is read one object after another, or
a field at a time when it was written that way — every object's first member,
then every object's second — which is how ROOT writes a container of plain C++
objects and a `TClonesArray` asked to bypass its streamer. A `vector` of pairs
comes back as the list of tuples it is. A class
inside an entry that this reader cannot walk — one the file does not describe,
or one that streams itself some way of its own — comes back as the name of its
class, because the length written in front of it says how to step over it
without touching anything else.

The bytes have to account for themselves: a class that streams itself in some
way of its own leaves the object a different length than the layout says, and
that is refused by name rather than read into a plausible wrong answer.

## Histograms

A `TH1`, `TH2` or `TH3` of any bin type comes back as a `Histogram`: bins,
edges and what is in them, counted the way Python counts.

```python
h = f["h1d"]
h.name, h.title, h.shape          # ('h1d', 'h1d', (10,))
h.values()                        # array('d', [6.6, 72.6, 543.4, ...])
h.errors()                        # the same shape, from the sums of squared weights
h.edges()                         # 11 edges for 10 bins
h.axes[0].centers()               # where a point would be drawn
h.sum(), h.entries                # 11000.0, 10004.0
len(h), len(h.axes[0])            # 10, 10
```

`values()[0]` is the first bin of the axis, not the underflow. ROOT keeps two
extra bins per axis for what fell off each end, and `values(flow=True)` gives
them back at the ends where ROOT keeps them — so `sum(flow=True)` is everything
that was ever filled and `sum()` is what landed on the axis.

A two-dimensional histogram gives a row per bin along x, so `values()[ix][iy]`
is the bin ROOT calls `GetBinContent(ix+1, iy+1)`, and a three-dimensional one
nests once more. Everything is an `array.array`, which `numpy.asarray` takes
without copying and which needs nothing installed to use as it is.

A histogram filled with weights keeps the sum of their squares, and that is
where `errors()` comes from; one filled without them keeps no such sum, and
then the error on a count of *n* is its square root, which is what ROOT itself
would give. An unevenly binned axis keeps every edge and those are given back
exactly; an evenly binned one keeps none, and they are worked out from the ends.

Every member the histogram was written with is still there under `h.members`,
under the name of the class that declared it, so nothing is hidden by being
tidied away.

## Graphs

A `TGraph`, `TGraphErrors`, `TGraphAsymmErrors` or `TGraphMultiErrors` comes
back as a `Graph`, which is a sequence of points:

```python
g = f["tge"]
len(g), g[0]                      # 4, (1.0, 2.0)
for x, y in g:
    ...
g.x, g.y                          # array('d', [...]) each, one value per point
g.points()                        # [(1.0, 2.0), (2.0, 4.0), ...]
below, above = g.yerr             # the bars either side, or None if none were kept
```

`xerr` and `yerr` are always a pair, low side first, so the same code reads a
graph whose bars are symmetric and one whose are not — a graph that kept one
array for both sides gives that array twice. A `TGraph` proper kept none, and
both are `None`.

A `TGraphMultiErrors` keeps its y errors in layers — statistical and
systematic, say — and `g.layers` is those pairs of bars in the order they were
added. Asking such a graph for `yerr` raises rather than picking a layer or
summing them for you, because how to combine them is physics, not format. For
every other graph `layers` is simply `(yerr,)`, or `()` when none were kept.

A graph or histogram met *inside* another object — in the list a `TMultiGraph`
keeps, behind a pointer in a `TEfficiency` — comes back as a `Graph` or
`Histogram` too, the same as it would standing in a key of its own.

## Drawing

ROOT draws through a `TCanvas`, which this library does not carry. What it
has instead is the two ways Python usually looks at data. `plot()` draws onto
matplotlib axes — made on demand, or brought along — and returns them, so
styling and saving carry on where it left off:

```python
ax = f["h1d"].plot()                       # steps for 1D, a shaded mesh for 2D
f["tge"].plot(ax=ax, color="crimson")      # points with their error bars
ax.figure.savefig("both.png")
```

matplotlib is not a dependency; `pip install pyxrootdclient[plot]` brings it,
and without it `plot()` refuses with both ways out by name. The other way is
`text()`, which needs nothing installed at all and goes anywhere a string
goes — a terminal, a log file, a CI transcript:

```python
print(f["h1d"].text())     # one line per bin: its edges, a bar and the value
print(f["h2d"].text())     # a shaded grid, y upward
print(f["tge"].text())     # a grid of stars with the axis ends labelled
```

A graph of layered error bars draws every layer over the same points; a
three-dimensional histogram has no honest flat picture and refuses both ways,
saying to slice `values()` down to the two dimensions you want to see.

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
| a string, `std::string` or `TString` | a `list[str]` |
| an STL container | a `list`, one Python object per entry |

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

## C++ classes and containers

A file written by a physics framework is usually a C++ class per entry, split
by ROOT into one branch per member. Those branches are columns here like any
other, under the names ROOT gave them:

```python
tree.keys()             # ['Muon.pt', 'Muon.eta', 'evt.N', 'evt.StlVecF64', ...]
tree["evt.StlVecF64"].array(0, 100)   # <Jagged 100 rows of ...>
```

The object itself is a branch too, holding nothing at all — every byte of it
is in the members — so asking for it gives back one dictionary per entry, and
an object nested inside it is a dictionary inside that:

```python
tree.groups()                  # ['evt', 'P3'] - the objects that were split
tree["evt"].array(1, 2)
# [{'I32': 1, 'Str': 'evt-001', 'P3': {'Px': 0, 'Py': 1.0, 'Pz': 0}, ...}]
```

This reads exactly the same baskets as asking for the members, and costs the
same; it is a shape, not a shortcut. `tree.arrays()` and `tree.iterate()`
leave the split objects out, because their members are already there under
their own names and taking both would read every basket twice. A member this
reader will not decode is absent from the dictionary and named in
`tree["evt"].unreadable`, with the same sentence `tree.unreadable` gives it.

A file written with splitting turned off puts the whole object in one branch,
with nothing under it. That reads as the same dictionary per entry, walked
member by member in the order the class declares them, using the layout the
file's own streamer information gives:

```python
tree.keys()                    # ['evt'] - the object, and nothing under it
tree["evt"].array(1, 2)
# [{'Beg': 'beg-001', 'I32': 1, 'P3': {'Px': 0, 'Py': 1.0, 'Pz': 0}, ...}]
```

A class that inherits keeps each base under the base's own name, because a
derived class is allowed to declare a member its base already declared and
flattening the two together would quietly drop one of them. `TObject`, which
nearly every ROOT class inherits, comes back as the identifier and bits it
really is:

```python
tree["p4"].array(0, 1)
# [{'TObject': {'fUniqueID': 0, 'fBits': 50331648},
#   'fP': {'TObject': {...}, 'fX': 0.0, 'fY': 1.0, 'fZ': 2.0}, 'fE': 3.0}]
```

Some classes stream themselves rather than being written out by the file's
streamer information - `TLorentzVector` is one - and then the entry is the
class's own record with a version in front of the members. Both are read, and
so is the older `TBranchObject`, which writes the name of the class in front
of every entry; a branch of that kind holding more than one class stops with a
message naming both rather than reading the one as the other.

The same events written the two ways read back the same values, member for
member. Splitting is still the cheaper way to have written them - a split file
lets you read one member without touching the rest, and an unsplit one cannot -
but neither is a file this reader has to refuse. A class the file's streamer
information does not describe is refused, because its layout is then not
knowable, and so is a member of a kind this reader has no reader for: an
unsplit object is read from first byte to last, so one member it cannot walk
past is the whole entry.

A member that is an STL container comes back as the Python thing it most
nearly is, one per entry:

| In the file | In Python |
| --- | --- |
| `std::vector<double>`, `list`, `deque`, `set` of numbers | a `Jagged` — rows of `array.array` |
| a container of strings | a `list[list[str]]` |
| `vector<vector<T>>` | a `list` of `list`s of `array.array` |
| `std::map<K, V>`, `unordered_map` | a `list[dict]`, one dict per entry |
| `std::string`, `TString` | a `list[str]` |
| `std::vector<bool>` | rows of 0 and 1, a byte an element, which is how ROOT wrote it |
| `ROOT::VecOps::RVec<T>` | whatever the same `std::vector<T>` gives, which is what it is written as |
| `std::bitset<N>` | rows of 0 and 1, `bs[0]` first, a byte a bit as ROOT wrote it |
| `TDatime` | a `datetime.datetime`, out of the one word it packs itself into |

`Double32_t` and `Float16_t` are floats squeezed into three or four bytes by a
recipe written as `[xmin,xmax,nbits]`, where the ends may be given in units of
`pi`. A branch of its own keeps the recipe in the leaf title; a member of a
split class keeps it in the trailing comment on the declaration, which is in
the file's streamer information — either way it is unpacked back to `float64`,
so `typenames()` says `float64` and nothing about the packing reaches you. A
packed member of a class the file does not describe is refused rather than
read at the default, because the wrong recipe gives plausible wrong numbers.

## Into PyTorch and TensorFlow

`xrd.root.ml` turns those batches into tensors. Neither framework is imported
until it is called, so nothing here costs anything on a machine with neither:

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
them, because neither framework's unsigned types beyond `uint8` are supported
well enough to hand anybody; a value too large for `int64` is refused by name
rather than wrapped around.

TensorFlow takes the same tree through `tf_dataset`, which declares its shapes
and types up front — from what the file says the columns are, without a first
pass over the data — so the rest of `tf.data` works on it:

```python
data = xrd.root.ml.tf_dataset(tree, ["Muon_pt"], step=8192).prefetch(2)
for batch in data:
    model(batch["Muon_pt"])
```

With no column names, either dataset takes every numeric column and leaves the
strings and objects out, rather than leaving them in to fail later.

## Writing

`create` makes a new ROOT file anywhere this library can put bytes — a local
path, any URL scheme it writes, or an already-open binary file, used as it is
and left open:

```python
import xrd.root

with xrd.root.create("root://eos.example.org//store/user/me/out.root") as f:
    f["counts"] = xrd.root.Histogram.new("counts", edges, values)
    f["scan"] = xrd.root.Graph.new("scan", xs, ys, yerr=bars)
    f["note"] = "made from run 4711"
    f["weights"] = array.array("d", weights)
```

The file is a mapping from name to object, closed with `with` or `close()`. It
is the real thing — keys, directory, free list, and the streamer information
describing its classes exactly as the ROOT the layouts were harvested from
would — so ROOT, uproot and this library's own reader all read it back by its
own self-description. A name written twice becomes a second cycle of itself,
exactly as in ROOT, and reading back gives the newest. Everything is built in
memory and written out in one piece at a clean close; a `with` block that
raises writes nothing at all, on the principle that no file is better than
half a file.

What can be written is what can be written *correctly*: trees of numbers,
histograms and graphs — read from another file, or built from plain numbers —
plus strings and `array.array`s of signed integers or floats, which become the
matching `TArray`. `Histogram.new` takes every bin edge, one value per bin (two more
fills the flow bins), and optionally per-bin errors and an entry count;
evenly spaced edges are stored the compact way ROOT stores an even axis.
`Graph.new` picks its own class: plain points make a `TGraph`, one bar per
point a `TGraphErrors`, and any `(low, high)` pair of runs a
`TGraphAsymmErrors`.

Anything else is refused by name rather than guessed at — a
`TGraphMultiErrors`, an unsigned array ROOT has no class for, a histogram
with fits attached or an axis with labels (empty them first, rather than
have them silently dropped), a name a reader could never ask back for —
because a plausible-looking file that ROOT misreads is worse than an error
message. Files past 2 GB, which need ROOT's wide layout, are refused too:
split the output across files.

`create(..., compression="zstd")` chooses the algorithm — zlib unless said
otherwise, or `lzma`, `lz4`, `zstd`, `None` to store raw — and `level` the
effort; an object that did not get smaller is stored raw anyway, exactly as
ROOT does.

### Trees

`f.tree(name, columns)` declares a tree, and `fill` gives it entries:

```python
with xrd.root.create("out.root") as f:
    tree = f.tree("events", {"energy": float, "hits": ("i", 4), "ok": bool})
    for event in run:
        tree.fill(energy=event.energy, hits=event.hits, ok=event.passed)
```

A column is a Python `bool`, `int` or `float`, an [`array`][array] type code
such as `'f'` for a narrower number, or a pair of either and how many values
every entry holds — `("i", 4)` is four `int32`s per entry. A Python `int` gets
the widest ROOT has, because a Python int has no width of its own and a column
that quietly stopped fitting halfway down the file would be worse. `extend`
takes an iterable of mappings for the same thing in bulk.

[array]: https://docs.python.org/3/library/array.html

Entries go out a basket at a time as they gather — `basket_size` bytes of a
column at a time, 32 kB unless said otherwise — so a tree can be far larger
than memory, and reading a range back later costs one read per basket it
touches rather than one read of everything. Each column fills its own baskets
at its own rate, so a wide image column and a one-byte label column do not
force each other's geometry. The tree's own record says where every basket
landed, so it is written when the file closes.

An entry that does not fit its columns is refused whole — nothing is kept for
any column, so the tree is exactly as it was — and it is refused by name:
which column, what it holds, and what arrived instead. Columns are fixed-size
by design; rows of varying length, strings and split C++ objects are refused
rather than approximated, on the same principle as the rest of the writer.

The result is a tree laid out the way ROOT lays one out, down to the record
versions and the `fLeaves` references pointing at the very leaves the branches
hold, so ROOT, uproot and this library's own reader all walk it the same way.

### MNIST, end to end

`xrd.root.mnist` is the worked example: it fetches the handwritten digits —
through this library, so the source can be a URL, a path or bytes already in
hand — and writes them as one tree per class. It is one of the eight in
[`xrd.root.datasets`](#the-datasets-everyone-teaches-with) under its own name,
because it is the one everybody starts with.

```python
from xrd.root import create, mnist

with create("mnist.root") as f:
    for split in ("train", "test"):
        print(mnist.convert(f, split=split))
# {'train_0': 5923, 'train_1': 6742, ... 'train_9': 5949}
# {'test_0': 980, 'test_1': 1135, ... 'test_9': 1009}
```

That is all 70,000 images in one 11.6 MB file — as small as the gzipped
originals, and unlike them addressable a range at a time. Each tree holds
`image` (784 `uint8`, the 28×28 picture flat), `label`, and `index`, the
entry's place in the original file, so any row can be traced back. One tree
per digit is the shape a training loop wants: sampling a class reads one part
of the file rather than seeking all over it, and every tree carries its label
anyway, so concatenating all ten and shuffling works exactly as well.

Training on it is the loader from [above](#into-pytorch-and-tensorflow), with
nothing in between:

```python
import torch, xrd.root, xrd.root.ml

with xrd.root.open_root("mnist.root") as f:
    loaders = [
        torch.utils.data.DataLoader(
            xrd.root.ml.dataset(f[f"train_{digit}"], ["image", "label"], step=64),
            batch_size=None,
        )
        for digit in range(10)
    ]
    for parts in zip(*loaders):            # 64 of each digit: a balanced batch
        x = torch.cat([part["image"] for part in parts]).float().div_(255)
        y = torch.cat([part["label"] for part in parts]).long()
        loss = torch.nn.functional.cross_entropy(model(x.view(-1, 1, 28, 28)), y)
```

The `image` column arrives shaped `(entries, 784)` because the file says it is
784 wide, so that `view` is the only shaping anybody has to write. Taking one
batch from each loader is what the per-class layout buys: every batch is class
balanced without a sampler, and the epoch ends with the smallest class.
Training on the ten trees as one stream instead is `itertools.chain`, and
shuffling across them is whatever your training would do anyway.

Point `open_root` at a `root://` or `https://` URL and the same loop trains
straight off a storage element, reading the baskets it needs and nothing else.

### The datasets everyone teaches with

MNIST is one of eight. `xrd.root.datasets` converts the sets machine learning
is actually taught and benchmarked with, all of them the same way — one tree
per class, the label beside the data, and the row's place in the original file
so any number can be traced back to where it came from.

```python
from xrd.root import datasets

datasets.convert("cifar10", "cifar10.root", split="train")
# {'train_airplane': 5000, 'train_automobile': 5000, ... 'train_truck': 5000}

datasets.convert("iris", "iris.root")
# {'setosa': 50, 'versicolor': 50, 'virginica': 50}
```

`describe()` prints the lot, with the licence and the source of each:

| name | what it is | licence | converted |
| --- | --- | --- | --- |
| `mnist` | 70,000 handwritten digits, 28×28 greyscale, 10 classes | CC BY-SA 3.0 | 11.6 MB |
| `fashion_mnist` | 70,000 clothing photographs, 28×28 greyscale, 10 classes | MIT | 30.7 MB |
| `kmnist` | 70,000 classical Japanese characters, 28×28 greyscale, 10 classes | CC BY-SA 4.0 | 21.6 MB |
| `cifar10` | 60,000 photographs, 32×32 colour, 10 classes | see below | 169.1 MB |
| `cifar100` | 60,000 photographs, 32×32 colour, 100 classes in 20 superclasses | see below | 167.3 MB |
| `iris` | 150 iris flowers measured four ways, 3 species | CC BY 4.0 | 9 kB |
| `penguins` | 344 penguins measured at Palmer Station, 3 species | CC0 | 13 kB |
| `covertype` | 581,012 patches of Colorado forest, 54 features, 7 cover types | CC BY 4.0 | 8.3 MB |

The last column is one file holding every split, written with the default
`zlib`, as measured on a conversion of all eight — both splits of a set that
has them, a tree a class, and the `about` key beside them.

Nothing is redistributed here. Each set is fetched from whoever publishes it,
on the machine doing the converting, and the licences above are what those
publishers say — read them before passing the converted file on. The CIFAR
sets have no formal licence at all; Krizhevsky asks that the tech report be
cited. So that a file outlives the program that made it, `convert` writes an
`about` key beside the trees holding the same statement:

```python
with xrd.root.open_root("cifar10.root") as f:
    print(f["train_about"])
# CIFAR-10: 60,000 photographs, 32x32 colour, 10 classes
# split: train
# licence: no formal licence; Krizhevsky asks that the tech report be cited
# source: https://www.cs.toronto.edu/~kriz/cifar.html
# converted by xrd.root.datasets, one tree per class
```

The CIFAR sets are taken in their **binary** distribution rather than the
Python one, on purpose: the Python one is a pickle, and unpickling a download
is a way to run somebody else's code. Every archive here — IDX, tar, zip,
gzip, CSV — is read with the standard library and nothing else.

Images come out exactly as the archive laid them: CIFAR is 3072 bytes an
entry, 1024 red then 1024 green then 1024 blue, which is what PyTorch wants,
so the loop is the MNIST one with a wider `view`:

```python
loaders = [
    torch.utils.data.DataLoader(
        xrd.root.ml.dataset(f[f"train_{cls}"], ["image", "label"], step=64),
        batch_size=None,
    )
    for cls in ("airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck")
]
for parts in zip(*loaders):
    x = torch.cat([part["image"] for part in parts]).float().div_(255)
    loss = criterion(model(x.view(-1, 3, 32, 32)), ...)
```

The tabular sets become one column per field instead of one wide array:

```python
with xrd.root.open_root("penguins.root") as f:
    print(xrd.root.ml.numeric(f["gentoo"]))
# ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
#  'body_mass_g', 'sex', 'year', 'label', 'index']
```

Trees hold numbers, so the categories are numbered: `datasets.DATASETS[
"penguins"].codes` says which is which, and a category nobody declared is
refused rather than guessed at. A measurement nobody took becomes a NaN, which
is what every reader downstream already means by it; a category nobody
recorded becomes `-1`, because there is no such code.

Converting is one pass, and `parts=` takes archives already on disk when you
would rather not download them twice:

```python
datasets.convert("cifar100", "cifar100.root", split="test",
                 parts={"archive": "cifar-100-binary.tar.gz"})
```

`base=` points the downloads at a mirror of your own, and `target` may be a
`WritableFile` already open, which is how several splits end up in one file.
Adding a dataset of your own is an `Images`, `CIFAR` or `Table` in the
registry — each is a frozen dataclass saying where the data is and what its
fields mean, with no code to write for the common shapes.

## Compression

Every algorithm ROOT writes with is read here — and written: zlib, lzma, LZ4,
zstd, and the bare-deflate blocks ROOT wrote before 2005 (read only). The LZ4
coder is Python both ways, checksum and all, because a physics file should not
need a wheel; zstd uses Python 3.14's own `compression.zstd` where there is
one and the `zstandard` package otherwise, and is the one case where a file
may need something installed.

## Old files

A tree written by ROOT 4 opens like any other. Those files count entries in
doubles and keep their seek points in 32-bit integers, and one small enough
never to have been flushed holds its baskets inside the branch record rather
than out in the file — all of which is read here, so a decade-old Geant4 run
needs no copying forward first. ROOT 3 and older are refused by name.

## What it refuses, and why by name

A plausible misreading of physics data is worse than a refusal, so anything
this reader does not understand is named rather than guessed at. A column that
cannot be decoded is listed in `tree.unreadable` with the reason, and raises
with the same sentence if it is asked for:

```python
>>> tree.unreadable
{'vtx': 'Vertex, which is a C++ type this reader does not decode: this '
        "file's streamer information does not describe its layout, and a "
        'file written split would have its members as branches of their own'}
```

What is named that way:

- a class whose layout the file's streamer information does not describe, so
  that reading it whole would be a guess;
- a member of an unsplit object of a kind this reader has no reader for, which
  stops the entry it is in, because an unsplit entry is read from first byte to
  last and there is no length in front of a member to step over it by;
- a `multimap`, whose duplicate keys a `dict` would silently drop, and a map
  keyed by a container or nested inside one;
- a `pair` anywhere but directly under its own container, and a container of
  pairs written pair by pair, neither of which any file this reader has met
  writes;
- a graph of layered y errors asked for `yerr`, because summing the layers
  would be an answer this reader made up;
- trees written by ROOT 3 or older.

A class the file describes as having no members at all — `TLimit` is one —
reads as the empty `dict` it honestly is, rather than being refused.

## Errors

`xrd.root.ROOTError` is an `XRootDError`, so it is an `OSError` like
everything else this library raises.

| Exception | Means |
| --- | --- |
| `FormatError` | the bytes are not the ROOT format they claim — a truncated download and an HTML error page both look like this |
| `UnsupportedFeatureError` | a valid file using something this reader does not do, named rather than guessed at |
