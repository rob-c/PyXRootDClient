# ROOT files used by the tests

Fourteen small ROOT files, taken unchanged from the [go-hep](https://github.com/go-hep/hep)
project's `groot/testdata`, and used here to check that `xrd.root` reads what
ROOT wrote. They are here rather than generated because nothing in this
library can write a ROOT file: the only honest test of a reader is bytes
somebody else's writer produced.

| File | What it is there for |
| --- | --- |
| `simple.root` | one tree, an int, a float and a string |
| `small-flat-tree.root` | every numeric width, fixed-size arrays and variable-length ones |
| `leaves.root` | the same again with the leaf classes at the edges, including the `Double32_t` and `Float16_t` packings |
| `padding.root` | branches holding several leaves each, where the entry record has holes in it |
| `tntuple.root` | a `TNtuple`, which is a tree with another record wrapped round it |
| `dirs-6.14.00.root` | nested directories, and a histogram to be refused by name |
| `embedded-std-vector.root` | a `std::vector` member split out of a C++ class, read as rows |
| `pod-advanced.root` | a branch written in two baskets, so that a range crossing the boundary is read from both |
| `std-map-split1.root` | an object split into sub-branches, five kinds of `std::map` among them |
| `std-containers-split00.root` | forty columns of every STL container ROOT will write, unsplit |
| `small-evnt-tree-fullsplit.root` | a C++ class split all the way down: members, arrays, slices, strings and vectors |
| `small-evnt-tree-nosplit.root` | the same class and the same hundred events written whole, one object per entry |
| `std-map-split0.root` | the same maps as `std-map-split1.root`, written whole rather than split |
| `stdvec-bool-fullsplit-6.10.08.root` | `std::vector<bool>`, which is a byte per element and not a bit |

go-hep is BSD-3-Clause; the licence is in `LICENSE.go-hep` next to these
files, and it is the whole of what is required to redistribute them.
