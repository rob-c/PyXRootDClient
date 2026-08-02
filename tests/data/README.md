# ROOT files used by the tests

Thirty-one small ROOT files, taken unchanged from the [go-hep](https://github.com/go-hep/hep)
project's `groot/testdata`, and used here to check that `xrd.root` reads what
ROOT wrote. They are here rather than generated because the only honest test
of a reader is bytes somebody else's writer produced. The writer this library
grew later is gated on the same foreign bytes the other way round: what it
writes must round-trip the histograms and graphs read from these files, its
streamer descriptions must equal theirs member by member, and its LZ4
checksums must match ones ROOT itself computed (in `dirs-6.14.00.root`).
The layouts it writes trees with were harvested the same way — from
`small-flat-tree.root`, and from `leaves.root` for the three leaf classes no
6.08 donor here happens to contain — rather than typed out from the sources.

| File | What it is there for |
| --- | --- |
| `simple.root` | one tree, an int, a float and a string |
| `small-flat-tree.root` | every numeric width, fixed-size arrays and variable-length ones |
| `leaves.root` | the same again with the leaf classes at the edges, including the `Double32_t` and `Float16_t` packings |
| `padding.root` | branches holding several leaves each, where the entry record has holes in it |
| `tntuple.root` | a `TNtuple`, which is a tree with another record wrapped round it |
| `dirs-6.14.00.root` | nested directories, and a histogram filled without weights, whose errors are the root of its counts |
| `embedded-std-vector.root` | a `std::vector` member split out of a C++ class, read as rows |
| `pod-advanced.root` | a branch written in two baskets, so that a range crossing the boundary is read from both |
| `std-map-split1.root` | an object split into sub-branches, five kinds of `std::map` among them |
| `std-containers-split00.root` | forty columns of every STL container ROOT will write, unsplit |
| `small-evnt-tree-fullsplit.root` | a C++ class split all the way down: members, arrays, slices, strings and vectors |
| `small-evnt-tree-nosplit.root` | the same class and the same hundred events written whole, one object per entry |
| `std-map-split0.root` | the same maps as `std-map-split1.root`, written whole rather than split |
| `tlv-split99.root` | a `TLorentzVector`, which inherits `TObject` and streams itself |
| `tlv-split00.root` | the same ten of them in a `TBranchObject`, which names its class every entry |
| `tbase.root` | two classes deriving from one base, one of them redeclaring a member of it |
| `rvec.root` | forty-odd `ROOT::VecOps::RVec` branches, the vector an `RDataFrame` writes |
| `tdatime.root` | `TDatime`, in a key and in a branch, which streams itself and no record |
| `string-example.root` | a `std::string` standing on its own in a key |
| `std-bitset.root` | `std::bitset<8>`, on its own and in a vector |
| `g4-like.root` | a tree written by ROOT 4, whose baskets are inside the branch rather than out in the file |
| `stdvec-bool-fullsplit-6.10.08.root` | `std::vector<bool>`, which is a byte per element and not a bit |
| `graphs.root` | a graph, one with error bars and one whose bars differ each side |
| `tclonesarray-no-streamerbypass.root` | a `TClonesArray`, which names the one class it holds at the front |
| `tclonesarray-with-streamerbypass.root` | the same array written field by field, read back a field at a time |
| `gauss-h1.root` | one-dimensional histograms, evenly binned and with every edge written out |
| `gauss-h2.root` | two-dimensional histograms, for the order the bins are written in |
| `streamers.root` | objects held by pointer, arrays of them, and the class names written in front |
| `tconfidence-level.root` | a `TObjArray` member, a class of no members at all, and a `vector<pair<double,double>>` |
| `tformula.root` | a `vector<TF1*>`, and a member of a class no reader here can walk |
| `tgme.root` | a `TGraphMultiErrors` whose containers are written field by field |

go-hep is BSD-3-Clause; the licence is in `LICENSE.go-hep` next to these
files, and it is the whole of what is required to redistribute them.
