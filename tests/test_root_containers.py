"""The C++ side of a ROOT file: split members, containers and packed floats.

Everything asserted against a file here is a value go-hep reads out of the
same bytes, which is the only way to be sure a reader is right rather than
merely self-consistent. The crafted records are for the shapes the corpus has
no example of - a streamer element from 2001, a container written field by
field - where a refusal has to be provoked to be checked.
"""

from __future__ import annotations

import array
import io
import pathlib
import struct

import pytest

from xrd.root import FormatError, UnsupportedFeatureError, open_root
from xrd.root.buffer import BYTE_COUNT_MASK, Buffer
from xrd.root.cxx import Mapping, Prim, Seq, Str, parse, py_name
from xrd.root.file import Source
from xrd.root.interp import KINDS, Flat, Refused, Rows, Values, _packed, _range, build
from xrd.root.objects import (
    BranchRecord,
    LeafRecord,
    read_branch_element,
    read_derived_branch,
)
from xrd.root.streamers import Member, read_element, read_info, read_streamers

DATA = pathlib.Path(__file__).parent / "data"


def opened(name: str):
    return open_root(str(DATA / f"{name}.root"))


@pytest.fixture
def containers():
    with opened("std-containers-split00") as handle:
        yield handle["tree"]


@pytest.fixture
def event():
    with opened("small-evnt-tree-fullsplit") as handle:
        yield handle["tree"]


# -- crafted bytes --------------------------------------------------------


def tstring(text: str) -> bytes:
    raw = text.encode()
    return bytes([len(raw)]) + raw


def record(version: int, body: bytes = b"") -> bytes:
    return struct.pack(">IH", BYTE_COUNT_MASK | (2 + len(body)), version) + body


def named_bytes(name: str, title: str = "") -> bytes:
    return record(1, struct.pack(">HII", 1, 0, 0) + tstring(name) + tstring(title))


def element_bytes(version: int, name: str, typename: str, stype: int = 3) -> bytes:
    """A ``TStreamerElement`` of the given version, under a subclass record."""
    body = named_bytes(name, "") + struct.pack(">iiii", stype, 4, 1, 1)
    body += struct.pack(">ii", 1, 0) if version == 1 else struct.pack(">5i", 0, 0, 0, 0, 0)
    return record(4, record(version, body + tstring(typename)))


def branch_bytes() -> bytes:
    """A ``TBranch`` with no baskets in it, which is the part every branch has."""
    body = named_bytes("b", "b/I") + record(1)
    body += struct.pack(">iiii", 1, 4096, 0, 0)
    body += struct.pack(">q", 0)
    body += struct.pack(">iii", 0, 0, 0)
    body += struct.pack(">qqq", 0, 0, 0)
    empty = record(3, struct.pack(">HII", 1, 0, 0) + tstring("") + struct.pack(">ii", 0, 0))
    return record(10, body + empty * 3 + b"\x01\x01\x01" + tstring(""))


def branch_element_bytes(version: int, classname: str) -> bytes:
    """A ``TBranchElement`` of a version old enough to say things differently."""
    body = branch_bytes() + tstring(classname)
    if version > 1:
        body += tstring("") + tstring("") + struct.pack(">I", 0)
    body += struct.pack(">I" if version < 10 else ">H", 0)
    return record(version, body)


class Nothing:
    """A source that describes no classes at all, which some files do not."""

    def streamers(self) -> dict[str, dict[str, Member]]:
        return {}


def column_of(typename: str, *, member: str = "", ltype: int = -1, source=None):
    """What :func:`build` makes of a branch declared to hold ``typename``."""
    record_ = BranchRecord()
    record_.name = record_.title = "b"
    record_.classname = typename
    leaf = LeafRecord("TLeafElement")
    leaf.name = member or "b"
    leaf.ltype = ltype
    record_.leaves = [leaf]
    return build(record_, leaf, source or Nothing())


# -- C++ type names -------------------------------------------------------


def test_every_spelling_of_a_fundamental_type_is_one_type():
    assert parse("unsigned long long int").typename == "uint64"
    assert parse("Float_t").typename == "float32"
    assert parse("std::vector<Double_t>*").item.typename == "float64"
    assert parse("Version_t").itemsize == 2


def test_both_kinds_of_string_are_strings_written_differently():
    assert parse("string").record is True
    assert parse("basic_string<char>").record is True
    assert parse("TString").record is False


def test_a_container_is_the_type_it_holds():
    assert py_name(parse("vector<vector<int> >".replace(" ", ""))) == "list[list[int32]]"
    assert py_name(parse("set<TString>")) == "list[str]"
    assert py_name(parse("unordered_map<string,deque<float> >")) == "dict[str, list[float32]]"
    assert py_name(parse("short")) == "int16"


def test_a_name_this_reader_has_no_idea_about_is_no_type_at_all():
    assert parse("TLorentzVector") is None
    assert parse("vector<TLorentzVector>") is None
    assert parse("multimap<int,int>") is None  # a dict would drop the duplicate keys
    assert parse("map<int>") is None
    assert parse("map<int,TLorentzVector>") is None
    assert parse("map<TLorentzVector,int>") is None
    assert parse("pair<int,int>") is None


def test_the_pieces_of_a_type_say_what_they_are():
    assert repr(parse("float")) == "<Prim float32>"
    assert repr(parse("string")) == "<Str>"
    assert repr(parse("vector<int>")) == "<Seq of <Prim int32>>"
    assert repr(parse("map<int,int>")) == "<Mapping <Prim int32> to <Prim int32>>"
    assert isinstance(parse("list<int>"), Seq)
    assert isinstance(parse("map<int,int>"), Mapping)
    assert isinstance(parse("TString"), Str)
    assert isinstance(parse("bool"), Prim)


# -- the classes a file describes -----------------------------------------


def test_a_file_says_what_its_own_classes_are_made_of():
    with opened("small-evnt-tree-fullsplit") as handle:
        handle["tree"]  # a split branch is what makes the streamers be read
        classes = handle._source.streamers()
        assert classes["Event"]["StlVecF64"].typename == "vector<double>"
        assert classes["Event"]["StdStr"].typename == "string"
        assert classes["Event"]["SliceF64"].typename == "double*"
        assert repr(classes["Event"]["StdStr"]) == "<Member string StdStr>"
        assert handle._source.streamers() is classes  # read once, then kept


def test_a_file_with_no_class_descriptions_describes_no_classes():
    source = Source(io.BytesIO(b""), "crafted", owned=True)
    assert read_streamers(source) == {}
    assert source.streamers() == {}


def test_a_streamer_element_from_2001_counted_its_own_dimensions():
    member = read_element(Buffer(element_bytes(1, "fN", "int")))
    assert (member.name, member.typename, member.stype, member.length) == ("fN", "int", 3, 1)


def test_a_class_written_with_no_member_list_has_no_members():
    body = named_bytes("Empty", "") + struct.pack(">Ii", 0, 1) + struct.pack(">I", 0)
    assert read_info(Buffer(record(9, body))) == ("Empty", {})


def test_a_branch_element_from_before_the_version_shrank_still_names_its_class():
    assert read_branch_element(Buffer(branch_element_bytes(9, "Event"))).classname == "Event"
    assert read_branch_element(Buffer(branch_element_bytes(1, "Event"))).classname == "Event"


def test_a_branch_of_a_kind_this_reader_cannot_follow_still_appears_by_name():
    """A ``TBranchObject`` and its two cousins: named and refused, not missing."""
    branch = read_derived_branch(Buffer(record(1, branch_bytes() + tstring("Event"))))
    assert branch.name == "b"
    assert branch.classname == ""  # nothing said what it holds, so nothing is guessed


# -- containers, against what go-hep reads --------------------------------


def test_a_vector_of_numbers_is_rows_of_numbers(containers):
    assert containers["vec_i32"].is_jagged
    assert containers["vec_i32"].array(0, 2).tolist() == [[-1], [-1, -2]]
    assert containers["vec_u32"].array(1, 2).tolist() == [[1, 2]]
    assert containers["lst_i32"].array(1, 2).tolist() == [[-1, -2]]
    assert containers["deq_i32"].array(1, 2).tolist() == [[-1, -2]]
    assert containers["set_i32"].array(1, 2).tolist() == [[-2, -1]]  # a set is sorted


def test_a_container_of_containers_is_a_list_of_arrays(containers):
    assert containers["vec_vec_i32"].typename == "list[list[int32]]"
    assert [list(map(list, e)) for e in containers["vec_vec_i32"].array(0, 2)] == [
        [[-1]],
        [[-1], [-1, -2]],
    ]
    assert [list(map(list, e)) for e in containers["vec_set_i32"].array(1, 2)] == [[[-1], [-2, -1]]]


def test_a_container_of_strings_is_a_list_of_strings(containers):
    assert containers["vec_str"].array(1, 2) == [["one", "two"]]
    assert containers["vec_tstr"].array(1, 2) == [["one", "two"]]
    assert containers["vec_vec_str"].array(1, 2) == [[["one"], ["one", "two"]]]
    assert containers["uset_str"].array(0, 1) == [["one"]]


def test_a_top_level_string_branch_is_written_bare(containers):
    assert containers["str"].array(0, 2) == ["one", "two"]
    assert containers["tstr"].array(0, 2) == ["one", "two"]
    assert containers["str"].typename == "str"


def test_a_map_comes_back_as_a_dict_whichever_way_it_is_keyed(containers):
    assert containers["map_i32_i16"].array(1, 2) == [{-2: -2, -1: -1}]
    assert containers["map_u32_u16"].array(1, 2) == [{1: 1, 2: 2}]
    assert containers["map_str_i16"].array(1, 2) == [{"one": -1, "two": -2}]
    assert containers["map_str_str"].array(1, 2) == [{"one": "ONE", "two": "TWO"}]
    assert containers["umap_str_str"].array(1, 2) == [{"one": "ONE", "two": "TWO"}]


def test_a_tstring_in_a_map_is_bytes_where_a_std_string_is_a_record(containers):
    both = {"one": "ONE", "two": "TWO"}
    assert containers["map_str_tstr"].array(1, 2) == [both]
    assert containers["map_tstr_str"].array(1, 2) == [both]
    assert containers["map_tstr_tstr"].array(1, 2) == [both]


def test_a_map_of_containers_keeps_the_containers(containers):
    assert containers["map_i32_vec_i16"].array(1, 2) == [
        {-2: array.array("h", [-1, -2]), -1: array.array("h", [-1])}
    ]
    assert containers["map_str_vec_str"].array(1, 2) == [
        {"one": ["one"], "two": ["one", "two"]}
    ]
    assert containers["map_i32_set_i16"].array(1, 2) == [
        {-2: array.array("h", [-2, -1]), -1: array.array("h", [-1])}
    ]
    nested = containers["map_i32_vec_vec_i16"].array(1, 2)[0]
    assert {key: [list(row) for row in value] for key, value in nested.items()} == {
        -2: [[-1], [-1, -2]],
        -1: [[-1]],
    }


def test_every_column_of_the_container_file_but_none_is_readable(containers):
    assert containers.unreadable == {}
    assert len(containers.readable()) == len(containers.keys()) == 40


# -- a class split all the way down ---------------------------------------


def test_a_split_member_is_a_column_of_its_own(event):
    assert event["I16"].array(1, 2).tolist() == [1]
    assert event["U64"].array(1, 2).tolist() == [1]
    assert event["ArrayF64[10]"].array(1, 2).tolist() == [1.0] * 10
    assert event["SliceF64"].array(1, 2).tolist() == [[1.0]]
    assert event["N"].array(1, 2).tolist() == [1]


def test_a_split_object_inside_an_object_is_its_members(event):
    assert [event[f"P3.P{axis}"].array(1, 2)[0] for axis in "xyz"] == [0, 1.0, 0]
    assert [event[f"P3.P{axis}"].array(0, 1)[0] for axis in "xyz"] == [-1, 0.0, -1]


def test_the_three_kinds_of_string_a_split_class_can_hold(event):
    assert event["Beg"].array(1, 2) == ["beg-001"]  # a TString member
    assert event["Str"].array(1, 2) == ["evt-001"]
    assert event["StdStr"].array(1, 2) == ["std-001"]  # a std::string member
    assert event["StlVecStr"].array(1, 2) == [["vec-001"]]
    assert event["End"].array(1, 2) == ["end-001"]


def test_a_vector_member_of_a_split_class_is_rows(event):
    assert event["StlVecF64"].array(1, 2).tolist() == [[1.0]]
    assert event["StlVecI16"].array(0, 2).lengths() == [0, 1]
    assert event["StlVecF32"].typename == "float32"


def test_a_vector_of_bool_is_a_byte_an_element_not_a_bit():
    with opened("stdvec-bool-fullsplit-6.10.08") as handle:
        tree = handle["tree"]
        assert tree["Bool"].array(0, 3).tolist() == [1, 0, 1]
        assert tree["ArrayBool[10]"].array(2, 3).tolist() == [1] * 10
        assert tree["StlVecBool"].array(0, 3).tolist() == [[], [0], [1, 1]]
        assert tree["SliceBool"].array(0, 3).tolist() == [[], [0], [1, 1]]


# -- refusing, by name ----------------------------------------------------


def test_a_class_with_no_layout_this_reader_knows_is_refused_by_its_name():
    column = column_of("TLorentzVector")
    assert isinstance(column, Refused)
    assert "TLorentzVector, which is a C++ type this reader does not decode" in column.reason
    assert repr(column) == "<Refused unreadable>"
    assert "an unnamed type" in column_of("").reason


def test_a_map_a_dict_cannot_be_is_refused_in_those_words():
    assert "keyed by a container" in column_of("map<vector<int>,int>").reason
    assert "a map inside another container" in column_of("vector<map<int,int> >").reason
    assert "a map inside another container" in column_of("map<int,vector<map<int,int> > >").reason


def test_a_member_the_file_never_described_cannot_be_guessed_at():
    column = column_of("Event", member="fMissing", ltype=300)
    assert "streamer information does not describe" in column.reason


def test_a_streamer_type_this_reader_will_not_decode_is_named_in_words():
    for stype, words in KINDS.items():
        reason = column_of("Event", ltype=stype).reason
        assert reason == f"{words}, which this reader does not decode"


def test_a_leaf_that_is_not_a_leaf_this_reader_knows_keeps_its_reason():
    leaf = LeafRecord("TLeafQuantum")
    branch = BranchRecord()
    branch.leaves = [leaf]
    assert "not a kind of leaf" in build(branch, leaf, Nothing()).reason


def test_a_container_written_field_by_field_is_refused_rather_than_guessed():
    column = column_of("vector<vector<int> >")
    assert isinstance(column, Values)
    with pytest.raises(UnsupportedFeatureError, match="field by field"):
        column.value(Buffer(record(0x4000 | 6)), 0)


def test_a_map_written_pair_by_pair_is_refused_rather_than_guessed():
    column = column_of("map<int,int>")
    with pytest.raises(UnsupportedFeatureError, match="pair by pair"):
        column.value(Buffer(record(6)), 0)


class Fake:
    """Just enough of a basket for a row to be measured inside it."""

    def __init__(self, data: bytes, end: int) -> None:
        self.data = data
        self._end = end

    def start_of(self, entry: int, offset: int) -> int:
        return 0

    def end_of(self, entry: int) -> int:
        return self._end


def test_a_row_claiming_more_values_than_it_holds_is_a_format_error():
    column = column_of("vector<int>")
    assert isinstance(column, Rows)
    basket = Fake(struct.pack(">IHI", BYTE_COUNT_MASK | 10, 6, 99), 14)
    with pytest.raises(FormatError, match="says it holds 99 values"):
        column.span(basket, 0, 0)


# -- floats packed into fewer bytes ---------------------------------------


def test_a_title_with_no_range_in_it_asks_for_the_default_packing():
    assert _range("") == (0.0, 0.0, 0.0)
    assert _range("x[10]") == (0.0, 0.0, 0.0)  # an array, which is not a range
    assert _range("f[0,0,16]") == (0.0, 0.0, 0.0)  # no slash: not a range either
    assert _range("x/d[10]") == (0.0, 0.0, 0.0)


def test_a_title_that_gives_a_range_gives_a_scale_factor():
    xmin, xmax, factor = _range("x/d[-1,1]")
    assert (xmin, xmax) == (-1.0, 1.0)
    assert factor == pytest.approx(0xFFFFFFFF / 2)
    assert _range("x/d[0,2,8]")[2] == pytest.approx(128.0)
    assert _range("[-1,1]")[:2] == (-1.0, 1.0)  # a title that is nothing but a range


def test_a_title_that_gives_only_a_bit_count_keeps_it_where_root_keeps_it():
    assert _range("x/d[0,0,10]")[0] == pytest.approx(10.1)
    assert _range("x/d[0,0,20]") == (0.0, 0.0, 0.0)  # too many bits to be kept


def test_a_range_can_be_written_in_units_of_pi():
    assert _range("x/d[-pi,2pi]")[:2] == pytest.approx((-3.141592653589793, 6.283185307179586))
    assert _range("x/d[-pi/4,pi/2]")[:2] == pytest.approx((-0.7853981633974483, 1.5707963267948966))
    assert _range("x/d[0,twopi]")[1] == pytest.approx(6.283185307179586)
    assert _range("x/d[0,2*pi]")[1] == pytest.approx(6.283185307179586)


def test_a_range_this_reader_cannot_read_is_refused_rather_than_ignored():
    assert _range("x/d[1,2,3,4]") is None
    assert _range("x/d[low,high]") is None
    assert _range("x/d[0,1,99]") is None
    leaf = LeafRecord("TLeafD32")
    leaf.title = "x/d[low,high]"
    assert "not a spelling this reader can turn into numbers" in _packed(leaf).reason


def test_a_scaled_double_comes_back_across_the_range_it_was_written_in():
    leaf = LeafRecord("TLeafD32")
    leaf.title, leaf.length = "x/d[0,4,2]", 3
    column = _packed(leaf)
    assert isinstance(column, Flat)
    assert column.itemsize == 4
    assert list(column.decode(struct.pack(">3I", 0, 2, 4))) == [0.0, 2.0, 4.0]


def test_a_packed_slice_is_rows_of_doubles():
    leaf = LeafRecord("TLeafF16")
    leaf.title = "x/f[0,0,10]"
    leaf.count = LeafRecord("TLeafI")
    column = _packed(leaf)
    assert isinstance(column, Rows)
    assert column.itemsize == 3
