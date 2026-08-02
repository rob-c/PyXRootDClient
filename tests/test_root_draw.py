"""Drawing histograms and graphs, with matplotlib and without it.

matplotlib is not a dependency of this library, so most of what is here
stands a small axes object in for it and checks the right calls arrive with
the right numbers - the part this library is responsible for. One test at the
end draws onto the real thing when it happens to be installed, headless.
"""

from __future__ import annotations

import sys
import types

import pytest

from xrd.root import Graph, Histogram, UnsupportedFeatureError, open_root
from xrd.root.draw import bar, missing_picture, shade

DATA = __file__.rsplit("/", 1)[0] + "/data"


class Axes:
    """Enough of matplotlib axes to see what was drawn on them."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return call


@pytest.fixture
def fresh(monkeypatch):
    """A stand-in matplotlib whose every ``subplots`` makes one of our axes."""
    made = []

    def subplots():
        made.append(Axes())
        return (None, made[-1])

    pyplot = types.SimpleNamespace(subplots=subplots)
    monkeypatch.setitem(sys.modules, "matplotlib", types.SimpleNamespace(pyplot=pyplot))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return made


# -- the terminal, which needs nothing installed ---------------------------


def test_a_bar_grows_an_eighth_of_a_character_at_a_time():
    assert bar(0.5, 4) == "██"
    assert bar(1.0, 4) == "████"
    assert bar(0.0, 4) == ""
    assert bar(-1.0, 4) == ""  # clamped, not mirrored
    assert bar(2.0, 4) == "████"
    assert bar(1 / 16, 2) == "▏"


def test_a_shade_darkens_in_five_steps():
    assert shade(0.0) == " "
    assert shade(-0.5) == " "
    assert shade(0.1) == "░"
    assert shade(0.6) == "▓"
    assert shade(0.9) == "█"
    assert shade(5.0) == "█"


def test_a_histogram_draws_itself_as_text_one_line_per_bin():
    hist = Histogram.new("h", [0, 1, 2], [4, 2])
    lines = hist.text(width=8).splitlines()
    assert lines[0].endswith(" 4")
    assert lines[1].endswith(" 2")
    assert "████████" in lines[0]
    assert "████" in lines[1]
    assert "[0, 1)" in lines[0] and "[1, 2)" in lines[1]


def test_an_empty_histogram_still_has_lines_just_no_bars():
    text = Histogram.new("h", [0, 1, 2], [0, 0]).text(width=8)
    assert "█" not in text


def test_a_two_dimensional_histogram_draws_as_a_grid_with_y_upward():
    with open_root(f"{DATA}/gauss-h2.root") as root:
        text = root["h2d"].text()
    lines = text.splitlines()
    assert len(lines) == 3  # three bins along y
    assert set("".join(lines)) <= set(" ░▒▓█")
    assert any(cell != " " for cell in "".join(lines))


def test_a_three_dimensional_histogram_refuses_both_pictures():
    with pytest.raises(UnsupportedFeatureError, match="3-dimensional histogram"):
        raise missing_picture("histogram", 3)


def test_a_graph_draws_itself_as_a_grid_of_stars():
    graph = Graph.new("g", [0, 1, 2], [0, 5, 10])
    lines = graph.text(width=11, height=5).splitlines()
    assert lines[0].startswith(f"{10:>10} |")
    assert lines[-2].startswith(f"{0:>10} |")
    assert lines[-1].endswith("2")
    assert "".join(lines).count("*") == 3
    assert lines[0][22] == "*"  # the highest point, top right
    assert lines[2][17] == "*"  # the middle one, midway
    assert lines[4][12] == "*"  # the lowest, bottom left


def test_a_flat_graph_still_draws_rather_than_dividing_by_nothing():
    text = Graph.new("g", [1, 1], [2, 2]).text(width=5, height=3)
    assert "*" in text


def test_a_graph_of_no_points_says_so_instead_of_an_empty_grid():
    assert Graph.new("g", [], []).text() == "(a graph of no points)"


# -- matplotlib, stood in for ----------------------------------------------


def test_a_histogram_plots_as_stairs_with_its_titles(fresh):
    hist = Histogram.new("h", [0, 1, 2], [4, 2], title="counts")
    hist.axes[0].title = "energy"
    ax = hist.plot(color="red")
    assert ax is fresh[0]
    names = [name for name, _, _ in ax.calls]
    assert names == ["stairs", "set_title", "set_xlabel"]
    _, (values, edges), options = ax.calls[0]
    assert list(values) == [4.0, 2.0]
    assert list(edges) == [0.0, 1.0, 2.0]
    assert options == {"color": "red"}
    assert ax.calls[1][1] == ("counts",)
    assert ax.calls[2][1] == ("energy",)


def test_a_two_dimensional_histogram_plots_as_a_mesh_with_the_grid_transposed(fresh):
    with open_root(f"{DATA}/gauss-h2.root") as root:
        hist = root["h2d"]
    hist.axes[1].title = "the other way"
    ax = hist.plot()
    name, (xs, ys, columns), _ = ax.calls[0]
    assert name == "pcolormesh"
    assert list(xs) == list(hist.edges(0))
    assert list(ys) == list(hist.edges(1))
    rows = [list(row) for row in hist.values()]
    assert columns == [list(column) for column in zip(*rows, strict=True)]
    assert ("set_ylabel", ("the other way",), {}) in ax.calls


def test_a_three_dimensional_histogram_refuses_to_plot_before_touching_matplotlib():
    hist = Histogram.new("h", [0, 1], [1])
    hist.axes = (None, None, None)
    with pytest.raises(UnsupportedFeatureError, match="no honest flat picture"):
        hist.plot()
    with pytest.raises(UnsupportedFeatureError, match="no honest flat picture"):
        hist.text()


def test_a_graph_plots_its_points_and_every_layer_of_bars(fresh):
    plain = Graph.new("g", [1, 2], [3, 4], title="scan")
    ax = plain.plot()
    name, (xs, ys), options = ax.calls[0]
    assert name == "errorbar"
    assert (list(xs), list(ys)) == ([1.0, 2.0], [3.0, 4.0])
    assert options["yerr"] is None and options["xerr"] is None
    assert options["fmt"] == "o" and options["markersize"] == 4
    assert ax.calls[1] == ("set_title", ("scan",), {})

    bars = Graph.new("g", [1, 2], [3, 4], xerr=[0.1, 0.1], yerr=[0.2, 0.3])
    ax = bars.plot(fmt="s")
    _, _, options = ax.calls[0]
    assert options["fmt"] == "s"  # the caller's style wins
    assert [list(side) for side in options["yerr"]] == [[0.2, 0.3], [0.2, 0.3]]
    assert options["xerr"] is not None


def test_axes_brought_along_are_drawn_on_rather_than_replaced(fresh):
    mine = Axes()
    assert Graph.new("g", [1], [2]).plot(ax=mine) is mine
    plain = Histogram.new("h", [0, 1], [1]).plot(ax=Axes())
    assert [name for name, _, _ in plain.calls] == ["stairs"]  # no titles to set
    assert fresh == []  # nothing was made behind the caller's back


def test_without_matplotlib_the_refusal_names_both_ways_out(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(UnsupportedFeatureError, match=r"pip install matplotlib.*text"):
        Histogram.new("h", [0, 1], [1]).plot()


# -- matplotlib, the real one, when it is there ----------------------------


def test_the_real_matplotlib_accepts_everything_we_send_it():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot

    with open_root(f"{DATA}/gauss-h2.root") as root:
        flat = root["h2d"]
    layered = Graph.new("g", [1, 2], [3, 4], yerr=([0.1, 0.2], [0.3, 0.4]))
    for thing in (Histogram.new("h", [0, 1, 2], [4, 2]), flat, layered):
        ax = thing.plot()
        assert ax.figure is not None
        pyplot.close(ax.figure)
