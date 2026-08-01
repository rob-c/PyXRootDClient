"""The public surface as a type checker sees it.

``open`` is overloaded so that a literal mode says whether ``read`` gives
``bytes`` or ``str``, and ``buffering=0`` hands back the raw layer. That is a
promise to callers which only a type checker can break, so it is checked here
rather than left to the next release to discover.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("mypy", reason="mypy is part of the dev extra")

SOURCE = '''
import xrd

url = "root://host//store/f.root"

reveal_type(xrd.open(url, "rb").read())
reveal_type(xrd.open(url, "r").read())
reveal_type(xrd.open(url, "rb", buffering=0))
reveal_type(xrd.open(url, mode_from_config).read())

fs = xrd.FileSystem("root://host")
reveal_type(fs.read_bytes("/f"))
reveal_type(fs.read_text("/f"))
reveal_type(xrd.XRootDPath(url).stat())
'''

EXPECTED = [
    'Revealed type is "bytes"',
    'Revealed type is "str"',
    'Revealed type is "xrd.io.raw.XRootDRawIO"',
    'Revealed type is "Any"',
    'Revealed type is "bytes"',
    'Revealed type is "str"',
    'Revealed type is "xrd.types.StatInfo"',
]


def test_the_open_overloads_say_what_comes_back(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "surface.py"
    # A non-literal mode is the escape hatch: it must still type-check, as
    # ``Any``, rather than being rejected.
    script.write_text("mode_from_config = str()\n" + SOURCE)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental", str(script)],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parent.parent,
    )
    revealed = [line for line in result.stdout.splitlines() if "Revealed type" in line]
    assert [line.split("note: ")[-1] for line in revealed] == EXPECTED, result.stdout
