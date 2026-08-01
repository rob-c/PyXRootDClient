"""Undoing ROOT's block compression.

A compressed ROOT object is a run of blocks, each with a nine-byte header
saying which algorithm made it and how long it is either way. Blocks are at
most 16 MB, so a large basket is several of them back to back.

zlib and lzma come from the standard library. LZ4 is decoded here, in Python,
because there is no LZ4 in the standard library and a physics file should not
need a wheel to be read - it is a small format and this is a small decoder.
zstd is used from the standard library on Python 3.14 and later, or from the
``zstandard`` package if it happens to be installed, and is otherwise refused
by name rather than guessed at.
"""

from __future__ import annotations

import lzma
import zlib

from .errors import FormatError, UnsupportedFeatureError

__all__ = ["decompress", "algorithm"]

HEADER = 9
#: What each tag in the block header means, for messages and for refusing well.
NAMES = {
    b"ZL": "zlib",
    b"XZ": "lzma",
    b"L4": "lz4",
    b"ZS": "zstd",
    b"CS": "the pre-2005 ROOT algorithm",
}


def algorithm(data: bytes) -> str:
    """The name of whatever compressed ``data``, for a message or a report."""
    return NAMES.get(bytes(data[:2]), "an unknown algorithm")


try:  # pragma: no cover - depends on the interpreter and an optional extra
    from compression.zstd import decompress as _unzstd  # type: ignore[import-not-found]

    def _zstd(block: bytes, size: int) -> bytes:
        return bytes(_unzstd(block))

except ImportError:  # pragma: no cover - before 3.14, where zstandard fills in
    try:
        from zstandard import ZstdDecompressor

        def _zstd(block: bytes, size: int) -> bytes:
            return bytes(ZstdDecompressor().decompress(block, max_output_size=size))

    except ImportError:

        def _zstd(block: bytes, size: int) -> bytes:
            raise UnsupportedFeatureError(
                "this file is zstd-compressed, which needs Python 3.14 or the zstandard "
                "package: pip install zstandard, or ask for the file written another way"
            )


def _lz4(src: bytes, size: int) -> bytes:
    """One LZ4 block, in Python.

    The format is a loop of tokens: a length of bytes to copy out verbatim,
    then a distance back into what has already been written and a length to
    repeat from there. The repeat is allowed to overlap what it is producing,
    which is how LZ4 spells a run, so that case copies byte by byte.
    """
    out = bytearray()
    pos, end = 0, len(src)
    try:
        while pos < end:
            token = src[pos]
            pos += 1
            literals = token >> 4
            if literals == 15:
                while (more := src[pos]) == 255:
                    literals += 255
                    pos += 1
                literals += more
                pos += 1
            out += src[pos : pos + literals]
            pos += literals
            if pos >= end:
                break  # the last sequence is literals only
            offset = src[pos] | (src[pos + 1] << 8)
            pos += 2
            match = (token & 0xF) + 4
            if token & 0xF == 15:
                while (more := src[pos]) == 255:
                    match += 255
                    pos += 1
                match += more
                pos += 1
            start = len(out) - offset
            if start < 0:
                raise FormatError(f"an LZ4 match points {-start} bytes before the block")
            if offset >= match:
                out += out[start : start + match]
            else:
                for index in range(start, start + match):
                    out.append(out[index])
    except IndexError:
        raise FormatError("an LZ4 block ends in the middle of a sequence") from None
    if len(out) != size:
        raise FormatError(f"an LZ4 block gave {len(out)} bytes where {size} were promised")
    return bytes(out)


def decompress(data: bytes, size: int) -> bytes:
    """The ``size`` bytes that ``data``'s blocks were made from."""
    out = bytearray()
    pos = 0
    while len(out) < size:
        if pos + HEADER > len(data):
            raise FormatError(
                f"the compressed object ran out after {len(out)} of {size} bytes"
            )
        tag = data[pos : pos + 2]
        packed = int.from_bytes(data[pos + 3 : pos + 6], "little")
        unpacked = int.from_bytes(data[pos + 6 : pos + 9], "little")
        block = data[pos + HEADER : pos + HEADER + packed]
        if len(block) != packed:
            raise FormatError(f"a block says it is {packed} bytes and only {len(block)} are here")
        pos += HEADER + packed

        if tag == b"ZL":
            out += zlib.decompress(block)
        elif tag == b"XZ":
            out += lzma.decompress(block)
        elif tag == b"L4":
            out += _lz4(block[8:], unpacked)  # after the checksum, which we do not need
        elif tag == b"ZS":
            out += _zstd(block, unpacked)
        else:
            raise UnsupportedFeatureError(
                f"this file is compressed with {algorithm(tag)}, which this reader does not "
                f"undo; rewrite it with hadd, or ask ROOT for zlib, lzma or lz4"
            )
    if len(out) != size:
        raise FormatError(f"decompressing gave {len(out)} bytes where {size} were promised")
    return bytes(out)
