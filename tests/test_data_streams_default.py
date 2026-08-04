"""The on-by-default data sub-streams.

A file opened over the binary protocol now binds a data sub-stream at open and
moves its bulk reads and writes over it, so a plain ``File.open`` is already
multi-stream. Three things have to hold:

* the default is one extra stream, and ``XRD_SUBSTREAMSPERCHANNEL`` sets it (the
  official client counts the control link, so its 1 is our 0);
* "arrival" routing puts the *whole* request frame on the bound data socket,
  not just a write's payload, so a server that keys a sub-stream on the
  connection a request came in on (BriX) serves it there;
* it is best-effort - a server that will not serve the bound op leaves the
  transfer byte-exact by falling back to the control link.

The success path against such a server is proven end-to-end against a live BriX
gateway; the fake server here is the standard, push-only kind, which is exactly
what exercises the fallback.
"""
from __future__ import annotations

import struct

from conftest import handshake_reply, login_body, ok, protocol_body

import xrd
from xrd.config import Config
from xrd.flags import OpenFlags
from xrd.proto import constants as c
from xrd.proto import machine as m
from xrd.proto import requests as r


# --------------------------------------------------------------------------
# The default
# --------------------------------------------------------------------------


def test_the_default_is_one_extra_stream():
    assert Config().data_streams == 1


def test_the_substreams_env_sets_the_extra_count(monkeypatch):
    # The official client's total includes the control link, so 1 is "no extra".
    monkeypatch.setenv("XRD_SUBSTREAMSPERCHANNEL", "1")
    assert Config().data_streams == 0
    monkeypatch.setenv("XRD_SUBSTREAMSPERCHANNEL", "4")
    assert Config().data_streams == 3
    monkeypatch.setenv("XRD_SUBSTREAMSPERCHANNEL", "0")
    assert Config().data_streams == 0


# --------------------------------------------------------------------------
# Arrival routing in the machine
# --------------------------------------------------------------------------


def _ready() -> m.SessionMachine:
    machine_ = m.SessionMachine(
        host="srv.example.org",
        config=Config(username="tester", auth_order=("host",)),
    )
    machine_.start()
    machine_.data_to_send()
    machine_.receive_data(handshake_reply())
    machine_.receive_data(ok(1, protocol_body()))
    machine_.receive_data(ok(2, login_body()))
    machine_.data_to_send()
    list(machine_.events())
    return machine_


def test_an_arrival_read_puts_the_whole_frame_on_the_path():
    machine_ = _ready()
    machine_.submit(r.Read(b"fh01", 0, 5, 1), arrive_on_path=True)
    # Nothing on the control link; the request frame is on the data socket.
    assert machine_.data_to_send() == b""
    frame = machine_.path_data_to_send(1)
    assert frame, "the read frame was not routed onto the path"
    assert struct.unpack(">H", frame[2:4])[0] == c.kXR_read


def test_an_arrival_write_carries_header_and_data_on_the_path():
    machine_ = _ready()
    machine_.submit(r.Write(b"fh01", 0, b"bulk", 1), arrive_on_path=True)
    assert machine_.data_to_send() == b""
    frame = machine_.path_data_to_send(1)
    assert struct.unpack(">H", frame[2:4])[0] == c.kXR_write
    assert frame.endswith(b"bulk"), "the write payload must ride the same socket"


def test_without_arrival_the_split_is_unchanged():
    # The standard model still stands for the manual bind_data_path API.
    machine_ = _ready()
    machine_.submit(r.Write(b"fh01", 0, b"bulk", 1))
    assert b"bulk" not in machine_.data_to_send()
    assert machine_.path_data_to_send(1) == b"bulk"


# --------------------------------------------------------------------------
# End to end against a standard (push-only) server: the fallback
# --------------------------------------------------------------------------


def _fast_multistream() -> Config:
    # data_streams on, with a tiny data-stream timeout so the bound attempt
    # against a server that will not serve it fails fast into the control-link
    # fallback instead of waiting out the default probe window.
    return Config(
        username="tester",
        auth_order=("host",),
        require_tls=False,
        data_streams=1,
        data_stream_timeout=0.3,
    )


def test_a_default_open_binds_a_stream(server):
    with xrd.File(f"{server.url}//data/a.root", _fast_multistream()) as fh:
        assert fh._data_paths, "open did not bind an automatic data stream"


def test_a_multistream_read_is_byte_exact_via_fallback(server):
    # The fake server is push-only, so the bound read is declined; the file must
    # still hand back exactly the bytes, on the control link.
    with xrd.File(f"{server.url}//data/a.root", _fast_multistream()) as fh:
        assert fh.read() == b"hello world"
        assert fh._multistream is False, "a declined bound op must latch off"


def test_a_multistream_write_is_byte_exact_via_fallback(server):
    path = "/data/ms_w.root"
    fh = xrd.File(f"{server.url}/{path}", _fast_multistream())
    fh.open(OpenFlags.NEW | OpenFlags.UPDATE)
    try:
        assert fh.write(b"payload", 0) == 7
    finally:
        fh.close()
    assert bytes(server.files[path]) == b"payload"
