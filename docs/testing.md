# Testing against it

`xrd.testing` ships three servers. They are part of the installed package, not
test-suite scaffolding, because the code that is hardest to test is yours -
the retry loop, the failure handler, the thing that has to cope with a data
server disappearing.

```python
from xrd.testing import FakeServer, FakeDAVServer, FaultProxy
```

Nothing here needs a daemon, a port you own, or root. Everything binds an
ephemeral loopback port and cleans up.

## `FakeServer` - the XRootD protocol

```python
import xrd
from xrd.testing import FakeServer

with FakeServer(files={"/store/f.root": b"payload"}, dirs=["/store/empty"]) as srv:
    fs = xrd.FileSystem(str(srv.url))
    assert fs.read_bytes("/store/f.root") == b"payload"
    fs.write_bytes("/store/g.root", b"new")

    assert bytes(srv.files["/store/g.root"]) == b"new"     # the test can look
```

A real server, in the sense that matters: it speaks the binary protocol,
frames responses, handles `kXR_open`/`read`/`write`/`readv`/`pgread`, dirlist,
stat, mkdir, rm, mv, truncate, chmod, xattrs, query, ping and the login and
authentication handshake.

### Making it behave badly

| Attribute | Effect |
| --- | --- |
| `srv.redirects[opcode] = (host, port, token)` | answer once with `kXR_redirect` |
| `srv.waits[opcode] = 3` | send three `kXR_wait` replies first |
| `srv.chunk_reads = 4096` | split reads into `kXR_oksofar` pieces |
| `srv.auth_rounds = 2` | demand two rounds of `kXR_authmore` |
| `srv.sec = "&P=gsi,v:10400&P=unix"` | what the login advertises |
| `srv.disconnect()` | drop live connections, keep listening |

And for assertions:

| Attribute | Holds |
| --- | --- |
| `srv.seen` | every opcode, in order |
| `srv.opened` | the raw path of each `kXR_open`, CGI included |
| `srv.arguments` | `(opcode, argument)` for every request that named a path |
| `srv.files`, `srv.dirs`, `srv.xattrs` | the namespace, mutated live |
| `srv.config_values` | what `kXR_query` config answers |

```python
from xrd.proto import constants as c

with FakeServer(files={"/f": b"x" * (1 << 20)}) as srv:
    srv.chunk_reads = 8192               # force reassembly
    srv.waits[c.kXR_open] = 1            # and one wait first
    assert xrd.Path(f"{srv.url}/f").read_bytes() == b"x" * (1 << 20)
    assert c.kXR_open in srv.seen
```

### Answering with something no server would send

`srv.handlers[opcode]` replaces one opcode's reply. A handler takes
`(connection, streamid, params, body)` and yields raw frames, which
`xrd.testing.frame` and `xrd.testing.error` build for you:

```python
from xrd.testing import FakeServer, error, frame
from xrd.proto import constants as c

def truncated(conn, sid, params, body):
    yield frame(sid, c.kXR_ok, b"short")      # fewer bytes than were asked for

with FakeServer(files={"/f": b"payload"}) as srv:
    srv.handlers[c.kXR_read] = truncated
    ...                                        # your retry loop, under a liar
```

A handler that yields nothing and closes `conn.sock` is a server that crashed
between doing the work and answering for it - the one case where a client must
*not* repeat a `write`, an `rm` or an `mv`.

## `FakeDAVServer` - HTTP and WebDAV

```python
with FakeDAVServer(files={"/store/f.root": b"payload"}) as dav:
    fs = xrd.FileSystem(str(dav.url))          # already http://host:port/
    fs.stat("/store/f.root")
    assert ("PROPFIND", "/store/f.root") in dav.seen
```

| Attribute | Effect |
| --- | --- |
| `dav.require_token = "abc"` | 401 unless that bearer token arrives |
| `dav.redirects[path] = url` | 307 before the real answer |
| `dav.no_dav = True` | refuse `PROPFIND`, like a plain HTTP server |
| `dav.ignore_ranges = True` | answer a ranged `GET` with the whole body |
| `dav.digests = False` | no `Want-Digest` support |
| `dav.macaroon = "..."` | what a macaroon request mints |
| `dav.no_tpc = True` | refuse `COPY`, like an endpoint without third-party copy |
| `dav.tpc_failure = "..."` | answer `202` and then fail, which is the awkward one |
| `dav.tpc_markers = 4` | how many performance markers a `COPY` sends |
| `dav.copies` | the headers of every `COPY` served |
| `dav.handlers["PROPFIND"] = fn` | answer one method yourself |
| `dav.bodies`, `dav.targets` | every request body, and every raw request target |

`handlers` is the HTTP twin of `FakeServer.handlers`: the function is handed
`(method, path, headers)` and returns `(status, body, headers)` to answer the
request, or `None` to record it and let the real implementation reply.

```python
def hostile(method, path, headers):
    return 207, b"<not xml", {"Content-Type": 'text/xml; charset="utf-8"'}

with FakeDAVServer(files={"/d/f.root": b"payload"}) as dav:
    dav.handlers["PROPFIND"] = hostile
    with pytest.raises(xrd.ProtocolError):
        xrd.FileSystem(str(dav.url)).stat("/d/f.root")
```

`ignore_ranges` is the one worth knowing about: caches in front of real
endpoints do exactly this, and a client that trusts a `200` where it asked for
a `206` silently returns the wrong bytes. This client detects it; your code
can be tested against it.

`COPY` is served for real - the server fetches from (or pushes to) whichever
endpoint the header names, which may well be a second `FakeDAVServer` - so a
third-party copy can be tested end to end in one process:

```python
with FakeDAVServer(files={"/d/f.root": b"payload"}) as src, FakeDAVServer(dirs=["/d"]) as dst:
    xrd.third_party(src.url / "d/f.root", dst.url / "d/f.root")
    assert dst.contents("/d/f.root") == b"payload"
```

## `FaultProxy` - breaking the network

A loopback TCP proxy that sits in front of anything with an address - a
`FakeServer`, a real daemon, a `(host, port)` pair - and can be told to
misbehave mid-connection.

```python
from xrd.testing import FakeServer, FaultProxy

with FakeServer(files={"/big": b"payload" * 1000}) as srv, FaultProxy(srv) as proxy:
    with xrd.open(proxy.url.with_path("/big"), "rb") as fh:
        head = fh.read(32)
        proxy.cut()                      # the data server goes away mid-read
        assert fh.read(32)               # re-opened underneath, no error
```

That is `recover_handles` doing its job: an open read-only handle whose
connection dies is re-opened at the same offset and the read is retried.
`File.recoveries` counts how often it happened, and
`Config(recover_handles=False)` turns it into a `TransientError` instead.

| Method | What it does |
| --- | --- |
| `drop_after(offset)` | close the connection after N bytes from the server |
| `stall_after(offset)` | stop forwarding, hold the socket open - a black hole |
| `delay(seconds, after=N)` | slow every chunk down |
| `corrupt(offset, mask)` | flip bits at a byte offset |
| `chop(size)` | forward in small pieces - reassembly, not failure |
| `rewrite(fn)` | pass each server chunk through a function |
| `refuse()` / `accept()` | stop and resume accepting connections |
| `cut()` | close every live connection now |
| `heal()` | disarm everything |

Every arming method returns the proxy, so they chain. Counters
(`proxy.connections`, `proxy.bytes_from_server`, `proxy.bytes_from_client`)
and `proxy.armed` are there for the assertion.

```python
proxy.chop(64).delay(0.01)               # slow and fragmented, still correct
assert xrd.FileSystem(str(proxy.url)).stat("/big").st_size == 7000

proxy.refuse()                           # now the endpoint is simply down
with pytest.raises(xrd.TransientError):
    xrd.FileSystem(str(proxy.url)).stat("/big")
proxy.accept().heal()
```

`refuse` bites on the *next* connection; sessions already established keep
working until something cuts them, which is what `cut()` is for.

`stall_after` tests timeouts; a client that only handles a *closed* socket
hangs forever on one that is merely silent.

## Suite markers

Two pytest markers gate the tests that need more than Python:

```console
$ pytest -m "not interop and not parity"     # the default, no daemon needed
$ pytest -m interop                          # needs xrootd on PATH
$ pytest -m parity                           # needs the official bindings too
```

`interop` runs the suite against a real `xrootd` daemon started on a
loopback port; `parity` runs the same operations through this client and the
official `XRootD` bindings and compares the answers field by field. Neither is
required to develop against the library, and both run in CI.

## Coverage

The default suite covers 100% of the package's statements *and* branches -
including the fake servers in `xrd.testing`, which are shipped code and so are
held to the same standard:

```console
$ pytest --cov                              # gated: proto/, crypto/, client/ at 100%
$ coverage run --branch -m pytest && coverage report -m --include='*/xrd/*'
```

The gate in `pyproject.toml` deliberately covers only the wire protocol, the
cryptography and the client surface, because coverage of the optional adapters
(fsspec, the CLI, `asyncio`) depends on which extras are installed. Run the
second form to see the whole package.
