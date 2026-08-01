# PyXRootDClient

A pure-Python client for XRootD. `root://`, `roots://`, `https://` and HEP
WebDAV, spoken by the same objects, with no compiled extension, no `libXrdCl`,
and no third-party import in the core.

```python
import xrd

with xrd.open("root://eos.example.org//store/data.root", "rb") as fh:
    header = fh.read(1024)

fs = xrd.FileSystem("root://eos.example.org")
for entry in fs.scandir("/store"):
    print(entry.name, entry.stat.st_size)

xrd.copy("root://a.example.org//store/f.root", "davs://b.example.org/store/f.root")
```

It is a Python library first and an XRootD binding second: files are real
`io` objects, errors are `OSError` subclasses, paths are `PurePath`-shaped,
and nothing returns a `(status, result)` pair.

## Install

```console
$ pip install pyxrootdclient                 # the whole library
$ pip install pyxrootdclient[fsspec]         # pandas / dask / pyarrow URLs
$ pip install pyxrootdclient[krb5]           # the Kerberos mechanism
```

Requires Python 3.10+. Almost nothing needs an extra: `http://`, `https://`
and WebDAV are `http.client`, and GSI/X.509 proxies are pure Python down to
the AES and RSA. Kerberos is the one exception — see below.

## What it does

**Files.** `xrd.open(url, mode)` returns something from the `io` stack:
seekable, buffered, iterable, context-managed, `read`/`write`/`readinto`,
text mode when you ask for it. Vector reads (`kXR_readv`), paged I/O with
CRC32c verification, checkpointed writes, and `sendfile`-shaped bulk copies
are on the underlying object when you want them.

```python
with xrd.open("root://host//store/f.root", "rb") as fh:
    for line in fh:            # buffered, like any other file
        ...
    fh.seek(-4096, 2)
    tail = fh.read()
```

**Namespaces.** `xrd.FileSystem` covers `stat`, `statx`, `statvfs`,
`scandir`, `walk`, `glob`, `mkdir`, `makedirs`, `rename`, `remove`,
`rmtree`, `truncate`, `chmod`, `touch`, `checksum`, `locate`, `deep_locate`,
`prepare`, `query_config`, extended attributes, and - where a server has been
taught the vendor opcodes - `symlink`, `link` and `readlink`.

```python
fs = xrd.FileSystem("davs://dav.example.org")
fs.makedirs("/store/user/me/2026", exist_ok=True)
print(fs.checksum("/store/user/me/f.root"))     # adler32:1a0b045d
```

**Paths.** `xrd.Path` is a `PurePosixPath` that knows its endpoint:

```python
p = xrd.Path("root://host//store") / "user" / "me"
p.mkdir(parents=True, exist_ok=True)
(p / "note.txt").write_text("hello")
sizes = {child.name: child.stat().st_size for child in p.iterdir()}
```

**Copies.** `xrd.copy`, `xrd.copy_tree` and `xrd.third_party` move data
between any two endpoints, local paths included, with checksum verification
on by default and a `progress=` callback that takes `(done, total)`. A tree
can be filtered (`include=`, `exclude=`), brought up to date rather than
recopied (`sync="size" | "mtime" | "checksum"`), pruned (`delete=True`),
rehearsed (`dry_run=True`) or moved (`remove_source=True`). An interrupted
transfer is continued rather than restarted with `resume=True`, or `xrd-cp -c`,
and a file long enough to be worth it is moved by `config.parallel_chunks`
connections at once, one span of the file each.

**Async.** `xrd.aio` mirrors the whole surface — same names, same arguments,
`await` in front. `import xrd` does not import `asyncio`; the facade is
resolved on first use.

```python
import asyncio, xrd.aio

async def main():
    async with xrd.aio.FileSystem("root://eos.example.org") as fs:
        async for entry in fs.iterdir("/store"):
            print(entry.name)
        names = await fs.listdir("/store")
        sizes = await asyncio.gather(*(fs.getsize(f"/store/{n}") for n in names))
        async with fs.open("/store/f.root") as fh:
            head, tail = await fh.readv([(0, 4096), (1 << 20, 4096)])

asyncio.run(main())
```

**Authentication.** `gsi`, `ztn`, `sss`, `unix` and `host` out of the box,
tried in that order against whatever the server offers, with every rejected
mechanism and its reason named in the final error. That means X.509 proxies
from `$X509_USER_PROXY` (RFC 3820 and legacy Globus, with the lifetime
checked *before* the round trip, so an expired proxy is a sentence and not a
timeout an hour into a job) and WLCG / SciTokens / macaroons. TLS on
`roots://`, `xroots://` and `davs://` — all three present the same proxy as
the client certificate, so mutual TLS costs no argument.

`krb5` is the one mechanism that needs an extra: it reads your credential
cache with no help at all, and will tell you when your ticket expired, but the
exchange itself goes through `gssapi` because a Kerberos token can only
honestly be tested against a live KDC.

At a terminal, a login with no proxy and no token asks for one — naming what
is missing, where it looked, and the command that produces it — instead of
failing; in a batch job it stays silent and puts the same explanation in the
error. `Config(prompt=False)`, `--no-prompt` and `$XRD_PROMPT=0` settle it
either way, and `Config(prompter=...)` moves the question into a GUI or a
notebook.

Credentials are redacted from logs, reprs and tracebacks — that is enforced by
a test, not a convention.

## Command line

```console
$ xrd-fs ls -l root://eos.example.org//store/user/me
$ xrd-fs stat --json davs://dav.example.org/store/f.root
$ xrd-fs checksum -a adler32 root://host//store/f.root
$ xrd-fs tail -f root://host//store/running.log
$ xrd-fs du root://host//store/run7
$ xrd-cp -r /tmp/results davs://dav.example.org/store/results
$ xrd-cp -r --sync size --delete /tmp/results root://host//store/results/
$ xrd-cp --tpc root://a//store/f.root root://b//store/f.root
```

Every subcommand takes whole URLs and understands `--json`, so a shell script
never has to parse columns. Exit codes are the usual three: `0` success,
`1` a runtime failure, `2` a usage error. Settings that never change can live
in `~/.config/xrd/config.ini` and be selected with `--alias`.

## fsspec

With the `[fsspec]` extra, `root://`, `roots://`, `xroot://`, `dav://`,
`davs://` and `webdav://` are registered URL schemes:

```python
import pandas as pd
df = pd.read_parquet("root://eos.example.org//store/t.parquet")
```

## Testing against it

`xrd.testing` ships the servers this library's own suite runs against — no
storage element required:

```python
from xrd.testing import FakeServer

with FakeServer(files={"/data/a.root": b"hello"}) as server:
    fs = xrd.FileSystem(server.url)
    assert fs.read_bytes("/data/a.root") == b"hello"
```

`FakeDAVServer` is the same idea for HTTP and WebDAV.

## Status

The wire protocol, session state machine, the whole authentication ladder,
file and namespace APIs, `pathlib` bindings, the async facade, HTTP/WebDAV,
the copy engine, the CLI and the fsspec bindings are implemented and tested —
2144 tests, of which the great majority need no network, no KDC and no
`openssl`. The remainder are the interoperability suite, which runs against a
real `xrootd` daemon and reads back what `xrdcp` and `xrdfs` write, and the
parity suite, which runs every operation through this client and the official
XRootD bindings and compares the answers field by field. Coverage is 100% of
statements *and* branches across the package, and the wire protocol, the
cryptography and the client surface are gated there; `ruff` and
`mypy --strict` pass clean and are hard gates too. The
package ships `py.typed`, and `xrd.open` is overloaded the way the builtin is,
so a literal mode tells your type checker whether you get bytes or text.

Third-party copy works in both dialects from one call: `xrd.third_party` sends
the `XrdOucTPC` rendezvous to a `root://` pair and the WLCG `COPY` dialect to
an `http(s)`/`dav(s)` one, so the bytes move server to server either way.

Connections are pooled across instances: a `FileSystem` that closes hands its
authenticated connection to the next one opened on the same server by the same
person, so a script that constructs one per file logs in once rather than a
thousand times. Reuse is keyed on the credentials as well as the endpoint, and
a connection that failed is discarded rather than passed on.

A file being streamed can put its bytes on a connection of their own:
`bind_data_path()` binds a second socket to the same session with `kXR_bind`,
and from then on reads and writes travel there while requests keep the
control link to themselves. The second connection inherits the session's
identity rather than logging in again.

Not yet: GSI's signed-DH path and X.509 delegation, both refused by name
rather than mis-answered, and HTTP/2.

Full documentation is in [`docs/`](docs/) (`mkdocs serve` to read it), with
[`SECURITY.md`](SECURITY.md) for the threat model,
[`benchmarks/bench.py`](benchmarks/bench.py) for the measurements,
[`docs/superpowers/plans/`](docs/superpowers/plans/) for the roadmap and
[`docs/superpowers/specs/`](docs/superpowers/specs/) for the design.

## Licence

LGPL-3.0-or-later: [`LICENSE`](LICENSE) is the Lesser terms, which apply on
top of the GPL text in [`COPYING`](COPYING).
