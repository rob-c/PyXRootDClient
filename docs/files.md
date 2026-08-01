# Files and paths

## `xrd.open` returns an `io` object

```python
fh = xrd.open("root://host//store/f.root", "rb")
```

`fh` is an `io.BufferedReader` wrapping an `XRootDRawIO`, which is an
`io.RawIOBase`. That single decision is what makes the rest of Python work
without adapters: anything that accepts a binary file object accepts this -
`numpy.load`, `zipfile.ZipFile`, `tarfile`, `pyarrow`, `uproot`, `json.load`
through a `TextIOWrapper`, `shutil.copyfileobj`.

| Argument | Meaning |
| --- | --- |
| `mode` | `r w x a` with `b t +`, exactly as the builtin. Default `"rb"` |
| `buffering` | `0` raw (binary only), `-1` a 1 MiB buffer, or a size |
| `encoding`, `errors`, `newline` | passed to `io.TextIOWrapper` in text mode |
| `config` | a [`Config`](config.md) for this file |
| `router` | share an existing connection |
| `posc` | persist-on-successful-close: the server discards a partial file if the connection dies |

The buffer defaults to a megabyte rather than the stdlib's 8 KiB because a
wide-area round trip costs several orders of magnitude more than a local one.

The signature is overloaded the way typeshed overloads the builtin, so a type
checker knows what came back:

```python
xrd.open(url, "rb").read()               # bytes
xrd.open(url, "r").read()                # str
xrd.open(url, "rb", buffering=0)         # XRootDRawIO, so .file is there
xrd.open(url, mode_from_config).read()   # Any - a non-literal mode is the escape hatch
```

The package ships a `py.typed` marker, so this reaches your code without a
stub package. See [Typing](typing.md).

### Modes, and what they do on the wire

| Mode | `kXR_open` options | Creates? |
| --- | --- | --- |
| `r` | `kXR_open_read` | no |
| `r+` | `kXR_open_updt` | no |
| `w`, `w+` | `kXR_open_updt \| kXR_delete` | yes (truncating) |
| `x`, `x+` | `kXR_open_updt \| kXR_new` | yes, fails if present |
| `a`, `a+` | `kXR_open_updt \| kXR_open_apnd` | yes |

!!! note "Append is the interesting one"
    XRootD's `kXR_open_apnd` opens for appending but does *not* create, and
    `kXR_new` is refused outright when the file already exists - so no single
    flag combination has Python's `"a"` semantics. The raw layer opens for
    append, and only if the server answers `kXR_NotFound` does it retry with
    `kXR_new`. This is one of the behaviours the
    [interoperability suite](interop.md) found; the fake server used to be
    more permissive than a real one and hid it.

## The handle underneath

`fh.raw.file` (or `fh.file` on a raw object) is an `xrd.File`, which is where
the operations that have no `io` equivalent live:

```python
handle = fh.raw.file
head, tail = handle.readv([(0, 4096), (1 << 30, 4096)])   # kXR_readv
page = handle.pgread(1 << 20, 0)                          # CRC32c per 4 KiB
print(page.corrupt_pages)                                 # [] when clean
handle.writev([(0, b"first"), (4096, b"second")])         # kXR_writev
handle.sync()
handle.truncate(1 << 20)
print(handle.checksum())                                  # adler32:1a0b045d
```

`File` can also be used on its own, with or without `with`:

```python
from xrd import File, OpenFlags

handle = File("root://host//store/f.root")
handle.open(OpenFlags.UPDATE | OpenFlags.NEW)
try:
    handle.write(b"...", 0)
finally:
    handle.close()
```

Entering a `File` that is not yet open opens it for reading; entering one you
have already opened is an error rather than a silent re-open.

### Vector reads

`readv()` returns one `bytes` per requested range, in the order you asked
for, no matter how the server batches or reorders the reply. Requests larger
than the server's advertised limits are split into as few round trips as the
limits allow.

```python
ranges = [(off, 128 << 10) for off in offsets]
for wanted, got in zip(ranges, handle.readv(ranges)):
    ...
```

This is the single biggest performance lever for HEP workloads: one round
trip for a hundred scattered basket reads instead of a hundred.

### Paged I/O

`pgread` and `pgwrite` carry a CRC32c per 4 KiB page. `pgread` verifies by
default and reports the offset of any page that failed:

```python
result = handle.pgread(1 << 20, 0)
if result.corrupt_pages:
    raise IOError(f"corrupt pages at {result.corrupt_pages}")
```

### Checkpointed writes

A checkpoint is a transaction over one handle: the server journals what you
write, and either keeps it or puts the file back the way it was.

```python
with xrd.File("root://host//store/f.root") as handle:
    handle.open(OpenFlags.UPDATE)
    with handle.checkpoint() as cp:
        handle.write(header, 0)
        handle.truncate(new_length)
        print(cp.query().free)      # bytes the journal still has room for
    # left cleanly -> committed; raised -> rolled back, and the error re-raised
```

Every `write`, `pgwrite` and `truncate` inside the block travels as
`kXR_ckpXeq` wrapping the real request, which is what makes it undoable.
`writev` is not one of the three a server can undo and is refused with
`UnsupportedError` rather than being let through unjournaled. Checkpoints do
not nest - the server keeps one per handle - and a server without
`kXR_chkpoint` raises on entry, before anything has been written.

`cp.query()` returns a `CheckpointInfo` with `capacity`, `used` and `free`.
A journal that fills up fails the write with `NoSpaceError`; the block still
rolls back on the way out.

### Recovery

With `Config(recover_handles=True)` - the default - a read-only handle whose
data server disappears is re-opened transparently and the read is retried.
`handle.recoverable` says whether a given handle qualifies. Writers are never
silently recovered, because a partially applied write is not something a
client can reason about on your behalf.

## `xrd.Path`

`xrd.Path` (also spelled `xrd.XRootDPath`) is the `pathlib` front door. It is
not a `PurePosixPath` subclass - it carries an endpoint - but it answers the
same questions:

```python
p = xrd.Path("root://host//store/user/me")

p.name, p.stem, p.suffix, p.parent, p.parts
p / "runs" / "run1.root"
p.with_suffix(".txt")
p.relative_to("root://host//store")

p.exists(), p.is_dir(), p.is_file()
p.stat().st_size
p.iterdir(), p.glob("**/*.root"), p.rglob("*.root"), p.walk()
p.mkdir(parents=True, exist_ok=True)
p.touch(), p.unlink(), p.rmdir(), p.rename(other), p.replace(other)
p.read_bytes(), p.write_text("hi"), p.open("rb")
p.checksum(), p.chmod(0o640), p.locate()
```

A path holds a connection once it has used one; `close()` returns it, and
`with` does that for you.

## Extended attributes

Present on both `File` and `FileSystem`:

```python
fs.setxattr("/store/f.root", "user.experiment", b"atlas")
fs.getxattr("/store/f.root", "user.experiment")
fs.listxattr("/store/f.root")
fs.xattrs("/store/f.root")          # the whole mapping
fs.removexattr("/store/f.root", "user.experiment")
```

A per-attribute failure travels inside an otherwise successful reply, and the
client raises it rather than dropping it: `setxattr(..., create_only=True)`
over an existing name is a `FileExistsError`, and removing an attribute that
is not there is an `AttrNotFoundError`.

!!! warning "Servers mangle names"
    A stock `xrootd` returns `user.experiment` from a listing as
    `.experiment`. The round trip through `setxattr`/`getxattr` is exact; the
    listing is the server's idea of the name, and this client does not
    second-guess it.
