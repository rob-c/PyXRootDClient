# Namespaces

`xrd.FileSystem` is one endpoint's namespace. It holds a connection, so use
it as a context manager or call `close()`.

```python
with xrd.FileSystem("root://eos.example.org") as fs:
    ...
```

Paths may be absolute or relative to the URL's own path.

## Reading the namespace

| Method | Notes |
| --- | --- |
| `stat(path)` | a `StatInfo` with `st_size`, `st_mtime`, `flags`, `id` |
| `statx(paths)` | many paths, one round trip |
| `statvfs(path)` | free space, utilisation, node counts |
| `exists`, `isdir`, `isfile`, `getsize` | the `os.path` spellings |
| `listdir(path)` | names only |
| `scandir(path)` | `DirEntry` objects with `stat` attached |
| `iterdir(path)` | the same, as an iterator |
| `walk(top)` | `os.walk`'s `(root, dirs, files)` triples |
| `glob(pattern, root="")` | `pathlib` semantics, absolute paths out |

```python
for entry in fs.scandir("/store"):
    print(entry.name, entry.is_dir(), entry.stat.st_size)
```

`DirEntry` is `os.PathLike`, so `open(entry)` and `os.fspath(entry)` work.

### glob

The pattern language is `pathlib.Path.glob`'s, not `fnmatch`'s: `*` and `?`
and `[...]` stop at a `/`, and `**/` is *zero or more* directories.

```python
fs.glob("/store/user/me/*.root")        # one level
fs.glob("/store/user/me/**/*.root")     # any depth, including none
fs.glob("run[0-9]*/*.root", root="/store/user/me")
```

A pattern whose magic is confined to its last component costs one `scandir`.
Anything deeper walks, but only from the deepest wildcard-free directory in
the pattern - `/store/user/me/**/*.root` never lists `/store/user`.

!!! note
    An earlier version matched recursive patterns against the path *relative*
    to the search root, which quietly returned nothing for an absolute
    pattern. It was found by running the same glob against a real daemon; see
    [interoperability](interop.md).

## Changing the namespace

```python
fs.mkdir("/store/user/me/2026")
fs.makedirs("/store/a/b/c", exist_ok=True)
fs.touch("/store/f.root", exist_ok=True)
fs.rename("/store/a.root", "/store/b.root")
fs.remove("/store/b.root")          # also spelled unlink
fs.rmdir("/store/empty")
fs.rmtree("/store/user/me/scratch")
fs.truncate("/store/f.root", 4096)
fs.chmod("/store/f.root", 0o640)
```

!!! note "`touch` is not `open(..., "a")`"
    A stock server refuses to create a file for `kXR_open_updt` alone, so
    `touch` asks for `kXR_new | kXR_open_updt | kXR_mkpath` and treats
    `FileExistsError` as success when `exist_ok`. Doing it the obvious way
    fails on a real daemon while passing against a lenient fake.

## Whole-file shortcuts

```python
fs.write_bytes("/store/f.root", payload)
data = fs.read_bytes("/store/f.root")
fs.write_text("/store/note.txt", "hello")
text = fs.read_text("/store/note.txt")
fh = fs.open("/store/f.root", "rb")     # shares this connection
```

## Asking the server about itself

```python
fs.ping()                     # returns None; raises if it is not there
fs.protocol()                 # ProtocolInfo: version, flags, security
fs.query_config("version", "role", "sitename")
fs.query(QueryCode.SPACE, "/store")
fs.checksum("/store/f.root")            # ChecksumInfo(algorithm, value)
fs.locate("/store/f.root")              # which servers hold it
fs.deep_locate("/store/f.root")         # follow managers to the data servers
fs.prepare(["/store/a.root", "/store/b.root"])   # stage from tape
fs.evict(["/store/a.root"])
fs.statvfs("/store")
```

`checksum()` asks for `config.preferred_checksum` (`adler32` by default) and
returns whatever the server actually computed, which is not always what you
asked for.

!!! warning "Servers cache checksums"
    A file rewritten in place can come back with the digest of its previous
    contents. If you need certainty after a write, checksum a fresh path.

## Extended attributes

```python
fs.setxattr(path, "user.experiment", b"atlas")
fs.setxattr(path, "user.run", b"1", create_only=True)   # refuses to overwrite
fs.getxattr(path, "user.experiment")
fs.listxattr(path)
fs.xattrs(path)
fs.removexattr(path, "user.experiment")
```

`kXR_fattr` reports one `errno` per attribute *inside* a successful reply, so
a request that failed for the attribute you named still comes back as
`kXR_ok`. The client reads that code and raises: `FileExistsError` for a
`create_only=True` name that is already there, `AttrNotFoundError` (an
`OSError` with `ENODATA`) for removing one that is not.

## Sharing a connection

Every object that talks to a server accepts an existing router, and
`FileSystem.open()` hands its own to the file it opens. Opening a hundred
files through one `FileSystem` costs one connection, not a hundred.

```python
with xrd.FileSystem("root://host") as fs:
    handles = [fs.open(f"/store/{n}", "rb") for n in names]
```

A handle that opens its own connection closes it, including when the open
itself fails - a subtlety that cost a socket leak per caught `FileExistsError`
until the interop suite caught it.
