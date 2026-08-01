# Copying

```python
xrd.copy(source, target, *, chunk_size=None, verify=None, algorithm=None,
         overwrite=True, progress=None, config=None) -> CopyResult
```

Either side may be a URL, a local path, an `xrd.Path`, or an already-open
binary file object. That covers every direction without a separate function
per case:

```python
xrd.copy("root://host//store/f.root", "/scratch/f.root")     # download
xrd.copy("/scratch/f.root", "root://host//store/f.root")     # upload
xrd.copy("root://a//store/f", "davs://b/store/f")            # across, via here
with open("/scratch/f", "wb") as fh:
    xrd.copy("root://host//store/f.root", fh)                # into a stream
```

The result says what happened:

```python
r = xrd.copy(src, dst)
r.size, r.seconds, r.rate, r.checksum, r.verified
print(r)   # root://... -> /scratch/f.root (4194304 bytes, 212.4 MB/s)
```

## Verification

`verify` defaults to `config.verify_checksums`, which is on. A digest is
computed while the bytes stream past and compared against the server's own
checksum - of the target when the target is remote, otherwise of the source.
A server that cannot checksum degrades quietly; `verify=True` makes that an
error instead.

```python
xrd.copy(src, dst, verify=True, algorithm="crc32c")
```

A mismatch raises `ChecksumMismatchError`, which carries both digests.

!!! warning
    A checksum is an integrity check, not authentication. A server that
    serves you wrong bytes can serve you the matching digest. See
    [Security](security.md).

## Progress

```python
def bar(done, total):
    print(f"\r{done * 100 // total}%", end="")

xrd.copy(src, dst, progress=bar)
```

`total` is the size the source reported, which for a stream source may be
zero.

## Recursive copies

```python
results = xrd.copy_tree("root://a//store/run7", "/scratch/run7")
print(sum(r.size for r in results))
```

Local destination directories are created as needed; remote ones come for
free, because a remote write asks for `kXR_mkpath`. Extra keyword arguments
are handed to `copy()` for each file.

Server-supplied names are validated before they are joined onto your
destination - a listing entry containing `/` or equal to `..` is refused
outright, so a hostile endpoint cannot steer a recursive download out of the
directory you named.

## Third-party copy

```python
xrd.third_party("root://a//store/f.root", "root://b//store/f.root")
xrd.third_party("davs://a/store/f.root", "davs://b/store/f.root")
```

The data moves between the two servers and never through this process. One
call, two dialects: the URLs decide which one is spoken.

| Endpoints | Dialect |
| --- | --- |
| two `root://` | the `XrdOucTPC` rendezvous - a key minted here, `tpc.src`/`tpc.dst` opaque, `kXR_sync` to trigger and to wait |
| two `http(s)`/`dav(s)` | WLCG `COPY`, the one FTS and Rucio use |

Both endpoints must speak the same protocol, because each dialect is one
server asking another for the file in a language it understands. A mixed
pair raises, and `copy()` is the answer - it streams through this process,
which is what a mixed pair needs anyway.

`root://` takes `token_mode` (the delegation style) and `posc`. HTTP takes
rather more, because the header set is the protocol:

```python
xrd.http.third_party(src, dst, mode="pull", overwrite=True, delegate=False,
                     verify=None, streams=None, remote_token=None,
                     transfer_headers={}, progress=None, timeout=None)
```

- **`mode`** - `"pull"` sends the `COPY` to the destination with a `Source:`
  header, which is what almost everything does. `"push"` sends it to the
  source with a `Destination:` header, for a destination that cannot make
  outbound connections.
- **the far side's token** travels in `TransferHeaderAuthorization`, and is
  taken from that URL's `authz` parameter first, so a pair of pre-signed URLs
  needs nothing else:

    ```python
    xrd.third_party(f"{src}?authz={read_token}", f"{dst}?authz={write_token}")
    ```

  The token is stripped from the URL the far side is given, since it belongs
  in the header the transfer authorises rather than in the other endpoint's
  request log.
- **`delegate=False`** sends `Credential: none`, which is what stops a server
  that supports X.509 delegation from waiting for a credential a
  token-authenticated transfer will never send.
- **`verify`** sets `RequireChecksumVerification`; left as `None` it says
  nothing and the server's own policy stands.
- **`progress`** is called with the running byte count from the performance
  markers, so a long transfer can be watched even though nothing is
  streaming through here.

The one thing to know about HTTP third-party copy is that a `202 Accepted`
means the copy *started*. The outcome is the last line of the response body,
after the performance markers, and a transfer that failed still arrived as a
`202`. This client reads to that line before returning, and turns a
`failure:` into the exception the status it quotes deserves - a source that
answers 403 raises `PermissionError`, exactly as a direct read would.

For a copy where the data must pass through you anyway, `copy()` is both
simpler and, on a fast network, not obviously slower - see
[Performance](performance.md).

## Tuning

| Setting | Effect |
| --- | --- |
| `config.chunk_size` | bytes per request, default 4 MiB (`XRD_CPCHUNKSIZE`) |
| `config.parallel_chunks` | requests in flight (`XRD_CPPARALLELCHUNKS`) |
| `config.verify_checksums` | default for `verify` |
| `config.preferred_checksum` | default for `algorithm` |

## From the command line

```console
$ xrd-cp /tmp/f.root root://host//store/f.root
$ xrd-cp -r /tmp/results davs://dav.example.org/store/results
$ xrd-cp --tpc root://a//store/f.root root://b//store/f.root
$ xrd-cp --tpc davs://a/store/f.root davs://b/store/f.root
$ xrd-cp --no-verify --progress root://host//store/big.root /scratch/
```
