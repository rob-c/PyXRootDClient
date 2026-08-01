# Performance

Pure Python framing a binary protocol sounds slow. On the transfers that
matter it is not, because the work is dominated by the network and by
`memoryview` slicing, not by interpreted bytecode.

## Measured

`benchmarks/bench.py`, 16 MiB file, real `xrootd` daemon on loopback,
best-of-3, against the official XRootD 5.9.6 Python bindings and the `xrdcp`
/ `xrdfs` command line tools. Lower is better; `1.00x` is the winner of each
row.

| Case | this client | bindings | CLI |
| --- | --- | --- | --- |
| read whole file | 1.63x | **1.00x** | 3.67x (`xrdcp`) |
| 256 x 64 KiB reads | **1.00x** | 7.91x | |
| vector read, 8 x 128 KiB | 4.43x | **1.00x** | |
| write whole file | 2.52x | **1.00x** | |
| `stat` | **1.00x** | 1.50x | 825x (`xrdfs`) |
| `listdir` | **1.00x** | 6.50x | |
| copy to local disk | **1.00x** | 1.13x | 3.64x (`xrdcp`) |

Read the table honestly: on one streaming read of a large file the C++ client
is ~1.6x faster, and on a vector read it is ~4.4x faster. On everything a real
analysis job does in a loop - many small reads, metadata, listing, copying -
this client is at least as fast, because the per-call overhead of crossing the
Python/C++ boundary costs more than parsing a header in Python does.

The `xrdfs stat` figure is process startup, not protocol. It is in the table
because 825x is what a shell loop over a thousand files actually pays.

## Running it yourself

```console
$ python benchmarks/bench.py --size 16 --repeat 3      # size in MiB
$ python benchmarks/bench.py --json results.json
$ python benchmarks/bench.py --url root://host//store/scratch   # a real endpoint
```

The harness starts its own `xrootd` on a loopback port, skips any comparison
whose counterpart is not installed, and reports both best-of-N and the median
so a single unlucky run is visible rather than averaged away. Metadata cases
are reported in operations per second, data cases in MiB/s.

## Where the time goes

Three things carry the load:

**No copy that can be avoided.** Response bodies are sliced out of the receive
buffer through a `memoryview`, so a multi-megabyte read is frozen to `bytes`
exactly once. The obvious spellings - `bytes(buf[:n])`, `pending + body` -
each copy twice; profiling this benchmark is how they were found.

**One socket, many requests.** Requests on a connection are multiplexed by
stream ID, so they do not wait for each other's responses, and a `FileSystem`
holds its connection open for its whole life. A thousand-file dataset read
through one `FileSystem` - or one `fsspec` instance, which is cached by its
constructor arguments - costs one login. Constructing a fresh `FileSystem` per
file costs one login *in total* as well, because a closed connection goes into
the pool and the next `FileSystem` for the same server and the same credential
takes it back out ([Pooling](config.md#pooling)). Hoisting it out of the loop
is still the clearer code, and it is the version that also avoids re-resolving
the URL; the pool is there for the code you did not write, like a helper that
takes a URL and returns bytes.

**Reads are batched when you let them.** `readv` is one round trip for many
ranges; `pgread` gets per-page CRC32C from the server for free.

## Making it faster

```python
cfg = xrd.Config(
    chunk_size=8 << 20,        # bigger writes, fewer round trips
    readahead=4 << 20,         # buffered reads pull more per request
    parallel_chunks=8,         # concurrent chunks in a copy
)
```

Defaults are 4 MiB, 1 MiB and 4. On a high-latency WAN link raise all three;
on loopback they make no difference.

For many ranges from one file, ask once:

```python
with xrd.open(url, "rb") as fh:
    blocks = fh.raw.file.readv([(off, 128 << 10) for off in offsets])
```

For one file being streamed hard, give its bytes a socket of their own:

```python
handle = fh.raw.file
handle.bind_data_path()        # kXR_bind; see Files -> a second connection
```

The reads still go out on the control link and the data comes back on the new
one, so a `stat` in another thread is not queued behind a 64 MiB read. It
costs a connection and a handshake, which is why it is a call and not a
default.

For many files, go wide rather than deep - one session per worker thread, or
`asyncio.gather` over separate handles ([Asynchronous use](async.md)). A
single session serialises its own calls.

Turn off what you are not using:

```python
xrd.copy(src, dst, config=cfg.evolve(verify_checksums=False))
```

Checksum verification costs a server-side digest per file. It is on by default
because silent corruption is worse than slow, but for scratch data it is
wasted work.

## What is not fast

Anything that must touch every byte in Python - computing a checksum locally,
say - runs at Python speed. Ask the server for the digest instead
(`fs.checksum(path)`), which is what `--verify` does.
