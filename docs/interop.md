# Interoperability

Everything in this library was written from the XRootD protocol
specification. A specification can be misread, and a test suite written from
the same misreading will pass happily forever. So two suites exist that cannot
be fooled that way.

## Against a real daemon

```console
$ pytest -m interop
```

Starts an unprivileged `xrootd` on a loopback port, exports a temporary
directory, and runs the client against it: open, read, write, `readv`,
`pgread`, truncate, dirlist, stat, mkdir, rm, mv, chmod, xattrs, query,
locate, checksums, third-party copy. Where it matters the stock `xrdcp` and
`xrdfs` binaries read back what this client wrote, and vice versa - bytes
written here are bytes the reference implementation sees.

It has already earned its keep. `kXR_writev` counted its data in `dlen`, which
the in-process fake accepted and the real server refused with
`kXR_ArgInvalid: Write vector is invalid`. The fake now checks the same thing.

Needs `xrootd` on `PATH`; skips otherwise. No root, no configuration file you
have to install.

## Against the official bindings

```console
$ pytest -m parity
```

Same daemon, same files, both clients, field by field. Anything this library
reports that `XRootD.client` does not is a bug in one of them, and the
disagreement is worth more than either answer alone.

The two libraries deliberately differ in shape - exceptions here, `(status,
result)` tuples there; `bytes` here, buffers there - so the suite asserts that
the *values* agree: the same size, the same mtime, the same flags, the same
checksum, the same directory listing, the same bytes at the same offsets.

Needs the official bindings installed as well; skips otherwise.

## Reading files other tools wrote

There is nothing to say here, which is the point. The wire format is the wire
format: files this client writes are ordinary files, checksums it computes
agree with `xrdfs query checksum`, and third-party copies it initiates are
the same `tpc.*` CGI handshake `xrdcp --tpc` performs. A site does not need to
know which client its users are running.

## What is deliberately not supported

| Not implemented | Why |
| --- | --- |
| GSI signed Diffie-Hellman | refused by name rather than mis-answered |
| X.509 delegation | same |
| `kXR_bind` split data sockets | one multiplexed stream is enough, and simpler |

Each raises an exception that names the feature. A server insisting on one of
them says so at the handshake, not three operations later.

Two more are absent because the protocol retired them, not because this client
declined: `kXR_verifyw` and `kXR_decrypt` were dropped from XProtocol, and
their request codes have since been reissued - 3026 is `kXR_pgwrite` today. A
client still sending the old opcodes would be writing pages while believing it
was verifying them. Their replacements, `pgread` and `pgwrite`, carry a CRC32C
per 4 KiB page and are implemented in full; older clients that still name
`verifyw` are describing a protocol no current server speaks.

## Beyond the specification

`symlink`, `link` and `readlink` are not in XProtocol at all. They are sent as
`kXR_symlink` (3501), `kXR_readlink` (3502) and `kXR_link` (3503), framed like
`kXR_mv` - the same numbers and framing XRootD.jl and XrdRust chose, so the
three clients agree with each other and a server taught one of them
understands all three. A stock daemon answers `kXR_Unsupported`, which is the
honest answer for a namespace with no links, and this client turns it into
`UnsupportedError` rather than pretending. The HTTP and WebDAV backends raise
the same exception without a round trip, because there is no verb to try.

## Version coverage

Tested against XRootD 5.x. The client announces protocol `0x520` (v5.2) in
`kXR_protocol`, which is what enables `kXR_pgread` and `kXR_pgwrite`. Those
two are asked for explicitly - `File.pgread` and `File.pgwrite` - and a server
that does not implement them says so rather than being guessed at; ordinary
`read`/`write` work everywhere and are what everything else uses.

## Coming from `pyxrootd`

See [Coming from pyxrootd](migrating.md) for the call-by-call mapping.
