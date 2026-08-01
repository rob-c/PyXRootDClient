# Errors

Every failure raises. There is no `(status, result)` tuple to check, because a
tuple you can forget to check is a bug waiting to happen, and because Python
already has a mechanism for this.

Server errors map onto the `OSError` subclass a local filesystem would have
raised, so code written against `/tmp` works against `root://` unchanged:

```python
try:
    data = xrd.Path("root://host//store/f.root").read_bytes()
except FileNotFoundError:
    ...
except PermissionError:
    ...
```

## The hierarchy

```
Exception
└── XRootDError
    ├── ProtocolError                 the peer is not speaking XRootD correctly
    ├── ConnectionError               (also builtins.ConnectionError)
    │   └── TransientError            worth retrying
    │       ├── TimeoutError          (also builtins.TimeoutError)
    │       └── WaitLimitError        the server kept saying "come back later"
    ├── AuthenticationError
    │   ├── NoMechanismError          nothing we have was accepted
    │   └── CredentialError
    │       └── TokenExpiredError
    ├── RedirectLimitError
    ├── ChecksumMismatchError
    └── ServerError                   a kXR_error response
        ├── NotFoundError             + FileNotFoundError   ENOENT
        ├── ExistsError               + FileExistsError     EEXIST
        ├── PermissionError           + PermissionError     EACCES
        ├── IsADirectoryError         + IsADirectoryError   EISDIR
        ├── NotADirectoryError        + NotADirectoryError  ENOTDIR
        ├── NoSpaceError              + OSError             ENOSPC
        ├── IOError                   + OSError             EIO
        ├── UnsupportedError          + OSError             ENOSYS
        ├── ReadOnlyError             + OSError             EROFS
        ├── QuotaError                + OSError             EDQUOT
        ├── AttrNotFoundError         + OSError             ENODATA
        ├── BusyError                 + OSError             EBUSY
        └── InvalidArgumentError      + OSError             EINVAL
```

Catch broadly with `xrd.XRootDError`, or narrowly with the builtin you already
know.

## What a `ServerError` carries

```python
try:
    fs.stat("/store/missing")
except xrd.ServerError as exc:
    exc.code        # 3011, the kXR_ code
    exc.message     # the server's own text
    exc.path        # what we asked about
    exc.errno       # errno.ENOENT
    str(exc)        # 'kXR_NotFound: no such file or directory [/store/missing]'
```

`str()` names the wire code, so a bug report says what the server actually
said. The `errno` is there because `OSError` promises it.

Exceptions pickle correctly, which matters when they cross a
`concurrent.futures` or `multiprocessing` boundary and you want the code and
the path on the other side, not a mangled `(errno, strerror)` pair.

## Retrying

`TransientError` is the marker: connection resets, timeouts, and
server-requested waits that outlived `wait_cap`. Everything else is a
permanent answer.

```python
for attempt in range(5):
    try:
        return xrd.Path(url).read_bytes()
    except xrd.TransientError:
        time.sleep(2 ** attempt)
raise
```

The client already reconnects and retries underneath - `connect_retries`,
`retry_backoff`, and `recover_handles` for a data server that disappears
mid-read. A `TransientError` reaching you means that budget was spent.

`WaitLimitError` is the one to single out. A server answering `kXR_wait` is
busy, not broken: the connection is fine, so the client sleeps and re-issues
until `wait_cap` is spent, and then stops rather than reconnecting - a new
socket would only ask the same overloaded server the same question. Catch it
if you want to back off on a human timescale, or let it surface as the
`TransientError` it is.

Note what is *not* retried. A request that changes something on the server -
`rm`, `mv`, `mkdir`, `truncate`, `chmod`, `write`, `close` - is sent at most
once. If the connection dies after the request left and before the reply came
back, the client cannot know whether the server did the work, so it reports
the failure instead of guessing; reads, stats and listings, which cost nothing
to repeat, are reissued on a fresh connection.

## Authentication failures

```python
except xrd.NoMechanismError as exc:
    print(exc)
    exc.offered    # ['gsi', 'unix'] - what the server said it would take
    exc.tried      # {'gsi': 'no proxy at /tmp/x509up_u1000', ...}
# no usable authentication mechanism (server offered: gsi, unix)
# [gsi: no proxy at /tmp/x509up_u1000; unix: refused]
```

One exception, every mechanism, and the reason each did not apply - rather
than a bare "authentication failed" that tells you nothing about which of the
six to fix.

## Unsupported operations

WebDAV has no `locate`, no `prepare`, no `statvfs`. Those raise
`UnsupportedError` naming the operation instead of returning something
invented, so the failure happens at the call rather than three functions later
when the invented answer turns out to be wrong.

## Checksums

```python
except xrd.ChecksumMismatchError as exc:
    exc.algorithm, exc.expected, exc.actual
```

Raised by a verified copy. It is a data-integrity signal, not an
authentication one - see [Security](security.md).
