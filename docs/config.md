# Configuration

`Config` is a frozen dataclass. Build one, pass it to whatever you are opening,
and derive variants with `evolve`.

```python
cfg = xrd.Config(request_timeout=60.0)
patient = cfg.evolve(request_timeout=600.0)

xrd.open("root://host//store/f.root", config=cfg)
xrd.FileSystem("root://host", config=cfg)
xrd.copy(src, dst, config=cfg)
```

Frozen because a configuration is shared by every session it reaches, and a
setting that changes underneath a live connection is a bug you find in
production. `evolve` returns a new one.

## Timeouts and retries

| Field | Default | Environment | Meaning |
| --- | --- | --- | --- |
| `connect_timeout` | `30.0` | `XRD_CONNECTIONWINDOW` | TCP connect and login |
| `request_timeout` | `300.0` | `XRD_REQUESTTIMEOUT` | one request/response |
| `stream_timeout` | `60.0` | `XRD_STREAMTIMEOUT` | idle socket before a keepalive |
| `connect_retries` | `3` | `XRD_CONNECTIONRETRY` | reconnection attempts |
| `retry_backoff` | `0.5` | `XRD_STREAMERRORWINDOW` | first backoff, then doubling |
| `redirect_limit` | `16` | `XRD_REDIRECTLIMIT` | redirects before giving up |
| `wait_cap` | `600.0` | | ceiling on a server-requested wait |
| `keepalive_interval` | `60.0` | | seconds between `kXR_ping`s |

## Transfers

| Field | Default | Environment |
| --- | --- | --- |
| `chunk_size` | 4 MiB | `XRD_CPCHUNKSIZE` |
| `readahead` | 1 MiB | `XRD_READAHEAD` |
| `parallel_chunks` | `4` | `XRD_CPPARALLELCHUNKS` |

## Pooling

| Field | Default | Environment |
| --- | --- | --- |
| `pool_size` | `8` | `XRD_POOLSIZE` |
| `pool_idle_ttl` | `120.0` | |

!!! warning "Reserved"
    These two are read from the environment and carried on the `Config`, but
    nothing consumes them yet - a cross-instance session pool is future work.
    Today one `FileSystem` (or one `fsspec` instance) owns one multiplexed
    connection and reuses it for every call, which is where the win already
    is; two `FileSystem` objects pointed at the same host log in twice.

## Security

| Field | Default | Environment |
| --- | --- | --- |
| `token` | `None` | (`$BEARER_TOKEN` is discovered separately) |
| `token_file` | `None` | `BEARER_TOKEN_FILE` |
| `keytab` | `None` | `XrdSecSSSKT`, `XrdSecsssKT` |
| `proxy` | `None` | `X509_USER_PROXY` |
| `ca_path` | `None` | `X509_CERT_DIR` |
| `ca_file` | `None` | `SSL_CERT_FILE` |
| `auth_order` | `("gsi", "ztn", "krb5", "sss", "unix", "host")` | |
| `verify_tls` | `True` | |
| `require_tls` | `False` | |

See [Authentication](auth.md) for what each mechanism looks for.

## Behaviour

| Field | Default | Meaning |
| --- | --- | --- |
| `recover_handles` | `True` | silently re-open a read-only file whose data server vanished mid-read |
| `verify_checksums` | `True` | compare checksums after a copy |
| `preferred_checksum` | `"adler32"` | algorithm asked for first |

`recover_handles=False` turns a lost data server into a `TransientError` at the
call that hit it, which is what you want when your job would rather fail than
re-read.

## Username

```python
Config(username="atlasprd")
```

Defaults to `$XRD_USER`, `$USER`, `$LOGNAME`, then `getpass.getuser()`, then
`"nobody"` - so it is never blank even in a container without a passwd entry.

## Secrets never print

```python
>>> xrd.Config(token="eyJhbGciOi...")
Config(username='me', ..., token='<redacted>', ...)
```

The `repr` redacts, and so does every log record - see [Security](security.md).

## Environment only

If you set nothing, the defaults above already read the `XRD_*` variables the
official client uses, so an existing site environment keeps working unchanged.
Environment values are read when the `Config` is constructed, not when it is
used.
