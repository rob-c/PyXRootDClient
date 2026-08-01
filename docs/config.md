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

One `FileSystem` (or one `fsspec` instance) owns one multiplexed connection
and reuses it for every call. When it closes, the connection is not dropped:
it goes into a process-wide pool, and the next `FileSystem` opened on the same
server *as the same person* picks it up instead of repeating the handshake,
the TLS negotiation and the login. A script that opens a `FileSystem` per file
pays for one bring-up rather than a hundred.

```python
from xrd.session import SESSIONS

len(SESSIONS)      # connections being held open right now
SESSIONS.clear()   # end them all, politely
```

Reuse is deliberately narrow, because sharing an authenticated connection with
the wrong caller is an authentication bug:

- The endpoint must match, TLS included.
- Every credential-bearing setting must match - `username`, `token`,
  `token_file`, `keytab`, `proxy`, `ca_path`, `ca_file`, `auth_order`,
  `verify_tls`, `require_tls`, plus any user in the URL. They are compared as
  a SHA-256 digest so that no key of the pool's ever holds a token.
- A connection that failed under a live handle is discarded, never pooled.
- Only idle connections are shared. Two `FileSystem` objects open at once are
  two connections; pooling reuses what is finished with, and does not
  multiplex what is not.

`pool_size` is how many idle connections are kept per server, `pool_idle_ttl`
how long one may sit unused before it is closed rather than handed on -
protection against a server that has forgotten a connection the client still
believes in. `pool_size = 0` turns pooling off entirely.

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
| `prompt` | `None` (ask only at a terminal) | `XRD_PROMPT` |
| `prompter` | `None` (ask on the terminal) | |

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

## The settings file

An INI file, read by `Config.from_file` and by both commands via `--config` /
`--alias`:

```ini
[defaults]
username = atlasprd
request_timeout = 600
verify_checksums = true

[alias eos]
token_file = /run/user/1000/bt_u1000
preferred_checksum = adler32
```

```python
cfg = xrd.Config.from_file()                  # the usual places
cfg = xrd.Config.from_file(alias="eos")       # [defaults], then [alias eos]
cfg = xrd.Config.from_file("./job.ini")       # exactly this file
```

```console
$ xrd-fs ls --alias eos root://eos.example.org//store/user/me
$ xrd-cp --config ./job.ini /tmp/f.root root://host//store/f.root
```

`[defaults]` is applied first and the alias overlays it, so an alias only says
what it changes. Field names are the `Config` field names, with `-` accepted
for `_`; values are typed by the field, and booleans take `configparser`'s
vocabulary (`true`, `yes`, `on`, `1`). Anything on the command line beats the
file, and the file beats the environment.

Looked for in this order:

| Where | Note |
| --- | --- |
| `$XRD_CONFIG` | wins outright, and must exist - a typo there is an error |
| `~/.config/xrd/config.ini` | |
| `~/.xrdrc` | |

No file at all means the defaults, which is what an absent dotfile should
mean. An `--alias` that the file does not define is an error naming the
aliases it does, because a typo there would quietly connect as somebody else.

Two settings are refused in a file: `prompter`, which is a callable and cannot
be spelled in INI, and `token`, because a literal bearer token in a dotfile is
a secret in every backup of that dotfile - say `token_file` instead.

## Environment only

If you set nothing, the defaults above already read the `XRD_*` variables the
official client uses, so an existing site environment keeps working unchanged.
Environment values are read when the `Config` is constructed, not when it is
used.
