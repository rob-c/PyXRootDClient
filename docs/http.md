# HTTP and WebDAV

HEP storage is spoken over `https://` and WebDAV as often as over `root://`.
The same three entry points cover both, dispatching on the URL scheme, so an
application changes a URL and nothing else.

```python
with xrd.open("davs://dav.example.org/store/f.root", "rb") as fh:
    header = fh.read(1024)

fs = xrd.FileSystem("https://dav.example.org")
for entry in fs.scandir("/store/user/me"):
    print(entry.name, entry.stat.st_size)

xrd.copy("root://a.example.org//store/f.root", "davs://b.example.org/store/f.root")
```

Schemes: `http`, `https`, `dav`, `davs`, `webdav`. Nothing here needs a wheel
- `http.client` and `xml.etree` do the work.

## What the WebDAV filesystem supports

| Operation | Method |
| --- | --- |
| `stat` | `PROPFIND` with `Depth: 0`, falling back to `HEAD` |
| `scandir`, `listdir`, `walk`, `glob` | `PROPFIND` with `Depth: 1` |
| `mkdir`, `makedirs` | `MKCOL` |
| `remove`, `rmdir`, `rmtree` | `DELETE` |
| `rename` | `MOVE` |
| `checksum` | RFC 3230 `Want-Digest` / `Digest` |
| `read_bytes`, `open` | `GET`, with `Range` for seeks |
| `write_bytes`, `open("wb")` | `PUT` |
| `third_party` | `COPY` with `Source:`/`Destination:` |

Operations with no HTTP equivalent - `locate`, `prepare`, `cancel_prepare`,
`query_config`, `query_stats`, `query_space`, `checksum_cancel`,
`set_property`, `appid`, `statvfs` - raise `UnsupportedError` naming the
operation rather than returning something invented. They are the XRootD
protocol talking to an XRootD server about itself; WebDAV has no vocabulary
for any of it, and a plausible-looking answer built out of `PROPFIND` would be
a worse outcome than a refusal.

## Ranged reads

`GET` with a `Range` header is how a seek is served, so an `xrd.open` over
`https://` is still a real seekable file object. Servers that ignore `Range`
are detected (a `200` where a `206` was asked for) and reported rather than
silently returning the whole file.

## Authentication

```python
Config(token="eyJ...")                # Authorization: Bearer ...
Config(proxy="/tmp/x509up_u1000")     # mutual TLS with the X.509 proxy
```

A bearer token goes in the `Authorization` header; an X.509 proxy is
presented as the client certificate for `davs://` and `https://` alike. The
same `Config` drives both protocols.

## Macaroons

```python
from xrd.http import macaroon

token = macaroon("davs://dav.example.org/store/user/me",
                 caveats=["activity:DOWNLOAD"], validity="PT10M")
xrd.copy("davs://dav.example.org/store/user/me/f.root", "/tmp/f.root",
         config=xrd.Config(token=token))
```

`validity` is an ISO 8601 duration, the spelling dCache and XRootD both use.
The result is a plain string on purpose: a macaroon is a bearer token, so it
goes wherever `Config.token` goes.

## Third-party copy

```python
xrd.third_party("davs://a.example.org/store/f.root",
                "davs://b.example.org/store/f.root")
```

`COPY` with a `Source:` header, the dialect FTS and Rucio speak, so the two
storage elements move the file between themselves. The outcome is in the
response body rather than the status line - a failed transfer still answers
`202 Accepted` - and this client reads through the performance markers to
that last line before returning. [Copying](copying.md#third-party-copy) has
the header set, the push mode, and how the far side's token travels.

## Lower-level pieces

```python
from xrd.http import HTTPClient, propfind, digest, open_http, status_code

client = HTTPClient(xrd.Config())
response = client.request("HEAD", url)
props = propfind(xrd.parse(url), depth=1, config=cfg)   # [(path, StatInfo)]
info = digest(url, "adler32", config=cfg)               # ChecksumInfo
status_code(403)                                        # the kXR_* code it means
```

These exist because WebDAV endpoints differ, and being able to send one
request and look at the answer beats guessing.
