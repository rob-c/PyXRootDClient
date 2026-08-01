# Command line

Two commands, `xrd-fs` and `xrd-cp`. Both take whole URLs, both understand
`--json`, and both use the same three exit codes: `0` success, `1` a runtime
failure, `2` a usage error.

Common options on every subcommand:

| Option | Meaning |
| --- | --- |
| `--json` | machine-readable output, so a script never parses columns |
| `-q`, `--quiet` | say nothing on success |
| `-v`, `-vv`, `-vvv` | warnings, info, the wire itself |
| `--token TOKEN` | bearer token to present |
| `--user NAME` | username to authenticate as |
| `--no-verify-tls` | do not verify the server certificate |
| `--prompt` | ask for missing credentials even when this is not a terminal |
| `--no-prompt` | never ask for credentials; fail instead |
| `--config FILE` | settings file to read instead of the usual places |
| `--alias NAME` | the `[alias NAME]` section of that file to overlay |

See [Configuration](config.md#the-settings-file) for the file itself.

## `xrd-fs`

```console
$ xrd-fs ls -l root://eos.example.org//store/user/me
$ xrd-fs ls -R root://host//store/run7
$ xrd-fs stat --json davs://dav.example.org/store/f.root
$ xrd-fs cat root://host//store/notes.txt
$ xrd-fs checksum -a adler32 root://host//store/f.root
$ xrd-fs mkdir -p root://host//store/a/b/c
$ xrd-fs rm -r -f root://host//store/scratch
$ xrd-fs mv root://host//store/a.root root://host//store/b.root
$ xrd-fs touch root://host//store/marker
$ xrd-fs df root://host//store
$ xrd-fs locate --deep root://host//store/f.root
$ xrd-fs ping root://host
$ xrd-fs query root://host version role sitename
$ xrd-fs xattr --set user.experiment=atlas root://host//store/f.root
$ xrd-fs tail -n 20 root://host//store/log.txt
$ xrd-fs tail -f root://host//store/running.log        # keep printing appends
$ xrd-fs du root://host//store/run7
$ xrd-fs chmod 640 root://host//store/f.root
$ xrd-fs truncate -s 0 root://host//store/scratch.bin
$ xrd-fs prepare root://host//store/f.root             # stage it onto disk
$ xrd-fs prepare --evict root://host//store/f.root     # drop the cached copy
$ xrd-fs prepare --status prep-0001 root://host//store/f.root   # is it there yet?
$ xrd-fs ln -s root://host//store/f.root root://host//store/latest
$ xrd-fs readlink root://host//store/latest
```

`tail -f` polls the size every `--interval` seconds and prints what appeared;
a file that shrinks is a different file, and it stops. `du` counts from the
listing, one request per directory, and falls back to one stat per entry when
the server lists without sizes. `ln`, `ln -s` and `readlink` are a **vendor
extension** - stock XRootD answers `kXR_Unsupported`, and HTTP has no verb for
them at all; see [Interoperability](interop.md).

## `xrd-cp`

```console
$ xrd-cp /tmp/f.root root://host//store/f.root
$ xrd-cp root://host//store/f.root /scratch/
$ xrd-cp -r /tmp/results davs://dav.example.org/store/results
$ xrd-cp --tpc root://a//store/f.root root://b//store/f.root
$ xrd-cp -n /tmp/f.root root://host//store/f.root      # never overwrite
$ xrd-cp --verify -a crc32c /tmp/f.root root://host//store/f.root
$ xrd-cp --chunk-size 8M --progress root://host//store/big.root /scratch/
$ xrd-cp -r --exclude '*.log' /tmp/results root://host//store/results
$ xrd-cp -r --include '*.root' --sync size /tmp/results root://host//store/results
$ xrd-cp -r --delete /tmp/results root://host//store/results
$ xrd-cp -r --dry-run /tmp/results root://host//store/results
$ xrd-cp -r --parallel 8 /tmp/many-small root://host//store/many-small
$ xrd-cp --remove-source /tmp/f.root root://host//store/f.root   # a move
$ xrd-cp -c root://host//store/big.root /scratch/big.root   # carry on, do not restart
```

Several sources are allowed when the destination is a directory. Progress is
shown on a tty and suppressed otherwise; `--progress` and `--no-progress`
override that either way.

`-c`, `--continue` keeps whatever is already at the destination and starts
from the end of it; the JSON record carries `resumed_at` so a script can see
how much was skipped. It works on a tree too, and refuses to combine with
`--tpc` or `-n`, which each forbid the partial destination it needs. See
[Resuming](copying.md#resuming) for what it does to verification.

For a tree:

| Option | Meaning |
| --- | --- |
| `--include PATTERN` | only paths matching one of these travel (repeatable) |
| `--exclude PATTERN` | paths matching one of these do not (wins over include) |
| `--sync {size,mtime,checksum}` | skip what is already there, judged that way |
| `--delete` | remove what is in the target but not in the source |
| `--parallel N` | copy N files at once (default 1) |
| `--dry-run` | print what would happen, transfer nothing |
| `--remove-source` | delete the source once the copy is verified - a move |

Patterns are `fnmatch` patterns matched against the path relative to the
source root, so `sub/*.root` means what it looks like. `--sync checksum` asks
both endpoints for a digest, which is exact and not free; `--sync size` is one
stat each. `--delete` never removes anything an `--exclude` hid, because it
was never a candidate in the first place.

`--parallel` is files in flight, not requests: a tree of small files is
round-trip-bound, so copying several at once is what makes it faster, while a
single large file is already spread over several connections by itself. It
needs `-r`, since without a tree there is nothing to run in parallel.

`cp` semantics decide the destination: a target that already exists as a
directory is copied *into*, so a second run of `cp -r tree /dest` writes
`/dest/tree/tree`. Give the target a trailing slash to say "into this" every
time, which is what makes `--sync` and `--delete` idempotent.

## Scripting with `--json`

```console
$ xrd-fs stat --json root://host//store/f.root | jq .size
$ xrd-fs ls --json root://host//store | jq -r '.[] | select(.is_dir) | .name'
$ xrd-cp --json /tmp/f.root root://host//store/f.root | jq .rate
```

Errors are reported as a sentence on stderr, not a traceback - a missing file
is `no such file or directory: /store/f.root`, and the exit code is `1`.

## The URL slash

An XRootD URL needs a **doubled** slash before an absolute path:

```console
$ xrd-fs ls root://host//store/user/me     # /store/user/me
$ xrd-fs ls root://host/store/user/me      # relative to the server's export
```

This is not this client's invention - stock `xrdcp` refuses the single-slash
form outright - but it catches everyone once.
