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
```

## `xrd-cp`

```console
$ xrd-cp /tmp/f.root root://host//store/f.root
$ xrd-cp root://host//store/f.root /scratch/
$ xrd-cp -r /tmp/results davs://dav.example.org/store/results
$ xrd-cp --tpc root://a//store/f.root root://b//store/f.root
$ xrd-cp -n /tmp/f.root root://host//store/f.root      # never overwrite
$ xrd-cp --verify -a crc32c /tmp/f.root root://host//store/f.root
$ xrd-cp --chunk-size 8M --progress root://host//store/big.root /scratch/
```

Several sources are allowed when the destination is a directory. Progress is
shown on a tty and suppressed otherwise; `--progress` and `--no-progress`
override that either way.

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
