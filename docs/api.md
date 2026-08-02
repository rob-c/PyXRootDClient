# API reference

Generated from the source. Everything below is importable from the top-level
`xrd` package unless the heading says otherwise.

## Entry points

::: xrd.open

::: xrd.FileSystem

::: xrd.File

::: xrd.Checkpoint

::: xrd.XRootDPath

## Copying

::: xrd.copy

::: xrd.copy_tree

::: xrd.third_party

::: xrd.CopyResult

::: xrd.SyncMode

## Configuration

::: xrd.Config

::: xrd.find_config_file

::: xrd.configure

::: xrd.current

::: xrd.override

## URLs

::: xrd.parse

::: xrd.XRootDURL

## Values

::: xrd.StatInfo

::: xrd.DirEntry

::: xrd.ChecksumInfo

::: xrd.CheckpointInfo

::: xrd.LocationInfo

::: xrd.PageResult

::: xrd.PrepareStatus

::: xrd.ProtocolInfo

::: xrd.ReadRange

::: xrd.SpaceInfo

::: xrd.VFSInfo

::: xrd.WriteChunk

## Flags

::: xrd.OpenFlags

::: xrd.Access

::: xrd.MkDirFlags

::: xrd.DirListFlags

::: xrd.QueryCode

::: xrd.StatInfoFlags

::: xrd.LocateFlags

::: xrd.PrepareFlags

## Authentication

::: xrd.auth.select

::: xrd.auth.require

::: xrd.auth.supply

::: xrd.auth.prompt.Ask

::: xrd.auth.prompt.ask_on_terminal

::: xrd.auth.prompt.forget

## Errors

::: xrd.errors

## Asynchronous

::: xrd.aio

## HTTP and WebDAV

::: xrd.http.third_party

::: xrd.http.macaroon

::: xrd.http.propfind

::: xrd.http.digest

::: xrd.http.HTTPClient

## S3

::: xrd.s3.S3FileSystem

::: xrd.s3.open_s3

::: xrd.s3.Credentials

::: xrd.s3.sign

## ROOT files

::: xrd.root.open_root

::: xrd.root.ROOTFile

::: xrd.root.Directory

::: xrd.root.TTree

::: xrd.root.Branch

::: xrd.root.Group

::: xrd.root.Jagged

::: xrd.root.Histogram

::: xrd.root.Axis

::: xrd.root.Graph

### Writing

::: xrd.root.create

::: xrd.root.WritableFile

::: xrd.root.WritableTree

### Datasets

::: xrd.root.datasets.convert

::: xrd.root.datasets.describe

::: xrd.root.datasets.Dataset

::: xrd.root.datasets.Images

::: xrd.root.datasets.CIFAR

::: xrd.root.datasets.Audio

::: xrd.root.datasets.Matrix

::: xrd.root.datasets.Table

::: xrd.root.datasets.read_idx

::: xrd.root.datasets.read_table

::: xrd.root.datasets.read_arff

::: xrd.root.datasets.read_xlsx

::: xrd.root.datasets.fetch

::: xrd.root.mnist.convert

### Into a framework

::: xrd.root.ml.to_tensor

::: xrd.root.ml.iter_tensors

::: xrd.root.ml.dataset

::: xrd.root.ml.to_tf_tensor

::: xrd.root.ml.tf_dataset

::: xrd.root.ml.numeric

## Diagnosing

::: xrd.diagnose

::: xrd.Report

::: xrd.Check

## Testing

::: xrd.testing.FakeServer

::: xrd.testing.from_directory

::: xrd.testing.FakeDAVServer

::: xrd.testing.FakeS3Server

::: xrd.testing.FaultProxy
