# Changelog

## Unreleased

### Features

* add portable bfloat16 storage and inference across CPU, Metal, CUDA, and HIP
* allow eager and graph matmul callers to request a bfloat16 output dtype

## 0.18.4 (2026-11-16)


### Miscellaneous Chores

* release 0.18.4

## 0.18.3 (2026-11-16)


### Bug Fixes

* **metal:** order delegated MPS calls behind the async batch

## 0.18.2 (2026-10-03)


### Bug Fixes

* **cuda:** build without cuDNN and other optional vendor libraries

## 0.18.1 (2026-08-14)


### Performance Improvements

* batch asynchronous Metal dispatches

## 0.18.0 (2026-08-14)


### Features

* add an explicit eager async submission API (Tilework.submit)
* defer intermediate frees in the graph so :nn and reductions stop waiting, closes #86

## 0.17.4 (2026-08-13)


### Bug Fixes

* close native resource leak paths

## 0.17.3 (2026-08-13)


### Bug Fixes

* enforce graph buffer ownership

## 0.17.2 (2026-08-13)


### Bug Fixes

* take delegated conv2d outputs from the buffer pool

## 0.17.1 (2026-08-12)


### Bug Fixes

* escape quotes in extconf.rb Makefile CFLAGS

## 0.17.0 (2026-08-11)


### Features

* add bfloat16 inference support

## 0.16.3 (2026-08-10)


### Bug Fixes

* **ci:** install libyaml for HIP build

## 0.16.2 (2026-08-10)


### Bug Fixes

* **ci:** package native HIP bridge

## 0.16.1 (2026-08-10)


### Bug Fixes

* **release:** trigger HIP package publication

## 0.16.0 (2026-08-10)


### Features

* **hip:** prepare HIP backend release

## 0.15.0 (2026-08-09)


### Features

* **nn:** add tiled flash attention

## 0.14.1 (2026-08-09)


### Bug Fixes

* **metal:** drain autoreleased buffer allocations
* **metal:** reclaim pages dirtied by host writes

## 0.14.0 (2026-08-06)


### Features

* add top-level Rakefile delegating to core/metal/cuda
* **conv:** add dilation and groups to 2D convolution

## 0.13.0 (2026-08-06)


### Features

* **graph:** checkpointing, and the measurement that closes activation offload (#22 Step 5)
* **graph:** gradients for gather and conv2d (#22 Step 2)
* **graph:** hand-written vjps for the row-wise NN ops (#22 Step 2)
* **graph:** optimizers and a deterministic scatter-add (#22 Steps 3 and 4)
* **graph:** reverse mode as a graph-to-graph transform (#22 Step 1)
* **nn:** add mse and cross_entropy losses (#22 Step 0)


### Bug Fixes

* **graph:** compare checkpointing relative to magnitude, and recompute random draws
* **graph:** row fusion absorbed a reduction into two clusters

## 0.12.0 (2026-08-06)


### Features

* page weights against a VRAM budget in the graph layer
* prefetch weights ahead of need instead of fetching on demand
* ride eviction fences along with nodes that already synchronize

## 0.11.0 (2026-08-05)


### Features

* make conv2d a Tier 1 graph op and answer requirements per op

## 0.10.0 (2026-08-05)


### Features

* add cast, where, iota and reshape to the graph IR
* add permute, slice, gather, pad and axis-aware concat
* broadcast operands and reduce along an axis in the graph
* fold broadcasts into the consumer's index expression
* fuse row-wise reductions into the kernel that consumes them
* give high-level ops a decomposition into primitives

## 0.9.0 (2026-08-05)


### Features

* draw random values inside a graph from the plan's seed
* fuse connected elementwise nodes into one kernel
* write graph state in place instead of copying into it


### Bug Fixes

* allocate delegated GEMM outputs through the execution context
* pass buffer-view offsets explicitly through the GEMM bridge ABI

## 0.8.0 (2026-08-05)


### Features

* accept graph tensors in NN so blocks read the same eager or recorded
* type float16, batched matmul and concat in the graph layer

## 0.7.0 (2026-08-04)


### Features

* add a graph layer with versioned mutable state
* launch graph plans asynchronously and reuse their buffers
* let reductions stay on the device with device: true
* wire reductions into the graph layer

## 0.6.0 (2026-08-04)


### Features

* add a portable RNG so the same seed gives the same sequence on every backend, closes #21


### Bug Fixes

* synchronize before host writes into device buffers, closes #13
* write the fp16 cast kernel in the dialect so it compiles on Metal, closes #16

## 0.5.1 (2026-08-03)


### Bug Fixes

* repair Capabilities#merge arity and publish untagged gems
* repair Capabilities#merge arity and publish untagged gems

## 0.5.0 (2026-08-03)


### ⚠ BREAKING CHANGES

* `Tilework.capabilities` returns a `Tilework::Capabilities` value object instead of a Hash. `caps[:key]`, `#to_h`, `#fetch` and `#key?` behave as before, but Hash-only methods (`#keys`, `#each`, `#dig`) are gone, and reading a key outside the contract now raises KeyError instead of returning nil.

### Features

* declare the capability contract as a schema

## 0.4.2 (2026-08-03)


### Bug Fixes

* pad metal random generation to a multiple of four
* split metal conv2d over 65536 output channels

## 0.4.1 (2026-07-31)


### Bug Fixes

* wrap Metal bridge MPS calls in autoreleasepool to stop footprint growth

## 0.4.0 (2026-07-31)


### Features

* add MPS-backed random generation to the metal backend
* add MPSGraph-backed 2D convolution to the metal backend
* add MPSGraph-backed FFT to the metal backend

## 0.3.0 (2026-07-31)


### Features

* add MPS-backed GEMM to the metal backend, enable float16 storage

## 0.2.1 (2026-07-31)


### Bug Fixes

* bump version.rb to 0.2.0 and tolerate duplicate GH Packages pushes
* let release-please create the tag and GitHub Release itself
* run publish as a job in the release-please run, not via tag push
* use gh release upload instead of action-gh-release

## 0.2.0 (2026-07-31)


### Features

* require Conventional Commits for release-please


### Bug Fixes

* scope release-please to the repo root, not core/
