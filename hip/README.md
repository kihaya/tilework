# tilework-hip

`tilework-hip` provides Tilework's AMD GPU backend through the HIP Runtime and
HIPRTC. Select it after requiring the gem's backend entry point:

```ruby
require "tilework"
require "tilework/backend/hip"
Tilework.backend = :hip
```

## Build

The initial qualification tuple is ROCm 7.2.0.70200, Radeon RX 9070
(`gfx1201`), Ubuntu 24.04 under WSL2, and Ruby 3.2.3. It is a recorded test
environment, not a product allow-list.

ROCm is optional at gem-install time. Without HIP Runtime and HIPRTC headers
and shared libraries beneath `ROCM_HOME` (default `/opt/rocm`), `extconf.rb`
creates a no-op Makefile so CPU-only, CUDA-only, and macOS installations remain
installable. Selecting `:hip` without the built extension raises an actionable
`Tilework::DeviceError`.

On a ROCm host install the matching Ruby development headers, then build:

```sh
sudo apt install ruby3.2-dev
cd hip
rake clean && rake compile
GPU_TESTS=1 GPU_BACKEND=hip rake test:hip
```

The base bridge needs HIP Runtime and HIPRTC. hipBLAS, hipRAND, hipFFT, and
MIOpen are detected independently; each capability remains unavailable until
its native execution probe succeeds.

## Runtime behavior

`TILEWORK_DEVICE` is a non-negative index into the process-visible HIP device
list. AMD visibility variables (for example `HIP_VISIBLE_DEVICES`) determine
that list before Tilework runs; `TILEWORK_DEVICE` indexes the resulting list.
An invalid value raises an error; it never falls back to device zero. Selection
is immutable once the native bridge has initialized.
HIPRTC targets the selected device's runtime-probed `gcnArchName`. HIP never
reports a CUDA compute capability and never enables TF32.

Inspect the verified runtime facts rather than inferring them from a GPU name:

```ruby
Tilework.backend = :hip
p Tilework::HIPBridge.device_info
# => { name: "AMD Radeon RX 9070", architecture: "gfx1201", index: 0 }

p Tilework.capabilities.to_h.slice(:warp_size, :memory_budget_bytes, :gemm_library,
                                   :rng, :fft, :conv, :conv_backward, :tensor_core)
# => { warp_size: 32, memory_budget_bytes: ..., gemm_library: :hipblas,
#      rng: :hiprand, fft: :hipfft, conv: :miopen, conv_backward: true,
#      tensor_core: true }
```

Optional-library values are `nil`/`false` unless their executable probes pass.

The HIP backend supports explicit volatile host offload:

```ruby
buffer.offload_to_host!
buffer.restore!
```

HostBacking owns pageable host memory through TypedData and uses a bounded 1 MiB
pinned staging chunk when available, with a synchronous pageable fallback when
pinned memory cannot be allocated. On the qualified HIP stack,
`host_backing_async: true` means `restore_async!` returns a completion token;
use `restore_async!(async: false)` to force the synchronous path. A buffer is
not resident until `token.wait` completes. Graph residency can select this
transport explicitly:

```ruby
plan = graph.compile(vram: vram_bytes, backing: :host, host_budget: host_bytes)
```

