<img src="brand/icon/tilework-icon-macos.png" width="128" alt="Tilework icon">

# tilework

CUDA-style GPU computing from Ruby. Write kernels in a CUDA-flavored dialect
(or as plain Ruby blocks), launch them with a small `Buffer` / `compile` /
`launch` API, and run the same code on Apple Silicon, NVIDIA, AMD, or a pure
Ruby reference backend.

## Backends

Each backend is its own gem in this monorepo. The core gem has no native
extension.

| Backend | Gem | Directory | Runs on |
|---|---|---|---|
| `:cpu` | `tilework` | [`core/`](core/) | Anywhere (pure Ruby reference implementation) |
| `:metal` | `tilework-metal` | [`metal/`](metal/) | Apple Silicon (Swift + Metal) |
| `:cuda` | `tilework-cuda` | [`cuda/`](cuda/) | NVIDIA GPU + CUDA Toolkit (Driver API + NVRTC) |
| `:hip` | `tilework-hip` | [`hip/`](hip/) | AMD GPU + ROCm (HIP Runtime + HIPRTC) |

Where available, the GPU backends delegate to vendor libraries: cuBLAS, cuDNN,
cuRAND and cuFFT on `:cuda`; hipBLAS, MIOpen, hipRAND and hipFFT on `:hip`.

## Requirements

- Ruby >= 3.1
- For a GPU backend, its toolchain at install time (the native bridge is
  built from source):
  - `:metal` — macOS on Apple Silicon, Xcode Command Line Tools
    (`xcode-select --install`)
  - `:cuda` — CUDA Toolkit (`CUDA_HOME`, default `/usr/local/cuda`)
  - `:hip` — ROCm (`ROCM_HOME`, default `/opt/rocm`)

If the toolchain is missing, the extension build is skipped rather than
failing, so the gem still installs. Selecting that backend then raises
`Tilework::DeviceError`.

## Installation

```sh
gem install tilework            # core and the :cpu backend
gem install tilework-metal      # or tilework-cuda / tilework-hip
```

A backend gem compiles its native extension when it is installed, so the
toolchain for that backend has to be present at that point.

From a checkout:

```sh
cd metal                    # or cuda / hip
bundle install
bundle exec rake compile    # builds lib/tilework/<backend>/bridge.<dlext>
```

Each backend gem's `Gemfile` resolves core through `path: "../core"`.

## Quick start

```ruby
require "tilework"

Tilework.backend = :cpu   # or :metal / :cuda / :hip

add = Tilework.kernel(out: 2) do |a, b, c|
  i = global_id_x
  c[i] = a[i] + b[i]
end

a = Tilework[1.0, 2.0, 3.0]
b = Tilework[4.0, 5.0, 6.0]
add.(a, b).to_a   # => [5.0, 7.0, 9.0]
```

The block is transpiled into a GPU kernel. `out: 2` marks `c` as the result
buffer, so it is allocated for you and returned. Selecting a backend loads its
gem on demand.

## Ruby-first API

Most code never needs a kernel string:

```ruby
require "tilework"
require "tilework/core_ext"   # optional: with_gpu and Array#to_gpu

Tilework.backend = :cpu

vec_add = Tilework.kernel(out: 2) do |a, b, c, n|
  i = global_id_x
  c[i] = a[i] + b[i] if i < n
end

result = with_gpu(xs, ys) do |x, y|   # Arrays in
  c = vec_add.(x, y, x.length)         # output allocated, grid/block computed
  (c * 2.0).clamp(0.0, 1.5)
end                                    # Array out; buffers freed here
```

- `with_gpu(*inputs) { ... }` (also `Tilework.run`) turns Arrays into buffers,
  turns returned buffers back into Arrays, and frees every buffer allocated
  in the block, even on exceptions.
- `Tilework.kernel { ... }` transpiles a Ruby block into the kernel dialect.
  Types are specialized from the call arguments and cached. `out: i` marks
  parameter `i` as the result buffer, which is then allocated for you.
  `out_dtype:` and `over:` override the defaults.
- 2D-shaped buffers launch a 2D grid automatically and support `buf[r, c]`,
  `buf.rows` and `buf.cols` inside the block.

The DSL supports locals, arithmetic and comparison operators, `if` / `unless`
/ ternary / `while`, `n.times` and range loops, `global_id_x` and friends,
`Math.exp/log/sqrt/sin/cos/tanh/pow`, inlined constants, and `unroll` for
transpile-time expansion. Anything else raises `Tilework::CompileError` with a
line number.

## Layers

From least to most control:

1. **Array operations** — no kernels at all.

   ```ruby
   a = Tilework[1.0, 2.0, 3.0]
   b = Tilework[4.0, 5.0, 6.0]
   (a + b).to_a          # => [5.0, 7.0, 9.0]
   (a * 2.0).sum         # => 12.0

   m = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
   n = Tilework::Buffer.from_matrix([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
   m.matmul(n).to_nested_a  # => [[58.0, 64.0], [139.0, 154.0]]
   ```

   Arithmetic, math functions, comparisons (`eq` / `ne` / `<` ...),
   `maximum` / `minimum` / `clamp`, reductions (`sum` / `min` / `max` / `mean`
   / `argmax` / `argmin`) and `matmul`.

2. **Ruby kernel DSL** — `Tilework.kernel { ... }`, shown in Quick start.

3. **Dialect kernel strings** — CUDA-style source, compiled at runtime. Not
   supported on `:cpu`.

   ```ruby
   src = <<~SRC
     GPU_KERNEL void vec_add(GPU_ARGS,
                             device const float* a,
                             device const float* b,
                             device float* c,
                             constant int& n) {
       int i = GLOBAL_ID_X;
       if (i < n) c[i] = a[i] + b[i];
     }
   SRC

   n = a.length
   c = Tilework::Buffer.new(:float32, n)
   k = Tilework.compile(src, name: "vec_add")
   k.launch(grid: (n / 256.0).ceil, block: 256, args: [a, b, c, n])
   ```

   `grid` is the number of blocks and `block` is the threads per block — the
   same meaning as CUDA's `<<<grid, block>>>`.

4. **Native source** — `Tilework.compile(src, name:, raw: true)` passes MSL /
   CUDA C++ / HIP C++ through untouched, for anything the dialect does not
   cover (warp/SIMD-group ops, atomics, ...).

On top of these sit built-in libraries:

- `Tilework::NN` — `softmax`, `layernorm`, `rmsnorm`, `groupnorm`,
  activations, `transpose`, `multi_head_attention`, `mse`, `cross_entropy`
- `Tilework::Conv`, `Tilework::FFT`, random numbers (Philox)
- `Tilework::Offload` — page buffers between VRAM and disk or host memory

## Kernel dialect

A per-backend compat header (`core/lib/tilework/compat/<backend>.h`) is
prepended to the source, so one kernel body compiles everywhere.

| Macro | CUDA equivalent |
|---|---|
| `GPU_KERNEL` | `__global__` |
| `GPU_ARGS` | (implicit thread indices) |
| `GLOBAL_ID_X` / `GLOBAL_ID_Y` | `blockIdx * blockDim + threadIdx` |
| `LOCAL_ID_X` | `threadIdx.x` |
| `GROUP_ID_X` | `blockIdx.x` |
| `GPU_SHARED` | `__shared__` |
| `GPU_BARRIER` | `__syncthreads()` |
| `GPU_HALF` / `GPU_BFLOAT` | `__half` / `__nv_bfloat16` |

Metal address-space qualifiers (`device`, `constant`) are written in the
signature and removed automatically on CUDA and HIP.

## Graphs

`Tilework::Graph` is an opt-in layer that records operations first and runs
them later as a compiled plan:

```ruby
plan = Tilework::Graph.build do |g|
  x = g.input(:x, [2, 2])
  g.output(Tilework::NN.softmax(x * 2.0 + 1.0))
end.compile

y = plan.run(x: Tilework::Buffer.from_matrix([[1.0, 2.0], [3.0, 4.0]]))
```

Plans launch asynchronously, reuse intermediate buffers, and fuse elementwise
chains into single kernels. The graph layer also provides mutable state (KV
caches, optimizer moments), reverse-mode autograd (`g.grad`), SGD / Adam,
checkpointing, and automatic weight offload under a VRAM budget
(`compile(vram: bytes)`).

For eager code, `Tilework.submit { ... }` queues every launch in the block and
waits once at the end:

```ruby
out = Tilework.submit { (a * 2.0 + b) }.wait
```

## API notes

- `Kernel#launch` is synchronous. `launch_async` and `Tilework.synchronize`
  are also available, and `Buffer#to_a` always waits for pending work.
- Scalars bind as `float` (Ruby `Float`) or `int` (Ruby `Integer`). Up to 31
  kernel arguments.
- Storage dtypes: `float32` and `int32` everywhere; `float16` on the GPU
  backends; `bfloat16` on all backends. Check `Tilework.capabilities` at
  runtime.
- `Tilework::CompileError` carries the full compiler output. Launch validation
  raises `Tilework::LaunchError` before dispatch.
- `Buffer#free` releases device memory immediately. Free temporaries
  explicitly, or use `with_gpu`; do not rely on GC to reclaim VRAM in time.

## The `:cpu` backend

```ruby
Tilework.backend = :cpu
```

Runs the same DSL kernels, Array operations, `NN` and graphs in plain Ruby, with
no GPU required. It is the reference implementation that the GPU backends are
checked against, not a fast path. It does not accept dialect kernel strings.

## Testing

Run rake inside each gem directory:

```sh
cd core  && bundle exec rake test        # unit tests + :cpu, no GPU needed
cd metal && bundle exec rake test:metal  # Apple Silicon
cd cuda  && bundle exec rake test:cuda   # NVIDIA GPU + CUDA Toolkit
cd hip   && bundle exec rake test:hip    # AMD GPU + ROCm
```

Each backend is verified against `:cpu` by a shared conformance suite
(`Tilework::Conformance`). Device tests are skipped unless `GPU_TESTS=1`
(the `test:<backend>` tasks set it). CI runs only the `:cpu` suite; GPU
backends are tested by hand on real hardware.

## Documentation

- [`hip/README.md`](hip/README.md) — HIP build and runtime details

## License

MIT
