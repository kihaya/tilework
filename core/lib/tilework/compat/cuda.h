// tilework compat header (CUDA) — compiled by NVRTC.
// Expands the CUDA-flavored dialect's macro set into CUDA C++. This set is the
// de facto kernel ABI; keep it 1:1 with compat/metal.h (add a macro only if
// its semantics can be guaranteed identical on every backend).
//
// NVRTC provides built-ins like blockIdx / threadIdx, so no include is needed.

// extern "C" is required: with C++ name mangling cuModuleGetFunction(name)
// would not match the dialect's name: and fail.
#define GPU_KERNEL  extern "C" __global__

// 32-bit unsigned integer, 1:1 with MSL's uint (CUDA has no `uint` spelling,
// so it expands to unsigned int).
#define GPU_UINT unsigned int
// Device-callable helper. Without the annotation it would be a host function
// that __global__ cannot call; this is the counterpart of Metal's inline.
#define GPU_FUNC __device__ inline

// Half precision, **as a storage type only** — arithmetic is done after
// widening to float (accumulation is always FP32). __half has float
// conversion operators from __CUDA_ARCH__ >= 530, so reads work with a
// (float) cast and writes through implicit conversion on assignment.
#include <cuda_fp16.h>
#define GPU_HALF __half
#include <cuda_bf16.h>
#define GPU_BFLOAT __nv_bfloat16
#define GPU_BFLOAT_TO_FLOAT(x) __bfloat162float(x)
#define GPU_FLOAT_TO_BFLOAT(x) __float2bfloat16(x)

// CUDA has no counterpart to Metal's implicit arguments
// (thread_position_in_grid etc.); built-in variables are readable anywhere,
// so the arguments expand to nothing. Dialect::CUDA_REWRITES removes
// `GPU_ARGS,` together with its comma, so only bare uses like `f(GPU_ARGS)`
// reach this macro.
#define GPU_ARGS

#define GLOBAL_ID_X ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define GLOBAL_ID_Y ((int)(blockIdx.y * blockDim.y + threadIdx.y))
#define LOCAL_ID_X  ((int)threadIdx.x)
#define GROUP_ID_X  ((int)blockIdx.x)
#define GPU_SHARED  __shared__
// Metal expands this to threadgroup_barrier(mem_threadgroup) (same semantics)
#define GPU_BARRIER __syncthreads()

// MSL address-space qualifiers (device / constant) are **not** erased here.
// `#define device` is tempting, but CUDA's host_defines.h defines
// `__device__` as `__location__(device)`, so an empty `device` macro would
// collapse __device__ into __location__() and leave every later function
// without an execution-space annotation, failing compilation. Dialect::
// CUDA_REWRITES strips the qualifiers from the dialect source only (CUDA's
// own headers are untouched, so there is no side effect).

#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif

// MSL has clamp but CUDA does not, so provide it. min/max/exp/log/sqrt/sin/
// cos/tanh/floor/ceil/round/pow/abs/fabs resolve to CUDA built-ins.
template <typename T>
__device__ __forceinline__ T clamp(T x, T lo, T hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// CUDA has erff natively. MSL does not, so compat/metal.h approximates it
// (GPU_ERF is 1:1 across the headers).
#define GPU_ERF(x) erff(x)
