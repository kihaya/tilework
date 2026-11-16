// tilework compat header (HIP) — compiled by HIPRTC.
// Same macro set as compat/cuda.h; only the fp16 / bf16 headers and type
// names differ. See compat/cuda.h for the reasoning behind each macro.

#define GPU_KERNEL  extern "C" __global__

#define GPU_UINT unsigned int
#define GPU_FUNC __device__ inline

#include <hip/hip_fp16.h>
#define GPU_HALF __half
#include <hip/hip_bf16.h>
#define GPU_BFLOAT __hip_bfloat16
#define GPU_BFLOAT_TO_FLOAT(x) __bfloat162float(x)
#define GPU_FLOAT_TO_BFLOAT(x) __float2bfloat16(x)

#define GPU_ARGS

#define GLOBAL_ID_X ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define GLOBAL_ID_Y ((int)(blockIdx.y * blockDim.y + threadIdx.y))
#define LOCAL_ID_X  ((int)threadIdx.x)
#define GROUP_ID_X  ((int)blockIdx.x)
#define GPU_SHARED  __shared__
#define GPU_BARRIER __syncthreads()

#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif

template <typename T>
__device__ __forceinline__ T clamp(T x, T lo, T hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

#define GPU_ERF(x) erff(x)
