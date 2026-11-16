// tilework compat header (Metal)
// Expands the CUDA-flavored dialect's macro set into MSL. This set is the de
// facto kernel ABI; add a macro only if its semantics can be guaranteed
// identical on every backend.
#include <metal_stdlib>
using namespace metal;

// Half precision, as a storage type only (1:1 with cuda.h's GPU_HALF).
// MSL's half converts to and from float implicitly.
#define GPU_HALF    half
#define GPU_BFLOAT  bfloat
#define GPU_BFLOAT_TO_FLOAT(x) ((float)(x))
#define GPU_FLOAT_TO_BFLOAT(x) ((bfloat)(x))

// 32-bit unsigned integer (wrapping arithmetic). For computations that must
// give bit-identical answers on every backend, such as counter-based RNG.
#define GPU_UINT    uint
// Device-callable helper. CUDA needs an execution-space annotation (without it
// the function is host-only and __global__ cannot call it); this absorbs that.
#define GPU_FUNC    inline

#define GPU_KERNEL  kernel
#define GPU_ARGS    uint3 __gid [[thread_position_in_grid]], \
                    uint3 __lid [[thread_position_in_threadgroup]], \
                    uint3 __bid [[threadgroup_position_in_grid]]
#define GLOBAL_ID_X ((int)__gid.x)
#define GLOBAL_ID_Y ((int)__gid.y)
#define LOCAL_ID_X  ((int)__lid.x)
#define GROUP_ID_X  ((int)__bid.x)
#define GPU_SHARED  threadgroup
// CUDA / HIP expand this to __syncthreads() (same semantics as a threadgroup barrier)
#define GPU_BARRIER threadgroup_barrier(mem_flags::mem_threadgroup)

// MSL has no erf (no counterpart to CUDA's erff), so approximate it with
// Abramowitz & Stegun 7.1.26 (max error 1.5e-7), well inside the 2e-3
// tolerance of its GELU (exact) caller.
inline float gpu_erf_approx(float x) {
    float s = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
                   - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - poly * exp(-ax * ax));
}
#define GPU_ERF(x) gpu_erf_approx(x)
