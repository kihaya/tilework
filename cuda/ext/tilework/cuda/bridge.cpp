// tilework native bridge (CUDA)
//
// Implements the gpu_bridge_* C ABI declared by gpu_bridge.c with the CUDA
// Driver API + NVRTC. Same contract as bridge.swift (Metal,
// ../../../metal/ext/tilework/metal/): mechanism only, with policy on the
// Ruby side (core/lib/tilework.rb).
//
// Loading run-time-compiled PTX needs the Driver API (cu*), so memory
// allocation uses the Driver API too, for uniformity.
//
// NVRTC compiles kernels at run time, so this file has no __global__. It is
// pure host code and builds with the host C++ compiler rather than nvcc
// (see extconf.rb).

#include <cuda.h>
#include <nvrtc.h>
// The vendor libraries are optional: extconf.rb defines TILEWORK_HAS_<LIB>
// only for the ones it finds, and a missing one leaves its gpu_bridge_*
// function undefined, so Ruby sees the weak symbol absent and falls back.
#ifdef TILEWORK_HAS_CUBLAS
#include <cublas_v2.h>
#endif
#ifdef TILEWORK_HAS_CURAND
#include <curand.h>
#endif
#ifdef TILEWORK_HAS_CUFFT
#include <cufft.h>
#endif
#ifdef TILEWORK_HAS_CUDNN
#include <cudnn.h>
#endif

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <map>
#include <string>
#include <tuple>
#include <thread>
#include <vector>

namespace {

CUcontext g_ctx = nullptr;
CUdevice g_dev = 0;
bool g_init_failed = false;
// A compute-only stream. On the default stream, cuCtxSynchronize waits for
// every stream in the context — including offload's transfer stream — so
// prefetch never overlapped compute (raising depth stayed at 1.0x in
// measurements). Compute and transfers use separate streams and waits are
// per stream.
CUstream g_compute = nullptr;
// Stream-ordered memory pool (CUDA 11.2+). cuMemFreeAsync returns memory to
// the pool rather than the OS, so the next same-size allocation returns at
// once. On Jetson's unified memory cuMemAlloc involves page mapping and is
// very slow; a 16 MB allocation measured 3.208 ms → 0.136 ms (23.6x).
//
// Where unavailable, g_use_pool stays false and cuMemAlloc/cuMemFree are used.
// **Allocation and free must always take the same path**, so the choice is
// made once at initialization and never changes (mixing them crashes).
bool g_use_pool = false;
CUmemoryPool g_pool = nullptr;

#ifdef TILEWORK_HAS_CUBLAS
// cuBLAS handle, created lazily (creation is expensive) and reused. Orders of
// magnitude faster than our WMMA kernel (core/lib/tilework/math/tensor_core.rb).
cublasHandle_t g_blas = nullptr;
bool g_blas_failed = false;

bool ensure_blas() {
  if (g_blas) return true;
  if (g_blas_failed) return false;
  if (cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) {
    g_blas = nullptr;
    g_blas_failed = true;
    return false;
  }
  // ★ Bind to the compute stream. The pool is stream-ordered, so on the
  // default stream ordering against allocation / zero-fill is not guaranteed.
  cublasSetStream(g_blas, g_compute);
  return true;
}
#endif // TILEWORK_HAS_CUBLAS

char *dup_cstr(const char *s) {
  // The caller (gpu_bridge.c) releases it with free(), so allocate with malloc.
  size_t n = strlen(s) + 1;
  char *p = (char *)malloc(n);
  if (p) memcpy(p, s, n);
  return p;
}

void set_error(char **error_out, CUresult e, const char *what) {
  if (!error_out) return;
  const char *detail = nullptr;
  cuGetErrorString(e, &detail);
  char buf[512];
  snprintf(buf, sizeof(buf), "%s: %s", what, detail ? detail : "unknown CUDA error");
  *error_out = dup_cstr(buf);
}

// cuInit + context creation happen once; later calls just make it current
// again (Ruby assumes a single queue, so one context and one stream suffice).
bool ensure_ctx() {
  if (g_ctx) {
    cuCtxSetCurrent(g_ctx);
    return true;
  }
  if (g_init_failed) return false;

  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&g_dev, 0) != CUDA_SUCCESS) {
    g_init_failed = true;
    return false;
  }
  if (cuCtxCreate(&g_ctx, 0, g_dev) != CUDA_SUCCESS) {
    g_ctx = nullptr;
    g_init_failed = true;
    return false;
  }
  if (cuStreamCreate(&g_compute, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
    // cuCtxCreate succeeded, so this function owns a live context even though
    // initialization as a whole failed. Dropping only the pointer leaks all
    // driver-side context state for the lifetime of the process.
    cuCtxDestroy(g_ctx);
    g_ctx = nullptr;
    g_compute = nullptr;
    g_init_failed = true;
    return false;
  }

  int pools_supported = 0;
  cuDeviceGetAttribute(&pools_supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, g_dev);
  if (pools_supported && cuDeviceGetDefaultMemPool(&g_pool, g_dev) == CUDA_SUCCESS) {
    // A pool hoarding without limit starves other allocations into OOM on
    // unified-memory machines. Cap it at 1/4 of free memory (anything beyond
    // goes back to the OS).
    size_t free_bytes = 0, total_bytes = 0;
    cuMemGetInfo(&free_bytes, &total_bytes);
    cuuint64_t threshold = (cuuint64_t)(free_bytes / 4);
    cuMemPoolSetAttribute(g_pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold);
    g_use_pool = true;
  }
  return true;
}

// Offload chunk size. 16 MB by default (a middle size balancing speedup,
// hiding quality and synchronization overhead).
static const size_t OFFLOAD_CHUNK = 16u << 20;
// Header length at the start of a file; the body follows it.
static const long OFFLOAD_HEADER = 4096;

struct CuPipeline {
  CUmodule mod;
  CUfunction fn;
};
struct HostBacking { void *data; long bytes; };

// NVRTC does not bundle mma.h (measured on CUDA 12.6: without `-I`, "could
// not open source file mma.h"), so CUDA's include directory is always passed
// for the Tensor Core path. Passing header text as strings is an alternative,
// but -I is confirmed to work on 12.6.
const char *cuda_include_dir() {
  static std::string dir;
  if (dir.empty()) {
    const char *home = getenv("CUDA_HOME");
    dir = std::string((home && *home) ? home : "/usr/local/cuda") + "/include";
  }
  return dir.c_str();
}

// The PTX target architecture is built from the device rather than fixed at
// build time (compute_87 on a Jetson Orin, for example).
void arch_option(char *out, size_t len) {
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_dev);
  if (major == 0) { major = 5; minor = 0; } // on failure rely on PTX forward compatibility
  snprintf(out, len, "--gpu-architecture=compute_%d%d", major, minor);
}

} // namespace

extern "C" {

char *gpu_bridge_device_name(void) {
  if (!ensure_ctx()) return nullptr;
  char name[256];
  if (cuDeviceGetName(name, sizeof(name), g_dev) != CUDA_SUCCESS) return nullptr;
  return dup_cstr(name);
}

int gpu_bridge_device_available(void) {
  return ensure_ctx() ? 1 : 0;
}

void *gpu_bridge_buffer_alloc(long bytes) {
  if (!ensure_ctx()) return nullptr;

  CUdeviceptr dptr = 0;
  if (g_use_pool) {
    // Allocate on the compute stream. Every later operation on this pointer
    // must be on g_compute or explicitly synchronized with it (a constraint of
    // stream-ordered allocation).
    CUresult e = cuMemAllocAsync(&dptr, (size_t)bytes, g_compute);
    if (e != CUDA_SUCCESS) {
      // Return what the pool holds to the OS and retry once; without this the
      // state "free memory exists but allocation fails" is possible.
      cuStreamSynchronize(g_compute);
      cuMemPoolTrimTo(g_pool, 0);
      e = cuMemAllocAsync(&dptr, (size_t)bytes, g_compute);
    }
    if (e != CUDA_SUCCESS) return nullptr;
  } else if (cuMemAlloc(&dptr, (size_t)bytes) != CUDA_SUCCESS) {
    return nullptr;
  }

  // Zero-initialize. The pool returns regions with old contents, so without
  // this code assuming "a new buffer is zero" would read different values each
  // run (Metal's storageModeShared buffers are effectively zero-initialized,
  // and this keeps behavior the same across backends). The async version is
  // used because synchronizing here would undo the faster allocation.
  if (g_use_pool) {
    cuMemsetD8Async(dptr, 0, (size_t)bytes, g_compute);
  } else {
    cuMemsetD8(dptr, 0, (size_t)bytes);
  }

  // CUdeviceptr is a 64-bit integer; it round-trips as a void* handle.
  return (void *)(uintptr_t)dptr;
}

void gpu_bridge_buffer_write(void *handle, const void *src, long bytes, long offset) {
  if (!ensure_ctx()) return;
  CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)handle;
  // Done on the same g_compute as allocation and zero-fill. g_compute is
  // NON_BLOCKING, so on the default stream implicit synchronization would not
  // apply and the write could land before allocation or zero-fill finishes.
  cuMemcpyHtoDAsync(dptr + (size_t)offset, src, (size_t)bytes, g_compute);
  cuStreamSynchronize(g_compute); // Ruby expects the write to be complete
}

void gpu_bridge_buffer_read(void *handle, void *dst, long bytes, long offset) {
  if (!ensure_ctx()) return;
  CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)handle;
  // Read on the same g_compute as allocation, zero-fill and kernels so the
  // value is right even if the previous launch_async has not finished (the
  // same semantics as Metal's contents seeing committed buffers). Only the
  // compute stream is waited on — no need to wait for in-flight offload transfers.
  cuMemcpyDtoHAsync(dst, dptr + (size_t)offset, (size_t)bytes, g_compute);
  cuStreamSynchronize(g_compute);
}

void gpu_bridge_buffer_free(void *handle) {
  if (!handle) return;
  if (!ensure_ctx()) return;

  // Return it through the same path it was allocated by; mixing them crashes,
  // hence the branch on g_use_pool.
  if (g_use_pool) {
    cuMemFreeAsync((CUdeviceptr)(uintptr_t)handle, g_compute);
  } else {
    cuMemFree((CUdeviceptr)(uintptr_t)handle);
  }
}

int gpu_bridge_host_backing_available(void) { return ensure_ctx() ? 1 : 0; }
void *gpu_bridge_host_backing_alloc(long bytes) {
  if (bytes <= 0) return nullptr;
  auto *backing = static_cast<HostBacking *>(malloc(sizeof(HostBacking)));
  if (!backing) return nullptr;
  backing->data = malloc(static_cast<size_t>(bytes));
  backing->bytes = bytes;
  if (!backing->data) { free(backing); return nullptr; }
  return backing;
}
void gpu_bridge_host_backing_free(void *handle) {
  auto *backing = static_cast<HostBacking *>(handle);
  if (!backing) return;
  free(backing->data);
  free(backing);
}
int gpu_bridge_host_backing_store(void *host, void *device, long bytes, long offset, char **error_out) {
  auto *backing = static_cast<HostBacking *>(host);
  if (!backing || !device || bytes < 0 || offset < 0 || offset + bytes > backing->bytes) {
    if (error_out) *error_out = dup_cstr("host backing store range is out of bounds");
    return 1;
  }
  if (!ensure_ctx()) { if (error_out) *error_out = dup_cstr("no CUDA device available"); return 1; }
  CUresult status = cuMemcpyDtoHAsync(static_cast<char *>(backing->data) + offset,
                                      (CUdeviceptr)(uintptr_t)device, static_cast<size_t>(bytes), g_compute);
  if (status == CUDA_SUCCESS) status = cuStreamSynchronize(g_compute);
  if (status != CUDA_SUCCESS) { set_error(error_out, status, "host backing device-to-host copy"); return 1; }
  return 0;
}
int gpu_bridge_host_backing_restore(void *host, void *device, long bytes, long offset, char **error_out) {
  auto *backing = static_cast<HostBacking *>(host);
  if (!backing || !device || bytes < 0 || offset < 0 || offset + bytes > backing->bytes) {
    if (error_out) *error_out = dup_cstr("host backing restore range is out of bounds");
    return 1;
  }
  if (!ensure_ctx()) { if (error_out) *error_out = dup_cstr("no CUDA device available"); return 1; }
  CUresult status = cuMemcpyHtoDAsync((CUdeviceptr)(uintptr_t)device,
                                      static_cast<char *>(backing->data) + offset, static_cast<size_t>(bytes), g_compute);
  if (status == CUDA_SUCCESS) status = cuStreamSynchronize(g_compute);
  if (status != CUDA_SUCCESS) { set_error(error_out, status, "host backing host-to-device copy"); return 1; }
  return 0;
}

void *gpu_bridge_pipeline_create(const char *src, const char *fn_name, char **error_out) {
  if (!ensure_ctx()) {
    if (error_out) *error_out = dup_cstr("no CUDA device available (cuInit/cuCtxCreate failed)");
    return nullptr;
  }

  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, src, "kernel.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) {
    if (error_out) *error_out = dup_cstr("nvrtcCreateProgram failed");
    return nullptr;
  }

  char arch[64];
  arch_option(arch, sizeof(arch));
  std::string include_opt = std::string("-I") + cuda_include_dir();
  const char *opts[] = { arch, "--std=c++14", include_opt.c_str() };

  if (nvrtcCompileProgram(prog, 3, opts) != NVRTC_SUCCESS) {
    // Pass the whole log with line numbers to Ruby (Tilework::CompileError) as is.
    size_t n = 0;
    nvrtcGetProgramLogSize(prog, &n);
    char *log = (char *)malloc(n > 0 ? n : 1);
    if (log) {
      log[0] = '\0';
      if (n > 0) nvrtcGetProgramLog(prog, log);
    }
    if (error_out) *error_out = log ? log : dup_cstr("NVRTC compile failed");
    else if (log) free(log);
    nvrtcDestroyProgram(&prog);
    return nullptr;
  }

  size_t ptx_size = 0;
  nvrtcGetPTXSize(prog, &ptx_size);
  char *ptx = (char *)malloc(ptx_size);
  if (!ptx) {
    nvrtcDestroyProgram(&prog);
    if (error_out) *error_out = dup_cstr("out of memory while retrieving PTX");
    return nullptr;
  }
  nvrtcGetPTX(prog, ptx);
  nvrtcDestroyProgram(&prog);

  CuPipeline *pl = new CuPipeline{};
  CUresult e = cuModuleLoadData(&pl->mod, ptx);
  free(ptx);
  if (e != CUDA_SUCCESS) {
    set_error(error_out, e, "cuModuleLoadData failed");
    delete pl;
    return nullptr;
  }

  e = cuModuleGetFunction(&pl->fn, pl->mod, fn_name);
  if (e != CUDA_SUCCESS) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "kernel function '%s' not found in the compiled module "
             "(is it declared with GPU_KERNEL so that extern \"C\" applies?)",
             fn_name);
    if (error_out) *error_out = dup_cstr(buf);
    cuModuleUnload(pl->mod);
    delete pl;
    return nullptr;
  }
  return pl;
}

long gpu_bridge_pipeline_max_threads(void *handle) {
  if (!ensure_ctx()) return 0;
  CuPipeline *pl = (CuPipeline *)handle;
  int v = 0;
  if (cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, pl->fn) != CUDA_SUCCESS) {
    return 0;
  }
  return (long)v;
}

void gpu_bridge_pipeline_free(void *handle) {
  if (!handle) return;
  CuPipeline *pl = (CuPipeline *)handle;
  if (ensure_ctx()) cuModuleUnload(pl->mod);
  delete pl;
}

int gpu_bridge_dispatch(void *pipeline,
                        const int32_t *arg_kinds, void *const *arg_ptrs,
                        const long *arg_lens, int32_t arg_count,
                        long gx, long gy, long gz,
                        long bx, long by, long bz,
                        int32_t wait, char **error_out) {
  if (!ensure_ctx()) {
    if (error_out) *error_out = dup_cstr("no CUDA device available");
    return -1;
  }
  CuPipeline *pl = (CuPipeline *)pipeline;

  // kernelParams is an array of pointers to argument values. A buffer (kind=0)
  // passes the address of a local holding the CUdeviceptr; a scalar (kind=1)
  // passes the pointer to its value bytes directly.
  std::vector<CUdeviceptr> devptrs(arg_count > 0 ? arg_count : 1);
  std::vector<void *> params(arg_count > 0 ? arg_count : 1);
  for (int32_t i = 0; i < arg_count; i++) {
    if (arg_kinds[i] == 0) {
      // For buffer arguments arg_lens is the view's byte offset (see gpu_bridge.c)
      devptrs[i] = (CUdeviceptr)(uintptr_t)arg_ptrs[i] + (size_t)arg_lens[i];
      params[i] = &devptrs[i];
    } else {
      params[i] = arg_ptrs[i];
    }
  }

  // grid is the number of blocks and block the threads per block. Ruby's
  // convention matches cuLaunchKernel's semantics, so no conversion.
  CUresult e = cuLaunchKernel(pl->fn,
                              (unsigned)gx, (unsigned)gy, (unsigned)gz,
                              (unsigned)bx, (unsigned)by, (unsigned)bz,
                              0, g_compute, params.data(), nullptr);
  if (e != CUDA_SUCCESS) {
    set_error(error_out, e, "cuLaunchKernel failed");
    return -1;
  }

  if (wait) {
    e = cuStreamSynchronize(g_compute);
    if (e != CUDA_SUCCESS) {
      set_error(error_out, e, "cuStreamSynchronize failed");
      return -1;
    }
  }
  return 0;
}

void gpu_bridge_synchronize(void) {
  if (!ensure_ctx()) return;
  cuStreamSynchronize(g_compute);
}

} // extern "C"

namespace {
// On file systems without O_DIRECT this silently falls back to ordinary I/O.
int open_maybe_direct(const char *path, int flags) {
  int fd = open(path, flags | O_DIRECT);
  if (fd < 0 && errno == EINVAL) fd = open(path, flags);
  return fd;
}

size_t staging_bytes(long bytes) {
  size_t padded = ((size_t)bytes + 4095) & ~(size_t)4095;
  if (padded == 0) padded = 4096;
  return padded < OFFLOAD_CHUNK ? padded : OFFLOAD_CHUNK;
}

// The synchronous core. The async version (transfer_*) just runs this on a worker thread.
int staged_save(CUdeviceptr dptr, const char *path, long bytes, long off, std::string *err) {
  // Staging takes the smaller of the chunk limit and the actual data length
  // (rounded up to 4096). Always taking 16 MB let allocation cost dominate
  // round trips of small buffers (measured: 1 KB and 64 KB both took 8 ms).
  const size_t chunk = staging_bytes(bytes);
  void *stage = nullptr;
  if (cuMemAllocHost(&stage, chunk) != CUDA_SUCCESS) {
    if (err) *err = "failed to allocate pinned staging buffer";
    return -1;
  }
  // ★ A transfer-only stream. On the default stream it would synchronize with
  // compute kernels and prefetch would not overlap compute.
  CUstream xfer = nullptr;
  cuStreamCreate(&xfer, CU_STREAM_NON_BLOCKING);
  int fd = open_maybe_direct(path, O_WRONLY);
  if (fd < 0) {
    cuMemFreeHost(stage);
    if (err) *err = strerror(errno);
    return -1;
  }

  cuStreamSynchronize(g_compute); // settle pending kernels' writes before reading
  int rc = 0;
  for (long done = 0; done < bytes;) {
    size_t n = (size_t)((bytes - done) < (long)chunk ? (bytes - done) : (long)chunk);
    if (cuMemcpyDtoHAsync(stage, dptr + (size_t)off + (size_t)done, n, xfer) != CUDA_SUCCESS ||
        cuStreamSynchronize(xfer) != CUDA_SUCCESS) {
      if (err) *err = "device-to-host copy failed during offload";
      rc = -1;
      break;
    }
    // O_DIRECT needs length, offset and address all 4096-aligned. The last
    // chunk writes through its rounded-up zero padding (matching the Ruby
    // path's padding, so both paths' files are bit-identical).
    size_t wn = (n + 4095) & ~(size_t)4095;
    if (wn > n) memset((char *)stage + n, 0, wn - n);
    if (pwrite(fd, stage, wn, OFFLOAD_HEADER + done) != (ssize_t)wn) {
      if (err) *err = strerror(errno);
      rc = -1;
      break;
    }
    done += (long)n;
  }

  if (rc == 0) fsync(fd);
  close(fd);
  if (xfer) cuStreamDestroy(xfer);
  cuMemFreeHost(stage);
  return rc;
}

int staged_load(CUdeviceptr dptr, const char *path, long bytes, long off, std::string *err) {
  // Staging is sized for the same reason as in staged_save.
  const size_t chunk = staging_bytes(bytes);
  void *stage = nullptr;
  if (cuMemAllocHost(&stage, chunk) != CUDA_SUCCESS) {
    if (err) *err = "failed to allocate pinned staging buffer";
    return -1;
  }
  CUstream xfer = nullptr; // ★ transfer-only stream (see staged_save)
  cuStreamCreate(&xfer, CU_STREAM_NON_BLOCKING);
  int fd = open_maybe_direct(path, O_RDONLY);
  if (fd < 0) {
    cuMemFreeHost(stage);
    if (err) *err = strerror(errno);
    return -1;
  }

  // The destination was allocated stream-ordered (on g_compute). This is a
  // different stream, so not explicitly waiting for allocation and zero-fill
  // is undefined behavior. staged_save does the same synchronization up front.
  cuStreamSynchronize(g_compute);

  int rc = 0;
  for (long done = 0; done < bytes;) {
    size_t n = (size_t)((bytes - done) < (long)chunk ? (bytes - done) : (long)chunk);
    size_t rn = (n + 4095) & ~(size_t)4095; // the file is padded
    ssize_t got = pread(fd, stage, rn, OFFLOAD_HEADER + done);
    if (got < (ssize_t)n) {
      if (err) *err = (got < 0) ? strerror(errno) : "offload file is shorter than expected";
      rc = -1;
      break;
    }
    if (cuMemcpyHtoDAsync(dptr + (size_t)off + (size_t)done, stage, n, xfer) != CUDA_SUCCESS ||
        cuStreamSynchronize(xfer) != CUDA_SUCCESS) {
      if (err) *err = "host-to-device copy failed during restore";
      rc = -1;
      break;
    }
    done += (long)n;
  }

  close(fd);
  if (xfer) cuStreamDestroy(xfer);
  cuMemFreeHost(stage);
  return rc;
}

// State of one async transfer. The bridge has no scheduler; it just runs one
// worker. The policy of what to prefetch in which order lives in Ruby.
struct CuTransfer {
  std::thread worker;
  std::atomic<int> status{0}; // 0 = in progress, 1 = done, -1 = error
  std::string err;
};

} // namespace

// --- Optional ABI: offload transfers ---
//
// Pinned staging + O_DIRECT. Ruby has already written the header (the first
// 4096 bytes) with ordinary I/O; this reads/writes only the body on 4096
// boundaries.
//
// cuFile (GPUDirect Storage) is not used: on the test machine
// cuFileDriverOpen succeeds, but nvidia-fs is not loaded, so it only reaches
// compatibility mode, falls back to staging internally and gains nothing.

extern "C" {

int gpu_bridge_buffer_save(void *handle, const char *path, long bytes, long off, char **err) {
  if (!ensure_ctx()) {
    if (err) *err = dup_cstr("no CUDA device available");
    return -1;
  }
  std::string msg;
  int rc = staged_save((CUdeviceptr)(uintptr_t)handle, path, bytes, off, &msg);
  if (rc != 0 && err) *err = dup_cstr(msg.c_str());
  return rc;
}

int gpu_bridge_buffer_load(void *handle, const char *path, long bytes, long off, char **err) {
  if (!ensure_ctx()) {
    if (err) *err = dup_cstr("no CUDA device available");
    return -1;
  }
  std::string msg;
  int rc = staged_load((CUdeviceptr)(uintptr_t)handle, path, bytes, off, &msg);
  if (rc != 0 && err) *err = dup_cstr(msg.c_str());
  return rc;
}

// dir: 0=save (VRAM -> SSD), 1=load (SSD -> VRAM)
void *gpu_bridge_transfer_begin(void *handle, const char *path, long bytes, long off, int dir) {
  if (!ensure_ctx()) return nullptr;

  CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)handle;
  std::string p(path); // copied so it does not depend on the caller's string lifetime
  auto *t = new CuTransfer();
  t->worker = std::thread([t, dptr, p, bytes, off, dir]() {
    // Bind the context to the worker thread too.
    cuCtxSetCurrent(g_ctx);
    std::string msg;
    int rc = (dir == 0) ? staged_save(dptr, p.c_str(), bytes, off, &msg)
                        : staged_load(dptr, p.c_str(), bytes, off, &msg);
    t->err = msg;
    t->status.store(rc == 0 ? 1 : -1);
  });
  return t;
}

int gpu_bridge_transfer_poll(void *handle) {
  auto *t = (CuTransfer *)handle;
  return t ? t->status.load() : -1;
}

int gpu_bridge_transfer_end(void *handle, char **err) {
  auto *t = (CuTransfer *)handle;
  if (!t) return -1;

  if (t->worker.joinable()) t->worker.join();
  int status = t->status.load();
  if (status != 1 && err) *err = dup_cstr(t->err.empty() ? "offload transfer failed" : t->err.c_str());
  delete t;
  return status == 1 ? 0 : -1;
}

} // extern "C"

#ifdef TILEWORK_HAS_CURAND
// --- Optional ABI: random generation delegated to cuRAND ---
//
// Philox is counter-based: the same seed deterministically gives the same
// sequence. Its state is small and generation fast, which makes it easier to
// handle than the default XORWOW.
namespace {
curandGenerator_t g_rng = nullptr;
bool g_rng_failed = false;

bool ensure_rng() {
  if (g_rng) return true;
  if (g_rng_failed) return false;
  if (curandCreateGenerator(&g_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10) != CURAND_STATUS_SUCCESS) {
    g_rng = nullptr;
    g_rng_failed = true;
    return false;
  }
  // ★ Bind to the compute stream (the memory pool is stream-ordered)
  curandSetStream(g_rng, g_compute);
  return true;
}
} // namespace

extern "C" {

// kind: 0 = uniform (0, 1], 1 = normal (mean, stddev)
int gpu_bridge_random(void *handle, long n, int kind, unsigned long long seed,
                      float mean, float stddev, char **err) {
  if (!ensure_ctx() || !ensure_rng()) {
    if (err) *err = dup_cstr("cuRAND is not available");
    return -1;
  }

  // Setting the seed alone does not reset the internal offset, so the same
  // seed would continue the sequence on every call. Reset it to 0 explicitly
  // for reproducibility.
  curandSetPseudoRandomGeneratorSeed(g_rng, seed);
  curandSetGeneratorOffset(g_rng, 0);
  float *dst = (float *)(uintptr_t)handle;
  curandStatus_t st;

  if (kind == 0) {
    st = curandGenerateUniform(g_rng, dst, (size_t)n);
  } else if ((n % 2) == 0) {
    st = curandGenerateNormal(g_rng, dst, (size_t)n, mean, stddev);
  } else {
    // curandGenerateNormal makes pairs via Box-Muller and rejects odd counts.
    // For an odd count, allocate one extra separately and copy the first n.
    CUdeviceptr tmp = 0;
    size_t padded = (size_t)n + 1;
    CUresult ce = g_use_pool ? cuMemAllocAsync(&tmp, padded * sizeof(float), g_compute)
                             : cuMemAlloc(&tmp, padded * sizeof(float));
    if (ce != CUDA_SUCCESS) {
      if (err) *err = dup_cstr("failed to allocate scratch for an odd-length normal draw");
      return -1;
    }
    st = curandGenerateNormal(g_rng, (float *)tmp, padded, mean, stddev);
    if (st == CURAND_STATUS_SUCCESS) {
      cuMemcpyDtoDAsync((CUdeviceptr)(uintptr_t)handle, tmp, (size_t)n * sizeof(float), g_compute);
    }
    if (g_use_pool) {
      cuMemFreeAsync(tmp, g_compute);
    } else {
      cuStreamSynchronize(g_compute);
      cuMemFree(tmp);
    }
  }

  if (st != CURAND_STATUS_SUCCESS) {
    char buf[128];
    snprintf(buf, sizeof(buf), "curandGenerate%s failed (status %d)",
             kind == 0 ? "Uniform" : "Normal", (int)st);
    if (err) *err = dup_cstr(buf);
    return -1;
  }
  return 0;
}

} // extern "C"
#endif // TILEWORK_HAS_CURAND

extern "C" {

#ifdef TILEWORK_HAS_CUDNN
// --- Optional ABI: 2D convolution delegated to cuDNN ---
//
// A hand-written im2col + GEMM spends 74–84% of its time in im2col and reaches
// only 240–613 GF/s (measured). im2col writes out k^2 times the data and is
// memory-bound, which becomes the bottleneck. cuDNN's implicit GEMM has no
// expansion at all and reaches fp16 4747 GF/s (7.7x) on the same shape.
namespace {
cudnnHandle_t g_dnn = nullptr;
bool g_dnn_failed = false;

bool ensure_dnn() {
  if (g_dnn) return true;
  if (g_dnn_failed) return false;
  if (cudnnCreate(&g_dnn) != CUDNN_STATUS_SUCCESS) {
    g_dnn = nullptr;
    g_dnn_failed = true;
    return false;
  }
  // ★ Bind to the compute stream (the memory pool is stream-ordered)
  cudnnSetStream(g_dnn, g_compute);
  return true;
}

// Descriptor and algorithm selection is expensive, so they are reused per
// configuration. Descriptors are shared by forward and backward (a gradient
// has its tensor's shape, so dy uses y's descriptor, dx x's, dw w's).
struct ConvPlan {
  cudnnTensorDescriptor_t x, y;
  cudnnFilterDescriptor_t w;
  cudnnConvolutionDescriptor_t conv;
  cudnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  size_t workspace;              // forward
  size_t bwd_data_workspace;
  size_t bwd_filter_workspace;
};
std::map<std::vector<int>, ConvPlan> g_conv_plans;

void conv_plan_destroy(ConvPlan *plan) {
  if (!plan) return;
  if (plan->conv) cudnnDestroyConvolutionDescriptor(plan->conv);
  if (plan->w) cudnnDestroyFilterDescriptor(plan->w);
  if (plan->y) cudnnDestroyTensorDescriptor(plan->y);
  if (plan->x) cudnnDestroyTensorDescriptor(plan->x);
  *plan = ConvPlan{};
}

// Workspace comes from the pool (allocation takes 0.016 ms, so per call is fine).
bool conv_workspace_alloc(size_t bytes, CUdeviceptr *out) {
  *out = 0;
  if (bytes == 0) return true;
  CUresult ce = g_use_pool ? cuMemAllocAsync(out, bytes, g_compute)
                           : cuMemAlloc(out, bytes);
  return ce == CUDA_SUCCESS;
}

void conv_workspace_free(CUdeviceptr ws) {
  if (!ws) return;
  if (g_use_pool) {
    cuMemFreeAsync(ws, g_compute);
  } else {
    cuStreamSynchronize(g_compute);
    cuMemFree(ws);
  }
}
} // namespace

extern "C" {

int gpu_bridge_conv2d(void *x, void *w, void *y,
                      int n, int cin, int h, int width,
                      int cout, int kh, int kw,
                      int pad_h, int pad_w, int stride_h, int stride_w,
                      int dilation_h, int dilation_w, int groups,
                      int dtype, char **err) {
  if (!ensure_ctx() || !ensure_dnn()) {
    if (err) *err = dup_cstr("cuDNN is not available");
    return -1;
  }

  std::vector<int> key = {n, cin, h, width, cout, kh, kw, pad_h, pad_w, stride_h, stride_w,
                          dilation_h, dilation_w, groups, dtype};
  auto it = g_conv_plans.find(key);
  if (it == g_conv_plans.end()) {
    ConvPlan p{};
    cudnnDataType_t dt = dtype == 1 ? CUDNN_DATA_HALF :
                             (dtype == 2 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT);

    if (cudnnCreateTensorDescriptor(&p.x) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&p.y) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateFilterDescriptor(&p.w) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateConvolutionDescriptor(&p.conv) != CUDNN_STATUS_SUCCESS) {
      conv_plan_destroy(&p);
      if (err) *err = dup_cstr("failed to create cuDNN descriptors");
      return -1;
    }

    cudnnSetTensor4dDescriptor(p.x, CUDNN_TENSOR_NCHW, dt, n, cin, h, width);
    cudnnSetFilter4dDescriptor(p.w, dt, CUDNN_TENSOR_NCHW, cout, cin / groups, kh, kw);
    // Accumulation is always FP32
    cudnnSetConvolution2dDescriptor(p.conv, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(p.conv, groups);
    if (dtype != 0) cudnnSetConvolutionMathType(p.conv, CUDNN_TENSOR_OP_MATH);

    int on, oc, oh, ow;
    cudnnGetConvolution2dForwardOutputDim(p.conv, p.x, p.w, &on, &oc, &oh, &ow);
    cudnnSetTensor4dDescriptor(p.y, CUDNN_TENSOR_NCHW, dt, on, oc, oh, ow);

    int count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    if (cudnnGetConvolutionForwardAlgorithm_v7(g_dnn, p.x, p.w, p.conv, p.y, 8, &count, perf)
          != CUDNN_STATUS_SUCCESS || count == 0) {
      conv_plan_destroy(&p);
      if (err) *err = dup_cstr("cuDNN found no forward convolution algorithm");
      return -1;
    }
    p.algo = perf[0].algo;
    cudnnGetConvolutionForwardWorkspaceSize(g_dnn, p.x, p.w, p.conv, p.y, p.algo, &p.workspace);

    // Backward algorithms are chosen from the same descriptors here too.
    // Selection is expensive, so it is folded into the plan instead of being
    // redone every training step.
    cudnnConvolutionBwdDataAlgoPerf_t dperf[8];
    count = 0;
    if (cudnnGetConvolutionBackwardDataAlgorithm_v7(g_dnn, p.w, p.y, p.conv, p.x, 8, &count, dperf)
          == CUDNN_STATUS_SUCCESS && count > 0) {
      p.bwd_data_algo = dperf[0].algo;
      cudnnGetConvolutionBackwardDataWorkspaceSize(g_dnn, p.w, p.y, p.conv, p.x,
                                                   p.bwd_data_algo, &p.bwd_data_workspace);
    } else {
      p.bwd_data_workspace = SIZE_MAX; // marks unavailable; errors when called
    }

    cudnnConvolutionBwdFilterAlgoPerf_t fperf[8];
    count = 0;
    if (cudnnGetConvolutionBackwardFilterAlgorithm_v7(g_dnn, p.x, p.y, p.conv, p.w, 8, &count, fperf)
          == CUDNN_STATUS_SUCCESS && count > 0) {
      p.bwd_filter_algo = fperf[0].algo;
      cudnnGetConvolutionBackwardFilterWorkspaceSize(g_dnn, p.x, p.y, p.conv, p.w,
                                                     p.bwd_filter_algo, &p.bwd_filter_workspace);
    } else {
      p.bwd_filter_workspace = SIZE_MAX;
    }

    g_conv_plans[key] = p;
    it = g_conv_plans.find(key);
  }

  ConvPlan &p = it->second;

  CUdeviceptr ws = 0;
  if (!conv_workspace_alloc(p.workspace, &ws)) {
    if (err) *err = dup_cstr("failed to allocate the cuDNN workspace");
    return -1;
  }

  const float alpha = 1.0f, beta = 0.0f;
  cudnnStatus_t st = cudnnConvolutionForward(g_dnn, &alpha, p.x, x, p.w, w, p.conv, p.algo,
                                             (void *)(uintptr_t)ws, p.workspace, &beta, p.y, y);
  conv_workspace_free(ws);

  if (st != CUDNN_STATUS_SUCCESS) {
    char buf[128];
    snprintf(buf, sizeof(buf), "cudnnConvolutionForward failed (status %d)", (int)st);
    if (err) *err = dup_cstr(buf);
    return -1;
  }
  return 0;
}

} // extern "C"
#endif // TILEWORK_HAS_CUDNN

#ifdef TILEWORK_HAS_CUFFT
// --- Optional ABI: FFT delegated to cuFFT ---
//
// Complex numbers are interleaved float32 (re, im, re, im, ...). That is
// exactly cufftComplex's layout, so a Buffer can be passed as is.
namespace {
// Plan creation is expensive, so plans are reused per (kind, n, batch).
std::map<std::tuple<int, long, long>, cufftHandle> g_fft_plans;

// kind: 0 = R2C, 1 = C2R, 2 = C2C
bool fft_plan(int kind, long n, long batch, cufftHandle *out) {
  auto key = std::make_tuple(kind, n, batch);
  auto it = g_fft_plans.find(key);
  if (it != g_fft_plans.end()) {
    *out = it->second;
    return true;
  }

  cufftType type = kind == 0 ? CUFFT_R2C : (kind == 1 ? CUFFT_C2R : CUFFT_C2C);
  cufftHandle plan;
  int len = (int)n;
  if (cufftPlanMany(&plan, 1, &len, nullptr, 1, 0, nullptr, 1, 0, type, (int)batch) != CUFFT_SUCCESS) {
    return false;
  }
  // ★ Bind to the compute stream (the memory pool is stream-ordered)
  cufftSetStream(plan, g_compute);
  g_fft_plans[key] = plan;
  *out = plan;
  return true;
}
} // namespace

extern "C" {

int gpu_bridge_fft(void *in, void *out, long n, long batch, int kind, int direction,
                   char **err) {
  if (!ensure_ctx()) {
    if (err) *err = dup_cstr("no CUDA device available");
    return -1;
  }

  cufftHandle plan;
  if (!fft_plan(kind, n, batch, &plan)) {
    if (err) *err = dup_cstr("cufftPlanMany failed");
    return -1;
  }

  cufftResult st;
  switch (kind) {
    case 0:
      st = cufftExecR2C(plan, (cufftReal *)(uintptr_t)in, (cufftComplex *)(uintptr_t)out);
      break;
    case 1:
      st = cufftExecC2R(plan, (cufftComplex *)(uintptr_t)in, (cufftReal *)(uintptr_t)out);
      break;
    default:
      st = cufftExecC2C(plan, (cufftComplex *)(uintptr_t)in, (cufftComplex *)(uintptr_t)out,
                        direction < 0 ? CUFFT_INVERSE : CUFFT_FORWARD);
      break;
  }

  if (st != CUFFT_SUCCESS) {
    char buf[128];
    snprintf(buf, sizeof(buf), "cufftExec failed (status %d)", (int)st);
    if (err) *err = dup_cstr(buf);
    return -1;
  }
  return 0;
}

} // extern "C"
#endif // TILEWORK_HAS_CUFFT

#ifdef TILEWORK_HAS_CUBLAS
// --- Optional ABI: GEMM delegated to cuBLAS ---
//
// Without it Ruby falls back to the existing kernel path (WMMA or SIMT).
// allow_tf32 enables TF32 tensor cores (10-bit mantissa, about 1e-3 relative).

// in_dtype: 0 = float32, 1 = float16, 2 = bfloat16. **The output is always
// float32** (FP32 accumulation, float32 output buffer).
int gpu_bridge_gemm(void *a, long a_off, void *b, long b_off, void *c, long c_off,
                    long m, long k, long n,
                    int allow_tf32, int in_dtype, char **err) {
  if (!ensure_ctx() || !ensure_blas()) {
    if (err) *err = dup_cstr("cuBLAS is not available");
    return -1;
  }
  a = (char *)a + a_off;
  b = (char *)b + b_off;
  c = (char *)c + c_off;

  cublasSetMathMode(g_blas, allow_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);

  // Accumulation is FP32, so alpha/beta are float; never pass __half.
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st;

  // Buffers are row-major but cuBLAS is column-major. Row-major C = A*B equals
  // computing C^T = B^T * A^T in column-major (no transpose copy needed).
  if (in_dtype == 1 || in_dtype == 2) {
    const cudaDataType_t input_type = in_dtype == 1 ? CUDA_R_16F : CUDA_R_16BF;
    // A/B are CUDA_R_16F and only C is CUDA_R_32F; easy to mix up.
    st = cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                      (int)n, (int)m, (int)k,
                      &alpha,
                      (const void *)(uintptr_t)b, input_type, (int)n,
                      (const void *)(uintptr_t)a, input_type, (int)k,
                      &beta,
                      (void *)(uintptr_t)c, CUDA_R_32F, (int)n,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  } else {
    st = cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                     (int)n, (int)m, (int)k,
                     &alpha,
                     (const float *)(uintptr_t)b, (int)n,
                     (const float *)(uintptr_t)a, (int)k,
                     &beta,
                     (float *)(uintptr_t)c, (int)n);
  }

  if (st != CUBLAS_STATUS_SUCCESS) {
    char buf[160];
    const char *operation = in_dtype == 1 ? "GemmEx (fp16)" :
                            (in_dtype == 2 ? "GemmEx (bf16)" : "Sgemm");
    snprintf(buf, sizeof(buf), "cublas%s failed (status %d)",
             operation, (int)st);
    if (err) *err = dup_cstr(buf);
    return -1;
  }
  return 0;
}

// --- Optional ABI: strided batched GEMM ---
//
// Issues `batch` GEMMs of the same shape in one call. Matrices spaced at even
// intervals need not be gathered (stride_* is how many elements to the next
// matrix). Multi-head attention's heads are laid out exactly like that, so
// slicing heads out and writing them back disappears entirely.
//
// ld_* is how many elements apart one column of the matrix is (the
// column-major leading dimension). To view a row-major [R, C] as part of a
// larger matrix, pass **the parent's column count** as ld; that is why heads
// can be views.
//
// Unlike gemm(), the caller builds m/k/n and ld/stride in column-major
// (reinterpreting from row-major stays in core/lib/tilework/math/nn.rb).
int gpu_bridge_gemm_strided_batched(void *a, long a_off, void *b, long b_off,
                                    void *c, long c_off,
                                    long m, long k, long n,
                                    long lda, long ldb, long ldc,
                                    long stride_a, long stride_b, long stride_c,
                                    int trans_a, int trans_b,
                                    long batch, double alpha_in,
                                    int allow_tf32, int in_dtype,
                                    char **err) {
  if (!ensure_ctx() || !ensure_blas()) {
    if (err) *err = dup_cstr("cuBLAS is not available");
    return -1;
  }
  a = (char *)a + a_off;
  b = (char *)b + b_off;
  c = (char *)c + c_off;

  cublasSetMathMode(g_blas, allow_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);

  // Accumulation is FP32, so alpha/beta are float. A scale after the product,
  // like attention's 1/sqrt(dh), is applied for free through alpha.
  const float alpha = (float)alpha_in, beta = 0.0f;
  const cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cudaDataType_t in_type = in_dtype == 1 ? CUDA_R_16F :
                                     (in_dtype == 2 ? CUDA_R_16BF : CUDA_R_32F);

  cublasStatus_t st = cublasGemmStridedBatchedEx(
      g_blas, op_a, op_b,
      (int)m, (int)n, (int)k,
      &alpha,
      (const void *)(uintptr_t)a, in_type, (int)lda, (long long)stride_a,
      (const void *)(uintptr_t)b, in_type, (int)ldb, (long long)stride_b,
      &beta,
      (void *)(uintptr_t)c, CUDA_R_32F, (int)ldc, (long long)stride_c,
      (int)batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (st != CUBLAS_STATUS_SUCCESS) {
    char buf[160];
    snprintf(buf, sizeof(buf),
             "cublasGemmStridedBatchedEx failed (status %d, batch %ld)", (int)st, batch);
    if (err) *err = dup_cstr(buf);
    return -1;
  }
  return 0;
}
#endif // TILEWORK_HAS_CUBLAS

// --- Optional ABI: capability queries ---
//
// Not a required ABI. gpu_bridge.c references it as a weak symbol, so linking
// a bridge without it (bridge.swift) does not break, and Ruby sees that the
// query is missing and falls back.
int gpu_bridge_device_capabilities(int32_t *warp_size, int32_t *cc_major, int32_t *cc_minor,
                                   int32_t *integrated, long *memory_budget_bytes) {
  if (!ensure_ctx()) return -1;

  int v = 0;
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_WARP_SIZE, g_dev);
  *warp_size = v;
  v = 0;
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_dev);
  *cc_major = v;
  v = 0;
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_dev);
  *cc_minor = v;
  // On Jetson and APUs the GPU and CPU share the same physical memory; this
  // feeds the offload staging limit.
  v = 0;
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_INTEGRATED, g_dev);
  *integrated = v;

  size_t free_bytes = 0, total_bytes = 0;
  *memory_budget_bytes = cuMemGetInfo(&free_bytes, &total_bytes) == CUDA_SUCCESS ? (long)free_bytes : 0;
  return 0;
}

#ifdef TILEWORK_HAS_CUDNN
// Backward. mode 0 = input gradient (dx), mode 1 = weight gradient (dw).
// The arguments mean different things per mode:
//   mode 0: a = w (weight), b = dy (output gradient), out = dx
//   mode 1: a = x (input),  b = dy (output gradient), out = dw
// Descriptors are shared with the forward plan (a gradient has its tensor's shape).
int gpu_bridge_conv2d_backward(void *a, void *b, void *out,
                               int n, int cin, int h, int width,
                               int cout, int kh, int kw,
                               int pad_h, int pad_w, int stride_h, int stride_w,
                               int dilation_h, int dilation_w, int groups,
                               int dtype, int mode, char **err) {
  if (!ensure_ctx() || !ensure_dnn()) {
    if (err) *err = dup_cstr("cuDNN is not available");
    return -1;
  }

  std::vector<int> key = {n, cin, h, width, cout, kh, kw, pad_h, pad_w, stride_h, stride_w,
                          dilation_h, dilation_w, groups, dtype};
  auto it = g_conv_plans.find(key);
  if (it == g_conv_plans.end()) {
    // The forward pass creates the plan. Training always runs forward first, so
    // a missing plan is rejected explicitly as a call-order error.
    if (err) *err = dup_cstr("no cached convolution plan; run the forward pass first");
    return -1;
  }

  ConvPlan &p = it->second;
  size_t need = mode == 0 ? p.bwd_data_workspace : p.bwd_filter_workspace;
  if (need == SIZE_MAX) {
    if (err) {
      *err = dup_cstr(mode == 0 ? "cuDNN found no backward-data convolution algorithm"
                                : "cuDNN found no backward-filter convolution algorithm");
    }
    return -1;
  }

  CUdeviceptr ws = 0;
  if (!conv_workspace_alloc(need, &ws)) {
    if (err) *err = dup_cstr("failed to allocate the cuDNN workspace");
    return -1;
  }

  const float alpha = 1.0f, beta = 0.0f;
  cudnnStatus_t st;
  if (mode == 0) {
    st = cudnnConvolutionBackwardData(g_dnn, &alpha, p.w, a, p.y, b, p.conv,
                                      p.bwd_data_algo, (void *)(uintptr_t)ws, need,
                                      &beta, p.x, out);
  } else {
    st = cudnnConvolutionBackwardFilter(g_dnn, &alpha, p.x, a, p.y, b, p.conv,
                                        p.bwd_filter_algo, (void *)(uintptr_t)ws, need,
                                        &beta, p.w, out);
  }
  conv_workspace_free(ws);

  if (st != CUDNN_STATUS_SUCCESS) {
    char buf[160];
    snprintf(buf, sizeof(buf), "%s failed (status %d)",
             mode == 0 ? "cudnnConvolutionBackwardData" : "cudnnConvolutionBackwardFilter",
             (int)st);
    if (err) *err = dup_cstr(buf);
    return -1;
  }
  return 0;
}
#endif // TILEWORK_HAS_CUDNN

// Ruby's end proc calls this while the CUDA runtime is still usable. These
// caches own native descriptors/plans independently of Ruby TypedData, so OS
// process teardown is not an adequate ownership boundary (embedding and
// backend reloads can keep the process alive).
void gpu_bridge_shutdown(void) {
  if (!g_ctx) return;
  cuCtxSetCurrent(g_ctx);
  if (g_compute) cuStreamSynchronize(g_compute);

#ifdef TILEWORK_HAS_CUDNN
  for (auto &entry : g_conv_plans) conv_plan_destroy(&entry.second);
  g_conv_plans.clear();
  if (g_dnn) {
    cudnnDestroy(g_dnn);
    g_dnn = nullptr;
  }
#endif
#ifdef TILEWORK_HAS_CUFFT
  for (auto &entry : g_fft_plans) cufftDestroy(entry.second);
  g_fft_plans.clear();
#endif
#ifdef TILEWORK_HAS_CUBLAS
  if (g_blas) {
    cublasDestroy(g_blas);
    g_blas = nullptr;
  }
#endif
#ifdef TILEWORK_HAS_CURAND
  if (g_rng) {
    curandDestroyGenerator(g_rng);
    g_rng = nullptr;
  }
#endif
}

} // extern "C"
