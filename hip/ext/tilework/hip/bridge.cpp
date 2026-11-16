#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#ifdef TILEWORK_HAS_HIPBLAS
#include <hipblas/hipblas.h>
#endif
#ifdef TILEWORK_HAS_HIPRAND
#include <hiprand/hiprand.h>
#endif
#ifdef TILEWORK_HAS_HIPFFT
#include <hipfft/hipfft.h>
#endif
#ifdef TILEWORK_HAS_MIOPEN
#include <miopen/miopen.h>
#endif

#include <charconv>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <atomic>
#include <map>
#include <string>
#include <tuple>
#include <thread>
#include <vector>

namespace {
struct Pipeline;
std::mutex g_mutex;
bool g_failed = false;
bool g_shutting_down = false;
int g_device = -1;
hipDeviceProp_t g_props{};
hipStream_t g_stream = nullptr;
std::string g_error;
constexpr size_t HOST_STAGING_BYTES = 1 << 20;
std::mutex g_staging_mutex;
void *g_staging = nullptr;
bool g_staging_attempted = false;
std::mutex g_matrix_mutex;
bool g_matrix_probed = false;
bool g_matrix_available = false;
Pipeline *g_matrix_pipeline = nullptr;
#ifdef TILEWORK_HAS_HIPBLAS
hipblasHandle_t g_blas = nullptr;
bool g_blas_failed = false;
bool g_blas_probed = false;
bool g_blas_available = false;
#endif
#ifdef TILEWORK_HAS_HIPFFT
std::map<std::tuple<int, long, long>, hipfftHandle> g_fft_plans;
bool g_fft_probed = false;
bool g_fft_available = false;
#endif
#ifdef TILEWORK_HAS_HIPRAND
hiprandGenerator_t g_rng = nullptr;
bool g_rng_failed = false;
bool g_rng_probed = false;
bool g_rng_available = false;
#endif
#ifdef TILEWORK_HAS_MIOPEN
struct MIOpenPlan {
  miopenTensorDescriptor_t x = nullptr, w = nullptr, y = nullptr;
  miopenConvolutionDescriptor_t conv = nullptr;
  miopenConvFwdAlgorithm_t algorithm{};
  void *workspace = nullptr;
  size_t workspace_bytes = 0;
  miopenConvBwdDataAlgorithm_t backward_data_algorithm{};
  void *backward_data_workspace = nullptr;
  size_t backward_data_workspace_bytes = 0;
  bool backward_data_ready = false;
  miopenConvBwdWeightsAlgorithm_t backward_weights_algorithm{};
  void *backward_weights_workspace = nullptr;
  size_t backward_weights_workspace_bytes = 0;
  bool backward_weights_ready = false;
};
miopenHandle_t g_miopen = nullptr;
bool g_miopen_failed = false;
bool g_miopen_probed = false;
bool g_miopen_available = false;
bool g_miopen_backward_probed = false;
bool g_miopen_backward_available = false;
std::map<std::string, MIOpenPlan *> g_miopen_plans;
#endif

#ifdef TILEWORK_HAS_HIPFFT
bool ensure_runtime();
std::string hipfft_error(const char *operation, hipfftResult status) {
  return std::string(operation) + ": hipFFT status " + std::to_string(static_cast<int>(status));
}

bool fft_plan(int kind, long n, long batch, hipfftHandle *out) {
  if (!ensure_runtime()) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto key = std::make_tuple(kind, n, batch);
  auto found = g_fft_plans.find(key);
  if (found != g_fft_plans.end()) { *out = found->second; return true; }
  hipfftType type = kind == 0 ? HIPFFT_R2C : (kind == 1 ? HIPFFT_C2R : HIPFFT_C2C);
  hipfftHandle plan{};
  int length = static_cast<int>(n);
  hipfftResult status = hipfftPlanMany(&plan, 1, &length, nullptr, 1, 0, nullptr, 1, 0, type, static_cast<int>(batch));
  if (status == HIPFFT_SUCCESS) status = hipfftSetStream(plan, g_stream);
  if (status != HIPFFT_SUCCESS) {
    if (plan) (void)hipfftDestroy(plan);
    g_error = hipfft_error("hipFFT plan creation", status);
    return false;
  }
  g_fft_plans.emplace(key, plan);
  *out = plan;
  return true;
}
#endif

bool ensure_runtime();
std::string hip_error(const char *operation, hipError_t status);

bool ensure_staging() {
  std::lock_guard<std::mutex> lock(g_staging_mutex);
  if (g_staging) return true;
  if (g_staging_attempted) return false;
  g_staging_attempted = true;
  hipError_t status = hipHostMalloc(&g_staging, HOST_STAGING_BYTES, hipHostMallocDefault);
  if (status != hipSuccess) {
    g_staging = nullptr;
    g_error = hip_error("HIP pinned staging allocation", status);
    return false;
  }
  return true;
}

#ifdef TILEWORK_HAS_MIOPEN
std::string miopen_error(const char *operation, miopenStatus_t status) {
  return std::string(operation) + ": MIOpen status " + std::to_string(static_cast<int>(status));
}

void miopen_plan_free(MIOpenPlan *plan) {
  if (!plan) return;
  if (plan->workspace) (void)hipFree(plan->workspace);
  if (plan->backward_data_workspace) (void)hipFree(plan->backward_data_workspace);
  if (plan->backward_weights_workspace) (void)hipFree(plan->backward_weights_workspace);
  if (plan->x) (void)miopenDestroyTensorDescriptor(plan->x);
  if (plan->w) (void)miopenDestroyTensorDescriptor(plan->w);
  if (plan->y) (void)miopenDestroyTensorDescriptor(plan->y);
  if (plan->conv) (void)miopenDestroyConvolutionDescriptor(plan->conv);
  delete plan;
}

bool ensure_miopen() {
  if (!ensure_runtime()) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_miopen) return true;
  if (g_miopen_failed) return false;
  miopenStatus_t status = miopenCreateWithStream(&g_miopen, reinterpret_cast<miopenAcceleratorQueue_t>(g_stream));
  if (status != miopenStatusSuccess) {
    g_miopen = nullptr;
    g_miopen_failed = true;
    g_error = miopen_error("miopenCreateWithStream", status);
    return false;
  }
  return true;
}

std::string miopen_key(int n, int cin, int h, int width, int cout, int kh, int kw,
                       int pad_h, int pad_w, int stride_h, int stride_w,
                       int dilation_h, int dilation_w, int groups, int dtype) {
  return std::to_string(n) + ":" + std::to_string(cin) + ":" + std::to_string(h) + ":" + std::to_string(width) + ":" +
         std::to_string(cout) + ":" + std::to_string(kh) + ":" + std::to_string(kw) + ":" +
         std::to_string(pad_h) + ":" + std::to_string(pad_w) + ":" + std::to_string(stride_h) + ":" + std::to_string(stride_w) + ":" +
         std::to_string(dilation_h) + ":" + std::to_string(dilation_w) + ":" + std::to_string(groups) + ":" + std::to_string(dtype);
}

bool miopen_plan(MIOpenPlan **out, void *x, void *w, void *y,
                 int n, int cin, int h, int width, int cout, int kh, int kw,
                 int pad_h, int pad_w, int stride_h, int stride_w,
                 int dilation_h, int dilation_w, int groups, int dtype) {
  if (!ensure_miopen()) return false;
  const std::string key = miopen_key(n, cin, h, width, cout, kh, kw, pad_h, pad_w, stride_h, stride_w,
                                     dilation_h, dilation_w, groups, dtype);
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_miopen_plans.find(key);
    if (found != g_miopen_plans.end()) { *out = found->second; return true; }
  }
  auto *plan = new MIOpenPlan;
  const miopenDataType_t data_type = dtype == 0 ? miopenFloat :
                                     (dtype == 1 ? miopenHalf : miopenBFloat16);
  int oh = 0, ow = 0, on = 0, oc = 0, returned = 0;
  miopenStatus_t status = miopenCreateTensorDescriptor(&plan->x);
  if (status == miopenStatusSuccess) status = miopenCreateTensorDescriptor(&plan->w);
  if (status == miopenStatusSuccess) status = miopenCreateTensorDescriptor(&plan->y);
  if (status == miopenStatusSuccess) status = miopenCreateConvolutionDescriptor(&plan->conv);
  if (status == miopenStatusSuccess) status = miopenSet4dTensorDescriptor(plan->x, data_type, n, cin, h, width);
  if (status == miopenStatusSuccess) status = miopenSet4dTensorDescriptor(plan->w, data_type, cout, cin / groups, kh, kw);
  if (status == miopenStatusSuccess) status = miopenInitConvolutionDescriptor(plan->conv, miopenConvolution,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  if (status == miopenStatusSuccess && groups != 1) status = miopenSetConvolutionGroupCount(plan->conv, groups);
  if (status == miopenStatusSuccess) status = miopenGetConvolutionForwardOutputDim(plan->conv, plan->x, plan->w, &on, &oc, &oh, &ow);
  if (status == miopenStatusSuccess && (on != n || oc != cout || oh <= 0 || ow <= 0)) status = miopenStatusBadParm;
  if (status == miopenStatusSuccess) status = miopenSet4dTensorDescriptor(plan->y, data_type, on, oc, oh, ow);
  if (status == miopenStatusSuccess) status = miopenConvolutionForwardGetWorkSpaceSize(g_miopen, plan->w, plan->x, plan->conv, plan->y, &plan->workspace_bytes);
  hipError_t hip_status = hipSuccess;
  if (status == miopenStatusSuccess && plan->workspace_bytes != 0) hip_status = hipMalloc(&plan->workspace, plan->workspace_bytes);
  miopenConvAlgoPerf_t perf{};
  if (status == miopenStatusSuccess && hip_status == hipSuccess)
    status = miopenFindConvolutionForwardAlgorithm(g_miopen, plan->x, x, plan->w, w, plan->conv, plan->y, y,
                                                   1, &returned, &perf, plan->workspace, plan->workspace_bytes, false);
  if (status == miopenStatusSuccess && returned != 1) status = miopenStatusUnknownError;
  if (status != miopenStatusSuccess || hip_status != hipSuccess) {
    g_error = status != miopenStatusSuccess ? miopen_error("MIOpen forward algorithm search", status)
                                            : hip_error("MIOpen workspace allocation", hip_status);
    miopen_plan_free(plan);
    return false;
  }
  plan->algorithm = perf.fwd_algo;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto [where, inserted] = g_miopen_plans.emplace(key, plan);
  if (!inserted) { miopen_plan_free(plan); }
  *out = where->second;
  return true;
}

bool miopen_prepare_backward_data(MIOpenPlan *plan, void *dy, void *w, void *dx) {
  if (plan->backward_data_ready) return true;
  miopenStatus_t status = miopenConvolutionBackwardDataGetWorkSpaceSize(g_miopen, plan->y, plan->w, plan->conv, plan->x,
                                                                          &plan->backward_data_workspace_bytes);
  hipError_t hip_status = hipSuccess;
  if (status == miopenStatusSuccess && plan->backward_data_workspace_bytes != 0)
    hip_status = hipMalloc(&plan->backward_data_workspace, plan->backward_data_workspace_bytes);
  miopenConvAlgoPerf_t perf{};
  int returned = 0;
  if (status == miopenStatusSuccess && hip_status == hipSuccess)
    status = miopenFindConvolutionBackwardDataAlgorithm(g_miopen, plan->y, dy, plan->w, w, plan->conv, plan->x, dx,
                                                         1, &returned, &perf, plan->backward_data_workspace,
                                                         plan->backward_data_workspace_bytes, false);
  if (status == miopenStatusSuccess && returned != 1) status = miopenStatusUnknownError;
  if (status != miopenStatusSuccess || hip_status != hipSuccess) {
    g_error = status != miopenStatusSuccess ? miopen_error("MIOpen backward-data algorithm search", status)
                                            : hip_error("MIOpen backward-data workspace allocation", hip_status);
    return false;
  }
  plan->backward_data_algorithm = perf.bwd_data_algo;
  plan->backward_data_ready = true;
  return true;
}

bool miopen_prepare_backward_weights(MIOpenPlan *plan, void *x, void *dy, void *dw) {
  if (plan->backward_weights_ready) return true;
  miopenStatus_t status = miopenConvolutionBackwardWeightsGetWorkSpaceSize(g_miopen, plan->y, plan->x, plan->conv, plan->w,
                                                                             &plan->backward_weights_workspace_bytes);
  hipError_t hip_status = hipSuccess;
  if (status == miopenStatusSuccess && plan->backward_weights_workspace_bytes != 0)
    hip_status = hipMalloc(&plan->backward_weights_workspace, plan->backward_weights_workspace_bytes);
  miopenConvAlgoPerf_t perf{};
  int returned = 0;
  if (status == miopenStatusSuccess && hip_status == hipSuccess)
    status = miopenFindConvolutionBackwardWeightsAlgorithm(g_miopen, plan->y, dy, plan->x, x, plan->conv, plan->w, dw,
                                                            1, &returned, &perf, plan->backward_weights_workspace,
                                                            plan->backward_weights_workspace_bytes, false);
  if (status == miopenStatusSuccess && returned != 1) status = miopenStatusUnknownError;
  if (status != miopenStatusSuccess || hip_status != hipSuccess) {
    g_error = status != miopenStatusSuccess ? miopen_error("MIOpen backward-filter algorithm search", status)
                                            : hip_error("MIOpen backward-filter workspace allocation", hip_status);
    return false;
  }
  plan->backward_weights_algorithm = perf.bwd_weights_algo;
  plan->backward_weights_ready = true;
  return true;
}

MIOpenPlan *miopen_existing_plan(int n, int cin, int h, int width, int cout, int kh, int kw,
                                 int pad_h, int pad_w, int stride_h, int stride_w,
                                 int dilation_h, int dilation_w, int groups, int dtype) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto found = g_miopen_plans.find(miopen_key(n, cin, h, width, cout, kh, kw, pad_h, pad_w, stride_h, stride_w,
                                              dilation_h, dilation_w, groups, dtype));
  if (found == g_miopen_plans.end()) {
    g_error = "MIOpen convolution backward requires the matching forward plan to run first";
    return nullptr;
  }
  return found->second;
}
#endif

const char *rocm_include_dir() {
  static std::string path;
  if (path.empty()) {
    const char *home = getenv("ROCM_HOME");
    path = std::string((home && *home) ? home : "/opt/rocm") + "/include";
  }
  return path.c_str();
}

char *dup(const std::string &s) {
  char *out = static_cast<char *>(malloc(s.size() + 1));
  if (out) memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

std::string hip_error(const char *operation, hipError_t status) {
  return std::string(operation) + ": " + hipGetErrorString(status) +
         " (" + std::to_string(static_cast<int>(status)) + ")";
}

bool ensure_runtime();

#ifdef TILEWORK_HAS_HIPBLAS
std::string hipblas_error(const char *operation, hipblasStatus_t status) {
  return std::string(operation) + ": hipBLAS status " + std::to_string(static_cast<int>(status));
}

bool ensure_blas() {
  if (!ensure_runtime()) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_blas) return true;
  if (g_blas_failed) return false;
  hipblasStatus_t status = hipblasCreate(&g_blas);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    g_blas = nullptr;
    g_blas_failed = true;
    g_error = hipblas_error("hipblasCreate", status);
    return false;
  }
  status = hipblasSetStream(g_blas, g_stream);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    (void)hipblasDestroy(g_blas);
    g_blas = nullptr;
    g_blas_failed = true;
    g_error = hipblas_error("hipblasSetStream", status);
    return false;
  }
  return true;
}

bool run_blas_probe() {
  if (g_blas_probed) return g_blas_available;
  g_blas_probed = true;
  if (!ensure_blas()) return false;
  const float a[4] = {1, 2, 3, 4};
  const float b[4] = {5, 6, 7, 8};
  float c[4] = {};
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  hipError_t hip_status = hipMalloc(&da, sizeof(a));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&db, sizeof(b));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&dc, sizeof(c));
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(da, a, sizeof(a), hipMemcpyHostToDevice, g_stream);
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(db, b, sizeof(b), hipMemcpyHostToDevice, g_stream);
  const float alpha = 1.0f, beta = 0.0f;
  hipblasStatus_t blas_status = hip_status == hipSuccess
    ? hipblasSgemm(g_blas, HIPBLAS_OP_N, HIPBLAS_OP_N, 2, 2, 2,
                   &alpha, db, 2, da, 2, &beta, dc, 2)
    : HIPBLAS_STATUS_INTERNAL_ERROR;
  if (hip_status == hipSuccess && blas_status == HIPBLAS_STATUS_SUCCESS)
    hip_status = hipMemcpyAsync(c, dc, sizeof(c), hipMemcpyDeviceToHost, g_stream);
  if (hip_status == hipSuccess && blas_status == HIPBLAS_STATUS_SUCCESS)
    hip_status = hipStreamSynchronize(g_stream);
  if (da) (void)hipFree(da);
  if (db) (void)hipFree(db);
  if (dc) (void)hipFree(dc);
  if (hip_status != hipSuccess) g_error = hip_error("hipBLAS execution probe", hip_status);
  else if (blas_status != HIPBLAS_STATUS_SUCCESS) g_error = hipblas_error("hipblasSgemm execution probe", blas_status);
  else if (c[0] != 19 || c[1] != 22 || c[2] != 43 || c[3] != 50)
    g_error = "hipBLAS execution probe returned an incorrect result";
  else g_blas_available = true;
  return g_blas_available;
}
#endif

#ifdef TILEWORK_HAS_HIPRAND
std::string hiprand_error(const char *operation, hiprandStatus_t status) {
  return std::string(operation) + ": hipRAND status " + std::to_string(static_cast<int>(status));
}

bool ensure_rng() {
  if (!ensure_runtime()) return false;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_rng) return true;
  if (g_rng_failed) return false;
  hiprandStatus_t status = hiprandCreateGenerator(&g_rng, HIPRAND_RNG_PSEUDO_PHILOX4_32_10);
  if (status == HIPRAND_STATUS_SUCCESS) status = hiprandSetStream(g_rng, g_stream);
  if (status != HIPRAND_STATUS_SUCCESS) {
    if (g_rng) (void)hiprandDestroyGenerator(g_rng);
    g_rng = nullptr;
    g_rng_failed = true;
    g_error = hiprand_error("hipRAND initialization", status);
    return false;
  }
  return true;
}

bool run_rng_probe() {
  if (g_rng_probed) return g_rng_available;
  g_rng_probed = true;
  if (!ensure_rng()) return false;
  float *device = nullptr;
  float host[4] = {};
  hipError_t hip_status = hipMalloc(&device, sizeof(host));
  hiprandStatus_t rand_status = hip_status == hipSuccess
    ? hiprandSetPseudoRandomGeneratorSeed(g_rng, 1234) : HIPRAND_STATUS_INTERNAL_ERROR;
  if (rand_status == HIPRAND_STATUS_SUCCESS) rand_status = hiprandSetGeneratorOffset(g_rng, 0);
  if (rand_status == HIPRAND_STATUS_SUCCESS) rand_status = hiprandGenerateUniform(g_rng, device, 4);
  if (hip_status == hipSuccess && rand_status == HIPRAND_STATUS_SUCCESS)
    hip_status = hipMemcpyAsync(host, device, sizeof(host), hipMemcpyDeviceToHost, g_stream);
  if (hip_status == hipSuccess && rand_status == HIPRAND_STATUS_SUCCESS)
    hip_status = hipStreamSynchronize(g_stream);
  if (device) (void)hipFree(device);
  if (hip_status != hipSuccess) g_error = hip_error("hipRAND execution probe", hip_status);
  else if (rand_status != HIPRAND_STATUS_SUCCESS) g_error = hiprand_error("hipRAND execution probe", rand_status);
  else if (!std::all_of(std::begin(host), std::end(host), [](float x) { return x > 0.0f && x <= 1.0f; }))
    g_error = "hipRAND execution probe returned values outside (0, 1]";
  else g_rng_available = true;
  return g_rng_available;
}
#endif
bool selected_device(int *out) {
  const char *value = getenv("TILEWORK_DEVICE");
  if (!value || !*value) { *out = 0; return true; }
  const char *end = value + strlen(value);
  int selected = 0;
  auto result = std::from_chars(value, end, selected, 10);
  if (result.ec != std::errc() || result.ptr != end || selected < 0) {
    g_error = std::string("invalid TILEWORK_DEVICE=") + value + "; expected a non-negative decimal index";
    return false;
  }
  *out = selected;
  return true;
}

bool ensure_runtime() {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_shutting_down) return false;
  if (g_stream) return hipSetDevice(g_device) == hipSuccess;
  if (g_failed) return false;

  int requested = 0, count = 0;
  if (!selected_device(&requested)) { g_failed = true; return false; }
  hipError_t status = hipGetDeviceCount(&count);
  if (status != hipSuccess) g_error = hip_error("hipGetDeviceCount", status);
  else if (requested >= count) {
    std::string names;
    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t p{};
      if (hipGetDeviceProperties(&p, i) == hipSuccess) {
        if (!names.empty()) names += ", ";
        names += p.name;
      }
    }
    g_error = "TILEWORK_DEVICE=" + std::to_string(requested) +
              " is outside the visible HIP device count (" + std::to_string(count) +
              "; devices: " + names + ")";
  } else if ((status = hipSetDevice(requested)) != hipSuccess) g_error = hip_error("hipSetDevice", status);
  else if ((status = hipGetDeviceProperties(&g_props, requested)) != hipSuccess) g_error = hip_error("hipGetDeviceProperties", status);
  else if ((status = hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking)) != hipSuccess) g_error = hip_error("hipStreamCreateWithFlags", status);
  else { g_device = requested; return true; }
  g_failed = true;
  if (g_stream) { (void)hipStreamDestroy(g_stream); g_stream = nullptr; }
  return false;
}

struct Pipeline { hipModule_t module; hipFunction_t function; };
struct HostBacking { void *data; long bytes; };
struct HostTransfer {
  std::thread worker;
  std::atomic<bool> done{false};
  int status = 0;
  std::string error;
};
std::atomic<int> g_live_host_transfers{0};
constexpr int MAX_HOST_TRANSFERS = 1;

void set_error(char **out, const std::string &message) { if (out) *out = dup(message); }
} // namespace

extern "C" {
char *gpu_bridge_device_name(void) {
  return ensure_runtime() ? dup(g_props.name) : nullptr;
}
int gpu_bridge_device_available(void) { return ensure_runtime() ? 1 : 0; }
char *gpu_bridge_last_error(void) { return g_error.empty() ? nullptr : dup(g_error); }
char *gpu_bridge_device_architecture(void) {
  return ensure_runtime() ? dup(g_props.gcnArchName) : nullptr;
}
int gpu_bridge_device_index(void) { return ensure_runtime() ? g_device : -1; }

void *gpu_bridge_buffer_alloc(long bytes) {
  if (bytes <= 0 || !ensure_runtime()) return nullptr;
  void *buffer = nullptr;
  if (hipMalloc(&buffer, static_cast<size_t>(bytes)) != hipSuccess) return nullptr;
  // The zero fill is queued on g_stream, and every kernel, copy and free this
  // handle can reach goes to that same stream, so ordering already guarantees
  // nobody observes the uninitialized memory.  Waiting here would put a host
  // synchronization in front of *every* allocation, which is exactly the cost
  // Tilework.submit exists to remove (an eager op allocates its output).
  if (hipMemsetAsync(buffer, 0, static_cast<size_t>(bytes), g_stream) != hipSuccess) {
    (void)hipFree(buffer);
    return nullptr;
  }
  return buffer;
}
void gpu_bridge_buffer_write(void *handle, const void *src, long bytes, long offset) {
  if (ensure_runtime() &&
      hipMemcpyAsync(static_cast<char *>(handle) + offset, src, bytes, hipMemcpyHostToDevice, g_stream) == hipSuccess) {
    // The Ruby ABI releases the source String after this call.  Complete the
    // pageable-host copy before returning so it cannot outlive that String.
    (void)hipStreamSynchronize(g_stream);
  }
}
void gpu_bridge_buffer_read(void *handle, void *dst, long bytes, long offset) {
  if (ensure_runtime() && hipMemcpyAsync(dst, static_cast<char *>(handle) + offset, bytes, hipMemcpyDeviceToHost, g_stream) == hipSuccess)
    (void)hipStreamSynchronize(g_stream);
}
void gpu_bridge_buffer_free(void *handle) { if (handle && ensure_runtime()) (void)hipFree(handle); }

int gpu_bridge_host_backing_available(void) { return ensure_runtime() ? 1 : 0; }
void *gpu_bridge_host_backing_alloc(long bytes) {
  if (bytes <= 0) { g_error = "host backing size must be positive"; return nullptr; }
  auto *backing = static_cast<HostBacking *>(malloc(sizeof(HostBacking)));
  if (!backing) { g_error = "failed to allocate host backing metadata"; return nullptr; }
  backing->data = malloc(static_cast<size_t>(bytes));
  backing->bytes = bytes;
  if (!backing->data) { free(backing); g_error = "failed to allocate pageable host backing"; return nullptr; }
  return backing;
}
void gpu_bridge_host_backing_free(void *handle) {
  auto *backing = static_cast<HostBacking *>(handle);
  if (!backing) return;
  free(backing->data);
  backing->data = nullptr;
  free(backing);
}
int gpu_bridge_host_backing_store(void *host, void *device, long bytes, long offset, char **error_out) {
  auto *backing = static_cast<HostBacking *>(host);
  if (!backing || !device || bytes < 0 || offset < 0 || offset + bytes > backing->bytes) {
    set_error(error_out, "host backing store range is out of bounds"); return 1;
  }
  if (!ensure_runtime()) { set_error(error_out, g_error); return 1; }
  hipError_t status = hipSuccess;
  if (ensure_staging()) {
    std::lock_guard<std::mutex> lock(g_staging_mutex);
    for (long copied = 0; copied < bytes && status == hipSuccess; ) {
      const size_t chunk = std::min(static_cast<size_t>(bytes - copied), HOST_STAGING_BYTES);
      status = hipMemcpyAsync(g_staging, static_cast<char *>(device) + copied, chunk, hipMemcpyDeviceToHost, g_stream);
      if (status == hipSuccess) status = hipStreamSynchronize(g_stream);
      if (status == hipSuccess) memcpy(static_cast<char *>(backing->data) + offset + copied, g_staging, chunk);
      copied += static_cast<long>(chunk);
    }
  } else {
    status = hipMemcpyAsync(static_cast<char *>(backing->data) + offset, device, static_cast<size_t>(bytes), hipMemcpyDeviceToHost, g_stream);
    if (status == hipSuccess) status = hipStreamSynchronize(g_stream);
  }
  if (status != hipSuccess) { set_error(error_out, hip_error("host backing device-to-host copy", status)); return 1; }
  return 0;
}
int gpu_bridge_host_backing_restore(void *host, void *device, long bytes, long offset, char **error_out) {
  auto *backing = static_cast<HostBacking *>(host);
  if (!backing || !device || bytes < 0 || offset < 0 || offset + bytes > backing->bytes) {
    set_error(error_out, "host backing restore range is out of bounds"); return 1;
  }
  if (!ensure_runtime()) { set_error(error_out, g_error); return 1; }
  hipError_t status = hipSuccess;
  if (ensure_staging()) {
    std::lock_guard<std::mutex> lock(g_staging_mutex);
    for (long copied = 0; copied < bytes && status == hipSuccess; ) {
      const size_t chunk = std::min(static_cast<size_t>(bytes - copied), HOST_STAGING_BYTES);
      memcpy(g_staging, static_cast<char *>(backing->data) + offset + copied, chunk);
      status = hipMemcpyAsync(static_cast<char *>(device) + copied, g_staging, chunk, hipMemcpyHostToDevice, g_stream);
      if (status == hipSuccess) status = hipStreamSynchronize(g_stream);
      copied += static_cast<long>(chunk);
    }
  } else {
    status = hipMemcpyAsync(device, static_cast<char *>(backing->data) + offset, static_cast<size_t>(bytes), hipMemcpyHostToDevice, g_stream);
    if (status == hipSuccess) status = hipStreamSynchronize(g_stream);
  }
  if (status != hipSuccess) { set_error(error_out, hip_error("host backing host-to-device copy", status)); return 1; }
  return 0;
}
void *gpu_bridge_host_backing_transfer_begin(void *host, void *device, long bytes, long offset, char **error_out) {
  if (g_live_host_transfers.fetch_add(1) >= MAX_HOST_TRANSFERS) {
    g_live_host_transfers.fetch_sub(1);
    set_error(error_out, "the HIP host-backing transfer limit is reached");
    return nullptr;
  }
  auto *transfer = new HostTransfer;
  transfer->worker = std::thread([transfer, host, device, bytes, offset] {
    char *error = nullptr;
    transfer->status = gpu_bridge_host_backing_restore(host, device, bytes, offset, &error);
    if (error) { transfer->error = error; free(error); }
    transfer->done.store(true, std::memory_order_release);
  });
  return transfer;
}
int gpu_bridge_host_backing_transfer_poll(void *handle) {
  auto *transfer = static_cast<HostTransfer *>(handle);
  return transfer && transfer->done.load(std::memory_order_acquire) ? 1 : 0;
}
int gpu_bridge_host_backing_transfer_end(void *handle, char **error_out) {
  auto *transfer = static_cast<HostTransfer *>(handle);
  if (!transfer) { set_error(error_out, "invalid HIP host-backing transfer token"); return 1; }
  if (transfer->worker.joinable()) transfer->worker.join();
  const int status = transfer->status;
  if (status != 0) set_error(error_out, transfer->error.empty() ? "HIP host-backing transfer failed" : transfer->error);
  delete transfer;
  g_live_host_transfers.fetch_sub(1);
  return status;
}

void *gpu_bridge_pipeline_create(const char *source, const char *name, char **error_out) {
  if (!ensure_runtime()) { set_error(error_out, g_error); return nullptr; }
  hiprtcProgram program{};
  hiprtcResult rc = hiprtcCreateProgram(&program, source, "tilework.hip", 0, nullptr, nullptr);
  if (rc != HIPRTC_SUCCESS) { set_error(error_out, "hiprtcCreateProgram failed"); return nullptr; }
  std::string arch = std::string("--gpu-architecture=") + g_props.gcnArchName;
  std::string include = std::string("-I") + rocm_include_dir();
  const char *options[] = { "--std=c++17", arch.c_str(), include.c_str() };
  rc = hiprtcCompileProgram(program, 3, options);
  size_t log_size = 0; hiprtcGetProgramLogSize(program, &log_size);
  std::string log(log_size, '\0'); if (log_size) hiprtcGetProgramLog(program, log.data());
  if (rc != HIPRTC_SUCCESS) { hiprtcDestroyProgram(&program); set_error(error_out, "HIPRTC compilation failed for " + arch + ":\n" + log); return nullptr; }
  size_t code_size = 0;
  if (hiprtcGetCodeSize(program, &code_size) != HIPRTC_SUCCESS) { hiprtcDestroyProgram(&program); set_error(error_out, "hiprtcGetCodeSize failed"); return nullptr; }
  std::vector<char> code(code_size);
  if (hiprtcGetCode(program, code.data()) != HIPRTC_SUCCESS) { hiprtcDestroyProgram(&program); set_error(error_out, "hiprtcGetCode failed"); return nullptr; }
  hiprtcDestroyProgram(&program);
  Pipeline *pipeline = new Pipeline{};
  hipError_t status = hipModuleLoadData(&pipeline->module, code.data());
  if (status == hipSuccess) status = hipModuleGetFunction(&pipeline->function, pipeline->module, name);
  if (status != hipSuccess) { if (pipeline->module) (void)hipModuleUnload(pipeline->module); delete pipeline; set_error(error_out, hip_error("HIP module/function load", status)); return nullptr; }
  return pipeline;
}
long gpu_bridge_pipeline_max_threads(void *handle) {
  // HIP's module API does not expose CUDA's per-function max-thread query on
  // all supported ROCm versions.  The device limit remains a safe upper bound;
  // HIP validates any stricter kernel-specific limit at launch.
  (void)handle;
  return g_props.maxThreadsPerBlock;
}
void gpu_bridge_pipeline_free(void *handle) { auto *p = static_cast<Pipeline *>(handle); if (p) { (void)hipModuleUnload(p->module); delete p; } }

int gpu_bridge_dispatch(void *pipeline, const int32_t *kinds, void *const *ptrs, const long *lens, int32_t count,
                        long gx, long gy, long gz, long bx, long by, long bz, int32_t wait, char **error_out) {
  if (!ensure_runtime()) { set_error(error_out, g_error); return 1; }
  std::vector<void *> values(count), params(count);
  for (int i = 0; i < count; ++i) {
    if (kinds[i] == 0) { values[i] = static_cast<char *>(ptrs[i]) + lens[i]; params[i] = &values[i]; }
    else params[i] = ptrs[i];
  }
  hipError_t status = hipModuleLaunchKernel(static_cast<Pipeline *>(pipeline)->function,
    gx, gy, gz, bx, by, bz, 0, g_stream, params.data(), nullptr);
  if (status == hipSuccess && wait) status = hipStreamSynchronize(g_stream);
  if (status != hipSuccess) { set_error(error_out, hip_error("hipModuleLaunchKernel", status)); return 1; }
  return 0;
}
void gpu_bridge_synchronize(void) { if (ensure_runtime()) (void)hipStreamSynchronize(g_stream); }
void gpu_bridge_shutdown(void) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_shutting_down = true;
  // All lazy library handles and plans are bound to this stream.  Drain it
  // before destroying any of them; hipFFT in particular can otherwise retain
  // work that touches plan-owned state during Ruby's end-proc sequence.
  if (g_stream) (void)hipStreamSynchronize(g_stream);
  {
    std::lock_guard<std::mutex> lock(g_staging_mutex);
    if (g_staging) {
      (void)hipHostFree(g_staging);
      g_staging = nullptr;
    }
  }
#ifdef TILEWORK_HAS_MIOPEN
  // Descriptor/workspace ownership belongs to MIOpen and must be released
  // while its handle and the HIP stream are still valid.
  for (auto &[_, plan] : g_miopen_plans) miopen_plan_free(plan);
  g_miopen_plans.clear();
  if (g_miopen) {
    (void)miopenDestroy(g_miopen);
    g_miopen = nullptr;
  }
#endif
#ifdef TILEWORK_HAS_HIPFFT
  // hipFFT must be released before the remaining ROCm library handles on this
  // WSL2 stack. MIOpen owns descriptor workspaces, so it is drained first.
  for (auto &[_, plan] : g_fft_plans) (void)hipfftDestroy(plan);
  g_fft_plans.clear();
#endif
#ifdef TILEWORK_HAS_HIPBLAS
  if (g_blas) {
    (void)hipblasDestroy(g_blas);
    g_blas = nullptr;
  }
#endif
#ifdef TILEWORK_HAS_HIPRAND
  if (g_rng) {
    (void)hiprandDestroyGenerator(g_rng);
    g_rng = nullptr;
  }
#endif
  if (g_stream) {
    (void)hipStreamDestroy(g_stream);
    g_stream = nullptr;
  }
  {
    std::lock_guard<std::mutex> matrix_lock(g_matrix_mutex);
    if (g_matrix_pipeline) {
      gpu_bridge_pipeline_free(g_matrix_pipeline);
      g_matrix_pipeline = nullptr;
    }
  }
  // MIOpen keeps process-global ROCm state whose own DSO teardown runs after
  // Ruby end-procs.  Resetting the device here races that teardown on ROCm
  // 7.2/WSL2; all extension-owned resources above are already released.
  // Leave final device destruction to the HIP runtime's process shutdown.
  g_device = -1;
}
int gpu_bridge_device_capabilities(int32_t *warp, int32_t *major, int32_t *minor, int32_t *integrated, long *budget) {
  if (!ensure_runtime()) return 1;
  size_t free_bytes = 0, total = 0;
  if (hipMemGetInfo(&free_bytes, &total) != hipSuccess) return 1;
  *warp = g_props.warpSize; *major = 0; *minor = 0; *integrated = g_props.integrated; *budget = static_cast<long>(free_bytes);
  return 0;
}

static const char *matrix_core_source() {
  return R"HIP(
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
extern "C" __global__ void tilework_rocwmma_gemm(const float16_t* a, const float16_t* b,
                                                  float* c, int m, int k, int n) {
  const int row = blockIdx.y * 16;
  const int col = blockIdx.x * 16;
  using FragA = fragment<matrix_a, 16, 16, 16, float16_t, row_major>;
  using FragB = fragment<matrix_b, 16, 16, 16, float16_t, row_major>;
  using Accum = fragment<accumulator, 16, 16, 16, float, row_major>;
  Accum acc;
  fill_fragment(acc, 0.0f);
  for (int kk = 0; kk < k; kk += 16) {
    FragA fa;
    FragB fb;
    load_matrix_sync(fa, a + row * k + kk, k);
    load_matrix_sync(fb, b + kk * n + col, n);
    mma_sync(acc, fa, fb, acc);
  }
  store_matrix_sync(c + row * n + col, acc, n);
}
)HIP";
}

bool ensure_matrix_pipeline(char **error_out) {
  std::lock_guard<std::mutex> lock(g_matrix_mutex);
  if (g_matrix_pipeline) return true;
  void *pipeline = gpu_bridge_pipeline_create(matrix_core_source(), "tilework_rocwmma_gemm", error_out);
  if (!pipeline) return false;
  g_matrix_pipeline = static_cast<Pipeline *>(pipeline);
  return true;
}

int gpu_bridge_matrix_core_gemm(void *a, long a_off, void *b, long b_off,
                                void *c, long c_off, long m, long k, long n,
                                char **error_out) {
  if (m <= 0 || k <= 0 || n <= 0 || m % 16 || k % 16 || n % 16) {
    set_error(error_out, "rocWMMA GEMM requires positive m, k, and n that are multiples of 16");
    return 1;
  }
  if (!ensure_runtime()) { set_error(error_out, g_error); return 1; }
  if (!ensure_matrix_pipeline(error_out)) return 1;
  int32_t kinds[] = {0, 0, 0, 1, 1, 1};
  int32_t dims[] = {static_cast<int32_t>(m), static_cast<int32_t>(k), static_cast<int32_t>(n)};
  void *args[] = {a, b, c, &dims[0], &dims[1], &dims[2]};
  long lens[] = {a_off, b_off, c_off, static_cast<long>(sizeof(int32_t)),
                 static_cast<long>(sizeof(int32_t)), static_cast<long>(sizeof(int32_t))};
  return gpu_bridge_dispatch(g_matrix_pipeline, kinds, args, lens, 6,
                             n / 16, m / 16, 1, 32, 1, 1, 1, error_out);
}

int gpu_bridge_matrix_core_available(void) {
  {
    std::lock_guard<std::mutex> lock(g_matrix_mutex);
    if (g_matrix_probed) return g_matrix_available ? 1 : 0;
  }

  char *compile_error = nullptr;
  bool ok = ensure_matrix_pipeline(&compile_error);
  if (!ok) {
    g_error = compile_error ? compile_error : "rocWMMA pipeline compilation failed";
    if (compile_error) free(compile_error);
  } else {
    std::vector<uint16_t> input(16 * 16, 0x3c00); // float16 1.0
    std::vector<float> output(16 * 16, 0.0f);
    void *a = nullptr, *b = nullptr, *c = nullptr;
    hipError_t status = hipMalloc(&a, input.size() * sizeof(uint16_t));
    if (status == hipSuccess) status = hipMalloc(&b, input.size() * sizeof(uint16_t));
    if (status == hipSuccess) status = hipMalloc(&c, output.size() * sizeof(float));
    if (status == hipSuccess) status = hipMemcpyAsync(a, input.data(), input.size() * sizeof(uint16_t), hipMemcpyHostToDevice, g_stream);
    if (status == hipSuccess) status = hipMemcpyAsync(b, input.data(), input.size() * sizeof(uint16_t), hipMemcpyHostToDevice, g_stream);
    char *dispatch_error = nullptr;
    if (status == hipSuccess)
      status = gpu_bridge_matrix_core_gemm(a, 0, b, 0, c, 0, 16, 16, 16, &dispatch_error) == 0 ? hipSuccess : hipErrorUnknown;
    if (status == hipSuccess) status = hipMemcpyAsync(output.data(), c, output.size() * sizeof(float), hipMemcpyDeviceToHost, g_stream);
    if (status == hipSuccess) status = hipStreamSynchronize(g_stream);
    if (a) (void)hipFree(a);
    if (b) (void)hipFree(b);
    if (c) (void)hipFree(c);
    if (status != hipSuccess) {
      g_error = dispatch_error ? dispatch_error : hip_error("rocWMMA execution probe", status);
      if (dispatch_error) free(dispatch_error);
    } else {
      ok = std::all_of(output.begin(), output.end(), [](float x) { return x == 16.0f; });
      if (!ok) g_error = "rocWMMA execution probe returned an incorrect result";
    }
  }
  std::lock_guard<std::mutex> lock(g_matrix_mutex);
  g_matrix_probed = true;
  g_matrix_available = ok;
  return ok ? 1 : 0;
}
#ifdef TILEWORK_HAS_HIPBLAS
int gpu_bridge_gemm_available(void) { return run_blas_probe() ? 1 : 0; }
int gpu_bridge_gemm(void *a, long a_off, void *b, long b_off, void *c, long c_off,
                    long m, long k, long n, int /*allow_tf32*/, int in_dtype, char **error_out) {
  if (!run_blas_probe()) { set_error(error_out, g_error); return 1; }
  const float alpha = 1.0f, beta = 0.0f;
  void *ap = static_cast<char *>(a) + a_off;
  void *bp = static_cast<char *>(b) + b_off;
  void *cp = static_cast<char *>(c) + c_off;
  hipblasStatus_t status;
  // Tilework buffers are row-major.  C^T = B^T * A^T lets hipBLAS consume
  // those same bytes as column-major matrices without a transpose kernel.
  if (in_dtype == 0) {
    status = hipblasSgemm(g_blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                          &alpha, static_cast<const float *>(bp), static_cast<int>(n),
                          static_cast<const float *>(ap), static_cast<int>(k),
                          &beta, static_cast<float *>(cp), static_cast<int>(n));
  } else if (in_dtype == 1) {
    status = hipblasGemmEx(g_blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                           static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                           &alpha, bp, HIP_R_16F, static_cast<int>(n),
                           ap, HIP_R_16F, static_cast<int>(k),
                           &beta, cp, HIP_R_32F, static_cast<int>(n),
                           HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  } else if (in_dtype == 2) {
    status = hipblasGemmEx(g_blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                           static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                           &alpha, bp, HIP_R_16BF, static_cast<int>(n),
                           ap, HIP_R_16BF, static_cast<int>(k),
                           &beta, cp, HIP_R_32F, static_cast<int>(n),
                           HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  } else {
    set_error(error_out, "hipBLAS GEMM received an unsupported input dtype");
    return 1;
  }
  if (status != HIPBLAS_STATUS_SUCCESS) {
    set_error(error_out, hipblas_error("hipBLAS GEMM", status));
    return 1;
  }
  return 0;
}
#endif
#ifdef TILEWORK_HAS_HIPRAND
int gpu_bridge_random_available(void) { return run_rng_probe() ? 1 : 0; }
int gpu_bridge_random(void *handle, long n, int kind, unsigned long long seed,
                      float mean, float stddev, char **error_out) {
  if (n <= 0) { set_error(error_out, "hipRAND draw length must be positive"); return 1; }
  if (!run_rng_probe()) { set_error(error_out, g_error); return 1; }
  hiprandStatus_t status = hiprandSetPseudoRandomGeneratorSeed(g_rng, seed);
  if (status == HIPRAND_STATUS_SUCCESS) status = hiprandSetGeneratorOffset(g_rng, 0);
  if (status != HIPRAND_STATUS_SUCCESS) { set_error(error_out, hiprand_error("hipRAND seed setup", status)); return 1; }
  if (kind == 0) {
    status = hiprandGenerateUniform(g_rng, static_cast<float *>(handle), static_cast<size_t>(n));
  } else if (kind == 1) {
    if ((n & 1) == 0) {
      status = hiprandGenerateNormal(g_rng, static_cast<float *>(handle), static_cast<size_t>(n), mean, stddev);
    } else {
      float *scratch = nullptr;
      hipError_t hip_status = hipMalloc(&scratch, static_cast<size_t>(n + 1) * sizeof(float));
      if (hip_status != hipSuccess) { set_error(error_out, hip_error("hipRAND normal scratch allocation", hip_status)); return 1; }
      status = hiprandGenerateNormal(g_rng, scratch, static_cast<size_t>(n + 1), mean, stddev);
      if (status == HIPRAND_STATUS_SUCCESS)
        hip_status = hipMemcpyAsync(handle, scratch, static_cast<size_t>(n) * sizeof(float), hipMemcpyDeviceToDevice, g_stream);
      (void)hipFree(scratch);
      if (hip_status != hipSuccess) { set_error(error_out, hip_error("hipRAND normal scratch copy", hip_status)); return 1; }
    }
  } else {
    set_error(error_out, "hipRAND received an unknown distribution kind");
    return 1;
  }
  if (status != HIPRAND_STATUS_SUCCESS) { set_error(error_out, hiprand_error("hipRAND generation", status)); return 1; }
  return 0;
}
#endif
#ifdef TILEWORK_HAS_HIPFFT
int gpu_bridge_fft(void *in, void *out, long n, long batch, int kind, int direction, char **error_out) {
  if (n <= 0 || batch <= 0 || kind < 0 || kind > 2) { set_error(error_out, "hipFFT received invalid transform dimensions or kind"); return 1; }
  hipfftHandle plan{};
  if (!fft_plan(kind, n, batch, &plan)) { set_error(error_out, g_error); return 1; }
  hipfftResult status;
  if (kind == 0) status = hipfftExecR2C(plan, static_cast<hipfftReal *>(in), static_cast<hipfftComplex *>(out));
  else if (kind == 1) status = hipfftExecC2R(plan, static_cast<hipfftComplex *>(in), static_cast<hipfftReal *>(out));
  else status = hipfftExecC2C(plan, static_cast<hipfftComplex *>(in), static_cast<hipfftComplex *>(out),
                              direction < 0 ? HIPFFT_BACKWARD : HIPFFT_FORWARD);
  if (status != HIPFFT_SUCCESS) { set_error(error_out, hipfft_error("hipFFT execution", status)); return 1; }
  return 0;
}

int gpu_bridge_fft_available(void) {
  if (g_fft_probed) return g_fft_available ? 1 : 0;
  g_fft_probed = true;
  float input[4] = {1, 0, 0, 0};
  float output[6] = {};
  void *in = nullptr, *out = nullptr;
  hipError_t hip_status = hipMalloc(&in, sizeof(input));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&out, sizeof(output));
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(in, input, sizeof(input), hipMemcpyHostToDevice, g_stream);
  char *fft_error_text = nullptr;
  if (hip_status == hipSuccess) hip_status = gpu_bridge_fft(in, out, 4, 1, 0, 1, &fft_error_text) == 0 ? hipSuccess : hipErrorUnknown;
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(output, out, sizeof(output), hipMemcpyDeviceToHost, g_stream);
  if (hip_status == hipSuccess) hip_status = hipStreamSynchronize(g_stream);
  if (in) (void)hipFree(in);
  if (out) (void)hipFree(out);
  if (hip_status != hipSuccess) {
    g_error = fft_error_text ? fft_error_text : hip_error("hipFFT execution probe", hip_status);
    if (fft_error_text) free(fft_error_text);
    return 0;
  }
  g_fft_available = output[0] == 1.0f && output[1] == 0.0f &&
                    output[2] == 1.0f && output[3] == 0.0f &&
                    output[4] == 1.0f && output[5] == 0.0f;
  if (!g_fft_available) g_error = "hipFFT execution probe returned an incorrect spectrum";
  return g_fft_available ? 1 : 0;
}
#endif
#ifdef TILEWORK_HAS_MIOPEN
int gpu_bridge_conv2d(void *x, void *w, void *y,
                      int n, int cin, int h, int width, int cout, int kh, int kw,
                      int pad_h, int pad_w, int stride_h, int stride_w,
                      int dilation_h, int dilation_w, int groups, int dtype, char **error_out) {
  if (n <= 0 || cin <= 0 || h <= 0 || width <= 0 || cout <= 0 || kh <= 0 || kw <= 0 ||
      pad_h < 0 || pad_w < 0 || stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 ||
      groups <= 0 || cin % groups != 0 || cout % groups != 0 || (dtype != 0 && dtype != 1 && dtype != 2)) {
    set_error(error_out, "MIOpen convolution received invalid dimensions or dtype");
    return 1;
  }
  MIOpenPlan *plan = nullptr;
  if (!miopen_plan(&plan, x, w, y, n, cin, h, width, cout, kh, kw, pad_h, pad_w, stride_h, stride_w,
                   dilation_h, dilation_w, groups, dtype)) {
    set_error(error_out, g_error);
    return 1;
  }
  const float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t status = miopenConvolutionForward(g_miopen, &alpha, plan->x, x, plan->w, w, plan->conv,
                                                    plan->algorithm, &beta, plan->y, y,
                                                    plan->workspace, plan->workspace_bytes);
  if (status != miopenStatusSuccess) {
    set_error(error_out, miopen_error("MIOpen convolution forward", status));
    return 1;
  }
  return 0;
}

int gpu_bridge_conv2d_backward(void *a, void *b, void *out,
                               int n, int cin, int h, int width, int cout, int kh, int kw,
                               int pad_h, int pad_w, int stride_h, int stride_w,
                               int dilation_h, int dilation_w, int groups, int dtype, int mode, char **error_out) {
  if (!ensure_miopen() || (mode != 0 && mode != 1)) {
    set_error(error_out, mode != 0 && mode != 1 ? "MIOpen convolution backward received an invalid mode" : g_error);
    return 1;
  }
  MIOpenPlan *plan = miopen_existing_plan(n, cin, h, width, cout, kh, kw, pad_h, pad_w, stride_h, stride_w,
                                          dilation_h, dilation_w, groups, dtype);
  if (!plan) { set_error(error_out, g_error); return 1; }
  const float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t status;
  if (mode == 0) {
    if (!miopen_prepare_backward_data(plan, b, a, out)) { set_error(error_out, g_error); return 1; }
    status = miopenConvolutionBackwardData(g_miopen, &alpha, plan->y, b, plan->w, a, plan->conv,
                                           plan->backward_data_algorithm, &beta, plan->x, out,
                                           plan->backward_data_workspace, plan->backward_data_workspace_bytes);
  } else {
    if (!miopen_prepare_backward_weights(plan, a, b, out)) { set_error(error_out, g_error); return 1; }
    status = miopenConvolutionBackwardWeights(g_miopen, &alpha, plan->y, b, plan->x, a, plan->conv,
                                              plan->backward_weights_algorithm, &beta, plan->w, out,
                                              plan->backward_weights_workspace, plan->backward_weights_workspace_bytes);
  }
  if (status != miopenStatusSuccess) {
    set_error(error_out, miopen_error(mode == 0 ? "MIOpen convolution backward-data" : "MIOpen convolution backward-filter", status));
    return 1;
  }
  return 0;
}

int gpu_bridge_conv2d_available(void) {
  if (g_miopen_probed) return g_miopen_available ? 1 : 0;
  g_miopen_probed = true;
  if (!ensure_miopen()) return 0;
  const float input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const float weight[1] = {2};
  float output[9] = {};
  void *x = nullptr, *w = nullptr, *y = nullptr;
  hipError_t hip_status = hipMalloc(&x, sizeof(input));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&w, sizeof(weight));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&y, sizeof(output));
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(x, input, sizeof(input), hipMemcpyHostToDevice, g_stream);
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(w, weight, sizeof(weight), hipMemcpyHostToDevice, g_stream);
  char *conv_error = nullptr;
  if (hip_status == hipSuccess)
    hip_status = gpu_bridge_conv2d(x, w, y, 1, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, &conv_error) == 0
      ? hipSuccess : hipErrorUnknown;
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(output, y, sizeof(output), hipMemcpyDeviceToHost, g_stream);
  if (hip_status == hipSuccess) hip_status = hipStreamSynchronize(g_stream);
  if (x) (void)hipFree(x);
  if (w) (void)hipFree(w);
  if (y) (void)hipFree(y);
  if (hip_status != hipSuccess) {
    g_error = conv_error ? conv_error : hip_error("MIOpen execution probe", hip_status);
    if (conv_error) free(conv_error);
    return 0;
  }
  g_miopen_available = std::equal(std::begin(output), std::end(output), std::begin(input),
                                  [](float got, float original) { return got == original * 2.0f; });
  if (!g_miopen_available) g_error = "MIOpen execution probe returned an incorrect result";
  return g_miopen_available ? 1 : 0;
}

int gpu_bridge_conv2d_backward_available(void) {
  if (g_miopen_backward_probed) return g_miopen_backward_available ? 1 : 0;
  g_miopen_backward_probed = true;
  if (!gpu_bridge_conv2d_available()) return 0;
  const float input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const float weight[1] = {2};
  const float gradient[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  float forward[9] = {}, dx[9] = {}, dw[1] = {};
  void *x = nullptr, *w = nullptr, *y = nullptr, *dy = nullptr, *device_dx = nullptr, *device_dw = nullptr;
  hipError_t hip_status = hipMalloc(&x, sizeof(input));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&w, sizeof(weight));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&y, sizeof(forward));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&dy, sizeof(gradient));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&device_dx, sizeof(dx));
  if (hip_status == hipSuccess) hip_status = hipMalloc(&device_dw, sizeof(dw));
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(x, input, sizeof(input), hipMemcpyHostToDevice, g_stream);
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(w, weight, sizeof(weight), hipMemcpyHostToDevice, g_stream);
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(dy, gradient, sizeof(gradient), hipMemcpyHostToDevice, g_stream);
  char *error = nullptr;
  if (hip_status == hipSuccess)
    hip_status = gpu_bridge_conv2d(x, w, y, 1, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, &error) == 0 ? hipSuccess : hipErrorUnknown;
  if (hip_status == hipSuccess)
    hip_status = gpu_bridge_conv2d_backward(w, dy, device_dx, 1, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, &error) == 0 ? hipSuccess : hipErrorUnknown;
  if (hip_status == hipSuccess)
    hip_status = gpu_bridge_conv2d_backward(x, dy, device_dw, 1, 1, 3, 3, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, &error) == 0 ? hipSuccess : hipErrorUnknown;
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(dx, device_dx, sizeof(dx), hipMemcpyDeviceToHost, g_stream);
  if (hip_status == hipSuccess) hip_status = hipMemcpyAsync(dw, device_dw, sizeof(dw), hipMemcpyDeviceToHost, g_stream);
  if (hip_status == hipSuccess) hip_status = hipStreamSynchronize(g_stream);
  if (x) (void)hipFree(x);
  if (w) (void)hipFree(w);
  if (y) (void)hipFree(y);
  if (dy) (void)hipFree(dy);
  if (device_dx) (void)hipFree(device_dx);
  if (device_dw) (void)hipFree(device_dw);
  if (hip_status != hipSuccess) {
    g_error = error ? error : hip_error("MIOpen backward execution probe", hip_status);
    if (error) free(error);
    return 0;
  }
  g_miopen_backward_available = std::all_of(std::begin(dx), std::end(dx), [](float value) { return value == 2.0f; }) && dw[0] == 45.0f;
  if (!g_miopen_backward_available) g_error = "MIOpen backward execution probe returned an incorrect result";
  return g_miopen_backward_available ? 1 : 0;
}
#endif
}
