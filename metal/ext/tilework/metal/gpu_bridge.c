#include <ruby.h>
#include <stdint.h>
#include <stdlib.h>

/* The C ABI the backend bridge exports. Mechanism only; policy lives in Ruby. */
extern char *gpu_bridge_device_name(void);
extern int gpu_bridge_device_available(void);
extern void *gpu_bridge_buffer_alloc(long bytes);
extern void gpu_bridge_buffer_write(void *handle, const void *src, long bytes, long offset);
extern void gpu_bridge_buffer_read(void *handle, void *dst, long bytes, long offset);
extern void gpu_bridge_buffer_free(void *handle);
extern void *gpu_bridge_pipeline_create(const char *src, const char *fn_name, char **error_out);
extern long gpu_bridge_pipeline_max_threads(void *handle);
extern void gpu_bridge_pipeline_free(void *handle);
extern int gpu_bridge_dispatch(void *pipeline,
                               const int32_t *arg_kinds, void *const *arg_ptrs,
                               const long *arg_lens, int32_t arg_count,
                               long gx, long gy, long gz,
                               long bx, long by, long bz,
                               int32_t wait, char **error_out);
extern void gpu_bridge_synchronize(void);
/* Optional: synchronize and report a failed asynchronous command. */
extern int gpu_bridge_synchronize_checked(char **err) __attribute__((weak));

/* Optional ABI. Weakly referenced, so linking a bridge that does not
   implement it leaves the symbol NULL instead of unresolved, and Ruby gets
   nil and falls back. */
extern int gpu_bridge_device_capabilities(int32_t *warp_size, int32_t *cc_major,
                                          int32_t *cc_minor, int32_t *integrated,
                                          long *memory_budget_bytes) __attribute__((weak));
extern char *gpu_bridge_device_architecture(void) __attribute__((weak));
extern int gpu_bridge_device_index(void) __attribute__((weak));
extern char *gpu_bridge_last_error(void) __attribute__((weak));
extern void gpu_bridge_shutdown(void) __attribute__((weak));

/* Optional ABI: offload transfers, two synchronous calls. Without them Ruby
   falls back to the plain D2H -> String -> File.write path. */
extern int gpu_bridge_buffer_save(void *handle, const char *path, long bytes,
                                  long off, char **err) __attribute__((weak));
extern int gpu_bridge_buffer_load(void *handle, const char *path, long bytes,
                                  long off, char **err) __attribute__((weak));

/* Optional ABI: three asynchronous transfer calls, used to hide prefetch.
   The bridge has no scheduler; what to prefetch in which order is Ruby's policy. */
extern void *gpu_bridge_transfer_begin(void *handle, const char *path, long bytes,
                                       long off, int dir) __attribute__((weak));
extern int gpu_bridge_transfer_poll(void *t) __attribute__((weak));
extern int gpu_bridge_transfer_end(void *t, char **err) __attribute__((weak));

/* Optional ABI: delegated GEMM. Without it the kernel path is used. */
extern int gpu_bridge_gemm(void *a, long a_off, void *b, long b_off, void *c, long c_off,
                           long m, long k, long n,
                           int allow_tf32, int in_dtype, char **err) __attribute__((weak));
extern int gpu_bridge_gemm_available(void) __attribute__((weak));

/* optional ABI: executable matrix-instruction path (rocWMMA etc.). */
extern int gpu_bridge_matrix_core_available(void) __attribute__((weak));
extern int gpu_bridge_matrix_core_gemm(void *a, long a_off, void *b, long b_off,
                                       void *c, long c_off, long m, long k, long n,
                                       char **err) __attribute__((weak));

/* Optional ABI: issue `batch` GEMMs of the same shape at once. Without it
   Ruby falls back to one gemm per batch. */
extern int gpu_bridge_gemm_strided_batched(void *a, long a_off, void *b, long b_off,
                                           void *c, long c_off,
                                           long m, long k, long n,
                                           long lda, long ldb, long ldc,
                                           long stride_a, long stride_b, long stride_c,
                                           int trans_a, int trans_b,
                                           long batch, double alpha,
                                           int allow_tf32, int in_dtype,
                                           char **err) __attribute__((weak));

/* Optional ABI: random generation (cuRAND). Without it Ruby's Random is used. */
extern int gpu_bridge_random(void *handle, long n, int kind, unsigned long long seed,
                             float mean, float stddev, char **err) __attribute__((weak));
extern int gpu_bridge_random_available(void) __attribute__((weak));

/* Optional ABI: FFT (cuFFT). Without it Ruby raises. */
extern int gpu_bridge_fft(void *in, void *out, long n, long batch, int kind,
                          int direction, char **err) __attribute__((weak));
extern int gpu_bridge_fft_available(void) __attribute__((weak));

/* Optional ABI: 2D convolution (cuDNN). Without it Ruby raises. */
extern int gpu_bridge_conv2d(void *x, void *w, void *y,
                             int n, int cin, int h, int width,
                             int cout, int kh, int kw,
                             int pad_h, int pad_w, int stride_h, int stride_w,
                             int dilation_h, int dilation_w, int groups,
                             int dtype, char **err) __attribute__((weak));
extern int gpu_bridge_conv2d_available(void) __attribute__((weak));

/* Optional ABI: convolution backward. mode 0 = input gradient, 1 = weight
   gradient. Descriptors are shared with the forward plan, so the forward
   pass must have run first. */
extern int gpu_bridge_conv2d_backward(void *a, void *b, void *out,
                                      int n, int cin, int h, int width,
                                      int cout, int kh, int kw,
                                      int pad_h, int pad_w, int stride_h, int stride_w,
                                      int dilation_h, int dilation_w, int groups,
                                      int dtype, int mode, char **err) __attribute__((weak));
extern int gpu_bridge_conv2d_backward_available(void) __attribute__((weak));

/* optional ABI: opaque pageable host backing, never a filesystem path. */
extern int gpu_bridge_host_backing_available(void) __attribute__((weak));
extern void *gpu_bridge_host_backing_alloc(long bytes) __attribute__((weak));
extern void gpu_bridge_host_backing_free(void *handle) __attribute__((weak));
extern int gpu_bridge_host_backing_store(void *host, void *device, long bytes, long offset, char **err) __attribute__((weak));
extern int gpu_bridge_host_backing_restore(void *host, void *device, long bytes, long offset, char **err) __attribute__((weak));
extern void *gpu_bridge_host_backing_transfer_begin(void *host, void *device, long bytes, long offset, char **err) __attribute__((weak));
extern int gpu_bridge_host_backing_transfer_poll(void *token) __attribute__((weak));
extern int gpu_bridge_host_backing_transfer_end(void *token, char **err) __attribute__((weak));

/* --- TypedData: buffer handle --- */

typedef struct {
    void *handle;  /* NULL after explicit free */
    long offset;   /* in bytes; nonzero only for a view */
    int owned;     /* 0 for a view: must not be freed */
    VALUE parent;  /* the BufferHandle a view points into (kept alive against GC) */
} bridge_buffer_t;

static void
buffer_dfree(void *ptr)
{
    bridge_buffer_t *data = ptr;
    /* A view only points at someone else's memory, so it frees nothing */
    if (data->handle && data->owned) gpu_bridge_buffer_free(data->handle);
    xfree(data);
}

static void
buffer_dmark(void *ptr)
{
    bridge_buffer_t *data = ptr;
    if (data->parent != Qnil) rb_gc_mark(data->parent);
}

static const rb_data_type_t buffer_data_type = {
    "Tilework::Bridge::BufferHandle",
    { buffer_dmark, buffer_dfree, NULL },
    NULL, NULL, 0 /* dmark rules out FREE_IMMEDIATELY */
};

/* For delegated libraries (cuBLAS/cuDNN/cuRAND/cuFFT). They take raw
   pointers, so pass the address with the view's offset added. */
static void *buffer_ptr(bridge_buffer_t *data)
{
    return (void *)((char *)data->handle + data->offset);
}

static bridge_buffer_t *
buffer_unwrap(VALUE obj)
{
    bridge_buffer_t *data;
    TypedData_Get_Struct(obj, bridge_buffer_t, &buffer_data_type, data);
    if (data->handle == NULL) rb_raise(rb_eRuntimeError, "buffer already freed");
    return data;
}

/* --- TypedData: pipeline handle --- */

typedef struct {
    void *handle;
} bridge_pipeline_t;

static void
pipeline_dfree(void *ptr)
{
    bridge_pipeline_t *data = ptr;
    if (data->handle) gpu_bridge_pipeline_free(data->handle);
    xfree(data);
}

static const rb_data_type_t pipeline_data_type = {
    "Tilework::Bridge::PipelineHandle",
    { NULL, pipeline_dfree, NULL },
    NULL, NULL, RUBY_TYPED_FREE_IMMEDIATELY
};

static bridge_pipeline_t *
pipeline_unwrap(VALUE obj)
{
    bridge_pipeline_t *data;
    TypedData_Get_Struct(obj, bridge_pipeline_t, &pipeline_data_type, data);
    if (data->handle == NULL) rb_raise(rb_eRuntimeError, "pipeline already freed");
    return data;
}

static VALUE cBufferHandle;
static VALUE cPipelineHandle;
static VALUE cHostBackingHandle;

typedef struct { void *handle; } bridge_host_backing_t;
static void host_backing_dfree(void *ptr) {
    bridge_host_backing_t *data = ptr;
    if (data->handle && gpu_bridge_host_backing_free != NULL) gpu_bridge_host_backing_free(data->handle);
    xfree(data);
}
static const rb_data_type_t host_backing_data_type = {
    "Tilework::Bridge::HostBackingHandle", { NULL, host_backing_dfree, NULL }, NULL, NULL, RUBY_TYPED_FREE_IMMEDIATELY
};
static bridge_host_backing_t *host_backing_unwrap(VALUE obj) {
    bridge_host_backing_t *data;
    TypedData_Get_Struct(obj, bridge_host_backing_t, &host_backing_data_type, data);
    if (!data->handle) rb_raise(rb_eRuntimeError, "host backing already freed");
    return data;
}

#ifndef TILEWORK_BRIDGE_MODULE
#define TILEWORK_BRIDGE_MODULE "MetalBridge"
#endif

static VALUE
raise_bridge_error(const char *fallback)
{
    char *error = gpu_bridge_last_error != NULL ? gpu_bridge_last_error() : NULL;
    VALUE message = rb_utf8_str_new_cstr(error ? error : fallback);
    if (error) free(error);
    rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, message));
    return Qnil;
}

static void
bridge_shutdown(VALUE _unused)
{
    if (gpu_bridge_shutdown != NULL) gpu_bridge_shutdown();
}

/* --- module functions --- */

static VALUE
bridge_device_name(VALUE self)
{
    char *name = gpu_bridge_device_name();
    if (name == NULL) return raise_bridge_error("no Tilework device available");
    VALUE str = rb_utf8_str_new_cstr(name);
    free(name);
    return str;
}

static VALUE
bridge_device_available_p(VALUE self)
{
    return gpu_bridge_device_available() ? Qtrue : Qfalse;
}

/* Diagnostic observations. Unlike Capabilities, never used for path selection. */
static VALUE
bridge_device_info(VALUE self)
{
    VALUE info = rb_hash_new();
    char *name = gpu_bridge_device_name();
    if (name != NULL) {
        rb_hash_aset(info, ID2SYM(rb_intern("name")), rb_utf8_str_new_cstr(name));
        free(name);
    }
    if (gpu_bridge_device_architecture != NULL) {
        char *architecture = gpu_bridge_device_architecture();
        if (architecture != NULL) {
            rb_hash_aset(info, ID2SYM(rb_intern("architecture")), rb_utf8_str_new_cstr(architecture));
            free(architecture);
        }
    }
    if (gpu_bridge_device_index != NULL) {
        int index = gpu_bridge_device_index();
        if (index >= 0) rb_hash_aset(info, ID2SYM(rb_intern("index")), INT2NUM(index));
    }
    if (gpu_bridge_last_error != NULL) {
        char *error = gpu_bridge_last_error();
        if (error != NULL) {
            rb_hash_aset(info, ID2SYM(rb_intern("last_error")), rb_utf8_str_new_cstr(error));
            free(error);
        }
    }
    return info;
}

static VALUE
bridge_buffer_alloc(VALUE self, VALUE bytesize)
{
    long bytes = NUM2LONG(bytesize);
    if (bytes <= 0) rb_raise(rb_eArgError, "buffer size must be positive");
    void *handle = gpu_bridge_buffer_alloc(bytes);
    if (handle == NULL) return raise_bridge_error("failed to allocate Tilework buffer");

    bridge_buffer_t *data = ALLOC(bridge_buffer_t);
    data->handle = handle;
    data->offset = 0;
    data->owned = 1;
    data->parent = Qnil;
    return TypedData_Wrap_Struct(cBufferHandle, &buffer_data_type, data);
}

/* Creates another handle to the same memory. It frees nothing, so it is valid
   while the original handle lives; holding parent protects that from GC. */
static VALUE
bridge_buffer_view(VALUE self, VALUE buf, VALUE byte_offset)
{
    bridge_buffer_t *base = buffer_unwrap(buf);

    bridge_buffer_t *data = ALLOC(bridge_buffer_t);
    data->handle = base->handle;
    data->offset = base->offset + NUM2LONG(byte_offset);
    data->owned = 0;
    data->parent = buf;
    return TypedData_Wrap_Struct(cBufferHandle, &buffer_data_type, data);
}

static VALUE bridge_host_backing_available_p(VALUE self) {
    return (gpu_bridge_host_backing_available != NULL && gpu_bridge_host_backing_available()) ? Qtrue : Qfalse;
}
static VALUE bridge_host_backing_alloc(VALUE self, VALUE bytesize) {
    if (gpu_bridge_host_backing_alloc == NULL) rb_raise(rb_eNotImpError, "host backing is not available");
    void *handle = gpu_bridge_host_backing_alloc(NUM2LONG(bytesize));
    if (!handle) return raise_bridge_error("failed to allocate host backing");
    bridge_host_backing_t *data = ALLOC(bridge_host_backing_t); data->handle = handle;
    return TypedData_Wrap_Struct(cHostBackingHandle, &host_backing_data_type, data);
}
static VALUE bridge_host_backing_free(VALUE self, VALUE backing) {
    bridge_host_backing_t *data = host_backing_unwrap(backing);
    if (gpu_bridge_host_backing_free != NULL) gpu_bridge_host_backing_free(data->handle);
    data->handle = NULL; return Qnil;
}
static VALUE bridge_host_backing_copy(VALUE self, VALUE backing, VALUE buffer, VALUE bytes, VALUE offset, int restore) {
    bridge_host_backing_t *host = host_backing_unwrap(backing); bridge_buffer_t *device = buffer_unwrap(buffer);
    char *error = NULL;
    int status = restore ? gpu_bridge_host_backing_restore(host->handle, buffer_ptr(device), NUM2LONG(bytes), NUM2LONG(offset), &error)
                         : gpu_bridge_host_backing_store(host->handle, buffer_ptr(device), NUM2LONG(bytes), NUM2LONG(offset), &error);
    if (status != 0) { VALUE msg = rb_utf8_str_new_cstr(error ? error : "host backing transfer failed"); if (error) free(error); rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg)); }
    return Qnil;
}
static VALUE bridge_host_backing_store(VALUE self, VALUE backing, VALUE buffer, VALUE bytes, VALUE offset) { return bridge_host_backing_copy(self, backing, buffer, bytes, offset, 0); }
static VALUE bridge_host_backing_restore(VALUE self, VALUE backing, VALUE buffer, VALUE bytes, VALUE offset) { return bridge_host_backing_copy(self, backing, buffer, bytes, offset, 1); }
static VALUE bridge_host_backing_async_available_p(VALUE self) {
    return (gpu_bridge_host_backing_transfer_begin != NULL && gpu_bridge_host_backing_transfer_poll != NULL && gpu_bridge_host_backing_transfer_end != NULL) ? Qtrue : Qfalse;
}
static VALUE bridge_host_backing_transfer_begin(VALUE self, VALUE backing, VALUE buffer, VALUE bytes, VALUE offset) {
    if (gpu_bridge_host_backing_transfer_begin == NULL) rb_raise(rb_eNotImpError, "async host backing is not available");
    bridge_host_backing_t *host = host_backing_unwrap(backing); bridge_buffer_t *device = buffer_unwrap(buffer); char *error = NULL;
    void *token = gpu_bridge_host_backing_transfer_begin(host->handle, buffer_ptr(device), NUM2LONG(bytes), NUM2LONG(offset), &error);
    if (!token) { VALUE msg = rb_utf8_str_new_cstr(error ? error : "async host backing transfer failed"); if (error) free(error); rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg)); }
    return ULL2NUM((unsigned long long)(uintptr_t)token);
}
static VALUE bridge_host_backing_transfer_poll(VALUE self, VALUE token) { return INT2NUM(gpu_bridge_host_backing_transfer_poll((void *)(uintptr_t)NUM2ULL(token))); }
static VALUE bridge_host_backing_transfer_end(VALUE self, VALUE token) {
    char *error = NULL; if (gpu_bridge_host_backing_transfer_end((void *)(uintptr_t)NUM2ULL(token), &error) != 0) { VALUE msg = rb_utf8_str_new_cstr(error ? error : "async host backing transfer failed"); if (error) free(error); rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg)); } return Qnil;
}

static VALUE
bridge_buffer_write(VALUE self, VALUE buf, VALUE bytes, VALUE offset)
{
    bridge_buffer_t *data = buffer_unwrap(buf);
    Check_Type(bytes, T_STRING);
    gpu_bridge_buffer_write(data->handle, RSTRING_PTR(bytes), RSTRING_LEN(bytes),
                            data->offset + NUM2LONG(offset));
    return Qnil;
}

static VALUE
bridge_buffer_read(VALUE self, VALUE buf, VALUE bytesize, VALUE offset)
{
    bridge_buffer_t *data = buffer_unwrap(buf);
    long bytes = NUM2LONG(bytesize);
    VALUE str = rb_str_new(NULL, bytes);
    gpu_bridge_buffer_read(data->handle, RSTRING_PTR(str), bytes, data->offset + NUM2LONG(offset));
    return str;
}

static VALUE
bridge_buffer_free(VALUE self, VALUE buf)
{
    bridge_buffer_t *data;
    TypedData_Get_Struct(buf, bridge_buffer_t, &buffer_data_type, data);
    if (data->handle && data->owned) gpu_bridge_buffer_free(data->handle);
    data->handle = NULL; /* mark it unusable even for a view */
    return Qnil;
}

static VALUE
bridge_pipeline_create(VALUE self, VALUE src, VALUE fn_name)
{
    Check_Type(src, T_STRING);
    Check_Type(fn_name, T_STRING);
    char *error = NULL;
    void *handle = gpu_bridge_pipeline_create(StringValueCStr(src), StringValueCStr(fn_name), &error);
    if (handle == NULL) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "unknown kernel compile error");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }

    bridge_pipeline_t *data = ALLOC(bridge_pipeline_t);
    data->handle = handle;
    return TypedData_Wrap_Struct(cPipelineHandle, &pipeline_data_type, data);
}

static VALUE
bridge_pipeline_max_threads(VALUE self, VALUE pipeline)
{
    bridge_pipeline_t *data = pipeline_unwrap(pipeline);
    return LONG2NUM(gpu_bridge_pipeline_max_threads(data->handle));
}

/* args: a Ruby Array of BufferHandle (setBuffer) or String (setBytes). */
static VALUE
bridge_dispatch(VALUE self, VALUE pipeline, VALUE args, VALUE grid, VALUE block, VALUE wait)
{
    bridge_pipeline_t *pso = pipeline_unwrap(pipeline);
    Check_Type(args, T_ARRAY);
    long argc = RARRAY_LEN(args);

    int32_t *kinds = ALLOCA_N(int32_t, argc > 0 ? argc : 1);
    void **ptrs = ALLOCA_N(void *, argc > 0 ? argc : 1);
    long *lens = ALLOCA_N(long, argc > 0 ? argc : 1);

    for (long i = 0; i < argc; i++) {
        VALUE arg = rb_ary_entry(args, i);
        if (rb_typeddata_is_kind_of(arg, &buffer_data_type)) {
            bridge_buffer_t *data = buffer_unwrap(arg);
            kinds[i] = 0;
            ptrs[i] = data->handle;
            /* arg_lens was unused for buffer arguments, so it carries the
               view's byte offset here. CUDA adds it to the pointer, Metal
               takes it via setBuffer(offset:) — no C ABI change needed. */
            lens[i] = data->offset;
        } else if (RB_TYPE_P(arg, T_STRING)) {
            kinds[i] = 1;
            ptrs[i] = RSTRING_PTR(arg);
            lens[i] = RSTRING_LEN(arg);
        } else {
            rb_raise(rb_eTypeError, "dispatch argument %ld must be a BufferHandle or packed String", i);
        }
    }

    Check_Type(grid, T_ARRAY);
    Check_Type(block, T_ARRAY);
    long gx = NUM2LONG(rb_ary_entry(grid, 0));
    long gy = NUM2LONG(rb_ary_entry(grid, 1));
    long gz = NUM2LONG(rb_ary_entry(grid, 2));
    long bx = NUM2LONG(rb_ary_entry(block, 0));
    long by = NUM2LONG(rb_ary_entry(block, 1));
    long bz = NUM2LONG(rb_ary_entry(block, 2));

    char *error = NULL;
    int rc = gpu_bridge_dispatch(pso->handle, kinds, ptrs, lens, (int32_t)argc,
                                 gx, gy, gz, bx, by, bz,
                                 RTEST(wait) ? 1 : 0, &error);
    RB_GC_GUARD(args);
    if (rc != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "kernel dispatch failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_synchronize(VALUE self)
{
    if (gpu_bridge_synchronize_checked == NULL) {
        gpu_bridge_synchronize();
        return Qnil;
    }

    char *error = NULL;
    if (gpu_bridge_synchronize_checked(&error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "asynchronous work failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

/* Whether the bridge has a native offload path; Ruby uses it to pick a path. */
static VALUE
bridge_offload_available_p(VALUE self)
{
    return (gpu_bridge_buffer_save != NULL && gpu_bridge_buffer_load != NULL) ? Qtrue : Qfalse;
}

static VALUE
bridge_buffer_save(VALUE self, VALUE buf, VALUE path, VALUE bytesize, VALUE offset)
{
    if (gpu_bridge_buffer_save == NULL) rb_raise(rb_eNotImpError, "native offload is not available");

    bridge_buffer_t *data = buffer_unwrap(buf);
    char *error = NULL;
    if (gpu_bridge_buffer_save(buffer_ptr(data), StringValueCStr(path),
                               NUM2LONG(bytesize), NUM2LONG(offset), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "offload save failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_buffer_load(VALUE self, VALUE buf, VALUE path, VALUE bytesize, VALUE offset)
{
    if (gpu_bridge_buffer_load == NULL) rb_raise(rb_eNotImpError, "native offload is not available");

    bridge_buffer_t *data = buffer_unwrap(buf);
    char *error = NULL;
    if (gpu_bridge_buffer_load(buffer_ptr(data), StringValueCStr(path),
                               NUM2LONG(bytesize), NUM2LONG(offset), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "offload load failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_transfer_available_p(VALUE self)
{
    return (gpu_bridge_transfer_begin != NULL && gpu_bridge_transfer_poll != NULL &&
            gpu_bridge_transfer_end != NULL) ? Qtrue : Qfalse;
}

/* A transfer handle is an opaque pointer. Ruby only touches it at the start
   and the end, so it is returned as an Integer rather than TypedData (GC
   freeing it would pull it out from under the running worker thread). */
static VALUE
bridge_transfer_begin(VALUE self, VALUE buf, VALUE path, VALUE bytesize, VALUE offset, VALUE dir)
{
    if (gpu_bridge_transfer_begin == NULL) rb_raise(rb_eNotImpError, "async offload is not available");

    bridge_buffer_t *data = buffer_unwrap(buf);
    void *t = gpu_bridge_transfer_begin(buffer_ptr(data), StringValueCStr(path),
                                        NUM2LONG(bytesize), NUM2LONG(offset), NUM2INT(dir));
    if (t == NULL) rb_raise(rb_eRuntimeError, "failed to start the offload transfer");
    return ULL2NUM((unsigned long long)(uintptr_t)t);
}

static VALUE
bridge_transfer_poll(VALUE self, VALUE t)
{
    if (gpu_bridge_transfer_poll == NULL) rb_raise(rb_eNotImpError, "async offload is not available");

    return INT2NUM(gpu_bridge_transfer_poll((void *)(uintptr_t)NUM2ULL(t)));
}

static VALUE
bridge_transfer_end(VALUE self, VALUE t)
{
    if (gpu_bridge_transfer_end == NULL) rb_raise(rb_eNotImpError, "async offload is not available");

    char *error = NULL;
    if (gpu_bridge_transfer_end((void *)(uintptr_t)NUM2ULL(t), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "offload transfer failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_random_available_p(VALUE self)
{
    return (gpu_bridge_random != NULL &&
            (gpu_bridge_random_available == NULL || gpu_bridge_random_available())) ? Qtrue : Qfalse;
}

static VALUE
bridge_random(VALUE self, VALUE buf, VALUE n, VALUE kind, VALUE seed, VALUE mean, VALUE stddev)
{
    if (gpu_bridge_random == NULL) rb_raise(rb_eNotImpError, "no RNG library is available");

    bridge_buffer_t *data = buffer_unwrap(buf);
    char *error = NULL;
    if (gpu_bridge_random(buffer_ptr(data), NUM2LONG(n), NUM2INT(kind), NUM2ULL(seed),
                          (float)NUM2DBL(mean), (float)NUM2DBL(stddev), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "random generation failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_conv2d_available_p(VALUE self)
{
    return (gpu_bridge_conv2d != NULL &&
            (gpu_bridge_conv2d_available == NULL || gpu_bridge_conv2d_available())) ? Qtrue : Qfalse;
}

static VALUE
bridge_conv2d_backward_available_p(VALUE self)
{
    return (gpu_bridge_conv2d_backward != NULL &&
            (gpu_bridge_conv2d_backward_available == NULL || gpu_bridge_conv2d_backward_available())) ? Qtrue : Qfalse;
}

/* Many arguments, so Ruby passes an Array (dims has 14 entries: pad/stride
   followed by dilation_h, dilation_w, groups). */
static VALUE
bridge_conv2d(VALUE self, VALUE x, VALUE w, VALUE y, VALUE dims, VALUE dtype)
{
    if (gpu_bridge_conv2d == NULL) rb_raise(rb_eNotImpError, "no convolution library is available");

    Check_Type(dims, T_ARRAY);
    if (RARRAY_LEN(dims) != 14) rb_raise(rb_eArgError, "dims must have 14 elements");
    int d[14];
    for (int i = 0; i < 14; i++) d[i] = NUM2INT(rb_ary_entry(dims, i));

    bridge_buffer_t *dx = buffer_unwrap(x);
    bridge_buffer_t *dw = buffer_unwrap(w);
    bridge_buffer_t *dy = buffer_unwrap(y);
    char *error = NULL;
    if (gpu_bridge_conv2d(buffer_ptr(dx), buffer_ptr(dw), buffer_ptr(dy),
                          d[0], d[1], d[2], d[3], d[4], d[5], d[6],
                          d[7], d[8], d[9], d[10], d[11], d[12], d[13],
                          NUM2INT(dtype), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "conv2d failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_conv2d_backward(VALUE self, VALUE a, VALUE b, VALUE out, VALUE dims, VALUE dtype, VALUE mode)
{
    if (gpu_bridge_conv2d_backward == NULL) {
        rb_raise(rb_eNotImpError, "no convolution library is available");
    }

    Check_Type(dims, T_ARRAY);
    if (RARRAY_LEN(dims) != 14) rb_raise(rb_eArgError, "dims must have 14 elements");
    int d[14];
    for (int i = 0; i < 14; i++) d[i] = NUM2INT(rb_ary_entry(dims, i));

    bridge_buffer_t *da = buffer_unwrap(a);
    bridge_buffer_t *db = buffer_unwrap(b);
    bridge_buffer_t *dout = buffer_unwrap(out);
    char *error = NULL;
    if (gpu_bridge_conv2d_backward(buffer_ptr(da), buffer_ptr(db), buffer_ptr(dout),
                                   d[0], d[1], d[2], d[3], d[4], d[5], d[6],
                                   d[7], d[8], d[9], d[10], d[11], d[12], d[13],
                                   NUM2INT(dtype), NUM2INT(mode), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "conv2d_backward failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_fft_available_p(VALUE self)
{
    return (gpu_bridge_fft != NULL &&
            (gpu_bridge_fft_available == NULL || gpu_bridge_fft_available())) ? Qtrue : Qfalse;
}

static VALUE
bridge_fft(VALUE self, VALUE in, VALUE out, VALUE n, VALUE batch, VALUE kind, VALUE direction)
{
    if (gpu_bridge_fft == NULL) rb_raise(rb_eNotImpError, "no FFT library is available");

    bridge_buffer_t *di = buffer_unwrap(in);
    bridge_buffer_t *dobj = buffer_unwrap(out);
    char *error = NULL;
    if (gpu_bridge_fft(buffer_ptr(di), buffer_ptr(dobj), NUM2LONG(n), NUM2LONG(batch),
                       NUM2INT(kind), NUM2INT(direction), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "fft failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_gemm_available_p(VALUE self)
{
    return (gpu_bridge_gemm != NULL &&
            (gpu_bridge_gemm_available == NULL || gpu_bridge_gemm_available())) ? Qtrue : Qfalse;
}

static VALUE
bridge_gemm(VALUE self, VALUE a, VALUE b, VALUE c, VALUE m, VALUE k, VALUE n,
            VALUE tf32, VALUE in_dtype)
{
    if (gpu_bridge_gemm == NULL) rb_raise(rb_eNotImpError, "no GEMM library is available");

    bridge_buffer_t *da = buffer_unwrap(a);
    bridge_buffer_t *db = buffer_unwrap(b);
    bridge_buffer_t *dc = buffer_unwrap(c);
    char *error = NULL;
    if (gpu_bridge_gemm(da->handle, da->offset, db->handle, db->offset, dc->handle, dc->offset,
                        NUM2LONG(m), NUM2LONG(k), NUM2LONG(n),
                        RTEST(tf32) ? 1 : 0, NUM2INT(in_dtype), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "gemm failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

static VALUE
bridge_gemm_batched_available_p(VALUE self)
{
    return gpu_bridge_gemm_strided_batched != NULL ? Qtrue : Qfalse;
}

static VALUE
bridge_matrix_core_available_p(VALUE self)
{
    return gpu_bridge_matrix_core_available != NULL && gpu_bridge_matrix_core_available() ? Qtrue : Qfalse;
}

static VALUE
bridge_matrix_core_gemm(VALUE self, VALUE a, VALUE b, VALUE c, VALUE m, VALUE k, VALUE n)
{
    if (gpu_bridge_matrix_core_gemm == NULL) rb_raise(rb_eNotImpError, "no matrix-core implementation is available");
    bridge_buffer_t *da = buffer_unwrap(a);
    bridge_buffer_t *db = buffer_unwrap(b);
    bridge_buffer_t *dc = buffer_unwrap(c);
    char *error = NULL;
    if (gpu_bridge_matrix_core_gemm(da->handle, da->offset, db->handle, db->offset,
                                    dc->handle, dc->offset, NUM2LONG(m), NUM2LONG(k), NUM2LONG(n),
                                    &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "matrix-core GEMM failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

/* 17 arguments exceed rb_define_module_function's fixed arity limit (15),
   so this takes argc/argv. The order matches the bridge's declaration. */
static VALUE
bridge_gemm_batched(int argc, VALUE *argv, VALUE self)
{
    if (gpu_bridge_gemm_strided_batched == NULL) {
        rb_raise(rb_eNotImpError, "no batched GEMM library is available");
    }
    if (argc != 18) rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 18)", argc);

    bridge_buffer_t *da = buffer_unwrap(argv[0]);
    bridge_buffer_t *db = buffer_unwrap(argv[1]);
    bridge_buffer_t *dc = buffer_unwrap(argv[2]);
    char *error = NULL;
    if (gpu_bridge_gemm_strided_batched(da->handle, da->offset, db->handle, db->offset,
                                        dc->handle, dc->offset,
                                        NUM2LONG(argv[3]), NUM2LONG(argv[4]), NUM2LONG(argv[5]),
                                        NUM2LONG(argv[6]), NUM2LONG(argv[7]), NUM2LONG(argv[8]),
                                        NUM2LONG(argv[9]), NUM2LONG(argv[10]), NUM2LONG(argv[11]),
                                        RTEST(argv[12]) ? 1 : 0, RTEST(argv[13]) ? 1 : 0,
                                        NUM2LONG(argv[14]), NUM2DBL(argv[15]),
                                        RTEST(argv[16]) ? 1 : 0,
                                        NUM2INT(argv[17]), &error) != 0) {
        VALUE msg = rb_utf8_str_new_cstr(error ? error : "batched gemm failed");
        if (error) free(error);
        rb_exc_raise(rb_exc_new_str(rb_eRuntimeError, msg));
    }
    return Qnil;
}

/* Returns raw device facts only. Turning them into the Tilework.capabilities
   contract is Ruby's policy (lib/tilework.rb). */
static VALUE
bridge_device_capabilities(VALUE self)
{
    if (gpu_bridge_device_capabilities == NULL) return Qnil;

    int32_t warp_size = 0, cc_major = 0, cc_minor = 0, integrated = 0;
    long memory_budget = 0;
    if (gpu_bridge_device_capabilities(&warp_size, &cc_major, &cc_minor,
                                       &integrated, &memory_budget) != 0) {
        return Qnil;
    }

    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("warp_size")), INT2NUM(warp_size));
    rb_hash_aset(h, ID2SYM(rb_intern("compute_capability")),
                 rb_ary_new_from_args(2, INT2NUM(cc_major), INT2NUM(cc_minor)));
    rb_hash_aset(h, ID2SYM(rb_intern("unified_memory")), integrated ? Qtrue : Qfalse);
    rb_hash_aset(h, ID2SYM(rb_intern("memory_budget_bytes")), LONG2NUM(memory_budget));
    return h;
}

void
Init_bridge(void)
{
    VALUE mTilework = rb_define_module("Tilework");
    VALUE mBridge = rb_define_module_under(mTilework, TILEWORK_BRIDGE_MODULE);

    cBufferHandle = rb_define_class_under(mBridge, "BufferHandle", rb_cObject);
    rb_undef_alloc_func(cBufferHandle);
    cPipelineHandle = rb_define_class_under(mBridge, "PipelineHandle", rb_cObject);
    rb_undef_alloc_func(cPipelineHandle);
    cHostBackingHandle = rb_define_class_under(mBridge, "HostBackingHandle", rb_cObject);
    rb_undef_alloc_func(cHostBackingHandle);

    rb_define_module_function(mBridge, "device_name", bridge_device_name, 0);
    rb_define_module_function(mBridge, "device_available?", bridge_device_available_p, 0);
    rb_define_module_function(mBridge, "device_info", bridge_device_info, 0);
    rb_define_module_function(mBridge, "buffer_alloc", bridge_buffer_alloc, 1);
    rb_define_module_function(mBridge, "buffer_write", bridge_buffer_write, 3);
    rb_define_module_function(mBridge, "buffer_read", bridge_buffer_read, 3);
    rb_define_module_function(mBridge, "buffer_free", bridge_buffer_free, 1);
    rb_define_module_function(mBridge, "buffer_view", bridge_buffer_view, 2);
    rb_define_module_function(mBridge, "host_backing_available?", bridge_host_backing_available_p, 0);
    rb_define_module_function(mBridge, "host_backing_alloc", bridge_host_backing_alloc, 1);
    rb_define_module_function(mBridge, "host_backing_free", bridge_host_backing_free, 1);
    rb_define_module_function(mBridge, "host_backing_store", bridge_host_backing_store, 4);
    rb_define_module_function(mBridge, "host_backing_restore", bridge_host_backing_restore, 4);
    rb_define_module_function(mBridge, "host_backing_async_available?", bridge_host_backing_async_available_p, 0);
    rb_define_module_function(mBridge, "host_backing_transfer_begin", bridge_host_backing_transfer_begin, 4);
    rb_define_module_function(mBridge, "host_backing_transfer_poll", bridge_host_backing_transfer_poll, 1);
    rb_define_module_function(mBridge, "host_backing_transfer_end", bridge_host_backing_transfer_end, 1);
    rb_define_module_function(mBridge, "pipeline_create", bridge_pipeline_create, 2);
    rb_define_module_function(mBridge, "pipeline_max_threads", bridge_pipeline_max_threads, 1);
    rb_define_module_function(mBridge, "dispatch", bridge_dispatch, 5);
    rb_define_module_function(mBridge, "synchronize", bridge_synchronize, 0);
    rb_define_module_function(mBridge, "device_capabilities", bridge_device_capabilities, 0);
    rb_define_module_function(mBridge, "offload_available?", bridge_offload_available_p, 0);
    rb_define_module_function(mBridge, "buffer_save", bridge_buffer_save, 4);
    rb_define_module_function(mBridge, "buffer_load", bridge_buffer_load, 4);
    rb_define_module_function(mBridge, "transfer_available?", bridge_transfer_available_p, 0);
    rb_define_module_function(mBridge, "transfer_begin", bridge_transfer_begin, 5);
    rb_define_module_function(mBridge, "transfer_poll", bridge_transfer_poll, 1);
    rb_define_module_function(mBridge, "transfer_end", bridge_transfer_end, 1);
    rb_define_module_function(mBridge, "conv2d_available?", bridge_conv2d_available_p, 0);
    rb_define_module_function(mBridge, "conv2d", bridge_conv2d, 5);
    rb_define_module_function(mBridge, "conv2d_backward", bridge_conv2d_backward, 6);
    rb_define_module_function(mBridge, "conv2d_backward_available?", bridge_conv2d_backward_available_p, 0);
    rb_define_module_function(mBridge, "fft_available?", bridge_fft_available_p, 0);
    rb_define_module_function(mBridge, "fft", bridge_fft, 6);
    rb_define_module_function(mBridge, "random_available?", bridge_random_available_p, 0);
    rb_define_module_function(mBridge, "random", bridge_random, 6);
    rb_define_module_function(mBridge, "gemm_available?", bridge_gemm_available_p, 0);
    rb_define_module_function(mBridge, "gemm", bridge_gemm, 8);
    rb_define_module_function(mBridge, "gemm_batched_available?", bridge_gemm_batched_available_p, 0);
    rb_define_module_function(mBridge, "gemm_batched", bridge_gemm_batched, -1);
    rb_define_module_function(mBridge, "matrix_core_available?", bridge_matrix_core_available_p, 0);
    rb_define_module_function(mBridge, "matrix_core_gemm", bridge_matrix_core_gemm, 6);
    rb_set_end_proc(bridge_shutdown, Qnil);
}
