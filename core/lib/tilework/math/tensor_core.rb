# frozen_string_literal: true

module Tilework
  # The Tensor Core (nvcuda::wmma) path.
  #
  # Policy:
  #   - Use the fragment API (nvcuda::wmma). Hand-written mma.sync / wgmma PTX
  #     and CUTLASS belong behind the raw: escape hatch, not here.
  #   - Start with TF32: FP32 buffers go in as is, so Tensor Cores are used
  #     without touching Buffer::DTYPES.
  #
  # Callers do not decide whether to take this path; eligible? judges from
  # Tilework.capabilities and the matrix shapes. Otherwise it falls through to
  # the existing SIMT kernel, so users see only a speed difference.
  module TensorCore
    # The TF32 fragment shape. K is 8, unlike FP16's 16, so the remainder
    # conditions depend on the dtype.
    FRAG_M = 16
    FRAG_N = 16
    FRAG_K = 8

    # Block tile: 4 warps (128 threads) per block in a 2x2 layout, each warp
    # owning 32x32 (2x2 of 16x16 fragments) → 64x64 per block. K loads two
    # fragments' worth into shared memory at a time.
    #
    # One tile per warp with no shared-memory reuse is dominated by global
    # reads and loses to SIMT (measured 0.91x); reuse is what makes Tensor
    # Cores pay.
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 16
    WARPS_M = 2
    WARPS_N = 2
    WARP = 32
    THREADS = WARPS_M * WARPS_N * WARP # 128

    class << self
      # A switch to turn the path off explicitly (for performance comparisons or
      # bisecting). On by default; eligible? decides from shapes and capabilities.
      attr_writer :enabled

      def enabled?
        return false if ENV["GPU_DISABLE_TENSOR_CORE"] == "1"

        @enabled.nil? ? true : @enabled
      end
    end

    module_function

    # Whether GEMM can be delegated to cuBLAS. It is an order of magnitude
    # faster than our WMMA kernel (measured 11.7x) and has **no shape
    # constraints** — small models that miss WMMA's multiple-of-64/16 rule
    # still reach the tensor cores.
    def gemm_library?
      return false unless Tilework.bridge.respond_to?(:gemm_available?)

      Tilework.bridge.gemm_available?
    end

    def gemm_library_name
      return Tilework.bridge.gemm_library_name if Tilework.bridge.respond_to?(:gemm_library_name)

      :cublas
    end

    # Input dtypes the cuBLAS path accepts and their codes for the bridge.
    # **The output is always float32.**
    GEMM_INPUT_CODES = { float32: 0, float16: 1, bfloat16: 2 }.freeze

    # When the cuBLAS path applies: any shape (no tile-boundary constraint
    # unlike our WMMA), as long as both sides share a dtype cuBLAS handles.
    def gemm_eligible?(a, b, _dtype)
      return false unless gemm_library?
      return false unless a.dtype == b.dtype
      supported = Tilework.capabilities.gemm_dtypes

      GEMM_INPUT_CODES.key?(a.dtype) && supported.include?(a.dtype)
    end

    # Backends with an executable matrix-instruction probe (rocWMMA on HIP)
    # own their native source and supported tile contract.  Core only asks the
    # bridge whether this dtype/shape is eligible; it never recognizes a GPU,
    # an ISA name, or a backend name.
    def matrix_core_eligible?(a, b, dtype)
      return false unless TensorCore.enabled?
      return false unless Tilework.bridge.respond_to?(:matrix_core_available?) && Tilework.bridge.matrix_core_available?
      return false unless dtype == :float32 && a.dtype == :float16 && b.dtype == :float16

      a.rows.positive? && a.rows % 16 == 0 && a.cols % 16 == 0 && b.cols % 16 == 0
    end

    def matrix_core_gemm(a, b, m, k, n)
      out = Ops.allocate_output(:float32, m * n, [m, n])
      Tilework.bridge.matrix_core_gemm(a.__handle__, b.__handle__, out.__handle__, m, k, n)
      out
    rescue StandardError
      out&.free
      raise
    end

    # C = A * B through cuBLAS. Whether TF32 is allowed follows enabled? (off
    # means FP32, keeping relative error around 1e-7). fp16 input still
    # accumulates in FP32 and outputs float32. The output is allocated through
    # the context: a graph plan **returns intermediates to the pool instead of
    # freeing them**, so an allocation bypassing the pool would add one slot per
    # run (which matters in decode loops). GEMM writes all of C (beta = 0), so a
    # reused, non-zeroed slot leaves no residue. Eager has no context, so it is
    # the same as Buffer.new.
    def gemm(a, b, m, k, n)
      out = Ops.allocate_output(:float32, m * n, [m, n])
      Tilework.bridge.gemm(a.__handle__, b.__handle__, out.__handle__, m, k, n,
                      enabled?, GEMM_INPUT_CODES.fetch(a.dtype))
      out
    rescue StandardError
      out&.free
      raise
    end

    # Whether `batch` GEMMs of the same shape can be issued at once.
    def gemm_batched?
      return false unless Tilework.bridge.respond_to?(:gemm_batched_available?)

      Tilework.bridge.gemm_batched_available?
    end

    # A low-level passthrough to cuBLAS's strided batched GEMM.
    #
    # **All arguments are column-major**; nothing is reinterpreted from row-major.
    # Callers who think in row-major keep the reinterpretation on their side, as
    # Tilework::NN.multi_head_attention does. Moving it here would explode into
    # combinations of transpose flags x which matrix.
    #
    # ld_* is the column-major leading dimension. To view part of a larger
    # matrix as one batch element, pass **the parent's width** as ld and step
    # by the stride. Buffer itself has no strides, so this is the only place
    # that information can be expressed.
    #
    # The output is always float32, as with gemm. alpha is a scale GEMM applies
    # for free; attention's 1/sqrt(dh) uses it, saving another round trip over
    # the score matrix.
    def gemm_batched(a, b, out, m:, k:, n:, lda:, ldb:, ldc:,
                     stride_a:, stride_b:, stride_c:, batch:,
                     trans_a: false, trans_b: false, alpha: 1.0)
      unless a.dtype == b.dtype && GEMM_INPUT_CODES.key?(a.dtype)
        raise ArgumentError,
              "batched GEMM は両辺が同じ対応済み float dtype のときだけ通る: " \
              "#{a.dtype.inspect} と #{b.dtype.inspect}"
      end
      unless Tilework.capabilities.gemm_dtypes.include?(a.dtype)
        raise ArgumentError, "batched GEMM does not support #{a.dtype.inspect} on this backend"
      end
      raise ArgumentError, "出力は float32 でなければならない" unless out.dtype == :float32

      Tilework.bridge.gemm_batched(a.__handle__, b.__handle__, out.__handle__,
                              m, k, n, lda, ldb, ldc,
                              stride_a, stride_b, stride_c,
                              trans_a, trans_b, batch, alpha.to_f,
                              enabled?, GEMM_INPUT_CODES.fetch(a.dtype))
      out
    end

    # Alignment: Buffers are flat, stride-less linear memory, so shapes off the
    # tile boundary never enter the WMMA path. Handling remainders with
    # in-kernel branches would diverge within a warp, making fragment
    # instructions undefined. Rejected shapes are computed correctly by SIMT.
    def eligible?(a, b, dtype)
      return false unless TensorCore.enabled?
      return false unless dtype == :float32 && a.dtype == :float32 && b.dtype == :float32
      return false unless Tilework.capabilities[:tf32]

      a.rows % BLOCK_M == 0 && b.cols % BLOCK_N == 0 && a.cols % BLOCK_K == 0
    end

    # A WMMA GEMM that reuses tiles through shared memory. __syncthreads() and
    # fragment instructions sit only where every thread in the block and every
    # lane in the warp arrive together.
    def matmul_source
      <<~CUDA
        #include <mma.h>
        using namespace nvcuda;

        #define BM #{BLOCK_M}
        #define BN #{BLOCK_N}
        #define BK #{BLOCK_K}
        #define WM #{WARPS_M}
        #define WN #{WARPS_N}
        #define FM #{FRAG_M}
        #define FN #{FRAG_N}
        #define FK #{FRAG_K}

        extern "C" __global__ void matmul_tf32(const float* a, const float* b, float* c,
                                               int m, int k, int n) {
          __shared__ float As[BM][BK];
          __shared__ float Bs[BK][BN];

          const int tid = threadIdx.x;
          const int warp = tid / #{WARP};
          const int warp_m = warp / WN;   // 0..WM-1
          const int warp_n = warp % WN;   // 0..WN-1

          const int row0 = blockIdx.y * BM; // top-left of this block's output
          const int col0 = blockIdx.x * BN;

          // 2x2 16x16 fragments per warp = 32x32
          wmma::fragment<wmma::accumulator, FM, FN, FK, float> acc[2][2];
          for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
              wmma::fill_fragment(acc[i][j], 0.0f);

          for (int t = 0; t < k; t += BK) {
            // Cooperative load into shared memory (128 threads, BM*BK = BK*BN = 1024 elements)
            for (int i = tid; i < BM * BK; i += #{THREADS}) {
              As[i / BK][i % BK] = a[(row0 + i / BK) * k + t + (i % BK)];
            }
            for (int i = tid; i < BK * BN; i += #{THREADS}) {
              Bs[i / BN][i % BN] = b[(t + i / BN) * n + col0 + (i % BN)];
            }
            __syncthreads();

            for (int kk = 0; kk < BK; kk += FK) {
              wmma::fragment<wmma::matrix_a, FM, FN, FK, wmma::precision::tf32, wmma::row_major> fa[2];
              wmma::fragment<wmma::matrix_b, FM, FN, FK, wmma::precision::tf32, wmma::row_major> fb[2];

              for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(fa[i], &As[warp_m * 32 + i * FM][kk], BK);
                // Round values read as FP32 to TF32 (10-bit mantissa, same 8-bit
                // exponent as FP32) before multiply-accumulate. The exponent range
                // matches FP32, so there is no FP16-style overflow/underflow.
                for (int e = 0; e < fa[i].num_elements; e++) fa[i].x[e] = wmma::__float_to_tf32(fa[i].x[e]);
              }
              for (int j = 0; j < 2; j++) {
                wmma::load_matrix_sync(fb[j], &Bs[kk][warp_n * 32 + j * FN], BN);
                for (int e = 0; e < fb[j].num_elements; e++) fb[j].x[e] = wmma::__float_to_tf32(fb[j].x[e]);
              }

              for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                  wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
            }
            __syncthreads();
          }

          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              int r = row0 + warp_m * 32 + i * FM;
              int cc = col0 + warp_n * 32 + j * FN;
              wmma::store_matrix_sync(c + r * n + cc, acc[i][j], n, wmma::mem_row_major);
            }
          }
        }
      CUDA
    end

    # The WMMA path. Allocated through the context for the same reason as gemm
    # (only tile-aligned shapes get here, so the output is fully written).
    def matmul(a, b, m, k, n)
      out = Ops.allocate_output(:float32, m * n, [m, n])
      # raw: true — plain CUDA including mma.h, so no compat header is prepended.
      kernel = Tilework.compile(matmul_source, name: "matmul_tf32", raw: true)
      Ops.launch_kernel(kernel, grid: [n / BLOCK_N, m / BLOCK_M, 1], block: [THREADS, 1, 1],
                            args: [a, b, out, m, k, n])
      out
    rescue StandardError
      out&.free
      raise
    end
  end
end
