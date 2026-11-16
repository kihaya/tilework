# frozen_string_literal: true

module Tilework
  # Neural-network operations implemented as fused kernels.
  #
  # Why the library owns them (when `Ops` combinations could express them):
  #
  # 1. **Ops multiply.** Each operation has a fixed cost of about 90 µs, so
  #    writing softmax as `exp` / `sum` / `/` takes several round trips per
  #    row; fused, it is one kernel.
  # 2. **Everyone gets the launch domain wrong.** Omitting `over:` in a
  #    row-wise reduction makes every column thread compute the whole row (6.9x
  #    waste at 346x346). External applications would make the same mistake.
  #
  # The implementation is **one row (or one group) per block**, with the
  # block's threads cooperating through shared memory. Arithmetic is always in
  # float; float16 buffers are storage only (the output dtype follows the input).
  module NN
    BLOCK = 256

    # Upper bounds on K/V columns held per flash-attention block and on head
    # dimension. K/V/Q and the output accumulator total about 17 KiB, well
    # within the 32 KiB of threadgroup/shared memory on both Metal and CUDA.
    # Heads above the bound fall back to the GEMM path (typical head_dim is
    # 64 or 128).
    FLASH_TILE_KEYS = 16
    FLASH_MAX_HEAD_DIM = 128
    FLASH_MIN_SCORES = 4096

    # Reduction identities. The float minimum avoids -INFINITY (a finite lower
    # bound suffices rather than guaranteeing identical spelling and behavior
    # across MSL/CUDA).
    NEG_MAX = "-3.402823466e+38f"

    module_function

    # Applies a numerically stable softmax over the final dimension.
    #
    # @param x [Buffer, Graph::Tensor] tensor with at least two dimensions
    # @return [Buffer, Graph::Tensor] probabilities with the same shape and dtype
    #
    # Any rank from 2D up — all leading dims fold into rows, and the shape
    # comes back unchanged.
    #
    #   probs = Tilework::NN.softmax(logits)   # [rows, cols] -> [rows, cols]
    #   probs = Tilework::NN.softmax(scores)   # [heads, n, m] works the same
    #
    # The max is subtracted before exp, so large logits do not overflow
    # (`exp(1000)` is inf).
    def softmax(x)
      rows, cols = rowwise_dims!(x, "softmax")
      return Graph.record_nn(:softmax, [x], shape: x.shape, dtype: x.dtype) if Graph.symbolic?(x)
      return cpu_softmax(x, rows, cols) if Tilework.host_fallback?

      out = Ops.allocate_output(x.dtype, x.length, x.shape)
      kernel = Tilework.compile(softmax_src(x.dtype), name: "nn_softmax")
      Ops.launch_kernel(kernel, grid: rows, block: BLOCK, args: [x, out, rows, cols])
      out
    end

    # Applies layer normalization over the final dimension.
    #
    # @param x [Buffer, Graph::Tensor]
    # @param gamma [Buffer, Graph::Tensor, nil] optional scale of length `x.shape.last`
    # @param beta [Buffer, Graph::Tensor, nil] optional bias of length `x.shape.last`
    # @param eps [Float] numerical stability term
    # @return [Buffer, Graph::Tensor]
    #
    # A standard transformer component. Like softmax, any rank from 2D up
    # ([batch, tokens, d] passes as is).
    #
    #   y = Tilework::NN.layernorm(x, gamma: g, beta: b)   # g, b of length cols
    #
    # gamma / beta may be omitted (omitting both means no affine transform).
    def layernorm(x, gamma: nil, beta: nil, eps: 1e-5)
      rows, cols = rowwise_dims!(x, "layernorm")
      check_affine!(gamma, beta, cols, x.dtype)
      if Graph.symbolic?(x, gamma, beta)
        return Graph.record_nn(:layernorm, [x], tensors: { gamma: gamma, beta: beta },
                                                opts: { eps: eps }, shape: x.shape, dtype: x.dtype)
      end
      return cpu_layernorm(x, rows, cols, gamma, beta, eps) if Tilework.host_fallback?

      affine = !gamma.nil? || !beta.nil?
      out = Ops.allocate_output(x.dtype, x.length, x.shape)
      kernel = Tilework.compile(layernorm_src(x.dtype, affine), name: "nn_layernorm")
      args = [x, out]
      temporaries = []
      if affine
        scale = gamma || ones_like(cols, x.dtype).tap { |buffer| temporaries << buffer }
        bias = beta || zeros_like(cols, x.dtype).tap { |buffer| temporaries << buffer }
        args += [scale, bias]
      end
      begin
        Ops.launch_kernel(kernel, grid: rows, block: BLOCK, args: args + [rows, cols, eps.to_f])
      ensure
        temporaries.each(&:free)
      end
      out
    end

    # Applies RMS normalization over the final dimension.
    #
    # @param x [Buffer, Graph::Tensor]
    # @param gamma [Buffer, Graph::Tensor, nil] optional scale of length `x.shape.last`
    # @param eps [Float] numerical stability term
    # @return [Buffer, Graph::Tensor]
    #
    # Divides by the root mean square without subtracting the mean, as used by
    # T5 / LLaMA-family models.
    #
    #   y = Tilework::NN.rmsnorm(x, gamma: g, eps: 1e-6)   # g of length cols
    #
    # Unlike layernorm **there is no mean subtraction**, so no shift (beta)
    # either. The sum of squares accumulates in float (in half it stops
    # growing around 2048).
    def rmsnorm(x, gamma: nil, eps: 1e-6)
      rows, cols = rowwise_dims!(x, "rmsnorm")
      check_affine!(gamma, nil, cols, x.dtype)
      if Graph.symbolic?(x, gamma)
        return Graph.record_nn(:rmsnorm, [x], tensors: { gamma: gamma },
                                              opts: { eps: eps }, shape: x.shape, dtype: x.dtype)
      end
      return cpu_rmsnorm(x, rows, cols, gamma, eps) if Tilework.host_fallback?

      out = Ops.allocate_output(x.dtype, x.length, x.shape)
      kernel = Tilework.compile(rmsnorm_src(x.dtype, !gamma.nil?), name: "nn_rmsnorm")
      args = [x, out]
      args += [gamma] if gamma
      Ops.launch_kernel(kernel, grid: rows, block: BLOCK, args: args + [rows, cols, eps.to_f])
      out
    end

    # Applies group normalization to an NCHW tensor.
    #
    # @param x [Buffer, Graph::Tensor] `[N, C, H, W]` tensor
    # @param groups [Integer] positive divisor of `C`
    # @param gamma [Buffer, Graph::Tensor, nil] optional per-channel scale
    # @param beta [Buffer, Graph::Tensor, nil] optional per-channel bias
    # @param eps [Float] numerical stability term
    # @return [Buffer, Graph::Tensor]
    #
    # Splits channels into `groups` and normalizes each group over all of
    # (C/groups) x H x W — used heavily in diffusion-model UNets.
    #
    #   y = Tilework::NN.groupnorm(x, groups: 32, gamma: g, beta: b)
    #
    # gamma / beta are **per channel** (length C). groups == C is instance
    # norm; groups == 1 is equivalent to layer norm.
    def groupnorm(x, groups:, gamma: nil, beta: nil, eps: 1e-5)
      n, c, h, w = nchw_dims!(x, "groupnorm")
      unless groups.is_a?(Integer) && groups.positive? && (c % groups).zero?
        raise ArgumentError, "groups #{groups.inspect} must be a positive divisor of #{c} channels"
      end

      check_affine!(gamma, beta, c, x.dtype)
      if Graph.symbolic?(x, gamma, beta)
        return Graph.record_nn(:groupnorm, [x], tensors: { gamma: gamma, beta: beta },
                                                opts: { groups: groups, eps: eps },
                                                shape: x.shape, dtype: x.dtype)
      end
      return cpu_groupnorm(x, [n, c, h, w], groups, gamma, beta, eps) if Tilework.host_fallback?

      affine = !gamma.nil? || !beta.nil?
      out = Ops.allocate_output(x.dtype, x.length, x.shape)
      kernel = Tilework.compile(groupnorm_src(x.dtype, affine), name: "nn_groupnorm")
      args = [x, out]
      temporaries = []
      if affine
        scale = gamma || ones_like(c, x.dtype).tap { |buffer| temporaries << buffer }
        bias = beta || zeros_like(c, x.dtype).tap { |buffer| temporaries << buffer }
        args += [scale, bias]
      end
      begin
        Ops.launch_kernel(kernel, grid: n * groups, block: BLOCK, args: args + [n, c, h * w, groups, eps.to_f])
      ensure
        temporaries.each(&:free)
      end
      out
    end

    # --- Activations ---------------------------------------------------------

    # Applies the SiLU (Swish) activation.
    # @param x [Buffer, Graph::Tensor]
    # @return [Buffer, Graph::Tensor]
    #
    # x * sigmoid(x), standard in diffusion models and recent LLMs. sigmoid is
    # rearranged to use exp(-|x|), so very negative x does not overflow (a naive
    # 1/(1+exp(-x)) gives exp(100) = inf at x = -100).
    def silu(x) = activation(x, :silu)

    # Applies GELU using the tanh approximation by default.
    # @param x [Buffer, Graph::Tensor]
    # @param exact [Boolean] use the slower `erf` formulation
    # @return [Buffer, Graph::Tensor]
    #
    # The tanh approximation is the default of GPT-2 and many implementations.
    def gelu(x, exact: false) = activation(x, exact ? :gelu_exact : :gelu_tanh)

    # Applies the ReLU activation.
    # @param x [Buffer, Graph::Tensor]
    # @return [Buffer, Graph::Tensor]
    def relu(x) = activation(x, :relu)

    # Applies the sigmoid activation.
    # @param x [Buffer, Graph::Tensor]
    # @return [Buffer, Graph::Tensor]
    def sigmoid(x) = activation(x, :sigmoid)

    # --- Losses ------------------------------------------------------------
    #
    # The entry point of training. **They end in a reduction**, so like
    # `Ops.reduce` there are two ways to receive the result (see #reduce in ops.rb):
    #
    #   NN.mse(y, t)                 # => Float; the queue drains here
    #   NN.mse(y, t, device: true)   # => a one-element Buffer, left on the device
    #   NN.mse(y, t, reduction: :none)  # => per-element / per-row loss Buffer
    #
    # In a graph they always run in the device: true form (dropping to the host
    # would end the plan). Backward (#22) requires that form, so **keeping the
    # loss a Tensor** is the point.
    #
    # reduction is :mean (default) / :sum / :none. It changes the gradient's
    # magnitude, so stating it explicitly beats relying on the default.

    LOSS_REDUCTIONS = %i[mean sum none].freeze

    # Calculates mean squared error.
    #
    # @param pred [Buffer, Graph::Tensor]
    # @param target [Buffer, Graph::Tensor] same shape and dtype as `pred`
    # @param reduction [:mean, :sum, :none] reduction mode
    # @param device [Boolean] return a one-element buffer instead of a host scalar
    # @return [Float, Buffer, Graph::Tensor]
    #
    #   loss = Tilework::NN.mse(pred, target)
    #
    # reduction: :none returns per-element squared errors (the input's shape).
    def mse(pred, target, reduction: :mean, device: false)
      check_loss_pair!(pred, target)
      check_reduction!(reduction)
      if Graph.symbolic?(pred, target)
        return Graph.record_nn(:mse, [pred, target], opts: { reduction: reduction, device: true },
                                                     shape: loss_shape(pred.shape, reduction, rowwise: false),
                                                     dtype: loss_dtype(pred.dtype))
      end

      reduce_loss(squared_error(pred, target), reduction, device: device)
    end

    # Calculates cross entropy from unnormalized logits and int32 class labels.
    #
    # @param logits [Buffer, Graph::Tensor] tensor whose final dimension is classes
    # @param labels [Buffer, Graph::Tensor] int32 tensor with the final logits dimension removed
    # @param reduction [:mean, :sum, :none] reduction mode
    # @param device [Boolean] return a one-element buffer instead of a host scalar
    # @return [Float, Buffer, Graph::Tensor]
    #
    #   loss = Tilework::NN.cross_entropy(logits, labels)
    #
    # **Takes logits** (values before softmax). It computes **log-sum-exp in one
    # kernel** rather than log after softmax: a naive `log(softmax(x))` hits
    # -inf where a probability underflows to 0, and its gradient does not take
    # the plain (softmax(x) - onehot) / n form.
    #
    # Out-of-range labels are clamped to the ends (the same rule as `gather`) —
    # an out-of-range read on a GPU is undefined rather than merely wrong, so
    # clamping beats breaking silently.
    def cross_entropy(logits, labels, reduction: :mean, device: false)
      rows, cols = rowwise_dims!(logits, "cross_entropy")
      check_labels!(labels, logits, rows)
      check_reduction!(reduction)
      if Graph.symbolic?(logits, labels)
        return Graph.record_nn(:cross_entropy, [logits, labels],
                               opts: { reduction: reduction, device: true },
                               shape: loss_shape(logits.shape, reduction, rowwise: true),
                               dtype: loss_dtype(logits.dtype))
      end

      rowwise = if Tilework.host_fallback?
                  cpu_cross_entropy(logits, labels, rows, cols)
                else
                  out = Ops.allocate_output(loss_dtype(logits.dtype), rows, [rows])
                  kernel = Tilework.compile(cross_entropy_src(logits.dtype), name: "nn_cross_entropy")
                  Ops.launch_kernel(kernel, grid: rows, block: BLOCK, args: [logits, labels, out, rows, cols])
                  out
                end
      reduce_loss(rowwise, reduction, device: device)
    end

    # Transposes a two-dimensional tensor.
    #
    # @param x [Buffer, Graph::Tensor] `[rows, columns]` tensor
    # @return [Buffer, Graph::Tensor] `[columns, rows]` tensor
    #
    # **Launches over the output's domain.** Launching over the input's domain
    # and writing `out[c, r]` leaves the tail of the output unwritten when rows
    # and columns differ.
    def transpose(x)
      rows, cols = matrix_dims!(x, "transpose")
      return Graph.record_nn(:transpose, [x], shape: [cols, rows], dtype: x.dtype) if Graph.symbolic?(x)
      return cpu_transpose(x, rows, cols) if Tilework.host_fallback?

      out = Ops.allocate_output(x.dtype, x.length, [cols, rows])
      kernel = Tilework.compile(transpose_src(x.dtype), name: "nn_transpose")
      # The grid covers the output [cols, rows]: x is the column and y the row, as in the DSL.
      grid = [(rows / 16.0).ceil, (cols / 16.0).ceil]
      Ops.launch_kernel(kernel, grid: grid, block: [16, 16], args: [x, out, rows, cols])
      out
    end

    # Calculates scaled dot-product multi-head attention.
    #
    # @param q [Buffer, Graph::Tensor] query matrix `[n, d]`
    # @param k [Buffer, Graph::Tensor] key matrix `[m, d]`
    # @param v [Buffer, Graph::Tensor] value matrix `[m, d]`
    # @param heads [Integer] number of heads dividing `d`
    # @param scale [Float, nil] score scale; defaults to `1 / sqrt(d / heads)`
    # @param bias [Buffer, Graph::Tensor, nil] optional broadcastable score bias
    # @return [Buffer, Graph::Tensor] attention output `[n, d]`
    #
    #   q: [n, d], k/v: [m, d], d = heads * head_dim; returns [n, d].
    #
    # Heads are **small, evenly spaced matrices** inside `[*, d]`, so with a
    # strided batched GEMM they never need slicing out or writing back: pass
    # the original width d as ld and head_dim as the stride and compute on
    # views. The score matrix [heads, n, m] is contiguous, so viewed as
    # [heads*n, m] **one softmax covers every head**.
    #
    # For one 16-head attention this cuts launches from about 100 to 4.
    #
    # scale multiplies the scores (default 1/sqrt(head_dim)). It goes to GEMM's
    # alpha, so it is **free**; writing `* (1.0/Math.sqrt(dh))` yourself costs
    # another full round trip over the score matrix.
    #
    # bias is added to the scores **before** softmax, shaped [heads * n, m]
    # (each head's [n, m] stacked vertically — the scores' own layout). T5's
    # relative position bias and additive padding masks are this.
    #
    # If padding sits at the end, **passing k/v cut to the valid length** is
    # faster than masking with bias (it removes the computation itself).
    def multi_head_attention(q, k, v, heads:, scale: nil, bias: nil)
      n, d = matrix_dims!(q, "multi_head_attention (q)")
      m, dk = matrix_dims!(k, "multi_head_attention (k)")
      mv, dv = matrix_dims!(v, "multi_head_attention (v)")
      unless d == dk && d == dv && m == mv
        raise ArgumentError,
              "q [#{n}, #{d}] に対し k は [m, #{d}]、v は [m, #{d}] でなければならない " \
              "(k: [#{m}, #{dk}], v: [#{mv}, #{dv}])"
      end
      raise ArgumentError, "d (#{d}) は heads (#{heads}) で割り切れなければならない" unless (d % heads).zero?

      dh = d / heads
      scale ||= 1.0 / Math.sqrt(dh)
      check_attention_bias!(bias, heads, n, m)
      if Graph.symbolic?(q, k, v, bias)
        # scale is fixed here before becoming an attribute (decided when the plan is built).
        return Graph.record_nn(:multi_head_attention, [q, k, v], tensors: { bias: bias },
                                                                 opts: { heads: heads, scale: scale },
                                                                 shape: [n, d], dtype: :float32)
      end
      return flash_attention(q, k, v, heads, dh, n, m, d, scale, bias) if flash_attention?(dh, n, m)

      unless TensorCore.gemm_batched? && q.dtype == k.dtype && q.dtype == v.dtype &&
             Tilework.capabilities.gemm_dtypes.include?(q.dtype)
        return attention_per_head(q, k, v, heads, dh, n, m, d, scale, bias)
      end

      # Column-major Cc = Kc^T * Ac. Kc/Ac are views inside the parent [*, d], so ld = d.
      scores = Buffer.new(:float32, heads * n * m, shape: [heads * n, m])
      TensorCore.gemm_batched(k, q, scores,
                              m: m, k: dh, n: n,
                              lda: d, ldb: d, ldc: m,
                              stride_a: dh, stride_b: dh, stride_c: n * m,
                              batch: heads, trans_a: true, alpha: scale)
      if bias
        biased = scores + bias
        scores.free
        scores = biased
      end
      probs = softmax(scores)
      scores.free

      # Column-major ctx^T = Vc * Pc. With ldc = d and stride_c = dh it writes
      # **directly** into the h-th section of the output [n, d] (no write-back kernel).
      probs_for_gemm = if Ops::LOW_PRECISION_DTYPES.include?(v.dtype)
                         probs.cast(v.dtype)
                       else
                         probs
                       end
      out = Ops.allocate_output(:float32, n * d, [n, d])
      TensorCore.gemm_batched(v, probs_for_gemm, out,
                              m: dh, k: m, n: n,
                              lda: d, ldb: m, ldc: d,
                              stride_a: dh, stride_b: n * m, stride_c: dh,
                              batch: heads)
      probs_for_gemm.free unless probs_for_gemm.equal?(probs)
      probs.free
      out
    rescue StandardError
      [scores, probs_for_gemm, probs, out].compact.uniq.each { |buffer| buffer.free unless buffer.freed? }
      raise
    end

    # ========================================================================
    # Internal
    # ========================================================================

    # Attention that never puts the [heads*n, m] scores in VRAM. Each block is
    # "one query row of one head": K/V tiles go through shared memory while an
    # online softmax updates the running max / running sum and the output
    # accumulator (the same numerical skeleton as FlashAttention). The working
    # set is O(tile * dh), and no Buffer proportional to n*m is allocated.
    def flash_attention(q, k, v, heads, dh, n, m, d, scale, bias)
      out = Ops.allocate_output(:float32, n * d, [n, d])
      kernel = Tilework.compile(flash_attention_src(q.dtype, k.dtype, v.dtype, bias&.dtype),
                                name: "nn_flash_attention")
      args = [q, k, v]
      args << bias if bias
      Ops.launch_kernel(kernel, grid: heads * n, block: BLOCK,
                            args: args + [out, heads, n, m, d, dh, scale.to_f])
      out
    rescue StandardError
      out&.free
      raise
    end

    def flash_attention?(dh, n, m)
      !Tilework.host_fallback? && Tilework.capabilities.dialect &&
        dh <= FLASH_MAX_HEAD_DIM && (n * m) >= FLASH_MIN_SCORES
    end

    def flash_attention_src(q_dtype, k_dtype, v_dtype, bias_dtype)
      qt = ctype(q_dtype)
      kt = ctype(k_dtype)
      vt = ctype(v_dtype)
      bias_param = bias_dtype ? ",\n                                        device const #{ctype(bias_dtype)}* bias" : ""
      bias_read = bias_dtype ? " + #{read(bias_dtype, 'bias[(head * n + query) * m + key]')}" : ""
      <<~KERNEL
        GPU_KERNEL void nn_flash_attention(GPU_ARGS,
                                            device const #{qt}* q,
                                            device const #{kt}* k,
                                            device const #{vt}* v#{bias_param},
                                            device float* out,
                                            constant int& heads,
                                            constant int& n,
                                            constant int& m,
                                            constant int& d,
                                            constant int& dh,
                                            constant float& scale) {
          int row = GROUP_ID_X;
          int lid = LOCAL_ID_X;
          int head = row / n;
          int query = row % n;
          if (head >= heads) return;

          GPU_SHARED float dot[#{BLOCK}];
          GPU_SHARED float qtile[#{FLASH_MAX_HEAD_DIM}];
          GPU_SHARED float ktile[#{FLASH_TILE_KEYS * FLASH_MAX_HEAD_DIM}];
          GPU_SHARED float vtile[#{FLASH_TILE_KEYS * FLASH_MAX_HEAD_DIM}];
          GPU_SHARED float acc[#{FLASH_MAX_HEAD_DIM}];
          // state = max, sum, old-scale, new-score-weight
          GPU_SHARED float state[4];

          for (int j = lid; j < dh; j += #{BLOCK}) {
            qtile[j] = #{read(q_dtype, 'q[query * d + head * dh + j]')};
            acc[j] = 0.0f;
          }
          if (lid == 0) { state[0] = #{NEG_MAX}; state[1] = 0.0f; }
          GPU_BARRIER;

          for (int tile = 0; tile < m; tile += #{FLASH_TILE_KEYS}) {
            int count = m - tile < #{FLASH_TILE_KEYS} ? m - tile : #{FLASH_TILE_KEYS};
            for (int i = lid; i < count * dh; i += #{BLOCK}) {
              int t = i / dh;
              int j = i % dh;
              ktile[i] = #{read(k_dtype, 'k[(tile + t) * d + head * dh + j]')};
              vtile[i] = #{read(v_dtype, 'v[(tile + t) * d + head * dh + j]')};
            }
            GPU_BARRIER;

            for (int t = 0; t < count; t++) {
              int key = tile + t;
              float partial = 0.0f;
              for (int j = lid; j < dh; j += #{BLOCK}) partial += qtile[j] * ktile[t * dh + j];
              dot[lid] = partial;
              #{block_reduce('dot[lid] + dot[lid + s]').gsub('sh[', 'dot[')}

              if (lid == 0) {
                float score = dot[0] * scale#{bias_read};
                float next_max = fmax(state[0], score);
                state[2] = exp(state[0] - next_max);
                state[3] = exp(score - next_max);
                state[0] = next_max;
                state[1] = state[1] * state[2] + state[3];
              }
              GPU_BARRIER;
              for (int j = lid; j < dh; j += #{BLOCK}) {
                acc[j] = acc[j] * state[2] + state[3] * vtile[t * dh + j];
              }
              GPU_BARRIER;
            }
          }

          for (int j = lid; j < dh; j += #{BLOCK}) out[query * d + head * dh + j] = acc[j] / state[1];
        }
      KERNEL
    end

    # For backends without batched GEMM (such as :cpu): materializes heads one
    # at a time. Slow, but the result is the same.
    def attention_per_head(q, k, v, heads, dh, n, m, d, scale, bias = nil)
      # head_scatter covers [n, d] across all heads, so a reused pool slot (not
      # zero-filled) leaves no residue.
      out = Ops.allocate_output(:float32, n * d, [n, d])
      heads.times do |h|
        qh = head_slice(q, h, dh, n, d)
        kh = head_slice(k, h, dh, m, d)
        vh = head_slice(v, h, dh, m, d)
        kt = transpose(kh)
        scores = qh.matmul(kt) * scale
        if bias
          # bias is [heads * n, m]; the h-th [n, m] is contiguous by rows, so a
          # view can take it (do not let it outlive its parent)
          biased = scores + bias.view(h * n * m, n * m, shape: [n, m])
          scores.free
          scores = biased
        end
        probs = softmax(scores)
        ctx = probs.matmul(vh)
        head_scatter(ctx, out, h, dh, n, d)
        [qh, kh, vh, kt, scores, probs, ctx].each(&:free)
      end
      out
    rescue StandardError
      [qh, kh, vh, kt, scores, probs, ctx, out].compact.uniq.each { |buffer| buffer.free unless buffer.freed? }
      raise
    end

    def check_attention_bias!(bias, heads, n, m)
      return if bias.nil?

      shaped!(bias, "bias")
      return if bias.shape == [heads * n, m]

      raise ArgumentError,
            "bias must be [heads * n, m] = #{[heads * n, m].inspect}, got #{bias.shape.inspect}"
    end

    # Extracts the h-th head section of [rows, d] as [rows, dh].
    def head_slice(x, head, dh, rows, _d)
      @head_gather ||= Tilework.kernel("nn_head_gather", out: 1) do |src, o, hd, dhh|
        j = global_id_x
        r = global_id_y
        o[r, j] = src[r, (hd * dhh) + j] if j < dhh && r < src.rows
      end
      @head_gather.call(x, head, dh, over: [dh, rows])
    end

    # The reverse direction. dst is not out:, so the guard is written by hand.
    def head_scatter(src, dst, head, dh, rows, _d)
      @head_scatter ||= Tilework.kernel("nn_head_scatter") do |s, o, hd, dhh|
        j = global_id_x
        r = global_id_y
        o[r, (hd * dhh) + j] = s[r, j] if j < dhh && r < s.rows
      end
      @head_scatter.call(src, dst, head, dh, over: [dh, rows])
    end

    def ctype(dtype) = Ops::CTYPE.fetch(dtype)

    # Half is a storage type, so it is widened to float before entering an expression.
    def read(dtype, expr) = Ops.read_dtype(dtype, expr)

    # Checks that look only at shape and dtype accept Buffers and Graph::Tensors
    # alike, so recording into a graph goes through **the same checks** —
    # splitting them would let expressions through on one path only.
    def shaped!(x, op)
      return x if x.is_a?(Buffer) || x.is_a?(Graph::Tensor)

      raise TypeError, "#{op} requires a Tilework::Buffer or Tilework::Graph::Tensor, got #{x.class}"
    end

    def matrix_dims!(x, op)
      shaped!(x, op)
      unless x.shape.length == 2
        raise ArgumentError, "#{op} requires a 2D buffer, got #{x.shape.inspect}"
      end

      x.shape
    end

    # The launch domain for row-wise reductions (softmax / layernorm /
    # rmsnorm): **the last dim is columns and everything before folds into
    # rows**. Kernels only see a flat [rows, cols], so 3D and up pass as is
    # ([batch, tokens, d] or attention's [heads, n, m]); the output keeps the
    # input's shape, so no dims are lost.
    def rowwise_dims!(x, op)
      shaped!(x, op)
      unless x.shape.length >= 2
        raise ArgumentError, "#{op} requires a 2D or higher buffer, got #{x.shape.inspect}"
      end

      [x.shape[0..-2].inject(:*), x.shape[-1]]
    end

    def nchw_dims!(x, op)
      shaped!(x, op)
      unless x.shape.length == 4
        raise ArgumentError, "#{op} requires a 4D NCHW buffer, got #{x.shape.inspect}"
      end

      x.shape
    end

    def check_affine!(gamma, beta, expected, dtype)
      [["gamma", gamma], ["beta", beta]].each do |name, buf|
        next if buf.nil?

        shaped!(buf, name)
        unless buf.length == expected
          raise ArgumentError, "#{name} must have #{expected} elements, got #{buf.length}"
        end
        unless buf.dtype == dtype
          raise ArgumentError, "#{name} dtype #{buf.dtype.inspect} does not match the input's #{dtype.inspect}"
        end
      end
    end

    # --- Shared loss helpers -------------------------------------------------

    def check_reduction!(reduction)
      return if LOSS_REDUCTIONS.include?(reduction)

      raise ArgumentError, "reduction must be one of #{LOSS_REDUCTIONS.inspect}, got #{reduction.inspect}"
    end

    def check_loss_pair!(pred, target)
      shaped!(pred, "mse (pred)")
      shaped!(target, "mse (target)")
      unless pred.shape == target.shape
        raise ArgumentError, "mse requires matching shapes (pred #{pred.shape.inspect}, " \
                             "target #{target.shape.inspect})"
      end
      unless pred.dtype == target.dtype
        raise ArgumentError, "mse requires matching dtypes (pred #{pred.dtype.inspect}, " \
                             "target #{target.dtype.inspect})"
      end
      return if FLOAT_DTYPES.include?(pred.dtype)

      raise ArgumentError, "mse requires a float dtype, got #{pred.dtype.inspect}"
    end

    # Labels are class indices (:int32). Probability-style inputs (one-hot /
    # label smoothing) are floats of the same shape, so a mix-up fails by name,
    # saying what is expected.
    def check_labels!(labels, logits, rows)
      shaped!(labels, "cross_entropy (labels)")
      unless labels.dtype == :int32
        raise ArgumentError, "cross_entropy labels must be :int32 class indices, got " \
                             "#{labels.dtype.inspect} (probability targets are not supported yet)"
      end
      expected = logits.shape[0..-2]
      return if labels.shape == expected || labels.length == rows

      raise ArgumentError, "cross_entropy labels must have shape #{expected.inspect} " \
                           "(logits #{logits.shape.inspect}), got #{labels.shape.inspect}"
    end

    FLOAT_DTYPES = %i[float32 float16 bfloat16].freeze

    # Losses are always kept in float. Accumulating in half piles up error as
    # the count grows, so even :none (no reduction) widens to float32 (the same
    # rule as Ops' accumulation).
    def loss_dtype(dtype) = Ops::LOW_PRECISION_DTYPES.include?(dtype) ? :float32 : dtype

    def loss_shape(shape, reduction, rowwise:)
      return [1] unless reduction == :none

      rowwise ? shape[0..-2] : shape
    end

    # Receives the reduced value; :none returns it as is.
    def reduce_loss(values, reduction, device:)
      return values if reduction == :none

      result = if reduction == :mean
                 Ops.mean_op(values, device: device)
               else
                 Ops.reduce(:sum, values, device: device)
               end
      values.free
      result
    end

    def squared_error(pred, target)
      dtype = loss_dtype(pred.dtype)
      if Tilework.host_fallback?
        values = pred.to_a.zip(target.to_a).map { |a, b| (a - b)**2 }
        out = Ops.allocate_output(dtype, values.length, pred.shape)
        out.write(values)
        return out
      end

      out = Ops.allocate_output(dtype, pred.length, pred.shape)
      kernel = Tilework.compile(squared_error_src(pred.dtype, dtype), name: "nn_squared_error")
      Ops.launch_kernel(kernel, grid: (pred.length / 256.0).ceil, block: 256, args: [pred, target, out, pred.length])
      out
    end

    def squared_error_src(dtype, acc)
      t = ctype(dtype)
      <<~KERNEL
        GPU_KERNEL void nn_squared_error(GPU_ARGS,
                                         device const #{t}* a,
                                         device const #{t}* b,
                                         device #{ctype(acc)}* y,
                                         constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < n) {
            float d = #{read(dtype, 'a[i]')} - #{read(dtype, 'b[i]')};
            y[i] = d * d;
          }
        }
      KERNEL
    end

    # One row per block, the same skeleton as softmax (max → sum), but only one
    # loss per row is written, so no third pass is needed.
    def cross_entropy_src(dtype)
      t = ctype(dtype)
      <<~KERNEL
        GPU_KERNEL void nn_cross_entropy(GPU_ARGS,
                                         device const #{t}* x,
                                         device const int* labels,
                                         device #{ctype(loss_dtype(dtype))}* loss,
                                         constant int& rows,
                                         constant int& cols) {
          int row = GROUP_ID_X;
          int lid = LOCAL_ID_X;
          if (row >= rows) return;
          GPU_SHARED float sh[#{BLOCK}];
          int base = row * cols;

          // 1. Row max (stabilizes log-sum-exp)
          float m = #{NEG_MAX};
          for (int i = lid; i < cols; i += #{BLOCK}) m = fmax(m, #{read(dtype, 'x[base + i]')});
          sh[lid] = m;
          #{block_reduce('fmax(sh[lid], sh[lid + s])')}
          m = sh[0];
          GPU_BARRIER; // sh is reused by the next reduction, so wait until everyone has read it

          // 2. Sum of exp
          float total = 0.0f;
          for (int i = lid; i < cols; i += #{BLOCK}) total += exp(#{read(dtype, 'x[base + i]')} - m);
          sh[lid] = total;
          #{block_reduce('sh[lid] + sh[lid + s]')}

          // 3. -log p[label] = log(sum exp(x - m)) - (x[label] - m)
          if (lid == 0) {
            int c = labels[row];
            c = c < 0 ? 0 : (c >= cols ? cols - 1 : c); // clamp out-of-range to the ends (as gather does)
            loss[row] = log(sh[0]) - (#{read(dtype, 'x[base + c]')} - m);
          }
        }
      KERNEL
    end

    def cpu_cross_entropy(logits, labels, rows, cols)
      flat = logits.to_a
      classes = labels.to_a
      values = Array.new(rows) do |row|
        slice = flat[row * cols, cols]
        m = slice.max
        c = classes[row].clamp(0, cols - 1)
        Math.log(slice.sum { |v| Math.exp(v - m) }) - (slice[c] - m)
      end
      out = Ops.allocate_output(loss_dtype(logits.dtype), rows, [rows])
      out.write(values)
      out
    end

    # Fills in when only one of gamma/beta is passed.
    def ones_like(n, dtype)
      buf = Buffer.new(dtype, n)
      buf.write(Array.new(n, 1.0))
      buf
    end

    def zeros_like(n, dtype) = Buffer.new(dtype, n) # zero-filled on allocation

    # In-block reduction: call with values in sh[] and the result lands in
    # sh[0]. To reuse sh afterwards, read it and then put a GPU_BARRIER in between.
    def block_reduce(combine)
      <<~SRC.chomp
        GPU_BARRIER;
              for (int s = #{BLOCK / 2}; s > 0; s >>= 1) {
                if (lid < s) { sh[lid] = #{combine}; }
                GPU_BARRIER;
              }
      SRC
    end

    def softmax_src(dtype)
      t = ctype(dtype)
      <<~KERNEL
        GPU_KERNEL void nn_softmax(GPU_ARGS,
                                   device const #{t}* x,
                                   device #{t}* y,
                                   constant int& rows,
                                   constant int& cols) {
          int row = GROUP_ID_X;
          int lid = LOCAL_ID_X;
          if (row >= rows) return;
          GPU_SHARED float sh[#{BLOCK}];
          int base = row * cols;

          // 1. Row max. Without subtracting it, exp overflows on large logits.
          float m = #{NEG_MAX};
          for (int i = lid; i < cols; i += #{BLOCK}) m = fmax(m, #{read(dtype, 'x[base + i]')});
          sh[lid] = m;
          #{block_reduce('fmax(sh[lid], sh[lid + s])')}
          m = sh[0];
          GPU_BARRIER; // sh is reused by the next reduction, so wait until everyone has read it

          // 2. Sum of exp
          float total = 0.0f;
          for (int i = lid; i < cols; i += #{BLOCK}) total += exp(#{read(dtype, 'x[base + i]')} - m);
          sh[lid] = total;
          #{block_reduce('sh[lid] + sh[lid + s]')}
          float denom = sh[0];

          // 3. Write back
          for (int i = lid; i < cols; i += #{BLOCK}) {
            y[base + i] = exp(#{read(dtype, 'x[base + i]')} - m) / denom;
          }
        }
      KERNEL
    end

    def layernorm_src(dtype, affine)
      t = ctype(dtype)
      params = ["device const #{t}* x", "device #{t}* y"]
      params += ["device const #{t}* gamma", "device const #{t}* beta"] if affine
      params += ["constant int& rows", "constant int& cols", "constant float& eps"]
      scale = affine ? " * #{read(dtype, 'gamma[i]')} + #{read(dtype, 'beta[i]')}" : ""
      <<~KERNEL
        GPU_KERNEL void nn_layernorm(GPU_ARGS,
                                     #{params.join(",\n                                     ")}) {
          int row = GROUP_ID_X;
          int lid = LOCAL_ID_X;
          if (row >= rows) return;
          GPU_SHARED float sh[#{BLOCK}];
          int base = row * cols;

          // 1. Mean
          float total = 0.0f;
          for (int i = lid; i < cols; i += #{BLOCK}) total += #{read(dtype, 'x[base + i]')};
          sh[lid] = total;
          #{block_reduce('sh[lid] + sh[lid + s]')}
          float mean = sh[0] / (float)cols;
          GPU_BARRIER;

          // 2. Variance, recomputing deviations in a second pass (E[x^2] - E[x]^2 cancels catastrophically).
          float sq = 0.0f;
          for (int i = lid; i < cols; i += #{BLOCK}) {
            float d = #{read(dtype, 'x[base + i]')} - mean;
            sq += d * d;
          }
          sh[lid] = sq;
          #{block_reduce('sh[lid] + sh[lid + s]')}
          float inv = 1.0f / sqrt(sh[0] / (float)cols + eps);

          // 3. Normalize
          for (int i = lid; i < cols; i += #{BLOCK}) {
            y[base + i] = (#{read(dtype, 'x[base + i]')} - mean) * inv#{scale};
          }
        }
      KERNEL
    end

    # The same one-row-per-block reduction as layernorm, but with no mean
    # subtraction it takes one pass (layernorm scans twice, for mean and variance).
    def rmsnorm_src(dtype, affine)
      t = ctype(dtype)
      params = ["device const #{t}* x", "device #{t}* y"]
      params += ["device const #{t}* gamma"] if affine
      params += ["constant int& rows", "constant int& cols", "constant float& eps"]
      scale = affine ? " * #{read(dtype, 'gamma[i]')}" : ""
      <<~KERNEL
        GPU_KERNEL void nn_rmsnorm(GPU_ARGS,
                                   #{params.join(",\n                                   ")}) {
          int row = GROUP_ID_X;
          int lid = LOCAL_ID_X;
          if (row >= rows) return;
          GPU_SHARED float sh[#{BLOCK}];
          int base = row * cols;

          // Sum of squares; always accumulate in float (in half it stops growing midway)
          float sq = 0.0f;
          for (int i = lid; i < cols; i += #{BLOCK}) {
            float v = #{read(dtype, 'x[base + i]')};
            sq += v * v;
          }
          sh[lid] = sq;
          #{block_reduce('sh[lid] + sh[lid + s]')}
          float inv = 1.0f / sqrt(sh[0] / (float)cols + eps);

          for (int i = lid; i < cols; i += #{BLOCK}) {
            y[base + i] = #{read(dtype, 'x[base + i]')} * inv#{scale};
          }
        }
      KERNEL
    end

    def groupnorm_src(dtype, affine)
      t = ctype(dtype)
      params = ["device const #{t}* x", "device #{t}* y"]
      params += ["device const #{t}* gamma", "device const #{t}* beta"] if affine
      params += ["constant int& n", "constant int& c", "constant int& hw",
                 "constant int& groups", "constant float& eps"]
      scale = affine ? " * #{read(dtype, 'gamma[ch]')} + #{read(dtype, 'beta[ch]')}" : ""
      <<~KERNEL
        GPU_KERNEL void nn_groupnorm(GPU_ARGS,
                                     #{params.join(",\n                                     ")}) {
          // One block = one (image, group). The reduction spans all of (C/groups) * H * W.
          int blk = GROUP_ID_X;
          int lid = LOCAL_ID_X;
          if (blk >= n * groups) return;
          int img = blk / groups;
          int grp = blk % groups;
          int per = c / groups;           // channels per group
          int count = per * hw;           // elements per group
          int base = (img * c + grp * per) * hw;
          GPU_SHARED float sh[#{BLOCK}];

          float total = 0.0f;
          for (int i = lid; i < count; i += #{BLOCK}) total += #{read(dtype, 'x[base + i]')};
          sh[lid] = total;
          #{block_reduce('sh[lid] + sh[lid + s]')}
          float mean = sh[0] / (float)count;
          GPU_BARRIER;

          float sq = 0.0f;
          for (int i = lid; i < count; i += #{BLOCK}) {
            float d = #{read(dtype, 'x[base + i]')} - mean;
            sq += d * d;
          }
          sh[lid] = sq;
          #{block_reduce('sh[lid] + sh[lid + s]')}
          float inv = 1.0f / sqrt(sh[0] / (float)count + eps);

          for (int i = lid; i < count; i += #{BLOCK}) {
            int ch = grp * per + i / hw;  // gamma/beta are per channel
            y[base + i] = (#{read(dtype, 'x[base + i]')} - mean) * inv#{scale};
          }
        }
      KERNEL
    end

    # Every activation is one thread per element; only the expression differs.
    ACTIVATIONS = {
      relu: "fmax(v, 0.0f)",
      # Writing 1/(1+exp(-v)) directly overflows exp for very negative v. This
      # form uses only exp(-|v|), so it is safe for either sign.
      sigmoid: "(v >= 0.0f ? 1.0f / (1.0f + exp(-v)) : exp(v) / (1.0f + exp(v)))",
      silu: "v * (v >= 0.0f ? 1.0f / (1.0f + exp(-v)) : exp(v) / (1.0f + exp(v)))",
      gelu_tanh: "0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)))",
      gelu_exact: "0.5f * v * (1.0f + GPU_ERF(v * 0.7071067811865476f))",
    }.freeze

    def activation(x, kind)
      shaped!(x, kind.to_s)
      return Graph.record_nn(:activation, [x], args: [kind], shape: x.shape, dtype: x.dtype) if Graph.symbolic?(x)
      return cpu_activation(x, kind) if Tilework.host_fallback?

      expr = ACTIVATIONS.fetch(kind)
      t = ctype(x.dtype)
      src = <<~KERNEL
        GPU_KERNEL void nn_#{kind}(GPU_ARGS,
                                   device const #{t}* x,
                                   device #{t}* y,
                                   constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < n) {
            float v = #{read(x.dtype, 'x[i]')};
            y[i] = #{expr};
          }
        }
      KERNEL
      out = Ops.allocate_output(x.dtype, x.length, x.shape)
      kernel = Tilework.compile(src, name: "nn_#{kind}")
      Ops.launch_kernel(kernel, grid: (x.length / 256.0).ceil, block: 256, args: [x, out, x.length])
      out
    end

    def transpose_src(dtype)
      t = ctype(dtype)
      <<~KERNEL
        GPU_KERNEL void nn_transpose(GPU_ARGS,
                                     device const #{t}* x,
                                     device #{t}* y,
                                     constant int& rows,
                                     constant int& cols) {
          // Indexed by the input [rows, cols] position; the output is [cols, rows].
          int r = GLOBAL_ID_X;
          int c = GLOBAL_ID_Y;
          if (r < rows && c < cols) y[c * rows + r] = x[r * cols + c];
        }
      KERNEL
    end

    # --- Reference implementations for the :cpu backend ----------------------
    #
    # :cpu has no generator for shared memory, so these compute on the host.
    # Slow, but they also **produce the expected values for tests** (GPU
    # results are compared against them).

    def rebuild(x, values, shape = x.shape)
      out = Ops.allocate_output(x.dtype, values.length, shape)
      out.write(values)
      out
    end

    def cpu_softmax(x, rows, cols)
      values = x.to_a.each_slice(cols).flat_map do |row|
        m = row.max
        exps = row.map { |v| Math.exp(v - m) }
        total = exps.sum
        exps.map { |e| e / total }
      end
      rebuild(x, values)
    end

    def cpu_layernorm(x, rows, cols, gamma, beta, eps)
      g = gamma&.to_a
      b = beta&.to_a
      values = x.to_a.each_slice(cols).flat_map do |row|
        mean = row.sum / cols.to_f
        var = row.sum { |v| (v - mean)**2 } / cols.to_f
        inv = 1.0 / Math.sqrt(var + eps)
        row.each_with_index.map do |v, i|
          ((v - mean) * inv * (g ? g[i] : 1.0)) + (b ? b[i] : 0.0)
        end
      end
      rebuild(x, values)
    end

    def cpu_rmsnorm(x, rows, cols, gamma, eps)
      g = gamma&.to_a
      values = x.to_a.each_slice(cols).flat_map do |row|
        inv = 1.0 / Math.sqrt((row.sum { |v| v * v } / cols.to_f) + eps)
        row.each_with_index.map { |v, i| v * inv * (g ? g[i] : 1.0) }
      end
      rebuild(x, values)
    end

    def cpu_groupnorm(x, shape, groups, gamma, beta, eps)
      n, c, h, w = shape
      hw = h * w
      per = c / groups
      g = gamma&.to_a
      b = beta&.to_a
      flat = x.to_a
      out = Array.new(flat.length)
      n.times do |img|
        groups.times do |grp|
          base = ((img * c) + (grp * per)) * hw
          slice = flat[base, per * hw]
          mean = slice.sum / slice.length.to_f
          var = slice.sum { |v| (v - mean)**2 } / slice.length.to_f
          inv = 1.0 / Math.sqrt(var + eps)
          slice.each_with_index do |v, i|
            ch = (grp * per) + (i / hw)
            out[base + i] = ((v - mean) * inv * (g ? g[ch] : 1.0)) + (b ? b[ch] : 0.0)
          end
        end
      end
      rebuild(x, out)
    end

    def cpu_activation(x, kind)
      sigmoid = ->(v) { v >= 0 ? 1.0 / (1.0 + Math.exp(-v)) : Math.exp(v) / (1.0 + Math.exp(v)) }
      fn = case kind
           when :relu then ->(v) { v.positive? ? v : 0.0 }
           when :sigmoid then sigmoid
           when :silu then ->(v) { v * sigmoid.(v) }
           when :gelu_tanh
             ->(v) { 0.5 * v * (1.0 + Math.tanh(0.7978845608028654 * (v + (0.044715 * v * v * v)))) }
           when :gelu_exact then ->(v) { 0.5 * v * (1.0 + Math.erf(v * 0.7071067811865476)) }
           end
      rebuild(x, x.to_a.map(&fn))
    end

    def cpu_transpose(x, rows, cols)
      flat = x.to_a
      values = Array.new(rows * cols)
      cols.times { |c| rows.times { |r| values[(c * rows) + r] = flat[(r * cols) + c] } }
      rebuild(x, values, [cols, rows])
    end
  end
end
