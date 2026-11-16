# frozen_string_literal: true

module Tilework
  # Immutable feature and device-property values for a backend.
  # The value object returned by Tilework.capabilities. FIELDS is the single
  # source of the key set, types, defaults and meanings; upper layers (ops.rb /
  # dsl.rb / buffer.rb / nn.rb / tensor_core.rb) pick paths from it.
  #
  # Plain Hashes were dropped because of typos: caps[:tensro_core] on a Hash
  # quietly reads nil ("unsupported"), but raises KeyError here. When adding a
  # backend, FIELDS is what tells you what to declare.
  #
  #   caps = Tilework.capabilities
  #   caps[:warp_size]   # => 32   (KeyError for an unknown key)
  #   caps.warp_size     # => 32   (preferred form)
  class Capabilities
    # type is a small validation vocabulary (:boolean / :boolean_or_nil /
    # :integer_or_nil / :name_or_nil / :array / :array_or_nil / :any); :any
    # means "not validated".
    Field = Struct.new(:name, :type, :default, :doc)

    FIELDS = [
      Field.new(:tensor_core, :boolean, false,
                "行列積専用ユニット (Tensor Core / SIMD group matrix) を使えるか"),
      Field.new(:tf32, :boolean, false,
                "FP32 バッファをそのまま Tensor Core に入力できるか (Ampere 以降)"),
      Field.new(:fp16, :boolean, false,
                "FP16 の Tensor Core 経路があるか"),
      Field.new(:dialect, :boolean, true,
                "CUDA 風方言のカーネル文字列 (compile の raw: false) を受け付けるか"),
      Field.new(:fp16_storage, :boolean, true,
                "生成カーネルが float16 を格納型として扱えるか (Ops#require_computable!)"),
      Field.new(:bf16_storage, :boolean, false,
                "生成カーネルが bfloat16 を格納型として扱えるか"),
      Field.new(:gemm_dtypes, :array, [].freeze,
                "委譲 GEMM が受け付ける入力 dtype"),
      Field.new(:conv_dtypes, :array, [].freeze,
                "委譲 convolution が受け付ける dtype"),
      Field.new(:dtypes, :array, %i[float32 int32].freeze,
                "Buffer.new で確保できる dtype (Buffer#check_dtype_supported!)"),
      Field.new(:offload, :boolean, true,
                "VRAM↔ディスクの退避経路があるか。素朴な to_a -> File.write 経路は常に使える"),
      Field.new(:offload_async, :boolean, false,
                "非同期転送 ABI (transfer_begin/poll/end) があるか"),
      Field.new(:offload_transport, :name_or_nil, nil,
                "退避に使う転送方式 (:posix / :direct など)"),
      Field.new(:host_backing, :boolean, false,
                "不揮発ファイルではない pageable CPU RAM への明示退避があるか"),
      Field.new(:host_backing_async, :boolean, false,
                "pageable host backing との転送を非同期に完了できるか"),
      Field.new(:gemm_library, :name_or_nil, nil,
                "GEMM を委譲するライブラリ名 (cuBLAS 等)。nil なら自前カーネル"),
      Field.new(:gemm_batched, :boolean, false,
                "同じ形の GEMM をまとめて投げられるか (multi-head attention が速くなる)"),
      Field.new(:rng, :name_or_nil, nil,
                "乱数生成を委譲するライブラリ (cuRAND 等)。nil なら Ruby 側で作る"),
      Field.new(:fft, :name_or_nil, nil,
                "FFT を委譲するライブラリ (cuFFT 等)。nil なら FFT は使えない"),
      Field.new(:conv, :name_or_nil, nil,
                "畳み込みを委譲するライブラリ (cuDNN 等)。nil なら畳み込みは使えない"),
      Field.new(:conv_backward, :boolean, false,
                "畳み込みの逆伝播があるか (学習に必須)"),
      Field.new(:warp_size, :integer_or_nil, nil,
                "ロックステップで動く最小スレッド束の幅 (CUDA の warpSize)"),
      Field.new(:unified_memory, :boolean_or_nil, nil,
                "ホストとデバイスでメモリを共有しているか"),
      Field.new(:memory_budget_bytes, :integer_or_nil, nil,
                "確保に使えるデバイスメモリのおおよその上限"),
      Field.new(:compute_capability, :array_or_nil, nil,
                "デバイスの世代 ([major, minor])。持たないバックエンドは nil"),
    ].freeze

    NAMES = FIELDS.map(&:name).freeze
    BY_NAME = FIELDS.to_h { |f| [f.name, f] }.freeze
    DEFAULTS = FIELDS.to_h { |f| [f.name, f.default] }.freeze

    class << self
      # Builds validated capability values.
      #
      # @api private
      # @param overrides [Hash{Symbol => Object}] values overriding documented defaults
      # @return [Capabilities]
      # @raise [ArgumentError] for unknown keys or incompatible values
      #
      # The only entry point for a backend's declarations; unknown keys and type
      # mismatches are rejected here so nothing silently reads as nil.
      def build(**overrides)
        validate!(overrides)
        new(DEFAULTS.merge(overrides))
      end

      # Drops keys outside the contract. Inputs that may carry keys we did not
      # define, such as the raw hash from a bridge's optional
      # device_capabilities, pass through this before build.
      def slice_known(hash)
        hash.select { |k, _| BY_NAME.key?(k) }
      end

      private

      def validate!(overrides)
        overrides.each do |key, value|
          field = BY_NAME[key] or
            raise ArgumentError, "unknown capability #{key.inspect}; known keys are #{NAMES.join(', ')}"
          next if valid_type?(field.type, value)

          raise ArgumentError,
                "capability #{key.inspect} expects #{field.type} but got #{value.inspect}"
        end
      end

      def valid_type?(type, value)
        case type
        when :boolean         then [true, false].include?(value)
        when :boolean_or_nil  then value.nil? || [true, false].include?(value)
        when :integer_or_nil  then value.nil? || value.is_a?(Integer)
        # A library name or transport identifier. core returns Symbols like
        # :cublas, but a bridge declaring its own may use a String.
        when :name_or_nil     then value.nil? || value.is_a?(Symbol) || value.is_a?(String)
        when :array           then value.is_a?(Array)
        when :array_or_nil    then value.nil? || value.is_a?(Array)
        else true # :any
        end
      end
    end

    FIELDS.each do |field|
      define_method(field.name) { @values[field.name] }
    end

    def initialize(values)
      @values = values.freeze
      freeze
    end

    # Looks up a capability by name.
    #
    # @param key [Symbol, String]
    # @return [Object]
    # @raise [KeyError] for an unknown capability
    def [](key)
      @values.fetch(key) do
        raise KeyError, "unknown capability #{key.inspect}; known keys are #{NAMES.join(', ')}"
      end
    end

    def fetch(key, *default, &block)
      @values.fetch(key, *default, &block)
    end

    def key?(key)
      BY_NAME.key?(key)
    end

    # Returns a new instance with validated overrides.
    #
    # @param overrides [Array<Hash{Symbol => Object}>]
    # @return [Capabilities]
    #
    # Takes several Hashes like Hash#merge, so callers keep the call style from
    # when this was a Hash.
    def merge(*overrides)
      self.class.build(**overrides.reduce(@values) { |acc, h| acc.merge(h) })
    end

    def to_h
      @values.dup
    end

    def ==(other)
      other.is_a?(Capabilities) && other.to_h == @values
    end
    alias eql? ==

    def hash
      @values.hash
    end

    def inspect
      "#<Tilework::Capabilities #{@values.map { |k, v| "#{k}=#{v.inspect}" }.join(' ')}>"
    end
  end
end
