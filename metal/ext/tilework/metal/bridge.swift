import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Foundation

// The C ABI exported to the Ruby C shim (gpu_bridge.c). All policy (type
// conversion, validation, caching, exception hierarchy) lives in Ruby; this
// provides Metal mechanism only. Handles hand ownership to the C side
// (TypedData's free) via Unmanaged.passRetained.

final class BridgeContext {
    static let shared = BridgeContext()
    let device: MTLDevice
    let queue: MTLCommandQueue
    private var pending: [MTLCommandBuffer] = []
    private var batchCommandBuffer: MTLCommandBuffer?
    private var batchEncoder: MTLComputeCommandEncoder?
    private var batchDispatches = 0
    private let batchLimit = max(1, Int(ProcessInfo.processInfo.environment["TILEWORK_METAL_BATCH_SIZE"] ?? "8") ?? 8)
    private var asyncError: String?
    private let lock = NSLock()

    private init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("tilework/metal: no Metal device available")
        }
        self.device = device
        self.queue = queue
    }

    func track(_ cb: MTLCommandBuffer) {
        lock.lock()
        pruneFinishedLocked()
        pending.append(cb)
        lock.unlock()
    }

    // Nothing waits on an asynchronous command buffer at the call that
    // submitted it, so its failure has no caller to return to. Keep the first
    // one and report it at the next synchronization boundary instead of
    // dropping it. Call with `lock` held.
    private func recordFailureLocked(_ cb: MTLCommandBuffer) {
        guard asyncError == nil, let error = cb.error else { return }
        asyncError = error.localizedDescription
    }

    private func pruneFinishedLocked() {
        pending.removeAll { cb in
            if cb.status == .error { recordFailureLocked(cb) }
            return cb.status == .completed || cb.status == .error
        }
    }

    // The first failure of an asynchronous command buffer since the last call.
    func takeAsyncError() -> String? {
        lock.lock()
        defer { lock.unlock() }
        let error = asyncError
        asyncError = nil
        return error
    }

    // Async dispatches in one Tilework.submit/Graph boundary share a command
    // buffer and compute encoder. setBytes copies scalar payloads at encode
    // time, so callers do not need to retain the Ruby strings after dispatch.
    func encoderForBatch() -> MTLComputeCommandEncoder? {
        lock.lock()
        defer { lock.unlock() }
        if let encoder = batchEncoder { return encoder }
        guard let cb = queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else { return nil }
        batchCommandBuffer = cb
        batchEncoder = encoder
        return encoder
    }

    private func commitBatch() {
        lock.lock()
        guard let cb = batchCommandBuffer, let encoder = batchEncoder else {
            lock.unlock()
            return
        }
        batchCommandBuffer = nil
        batchEncoder = nil
        batchDispatches = 0
        encoder.endEncoding()
        cb.commit()
        pending.append(cb)
        lock.unlock()
    }

    // Delegated MPS / MPSGraph operations use their own command buffer.
    // Commit any open custom-kernel batch first: an encoded but uncommitted
    // dispatch is not on the queue yet, so a delegated command buffer
    // committed now would run *before* the kernels that produce its inputs.
    // No host wait is needed, because the queue executes committed command
    // buffers in order. A path that touches `contents()` on the host must
    // call `synchronize()` instead.
    func flushBatch() {
        commitBatch()
    }

    func didEncodeBatchDispatch() {
        lock.lock()
        batchDispatches += 1
        let full = batchDispatches >= batchLimit
        lock.unlock()
        if full { commitBatch() }
    }

    func synchronize() {
        commitBatch()
        lock.lock()
        let toWait = pending
        pending.removeAll()
        lock.unlock()
        for cb in toWait { cb.waitUntilCompleted() }
        lock.lock()
        for cb in toWait { recordFailureLocked(cb) }
        lock.unlock()
    }

    // A shared MTLBuffer that was dirtied through `contents()` keeps its
    // system-memory pages resident after its last strong reference goes away.
    // Marking it empty tells Metal that both the resource and its contents can
    // be discarded.  It must happen after the last async command buffer that
    // could still use it; the command queue's FIFO ordering makes the newest
    // tracked buffer a sufficient fence for all earlier launches.
    func discard(_ buffer: MTLBuffer) {
        // The buffer may still be referenced by an open async encoder. Commit
        // that batch before deciding whether an ordered purge fence is needed.
        commitBatch()
        lock.lock()
        pruneFinishedLocked()
        let hasPendingWork = !pending.isEmpty
        lock.unlock()

        guard hasPendingWork else {
            _ = buffer.setPurgeableState(.empty)
            return
        }

        // Metal requires completion handlers to be registered before commit.
        // This empty command buffer is ordered after every previously
        // submitted command on `queue`, and so is a safe, non-blocking fence.
        guard let fence = queue.makeCommandBuffer() else {
            return
        }
        fence.addCompletedHandler { [buffer] _ in
            autoreleasepool {
                _ = buffer.setPurgeableState(.empty)
            }
        }
        fence.commit()
        track(fence)
    }
}

final class BufferBox {
    let buffer: MTLBuffer
    init(_ buffer: MTLBuffer) { self.buffer = buffer }
}

final class PipelineBox {
    let pso: MTLComputePipelineState
    init(_ pso: MTLComputePipelineState) { self.pso = pso }
}

private func setError(_ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?, _ message: String) {
    out?.pointee = strdup(message)
}

@_cdecl("gpu_bridge_device_name")
public func gpu_bridge_device_name() -> UnsafeMutablePointer<CChar>? {
    return strdup(BridgeContext.shared.device.name)
}

@_cdecl("gpu_bridge_device_available")
public func gpu_bridge_device_available() -> Int32 {
    return MTLCreateSystemDefaultDevice() != nil ? 1 : 0
}

// MARK: - Buffer

@_cdecl("gpu_bridge_buffer_alloc")
public func gpu_bridge_buffer_alloc(_ bytes: Int) -> UnsafeMutableRawPointer? {
    // Ruby invokes this entrypoint directly, without an AppKit/Foundation run
    // loop to drain Objective-C autorelease pools.  `makeBuffer` returns an
    // autoreleased Metal object; BufferBox retains it for the lifetime of the
    // Ruby handle, but the original autorelease must still be drained here.
    // Without this pool every allocation keeps an extra MTLBuffer reference
    // until process exit, so explicit Buffer#free cannot reclaim inference
    // weights.
    return autoreleasepool { () -> UnsafeMutableRawPointer? in
        guard let buf = BridgeContext.shared.device.makeBuffer(length: max(bytes, 1),
                                                               options: .storageModeShared) else {
            return nil
        }
        return Unmanaged.passRetained(BufferBox(buf)).toOpaque()
    }
}

// With storageModeShared, buffer_write / buffer_read are **raw memcpys from
// the host** and never go on the command queue, so calling them while an
// in-flight kernel touches the same buffer is a data race. Waiting is core's
// job (Buffer#write / #to_a etc. call Tilework.synchronize) — waiting here
// every time would drag synchronization into paths made of many small
// transfers, such as offload.
@_cdecl("gpu_bridge_buffer_write")
public func gpu_bridge_buffer_write(_ handle: UnsafeMutableRawPointer,
                                    _ src: UnsafeRawPointer,
                                    _ bytes: Int, _ offset: Int) {
    let box = Unmanaged<BufferBox>.fromOpaque(handle).takeUnretainedValue()
    box.buffer.contents().advanced(by: offset).copyMemory(from: src, byteCount: bytes)
}

@_cdecl("gpu_bridge_buffer_read")
public func gpu_bridge_buffer_read(_ handle: UnsafeMutableRawPointer,
                                   _ dst: UnsafeMutableRawPointer,
                                   _ bytes: Int, _ offset: Int) {
    let box = Unmanaged<BufferBox>.fromOpaque(handle).takeUnretainedValue()
    dst.copyMemory(from: box.buffer.contents().advanced(by: offset), byteCount: bytes)
}

@_cdecl("gpu_bridge_buffer_free")
public func gpu_bridge_buffer_free(_ handle: UnsafeMutableRawPointer) {
    let box = Unmanaged<BufferBox>.fromOpaque(handle).takeUnretainedValue()
    BridgeContext.shared.discard(box.buffer)
    Unmanaged<BufferBox>.fromOpaque(handle).release()
}

// MARK: - Pipeline

@_cdecl("gpu_bridge_pipeline_create")
public func gpu_bridge_pipeline_create(_ source: UnsafePointer<CChar>,
                                       _ fnName: UnsafePointer<CChar>,
                                       _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?)
    -> UnsafeMutableRawPointer? {
    // Like buffers, pipeline creation can be called repeatedly from Ruby with
    // no ambient autorelease pool. PipelineBox owns the compiled state; this
    // pool only releases the temporary library/function objects.
    return autoreleasepool { () -> UnsafeMutableRawPointer? in
        let ctx = BridgeContext.shared
        let src = String(cString: source)
        let name = String(cString: fnName)
        do {
            let library = try ctx.device.makeLibrary(source: src, options: nil)
            guard let fn = library.makeFunction(name: name) else {
                setError(errorOut, "kernel function '\(name)' not found in compiled library")
                return nil
            }
            let pso = try ctx.device.makeComputePipelineState(function: fn)
            return Unmanaged.passRetained(PipelineBox(pso)).toOpaque()
        } catch {
            setError(errorOut, error.localizedDescription)
            return nil
        }
    }
}

@_cdecl("gpu_bridge_pipeline_max_threads")
public func gpu_bridge_pipeline_max_threads(_ handle: UnsafeMutableRawPointer) -> Int {
    let box = Unmanaged<PipelineBox>.fromOpaque(handle).takeUnretainedValue()
    return box.pso.maxTotalThreadsPerThreadgroup
}

@_cdecl("gpu_bridge_pipeline_free")
public func gpu_bridge_pipeline_free(_ handle: UnsafeMutableRawPointer) {
    Unmanaged<PipelineBox>.fromOpaque(handle).release()
}

// MARK: - GEMM (MPSMatrixMultiplication)
//
// The two GEMM entrypoints have *different* buffer-layout contracts (see
// bridge.cpp, the CUDA reference impl, and tensor_core.rb):
//
//   - `gemm` (plain, below): a/b/c are Tilework::Buffer's native row-major
//     layout. MPS is natively row-major too, so this needs no reinterpretation
//     — read A/B/C directly as row-major matrices.
//
//   - `gemm_strided_batched` (encodeGemm, this section): a/b/c are genuinely
//     column-major (lda/ldb/ldc/trans_a/trans_b are the caller's cuBLAS-style
//     column-major description; NN.multi_head_attention builds these itself).
//     Since MPS is row-major, we read each column-major buffer directly as
//     its own row-major transpose instead of copying/transposing — a col-
//     major (r, c) buffer with leading dimension ld is bit-identical to a
//     row-major (c, r) buffer with rowBytes = ld * elemSize. Combined with
//     cuBLAS's own "swap A/B, swap m/n, keep trans flags" identity for
//     row-major libraries, this lets the call be encoded with zero extra
//     data movement:
//
//       leftMatrix  = rowview(B), transpose = trans_b   (represents op(B)^T)
//       rightMatrix = rowview(A), transpose = trans_a   (represents op(A)^T)
//       result      = rowview(C)                        (represents C^T, i.e. C's
//                                                         raw bytes read row-major)
//
// See core/lib/tilework/math/tensor_core.rb for the full ABI contract.

private func mpsDataType(_ code: Int32) -> MPSDataType {
    if code == 1 { return .float16 }
    if code == 2 { return .bFloat16 }
    return .float32
}

private func elemSize(_ code: Int32) -> Int {
    return code == 0 ? 4 : 2
}

// `handle` is always the base BufferBox pointer returned by buffer_alloc —
// never offset (see gpu_bridge_dispatch's comment in gpu_bridge.c: CUDA does
// pointer arithmetic, Metal takes the offset separately). Views pass their
// byte offset in `offset`, applied here via MPSMatrix's own offset param
// rather than by shifting the opaque handle, which is not a raw device
// pointer and cannot be offset like one.
private func mpsMatrix(_ handle: UnsafeMutableRawPointer, offset: Int, rows: Int, columns: Int,
                       rowBytes: Int, matrices: Int, matrixBytes: Int,
                       dataType: MPSDataType) -> MPSMatrix {
    let box = Unmanaged<BufferBox>.fromOpaque(handle).takeUnretainedValue()
    let desc = MPSMatrixDescriptor(rows: rows, columns: columns, matrices: matrices,
                                   rowBytes: rowBytes, matrixBytes: matrixBytes,
                                   dataType: dataType)
    return MPSMatrix(buffer: box.buffer, offset: offset, descriptor: desc)
}

private func encodeGemm(_ a: UnsafeMutableRawPointer, _ aOff: Int,
                        _ b: UnsafeMutableRawPointer, _ bOff: Int,
                        _ c: UnsafeMutableRawPointer, _ cOff: Int,
                        m: Int, k: Int, n: Int,
                        lda: Int, ldb: Int, ldc: Int,
                        strideA: Int, strideB: Int, strideC: Int,
                        transA: Bool, transB: Bool,
                        batch: Int, alpha: Double, inDtype: Int32,
                        errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    let ctx = BridgeContext.shared
    ctx.flushBatch()
    let inType = mpsDataType(inDtype)
    let inElem = elemSize(inDtype)
    let outElem = 4 // ABI contract: the output is always float32

    // Raw dimensions in BLAS convention (before transposes are applied).
    let aRawRows = transA ? k : m
    let aRawCols = transA ? m : k
    let bRawRows = transB ? n : k
    let bRawCols = transB ? k : n

    let leftMat = mpsMatrix(b, offset: bOff, rows: bRawCols, columns: bRawRows,
                            rowBytes: ldb * inElem, matrices: batch,
                            matrixBytes: strideB * inElem, dataType: inType)
    let rightMat = mpsMatrix(a, offset: aOff, rows: aRawCols, columns: aRawRows,
                             rowBytes: lda * inElem, matrices: batch,
                             matrixBytes: strideA * inElem, dataType: inType)
    let resultMat = mpsMatrix(c, offset: cOff, rows: n, columns: m,
                              rowBytes: ldc * outElem, matrices: batch,
                              matrixBytes: strideC * outElem, dataType: .float32)

    let mul = MPSMatrixMultiplication(device: ctx.device,
                                      transposeLeft: transB, transposeRight: transA,
                                      resultRows: n, resultColumns: m, interiorColumns: k,
                                      alpha: alpha, beta: 0.0)

    guard let cb = ctx.queue.makeCommandBuffer() else {
        setError(errorOut, "failed to create command buffer for gemm")
        return -1
    }
    mul.encode(commandBuffer: cb, leftMatrix: leftMat, rightMatrix: rightMat, resultMatrix: resultMat)
    cb.commit()
    ctx.track(cb)
    return 0
}

// MARK: - GEMM: tiled simdgroup_matrix kernel (Phase 2, fp32 only)
//
// MPS (above) covers every shape correctly, but for large aligned fp32
// matmuls a hand-tiled kernel using simdgroup_float8x8 fragments can beat it
// by reusing each global-memory read across a whole threadgroup tile instead
// of paying MPS's more general dispatch overhead. This kernel deliberately
// mirrors the existing WMMA path's own tradeoff (see TensorCore's BLOCK_M/N/K
// comments in tensor_core.rb): it only accepts shapes that tile evenly, and
// anything else falls straight back to MPS — no partial-tile bounds checks,
// no attempt to be a general-purpose kernel.
//
// One simdgroup (32 threads) computes one 32x32 output tile as a 4x4 grid of
// 8x8 fragments, staging a 32x32 slice of A and B through threadgroup memory
// per K-step so the 32x32=1024 global loads are shared across all 16
// fragment multiplies that use them, instead of re-reading per fragment.
private let simdgroupGemmTile = 32
private let simdgroupGemmFrag = 8

private let simdgroupGemmSource = """
#include <metal_stdlib>
using namespace metal;

constant int TILE = \(simdgroupGemmTile);
constant int FRAG = \(simdgroupGemmFrag);
constant int FRAGS = TILE / FRAG;

kernel void gemm_simdgroup_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    int tileRow = int(tgid.y) * TILE;
    int tileCol = int(tgid.x) * TILE;

    simdgroup_float8x8 acc[FRAGS][FRAGS];
    for (int i = 0; i < FRAGS; i++)
        for (int j = 0; j < FRAGS; j++)
            acc[i][j] = simdgroup_float8x8(0);

    // Caller (gpu_bridge_gemm) only takes this path when M/N/K are all
    // multiples of TILE, so every tile below is fully in-bounds.
    for (int k0 = 0; k0 < K; k0 += TILE) {
        for (int e = int(lane); e < TILE * TILE; e += 32) {
            int r = e / TILE, c = e % TILE;
            As[r][c] = A[(tileRow + r) * K + (k0 + c)];
            Bs[r][c] = B[(k0 + r) * N + (tileCol + c)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < TILE; kk += FRAG) {
            simdgroup_float8x8 aFrag[FRAGS], bFrag[FRAGS];
            for (int i = 0; i < FRAGS; i++)
                simdgroup_load(aFrag[i], &As[i * FRAG][kk], TILE);
            for (int j = 0; j < FRAGS; j++)
                simdgroup_load(bFrag[j], &Bs[kk][j * FRAG], TILE);
            for (int i = 0; i < FRAGS; i++)
                for (int j = 0; j < FRAGS; j++)
                    simdgroup_multiply_accumulate(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < FRAGS; i++)
        for (int j = 0; j < FRAGS; j++)
            simdgroup_store(acc[i][j], &C[(tileRow + i * FRAG) * N + (tileCol + j * FRAG)], N);
}
"""

private final class SimdgroupGemmKernel {
    let pso: MTLComputePipelineState
    private init(pso: MTLComputePipelineState) { self.pso = pso }

    // Compiled once and cached. Where compilation itself fails (unsupported GPU
    // family etc.) it is nil and callers quietly fall back to MPS — it is an
    // optimization for supported GPUs only, so there is no reason to error.
    static let shared: SimdgroupGemmKernel? = {
        let ctx = BridgeContext.shared
        guard let library = try? ctx.device.makeLibrary(source: simdgroupGemmSource, options: nil),
              let fn = library.makeFunction(name: "gemm_simdgroup_f32"),
              let pso = try? ctx.device.makeComputePipelineState(function: fn) else {
            return nil
        }
        return SimdgroupGemmKernel(pso: pso)
    }()
}

// Measured (M4-class, fp32, square 256/512/1024) at only 0.2–0.5x of MPS on
// every size (no double buffering or register blocking). The code is kept as
// a fallback for when MPS is unavailable, with the threshold at infinity so it
// is never chosen. Restore a measured value if multi-simdgroup tiling and
// double buffering bring it close to MPS.
private let simdgroupGemmMinElements = Int.max

private func encodeSimdgroupGemm(_ kernel: SimdgroupGemmKernel,
                                 _ a: UnsafeMutableRawPointer, _ b: UnsafeMutableRawPointer,
                                 _ c: UnsafeMutableRawPointer,
                                 m: Int, k: Int, n: Int,
                                 errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    let ctx = BridgeContext.shared
    ctx.flushBatch()
    let boxA = Unmanaged<BufferBox>.fromOpaque(a).takeUnretainedValue()
    let boxB = Unmanaged<BufferBox>.fromOpaque(b).takeUnretainedValue()
    let boxC = Unmanaged<BufferBox>.fromOpaque(c).takeUnretainedValue()

    guard let cb = ctx.queue.makeCommandBuffer(),
          let encoder = cb.makeComputeCommandEncoder() else {
        setError(errorOut, "failed to create command buffer/encoder for simdgroup gemm")
        return -1
    }
    var m32 = Int32(m), k32 = Int32(k), n32 = Int32(n)
    encoder.setComputePipelineState(kernel.pso)
    encoder.setBuffer(boxA.buffer, offset: 0, index: 0)
    encoder.setBuffer(boxB.buffer, offset: 0, index: 1)
    encoder.setBuffer(boxC.buffer, offset: 0, index: 2)
    encoder.setBytes(&m32, length: 4, index: 3)
    encoder.setBytes(&k32, length: 4, index: 4)
    encoder.setBytes(&n32, length: 4, index: 5)
    encoder.dispatchThreadgroups(MTLSize(width: n / simdgroupGemmTile, height: m / simdgroupGemmTile, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    cb.commit()
    ctx.track(cb)
    return 0
}

// Plain GEMM: a/b/c are row-major (see the GEMM section above).
@_cdecl("gpu_bridge_gemm")
public func gpu_bridge_gemm(_ a: UnsafeMutableRawPointer, _ aOff: Int,
                            _ b: UnsafeMutableRawPointer, _ bOff: Int,
                            _ c: UnsafeMutableRawPointer, _ cOff: Int,
                            _ m: Int, _ k: Int, _ n: Int,
                            _ allowTF32: Int32, _ inDtype: Int32,
                            _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    return autoreleasepool { () -> Int32 in
        // allowTF32 is CUDA-specific (TensorFloat-32) and meaningless on Metal, so it is ignored.
        // The hand-tiled kernel does not accept views (offset != 0), so any call
        // with an offset goes to MPS unconditionally.
        let tileAligned = m % simdgroupGemmTile == 0 && n % simdgroupGemmTile == 0 && k % simdgroupGemmTile == 0
        if inDtype == 0, tileAligned, aOff == 0, bOff == 0, cOff == 0,
           m * n * k >= simdgroupGemmMinElements,
           let kernel = SimdgroupGemmKernel.shared {
            return encodeSimdgroupGemm(kernel, a, b, c, m: m, k: k, n: n, errorOut: errorOut)
        }

        let ctx = BridgeContext.shared
        ctx.flushBatch()
        let inType = mpsDataType(inDtype)
        let inElem = elemSize(inDtype)

        let matA = mpsMatrix(a, offset: aOff, rows: m, columns: k, rowBytes: k * inElem,
                             matrices: 1, matrixBytes: m * k * inElem, dataType: inType)
        let matB = mpsMatrix(b, offset: bOff, rows: k, columns: n, rowBytes: n * inElem,
                             matrices: 1, matrixBytes: k * n * inElem, dataType: inType)
        let matC = mpsMatrix(c, offset: cOff, rows: m, columns: n, rowBytes: n * 4,
                             matrices: 1, matrixBytes: m * n * 4, dataType: .float32)

        let mul = MPSMatrixMultiplication(device: ctx.device,
                                          transposeLeft: false, transposeRight: false,
                                          resultRows: m, resultColumns: n, interiorColumns: k,
                                          alpha: 1.0, beta: 0.0)

        guard let cb = ctx.queue.makeCommandBuffer() else {
            setError(errorOut, "failed to create command buffer for gemm")
            return -1
        }
        mul.encode(commandBuffer: cb, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
        cb.commit()
        ctx.track(cb)
        return 0
    }
}

@_cdecl("gpu_bridge_gemm_strided_batched")
public func gpu_bridge_gemm_strided_batched(_ a: UnsafeMutableRawPointer, _ aOff: Int,
                                            _ b: UnsafeMutableRawPointer, _ bOff: Int,
                                            _ c: UnsafeMutableRawPointer, _ cOff: Int,
                                            _ m: Int, _ k: Int, _ n: Int,
                                            _ lda: Int, _ ldb: Int, _ ldc: Int,
                                            _ strideA: Int, _ strideB: Int, _ strideC: Int,
                                            _ transA: Int32, _ transB: Int32,
                                            _ batch: Int, _ alpha: Double,
                                            _ allowTF32: Int32, _ inDtype: Int32,
                                            _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    return autoreleasepool { () -> Int32 in
        return encodeGemm(a, aOff, b, bOff, c, cOff, m: m, k: k, n: n,
                          lda: lda, ldb: ldb, ldc: ldc,
                          strideA: strideA, strideB: strideB, strideC: strideC,
                          transA: transA != 0, transB: transB != 0,
                          batch: batch, alpha: alpha, inDtype: inDtype,
                          errorOut: errorOut)
    }
}

// MARK: - Random (MPSMatrixRandomPhilox, cuRAND analogue)
//
// See random.rb for kind/mean/stddev. UNIFORM = 0, NORMAL = 1.
// cuRAND's convention makes uniform **(0, 1]** (excluding 0, including 1),
// but MPS's uniformDistributionDescriptor is [minimum, maximum) — the open end
// is on the other side, so the result is flipped with 1.0 - x afterwards.

private let RANDOM_KIND_UNIFORM: Int32 = 0

@_cdecl("gpu_bridge_random")
public func gpu_bridge_random(_ handle: UnsafeMutableRawPointer, _ n: Int, _ kind: Int32,
                              _ seed: UInt64, _ mean: Float, _ stddev: Float,
                              _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    return autoreleasepool { () -> Int32 in
        let ctx = BridgeContext.shared
        let box = Unmanaged<BufferBox>.fromOpaque(handle).takeUnretainedValue()

        let desc: MPSMatrixRandomDistributionDescriptor
        if kind == RANDOM_KIND_UNIFORM {
            desc = MPSMatrixRandomDistributionDescriptor.uniformDistributionDescriptor(withMinimum: 0.0, maximum: 1.0)
        } else {
            desc = MPSMatrixRandomDistributionDescriptor.normalDistributionDescriptor(withMean: mean, standardDeviation: stddev)
        }

        let rng = MPSMatrixRandomPhilox(device: ctx.device, destinationDataType: .float32,
                                        seed: Int(bitPattern: UInt(truncatingIfNeeded: seed)),
                                        distributionDescriptor: desc)

        // MPSMatrixRandomPhilox emits four elements at a time. A count that is
        // not a multiple of four **kills the process with an assert** (not an
        // exception, so Ruby cannot catch it). Generate into a scratch buffer
        // rounded up to a multiple of four and copy back the first n elements.
        let padded = (n + 3) & ~3
        let scratch: MTLBuffer?
        if padded == n {
            scratch = nil
        } else {
            guard let buf = ctx.device.makeBuffer(length: padded * 4, options: .storageModeShared) else {
                setError(errorOut, "failed to allocate \(padded * 4) bytes for random")
                return -1
            }
            scratch = buf
        }
        let dest = scratch ?? box.buffer

        let matDesc = MPSMatrixDescriptor(rows: 1, columns: padded, rowBytes: padded * 4, dataType: .float32)
        let matrix = MPSMatrix(buffer: dest, descriptor: matDesc)

        guard let cb = ctx.queue.makeCommandBuffer() else {
            setError(errorOut, "failed to create command buffer for random")
            return -1
        }
        ctx.flushBatch()
        rng.encode(commandBuffer: cb, destinationMatrix: matrix)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            setError(errorOut, error.localizedDescription)
            return -1
        }

        if let scratch = scratch {
            memcpy(box.buffer.contents(), scratch.contents(), n * 4)
        }

        if kind == RANDOM_KIND_UNIFORM {
            let ptr = box.buffer.contents().bindMemory(to: Float.self, capacity: n)
            for i in 0..<n { ptr[i] = 1.0 - ptr[i] }
        }

        return 0
    }
}

// MARK: - FFT (MPSGraph, cuFFT analogue)
//
// See fft.rb for kind/direction. R2C = 0, C2R = 1, C2C = 2, FORWARD = 1,
// INVERSE = -1. Normalization (1/n) is fft.rb's job, so this always uses
// MPSGraphFFTScalingModeNone (no scaling).
//
// Complex numbers are interleaved float32 ([re0, im0, re1, im1, ...]) per the
// core contract. MPSDataTypeComplexFloat32 is a pair of 2x32-bit (re, im), so
// a Buffer's raw bytes can be read as complexFloat32 directly — no copy.
//
// Graph compilation is expensive, so graphs are cached per shape key (kind,
// direction, n, batch), the same idea as cuFFT's plan cache.

private struct FFTPlanKey: Hashable {
    let kind: Int32
    let direction: Int32
    let n: Int
    let batch: Int
}

private final class FFTPlan {
    let graph: MPSGraph
    let inputTensor: MPSGraphTensor
    let outputTensor: MPSGraphTensor
    let inputShape: [NSNumber]
    let outputShape: [NSNumber]
    let inputDataType: MPSDataType
    let outputDataType: MPSDataType

    init(graph: MPSGraph, inputTensor: MPSGraphTensor, outputTensor: MPSGraphTensor,
        inputShape: [NSNumber], outputShape: [NSNumber],
        inputDataType: MPSDataType, outputDataType: MPSDataType) {
        self.graph = graph
        self.inputTensor = inputTensor
        self.outputTensor = outputTensor
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.inputDataType = inputDataType
        self.outputDataType = outputDataType
    }
}

private let FFT_KIND_R2C: Int32 = 0
private let FFT_KIND_C2R: Int32 = 1
private let FFT_DIR_INVERSE: Int32 = -1

private var fftPlanCache: [FFTPlanKey: FFTPlan] = [:]
private let fftPlanLock = NSLock()

private func fftPlan(kind: Int32, direction: Int32, n: Int, batch: Int) -> FFTPlan {
    let key = FFTPlanKey(kind: kind, direction: direction, n: n, batch: batch)

    fftPlanLock.lock()
    defer { fftPlanLock.unlock() }
    if let cached = fftPlanCache[key] { return cached }

    let graph = MPSGraph()
    let desc = MPSGraphFFTDescriptor()
    desc.scalingMode = .none
    desc.inverse = direction == FFT_DIR_INVERSE

    let plan: FFTPlan
    switch kind {
    case FFT_KIND_R2C:
        let inShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: n)]
        let input = graph.placeholder(shape: inShape, dataType: .float32, name: nil)
        let output = graph.realToHermiteanFFT(input, axes: [1], descriptor: desc, name: nil)
        let outShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: n / 2 + 1)]
        plan = FFTPlan(graph: graph, inputTensor: input, outputTensor: output,
                       inputShape: inShape, outputShape: outShape,
                       inputDataType: .float32, outputDataType: .complexFloat32)
    case FFT_KIND_C2R:
        let inSize = n / 2 + 1
        let inShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: inSize)]
        desc.roundToOddHermitean = n % 2 != 0
        let input = graph.placeholder(shape: inShape, dataType: .complexFloat32, name: nil)
        let output = graph.HermiteanToRealFFT(input, axes: [1], descriptor: desc, name: nil)
        let outShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: n)]
        plan = FFTPlan(graph: graph, inputTensor: input, outputTensor: output,
                       inputShape: inShape, outputShape: outShape,
                       inputDataType: .complexFloat32, outputDataType: .float32)
    default: // C2C
        let inShape: [NSNumber] = [NSNumber(value: batch), NSNumber(value: n)]
        let input = graph.placeholder(shape: inShape, dataType: .complexFloat32, name: nil)
        let output = graph.fastFourierTransform(input, axes: [1], descriptor: desc, name: nil)
        plan = FFTPlan(graph: graph, inputTensor: input, outputTensor: output,
                       inputShape: inShape, outputShape: inShape,
                       inputDataType: .complexFloat32, outputDataType: .complexFloat32)
    }

    fftPlanCache[key] = plan
    return plan
}

@_cdecl("gpu_bridge_fft")
public func gpu_bridge_fft(_ inHandle: UnsafeMutableRawPointer, _ outHandle: UnsafeMutableRawPointer,
                          _ n: Int, _ batch: Int, _ kind: Int32, _ direction: Int32,
                          _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    return autoreleasepool { () -> Int32 in
        let ctx = BridgeContext.shared
        let inBox = Unmanaged<BufferBox>.fromOpaque(inHandle).takeUnretainedValue()
        let outBox = Unmanaged<BufferBox>.fromOpaque(outHandle).takeUnretainedValue()

        ctx.flushBatch()
        let plan = fftPlan(kind: kind, direction: direction, n: n, batch: batch)
        let inputData = MPSGraphTensorData(inBox.buffer, shape: plan.inputShape, dataType: plan.inputDataType)
        let outputData = MPSGraphTensorData(outBox.buffer, shape: plan.outputShape, dataType: plan.outputDataType)

        plan.graph.run(with: ctx.queue, feeds: [plan.inputTensor: inputData],
                       targetOperations: nil,
                       resultsDictionary: [plan.outputTensor: outputData])

        return 0
    }
}

// MARK: - Conv2D (MPSGraph, cuDNN analogue)
//
// The layout is NCHW / weights OIHW per the core contract (see conv.rb). dtype
// uses the same codes as gemm (0 = float32, 1 = float16, 2 = bfloat16), so
// mpsDataType(_:) is reused. Unlike gemm, the output dtype follows the input
// (accumulation precision is left to MPSGraph's implementation).
//
// For mode see gpu_bridge_conv2d_backward's calling convention (conv.rb):
// 0 = input gradient (a=weights, b=grad_output), 1 = weight gradient
// (a=input, b=grad_output). All three are (tensorA, tensorB) -> output with
// two inputs and one output, so one ConvPlan struct represents them.

private let CONV_OP_FORWARD: Int32 = 0
private let CONV_OP_BACKWARD_DATA: Int32 = 1
private let CONV_OP_BACKWARD_FILTER: Int32 = 2

private struct ConvPlanKey: Hashable {
    let op: Int32
    let n: Int32, cin: Int32, h: Int32, w: Int32
    let cout: Int32, kh: Int32, kw: Int32
    let padH: Int32, padW: Int32, strideH: Int32, strideW: Int32
    let dilationH: Int32, dilationW: Int32, groups: Int32
    let dtype: Int32
}

private final class ConvPlan {
    let graph: MPSGraph
    let tensorA: MPSGraphTensor
    let tensorB: MPSGraphTensor
    let output: MPSGraphTensor
    let shapeA: [NSNumber]
    let shapeB: [NSNumber]
    let outputShape: [NSNumber]
    let dataType: MPSDataType

    init(graph: MPSGraph, tensorA: MPSGraphTensor, tensorB: MPSGraphTensor, output: MPSGraphTensor,
        shapeA: [NSNumber], shapeB: [NSNumber], outputShape: [NSNumber], dataType: MPSDataType) {
        self.graph = graph
        self.tensorA = tensorA
        self.tensorB = tensorB
        self.output = output
        self.shapeA = shapeA
        self.shapeB = shapeB
        self.outputShape = outputShape
        self.dataType = dataType
    }
}

private var convPlanCache: [ConvPlanKey: ConvPlan] = [:]
private let convPlanLock = NSLock()

// The Ruby shim registers this as an end proc. MPSGraph plans are held by
// process-global dictionaries rather than Ruby TypedData, so release them
// explicitly after queued work has drained. Avoid touching BridgeContext when
// neither cache was ever populated; shutdown must not initialize Metal merely
// because the extension was required.
@_cdecl("gpu_bridge_shutdown")
public func gpu_bridge_shutdown() {
    fftPlanLock.lock()
    let hasFFTPlans = !fftPlanCache.isEmpty
    fftPlanLock.unlock()
    convPlanLock.lock()
    let hasConvPlans = !convPlanCache.isEmpty
    convPlanLock.unlock()
    guard hasFFTPlans || hasConvPlans else { return }

    BridgeContext.shared.synchronize()
    autoreleasepool {
        fftPlanLock.lock()
        fftPlanCache.removeAll()
        fftPlanLock.unlock()
        convPlanLock.lock()
        convPlanCache.removeAll()
        convPlanLock.unlock()
    }
}

private func convPlan(op: Int32, n: Int32, cin: Int32, h: Int32, w: Int32,
                      cout: Int32, kh: Int32, kw: Int32,
                      padH: Int32, padW: Int32, strideH: Int32, strideW: Int32,
                      dilationH: Int32, dilationW: Int32, groups: Int32, dtype: Int32) -> ConvPlan {
    let key = ConvPlanKey(op: op, n: n, cin: cin, h: h, w: w, cout: cout, kh: kh, kw: kw,
                         padH: padH, padW: padW, strideH: strideH, strideW: strideW,
                         dilationH: dilationH, dilationW: dilationW, groups: groups, dtype: dtype)

    convPlanLock.lock()
    defer { convPlanLock.unlock() }
    if let cached = convPlanCache[key] { return cached }

    let graph = MPSGraph()
    let mpsType = mpsDataType(dtype)
    let effectiveKh = (Int(kh) - 1) * Int(dilationH) + 1
    let effectiveKw = (Int(kw) - 1) * Int(dilationW) + 1
    let oh = (Int(h) + 2 * Int(padH) - effectiveKh) / Int(strideH) + 1
    let ow = (Int(w) + 2 * Int(padW) - effectiveKw) / Int(strideW) + 1

    let desc = MPSGraphConvolution2DOpDescriptor(strideInX: Int(strideW), strideInY: Int(strideH),
                                                 dilationRateInX: Int(dilationW), dilationRateInY: Int(dilationH),
                                                 groups: Int(groups),
                                                 paddingLeft: Int(padW), paddingRight: Int(padW),
                                                 paddingTop: Int(padH), paddingBottom: Int(padH),
                                                 paddingStyle: .explicit,
                                                 dataLayout: .NCHW, weightsLayout: .OIHW)!

    let cinPerGroup = cin / groups
    let srcShape: [NSNumber] = [n, cin, h, w].map { NSNumber(value: $0) }
    let weightsShape: [NSNumber] = [cout, cinPerGroup, kh, kw].map { NSNumber(value: $0) }
    let outShape: [NSNumber] = [n, cout, Int32(oh), Int32(ow)].map { NSNumber(value: $0) }

    let plan: ConvPlan
    switch op {
    case CONV_OP_BACKWARD_DATA:
        let weights = graph.placeholder(shape: weightsShape, dataType: mpsType, name: nil)
        let gradOutput = graph.placeholder(shape: outShape, dataType: mpsType, name: nil)
        let dx = graph.convolution2DDataGradient(gradOutput, weights: weights, outputShape: srcShape,
                                                 forwardConvolutionDescriptor: desc, name: nil)
        plan = ConvPlan(graph: graph, tensorA: weights, tensorB: gradOutput, output: dx,
                        shapeA: weightsShape, shapeB: outShape, outputShape: srcShape, dataType: mpsType)
    case CONV_OP_BACKWARD_FILTER:
        let src = graph.placeholder(shape: srcShape, dataType: mpsType, name: nil)
        let gradOutput = graph.placeholder(shape: outShape, dataType: mpsType, name: nil)
        let dw = graph.convolution2DWeightsGradient(gradOutput, source: src, outputShape: weightsShape,
                                                    forwardConvolutionDescriptor: desc, name: nil)
        plan = ConvPlan(graph: graph, tensorA: src, tensorB: gradOutput, output: dw,
                        shapeA: srcShape, shapeB: outShape, outputShape: weightsShape, dataType: mpsType)
    default: // CONV_OP_FORWARD
        let src = graph.placeholder(shape: srcShape, dataType: mpsType, name: nil)
        let weights = graph.placeholder(shape: weightsShape, dataType: mpsType, name: nil)
        let out = graph.convolution2D(src, weights: weights, descriptor: desc, name: nil)
        plan = ConvPlan(graph: graph, tensorA: src, tensorB: weights, output: out,
                        shapeA: srcShape, shapeB: weightsShape, outputShape: outShape, dataType: mpsType)
    }

    convPlanCache[key] = plan
    return plan
}

private func runConvPlan(_ plan: ConvPlan, bufA: MTLBuffer, bufB: MTLBuffer, bufOut: MTLBuffer) {
    let ctx = BridgeContext.shared
    ctx.flushBatch()
    let dataA = MPSGraphTensorData(bufA, shape: plan.shapeA, dataType: plan.dataType)
    let dataB = MPSGraphTensorData(bufB, shape: plan.shapeB, dataType: plan.dataType)
    let dataOut = MPSGraphTensorData(bufOut, shape: plan.outputShape, dataType: plan.dataType)

    plan.graph.run(with: ctx.queue, feeds: [plan.tensorA: dataA, plan.tensorB: dataB],
                   targetOperations: nil,
                   resultsDictionary: [plan.output: dataOut])
}

private func runConvPlan(_ plan: ConvPlan, a: UnsafeMutableRawPointer, b: UnsafeMutableRawPointer,
                         out: UnsafeMutableRawPointer) {
    let boxA = Unmanaged<BufferBox>.fromOpaque(a).takeUnretainedValue()
    let boxB = Unmanaged<BufferBox>.fromOpaque(b).takeUnretainedValue()
    let boxOut = Unmanaged<BufferBox>.fromOpaque(out).takeUnretainedValue()
    runConvPlan(plan, bufA: boxA.buffer, bufB: boxB.buffer, bufOut: boxOut.buffer)
}

// MPSGraph's convolution **silently breaks** when output channels exceed 65536
// (rows past 65536 are never written and the first few rows differ; no error
// or warning).
//
// Up to exactly 65536 it is correct, so cout is split into chunks of that
// size. Buffers are storageModeShared, so slicing is a memcpy (graph.run is
// synchronous).
private let CONV_MAX_COUT: Int32 = 65536

private func convChunkBuffer(_ bytes: Int, _ what: String,
                             _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> MTLBuffer? {
    guard let buf = BridgeContext.shared.device.makeBuffer(length: max(bytes, 1), options: .storageModeShared) else {
        setError(errorOut, "failed to allocate \(bytes) bytes for the chunked conv2d \(what)")
        return nil
    }
    return buf
}

// Slice / write back along cout of [n, cout, plane]. plane is in bytes
// (oh*ow*elemSize), not elements.
private func convCopyChannels(dst: UnsafeMutableRawPointer, dstChannels: Int, dstOffset: Int,
                              src: UnsafeMutableRawPointer, srcChannels: Int, srcOffset: Int,
                              count: Int, batch: Int, planeBytes: Int) {
    for i in 0..<batch {
        let d = dst + (((i * dstChannels) + dstOffset) * planeBytes)
        let s = src + (((i * srcChannels) + srcOffset) * planeBytes)
        memcpy(d, s, count * planeBytes)
    }
}

// dst += src elementwise. Only backward_data has chunk results landing on the
// same dx (splitting by output channel makes the input gradient the sum over
// all chunks), so it must accumulate rather than write.
private func convAccumulate(dst: UnsafeMutableRawPointer, src: UnsafeMutableRawPointer,
                            count: Int, dtype: Int32) {
    if dtype == 1 {
        let d = dst.bindMemory(to: Float16.self, capacity: count)
        let s = src.bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { d[i] += s[i] }
    } else {
        let d = dst.bindMemory(to: Float.self, capacity: count)
        let s = src.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { d[i] += s[i] }
    }
}

// Taken only when cout > CONV_MAX_COUT. Ops differ only in what they slice and
// write back; the split (cout in steps of CONV_MAX_COUT) is shared.
private func runConvChunked(op: Int32, a: UnsafeMutableRawPointer, b: UnsafeMutableRawPointer,
                            out: UnsafeMutableRawPointer,
                            n: Int32, cin: Int32, h: Int32, width: Int32,
                            cout: Int32, kh: Int32, kw: Int32,
                            padH: Int32, padW: Int32, strideH: Int32, strideW: Int32,
                            dilationH: Int32, dilationW: Int32, groups: Int32,
                            dtype: Int32,
                            errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    // Grouped convolution (groups > 1) mixes up channels unless chunk
    // boundaries align with group boundaries. A grouped conv with cout > 65536
    // is unrealistic, so reject it explicitly rather than break silently.
    if groups != 1 {
        setError(errorOut, "conv2d: groups > 1 with cout > \(CONV_MAX_COUT) is not supported " +
                 "(chunked cout path only handles groups == 1)")
        return -1
    }

    let boxA = Unmanaged<BufferBox>.fromOpaque(a).takeUnretainedValue()
    let boxB = Unmanaged<BufferBox>.fromOpaque(b).takeUnretainedValue()
    let boxOut = Unmanaged<BufferBox>.fromOpaque(out).takeUnretainedValue()

    // This path slices its operands with memcpy on the host, so queue order is
    // not enough here: everything that writes them must have finished.
    BridgeContext.shared.synchronize()

    let es = elemSize(dtype)
    let effectiveKh = (Int(kh) - 1) * Int(dilationH) + 1
    let effectiveKw = (Int(kw) - 1) * Int(dilationW) + 1
    let oh = (Int(h) + 2 * Int(padH) - effectiveKh) / Int(strideH) + 1
    let ow = (Int(width) + 2 * Int(padW) - effectiveKw) / Int(strideW) + 1
    let outPlane = oh * ow * es          // bytes per output channel
    let weightPlane = Int(cin) * Int(kh) * Int(kw) * es // bytes per output channel of the weight
    let batch = Int(n)

    // backward_data accumulates, so zero the destination first.
    if op == CONV_OP_BACKWARD_DATA {
        memset(boxOut.buffer.contents(), 0, batch * Int(cin) * Int(h) * Int(width) * es)
    }

    var done: Int32 = 0
    while done < cout {
        let chunk = min(CONV_MAX_COUT, cout - done)
        let plan = convPlan(op: op, n: n, cin: cin, h: h, w: width,
                            cout: chunk, kh: kh, kw: kw, padH: padH, padW: padW,
                            strideH: strideH, strideW: strideW,
                            dilationH: dilationH, dilationW: dilationW, groups: 1, dtype: dtype)

        switch op {
        case CONV_OP_BACKWARD_DATA:
            // a = weight [cout, cin, kh, kw] (contiguous), b = output gradient [n, cout, oh, ow]
            guard let wTmp = convChunkBuffer(Int(chunk) * weightPlane, "weights", errorOut),
                  let gTmp = convChunkBuffer(batch * Int(chunk) * outPlane, "grad_output", errorOut),
                  let dxTmp = convChunkBuffer(batch * Int(cin) * Int(h) * Int(width) * es, "grad_input", errorOut)
            else { return -1 }
            memcpy(wTmp.contents(), boxA.buffer.contents() + Int(done) * weightPlane, Int(chunk) * weightPlane)
            convCopyChannels(dst: gTmp.contents(), dstChannels: Int(chunk), dstOffset: 0,
                             src: boxB.buffer.contents(), srcChannels: Int(cout), srcOffset: Int(done),
                             count: Int(chunk), batch: batch, planeBytes: outPlane)
            runConvPlan(plan, bufA: wTmp, bufB: gTmp, bufOut: dxTmp)
            convAccumulate(dst: boxOut.buffer.contents(), src: dxTmp.contents(),
                           count: batch * Int(cin) * Int(h) * Int(width), dtype: dtype)

        case CONV_OP_BACKWARD_FILTER:
            // a = input (all of it), b = output gradient, output = weight gradient [cout, cin, kh, kw]
            guard let gTmp = convChunkBuffer(batch * Int(chunk) * outPlane, "grad_output", errorOut),
                  let dwTmp = convChunkBuffer(Int(chunk) * weightPlane, "grad_weight", errorOut)
            else { return -1 }
            convCopyChannels(dst: gTmp.contents(), dstChannels: Int(chunk), dstOffset: 0,
                             src: boxB.buffer.contents(), srcChannels: Int(cout), srcOffset: Int(done),
                             count: Int(chunk), batch: batch, planeBytes: outPlane)
            runConvPlan(plan, bufA: boxA.buffer, bufB: gTmp, bufOut: dwTmp)
            memcpy(boxOut.buffer.contents() + Int(done) * weightPlane, dwTmp.contents(), Int(chunk) * weightPlane)

        default: // CONV_OP_FORWARD
            // a = input (all of it), b = weight [cout, cin, kh, kw] (contiguous)
            guard let wTmp = convChunkBuffer(Int(chunk) * weightPlane, "weights", errorOut),
                  let yTmp = convChunkBuffer(batch * Int(chunk) * outPlane, "output", errorOut)
            else { return -1 }
            memcpy(wTmp.contents(), boxB.buffer.contents() + Int(done) * weightPlane, Int(chunk) * weightPlane)
            runConvPlan(plan, bufA: boxA.buffer, bufB: wTmp, bufOut: yTmp)
            convCopyChannels(dst: boxOut.buffer.contents(), dstChannels: Int(cout), dstOffset: Int(done),
                             src: yTmp.contents(), srcChannels: Int(chunk), srcOffset: 0,
                             count: Int(chunk), batch: batch, planeBytes: outPlane)
        }

        done += chunk
    }
    return 0
}

@_cdecl("gpu_bridge_conv2d")
public func gpu_bridge_conv2d(_ x: UnsafeMutableRawPointer, _ w: UnsafeMutableRawPointer, _ y: UnsafeMutableRawPointer,
                              _ n: Int32, _ cin: Int32, _ h: Int32, _ width: Int32,
                              _ cout: Int32, _ kh: Int32, _ kw: Int32,
                              _ padH: Int32, _ padW: Int32, _ strideH: Int32, _ strideW: Int32,
                              _ dilationH: Int32, _ dilationW: Int32, _ groups: Int32,
                              _ dtype: Int32,
                              _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    return autoreleasepool { () -> Int32 in
        if cout > CONV_MAX_COUT {
            return runConvChunked(op: CONV_OP_FORWARD, a: x, b: w, out: y,
                                  n: n, cin: cin, h: h, width: width, cout: cout, kh: kh, kw: kw,
                                  padH: padH, padW: padW, strideH: strideH, strideW: strideW,
                                  dilationH: dilationH, dilationW: dilationW, groups: groups,
                                  dtype: dtype, errorOut: errorOut)
        }
        let plan = convPlan(op: CONV_OP_FORWARD, n: n, cin: cin, h: h, w: width,
                           cout: cout, kh: kh, kw: kw, padH: padH, padW: padW,
                           strideH: strideH, strideW: strideW,
                           dilationH: dilationH, dilationW: dilationW, groups: groups, dtype: dtype)
        runConvPlan(plan, a: x, b: w, out: y)
        return 0
    }
}

@_cdecl("gpu_bridge_conv2d_backward")
public func gpu_bridge_conv2d_backward(_ a: UnsafeMutableRawPointer, _ b: UnsafeMutableRawPointer,
                                       _ out: UnsafeMutableRawPointer,
                                       _ n: Int32, _ cin: Int32, _ h: Int32, _ width: Int32,
                                       _ cout: Int32, _ kh: Int32, _ kw: Int32,
                                       _ padH: Int32, _ padW: Int32, _ strideH: Int32, _ strideW: Int32,
                                       _ dilationH: Int32, _ dilationW: Int32, _ groups: Int32,
                                       _ dtype: Int32, _ mode: Int32,
                                       _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    return autoreleasepool { () -> Int32 in
        let op = mode == 0 ? CONV_OP_BACKWARD_DATA : CONV_OP_BACKWARD_FILTER
        if cout > CONV_MAX_COUT {
            return runConvChunked(op: op, a: a, b: b, out: out,
                                  n: n, cin: cin, h: h, width: width, cout: cout, kh: kh, kw: kw,
                                  padH: padH, padW: padW, strideH: strideH, strideW: strideW,
                                  dilationH: dilationH, dilationW: dilationW, groups: groups,
                                  dtype: dtype, errorOut: errorOut)
        }
        let plan = convPlan(op: op, n: n, cin: cin, h: h, w: width,
                           cout: cout, kh: kh, kw: kw, padH: padH, padW: padW,
                           strideH: strideH, strideW: strideW,
                           dilationH: dilationH, dilationW: dilationW, groups: groups, dtype: dtype)
        runConvPlan(plan, a: a, b: b, out: out)
        return 0
    }
}

// MARK: - Dispatch

// args is an array of kinds: kind=0 is an MTLBuffer handle, kind=1 raw setBytes bytes.
@_cdecl("gpu_bridge_dispatch")
public func gpu_bridge_dispatch(_ pipelineHandle: UnsafeMutableRawPointer,
                                _ argKinds: UnsafePointer<Int32>?,
                                _ argPtrs: UnsafePointer<UnsafeMutableRawPointer?>?,
                                _ argLens: UnsafePointer<Int>?,
                                _ argCount: Int32,
                                _ gx: Int, _ gy: Int, _ gz: Int,
                                _ bx: Int, _ by: Int, _ bz: Int,
                                _ wait: Int32,
                                _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    return autoreleasepool { () -> Int32 in
        let ctx = BridgeContext.shared
        let pso = Unmanaged<PipelineBox>.fromOpaque(pipelineHandle).takeUnretainedValue().pso

        let cb: MTLCommandBuffer?
        let encoder: MTLComputeCommandEncoder
        if wait == 0 {
            cb = nil
            guard let shared = ctx.encoderForBatch() else {
                setError(errorOut, "failed to create batched command buffer/encoder")
                return -1
            }
            encoder = shared
        } else {
            // Preserve queue order if an async batch is still open.
            ctx.synchronize()
            guard let commandBuffer = ctx.queue.makeCommandBuffer(),
                  let synchronous = commandBuffer.makeComputeCommandEncoder() else {
                setError(errorOut, "failed to create command buffer/encoder")
                return -1
            }
            cb = commandBuffer
            encoder = synchronous
        }
        encoder.setComputePipelineState(pso)

        for i in 0..<Int(argCount) {
            guard let ptr = argPtrs?[i] else {
                if wait != 0 { encoder.endEncoding() }
                setError(errorOut, "argument \(i) is null")
                return -1
            }
            if argKinds?[i] == 0 {
                let box = Unmanaged<BufferBox>.fromOpaque(ptr).takeUnretainedValue()
                // For buffer arguments argLens carries the view's byte offset
                // (see gpu_bridge.c); 0 for an ordinary buffer.
                encoder.setBuffer(box.buffer, offset: argLens?[i] ?? 0, index: i)
            } else {
                encoder.setBytes(ptr, length: argLens?[i] ?? 0, index: i)
            }
        }

        encoder.dispatchThreadgroups(MTLSize(width: gx, height: gy, depth: gz),
                                     threadsPerThreadgroup: MTLSize(width: bx, height: by, depth: bz))
        guard let cb else {
            ctx.didEncodeBatchDispatch()
            return 0
        }
        encoder.endEncoding()
        cb.commit()
        if wait != 0 {
            cb.waitUntilCompleted()
            if let error = cb.error {
                setError(errorOut, error.localizedDescription)
                return -1
            }
        } else {
            ctx.track(cb)
        }
        return 0
    }
}

@_cdecl("gpu_bridge_synchronize")
public func gpu_bridge_synchronize() {
    BridgeContext.shared.synchronize()
}

// Optional ABI: synchronize, and report the first asynchronous command buffer
// that failed since the last call. GEMM and wait:false dispatches return
// before they run, so this boundary is the only place their failure can
// surface.
@_cdecl("gpu_bridge_synchronize_checked")
public func gpu_bridge_synchronize_checked(
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Int32 {
    let ctx = BridgeContext.shared
    ctx.synchronize()
    guard let error = ctx.takeAsyncError() else { return 0 }
    setError(errorOut, error)
    return -1
}
