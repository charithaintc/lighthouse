# Attention Kernel Transformation Analysis

This document describes the step-by-step transformation of an attention kernel implementation through MLIR compiler passes. The attention mechanism computes: `attention(Q, K, V) = softmax(Q × K^T) × V`

---

## Stage 1: Initial Attention Computation

```mlir
func.func @attention(%arg0: memref<128x64xf16>, %arg1: memref<4096x64xf16>,
                     %arg2: memref<4096x64xf16>, %arg3: memref<128x64xf16>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<128x64xf16> to tensor<128x64xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>

  // Allocate output tensors
  %3 = tensor.empty() : tensor<64x4096xf16>
  %4 = tensor.empty() : tensor<128x4096xf16>
  %5 = tensor.empty() : tensor<128x4096xf16>
  %6 = tensor.empty() : tensor<128x64xf16>

  // Transpose K: 4096x64 -> 64x4096
  %transposed = linalg.transpose ins(%1 : tensor<4096x64xf16>)
                                 outs(%3 : tensor<64x4096xf16>)
                                 permutation = [1, 0]

  // First matmul: Q × K^T
  %cst = arith.constant 0.000000e+00 : f16
  %7 = linalg.fill ins(%cst : f16) outs(%4 : tensor<128x4096xf16>) -> tensor<128x4096xf16>
  %8 = linalg.matmul ins(%0, %transposed : tensor<128x64xf16>, tensor<64x4096xf16>)
                     outs(%7 : tensor<128x4096xf16>) -> tensor<128x4096xf16>

  // Softmax operation (high-level)
  %9 = linalg.softmax dimension(1) ins(%8 : tensor<128x4096xf16>)
                                    outs(%5 : tensor<128x4096xf16>) -> tensor<128x4096xf16>

  // Second matmul: softmax_out × V
  %10 = linalg.fill ins(%cst : f16) outs(%6 : tensor<128x64xf16>) -> tensor<128x64xf16>
  %11 = linalg.matmul ins(%9, %2 : tensor<128x4096xf16>, tensor<4096x64xf16>)
                      outs(%10 : tensor<128x64xf16>) -> tensor<128x64xf16>

  bufferization.materialize_in_destination %11 in restrict writable %arg3 :
    (tensor<128x64xf16>, memref<128x64xf16>) -> ()
  return
}
```

---

## Stage 2: After Softmax Decomposition

Softmax decomposition into constituent operations

This is the numerically stable softmax: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

```mlir
func.func @attention(%arg0: memref<128x64xf16>, %arg1: memref<4096x64xf16>,
                     %arg2: memref<4096x64xf16>, %arg3: memref<128x64xf16>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<128x64xf16> to tensor<128x64xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %3 = tensor.empty() : tensor<64x4096xf16>
  %4 = tensor.empty() : tensor<128x4096xf16>
  %5 = tensor.empty() : tensor<128x4096xf16>
  %6 = tensor.empty() : tensor<128x64xf16>

  %transposed = linalg.transpose ins(%1 : tensor<4096x64xf16>)
                                 outs(%3 : tensor<64x4096xf16>)
                                 permutation = [1, 0]

  // First matmul (unchanged)
  %cst = arith.constant 0.000000e+00 : f16
  %7 = linalg.fill ins(%cst : f16) outs(%4 : tensor<128x4096xf16>) -> tensor<128x4096xf16>
  %8 = linalg.matmul ins(%0, %transposed : tensor<128x64xf16>, tensor<64x4096xf16>)
                     outs(%7 : tensor<128x4096xf16>) -> tensor<128x4096xf16>

  // === SOFTMAX DECOMPOSITION BEGINS ===

  // Step 1: Find max along dimension 1 (reduction over 4096 elements)
  %9 = tensor.empty() : tensor<128xf16>
  %cst_0 = arith.constant 0xFE00 : f16  // -inf in f16
  %10 = linalg.fill ins(%cst_0 : f16) outs(%9 : tensor<128xf16>) -> tensor<128xf16>
  %11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%8 : tensor<128x4096xf16>) outs(%10 : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %18 = arith.maxnumf %in, %out : f16
    linalg.yield %18 : f16
  } -> tensor<128xf16>

  // Step 2: Subtract max and compute exp
  %12 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%8, %11 : tensor<128x4096xf16>, tensor<128xf16>)
    outs(%5 : tensor<128x4096xf16>) {
  ^bb0(%in: f16, %in_2: f16, %out: f16):
    %18 = arith.subf %in, %in_2 : f16
    %19 = math.exp %18 : f16
    linalg.yield %19 : f16
  } -> tensor<128x4096xf16>

  // Step 3: Sum exponentials (reduction)
  %cst_1 = arith.constant 0.000000e+00 : f16
  %13 = linalg.fill ins(%cst_1 : f16) outs(%9 : tensor<128xf16>) -> tensor<128xf16>
  %14 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%12 : tensor<128x4096xf16>) outs(%13 : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %18 = arith.addf %in, %out : f16
    linalg.yield %18 : f16
  } -> tensor<128xf16>

  // Step 4: Divide by sum for normalization
  %15 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%12, %14 : tensor<128x4096xf16>, tensor<128xf16>)
    outs(%5 : tensor<128x4096xf16>) {
  ^bb0(%in: f16, %in_2: f16, %out: f16):
    %18 = arith.divf %in, %in_2 : f16
    linalg.yield %18 : f16
  } -> tensor<128x4096xf16>

  // === SOFTMAX DECOMPOSITION ENDS ===

  // Second matmul (unchanged)
  %16 = linalg.fill ins(%cst : f16) outs(%6 : tensor<128x64xf16>) -> tensor<128x64xf16>
  %17 = linalg.matmul ins(%15, %2 : tensor<128x4096xf16>, tensor<4096x64xf16>)
                      outs(%16 : tensor<128x64xf16>) -> tensor<128x64xf16>

  bufferization.materialize_in_destination %17 in restrict writable %arg3 :
    (tensor<128x64xf16>, memref<128x64xf16>) -> ()
  return
}
```

---

## Stage 3: After Matmul and Transpose Generalization

Convert named operations (`linalg.matmul`, `linalg.transpose`) to `linalg.generic`

```mlir
func.func @attention(%arg0: memref<128x64xf16>, %arg1: memref<4096x64xf16>,
                     %arg2: memref<4096x64xf16>, %arg3: memref<128x64xf16>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<128x64xf16> to tensor<128x64xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %3 = tensor.empty() : tensor<64x4096xf16>
  %4 = tensor.empty() : tensor<128x4096xf16>
  %5 = tensor.empty() : tensor<128x4096xf16>
  %6 = tensor.empty() : tensor<128x64xf16>

  // Transpose converted to generic form
  %7 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,    // Input: swapped (4096×64 accessed as d1,d0)
                     affine_map<(d0, d1) -> (d0, d1)>],   // Output: 64×4096 (normal d0,d1)
    iterator_types = ["parallel", "parallel"]
  } ins(%1 : tensor<4096x64xf16>) outs(%3 : tensor<64x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<64x4096xf16>

  // First matmul converted to generic form
  %cst = arith.constant 0.000000e+00 : f16
  %8 = linalg.fill ins(%cst : f16) outs(%4 : tensor<128x4096xf16>) -> tensor<128x4096xf16>
  %9 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,    // Q: 128×64
                     affine_map<(d0, d1, d2) -> (d2, d1)>,    // K^T: 64×4096
                     affine_map<(d0, d1, d2) -> (d0, d1)>],   // Out: 128×4096
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%0, %7 : tensor<128x64xf16>, tensor<64x4096xf16>)
    outs(%8 : tensor<128x4096xf16>) {
  ^bb0(%in: f16, %in_2: f16, %out: f16):
    %19 = arith.mulf %in, %in_2 : f16
    %20 = arith.addf %out, %19 : f16
    linalg.yield %20 : f16
  } -> tensor<128x4096xf16>

  // Softmax decomposition (unchanged)
  // ....

  // Second matmul converted to generic form
  %17 = linalg.fill ins(%cst : f16) outs(%6 : tensor<128x64xf16>) -> tensor<128x64xf16>
  %18 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,    // Softmax: 128×4096
                     affine_map<(d0, d1, d2) -> (d2, d1)>,    // V: 4096×64
                     affine_map<(d0, d1, d2) -> (d0, d1)>],   // Out: 128×64
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%16, %2 : tensor<128x4096xf16>, tensor<4096x64xf16>)
    outs(%17 : tensor<128x64xf16>) {
  ^bb0(%in: f16, %in_2: f16, %out: f16):
    %19 = arith.mulf %in, %in_2 : f16
    %20 = arith.addf %out, %19 : f16
    linalg.yield %20 : f16
  } -> tensor<128x64xf16>

  bufferization.materialize_in_destination %18 in restrict writable %arg3 :
    (tensor<128x64xf16>, memref<128x64xf16>) -> ()
  return
}
```

---

## Stage 4: After Elementwise Fusion (Final)

Fusion of generic ops.

### Notes
This is the most optimized form. Multiple transformations occur:
1. **Transpose + Matmul fusion**: The transpose operation is fused into the first matmul, directly reading K with transposed indexing
2. **Softmax reduction fusion**: The exp computation and sum reduction are fused into a single operation
3. **Final matmul fusion**: The normalization division is fused with the final matmul, computing `(exp(x - max) / sum) × V` in one pass

```mlir
func.func @attention(%arg0: memref<128x64xf16>, %arg1: memref<4096x64xf16>,
                     %arg2: memref<4096x64xf16>, %arg3: memref<128x64xf16>) {
  %cst = arith.constant 0xFE00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16

  %0 = bufferization.to_tensor %arg0 restrict writable : memref<128x64xf16> to tensor<128x64xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %3 = tensor.empty() : tensor<128x4096xf16>
  %4 = tensor.empty() : tensor<128x64xf16>

  // === FUSED: Transpose + First Matmul ===
  // Previously: separate transpose (4096×64 → 64×4096) + matmul
  // Now: matmul directly reads K with transposed indexing
  %5 = linalg.fill ins(%cst_0 : f16) outs(%3 : tensor<128x4096xf16>) -> tensor<128x4096xf16>
  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,    // Q: 128×64
                     affine_map<(d0, d1, d2) -> (d1, d2)>,    // K: 4096×64 (transpose on-the-fly!)
                     affine_map<(d0, d1, d2) -> (d0, d1)>],   // Out: 128×4096
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%0, %1 : tensor<128x64xf16>, tensor<4096x64xf16>)
    outs(%5 : tensor<128x4096xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %14 = arith.mulf %in, %in_1 : f16
    %15 = arith.addf %out, %14 : f16
    linalg.yield %15 : f16
  } -> tensor<128x4096xf16>

  // Max reduction (unchanged)
  %7 = tensor.empty() : tensor<128xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<128xf16>) -> tensor<128xf16>
  %9 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%6 : tensor<128x4096xf16>) outs(%8 : tensor<128xf16>) {
  ^bb0(%in: f16, %out: f16):
    %14 = arith.maxnumf %in, %out : f16
    linalg.yield %14 : f16
  } -> tensor<128xf16>

  // === FUSED: exp and sum reduction ===
  // Previously: separate exp operation + sum reduction
  // Now: compute exp(x - max) and accumulate sum in one pass
  %10 = linalg.fill ins(%cst_0 : f16) outs(%7 : tensor<128xf16>) -> tensor<128xf16>
  %11 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,   // Input: QK^T
                     affine_map<(d0, d1) -> (d0)>,       // Max values
                     affine_map<(d0, d1) -> (d0)>],      // Sum accumulator (output)
    iterator_types = ["parallel", "reduction"]
  } ins(%6, %9 : tensor<128x4096xf16>, tensor<128xf16>)
    outs(%10 : tensor<128xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %14 = arith.subf %in, %in_1 : f16      // x - max
    %15 = math.exp %14 : f16                // exp(x - max)
    %16 = arith.addf %15, %out : f16        // accumulate sum
    linalg.yield %16 : f16
  } -> tensor<128xf16>

  // === FUSED: softmax normalization + second matmul ===
  // Previously: separate divide operation + matmul
  // Now: compute (exp(x - max) / sum) * V in one fused kernel
  %12 = linalg.fill ins(%cst_0 : f16) outs(%4 : tensor<128x64xf16>) -> tensor<128x64xf16>
  %13 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,   // QK^T scores
                     affine_map<(d0, d1, d2) -> (d0)>,       // Max values
                     affine_map<(d0, d1, d2) -> (d0)>,       // Sum values
                     affine_map<(d0, d1, d2) -> (d2, d1)>,   // V matrix
                     affine_map<(d0, d1, d2) -> (d0, d1)>],  // Output
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%6, %9, %11, %2 : tensor<128x4096xf16>, tensor<128xf16>,
                          tensor<128xf16>, tensor<4096x64xf16>)
    outs(%12 : tensor<128x64xf16>) {
  ^bb0(%in: f16, %in_1: f16, %in_2: f16, %in_3: f16, %out: f16):
    %14 = arith.subf %in, %in_1 : f16       // x - max
    %15 = math.exp %14 : f16                 // exp(x - max)
    %16 = arith.divf %15, %in_2 : f16       // exp(x - max) / sum  [softmax]
    %17 = arith.mulf %16, %in_3 : f16       // softmax * V
    %18 = arith.addf %out, %17 : f16        // accumulate result
    linalg.yield %18 : f16
  } -> tensor<128x64xf16>

  bufferization.materialize_in_destination %13 in restrict writable %arg3 :
    (tensor<128x64xf16>, memref<128x64xf16>) -> ()
  return
}
```

---

## Stage 5: Online Softmax Optimization (Max-Sum Fusion)

The previous stage still computes max and sum in two separate passes over the attention scores. We can fuse these into a single pass using the **online softmax algorithm**.

```mlir
func.func @attention_max_sum_fused(%arg0: memref<128x64xf16>, %arg1: memref<4096x64xf16>,
                                    %arg2: memref<4096x64xf16>, %arg3: memref<128x64xf16>) {
  %cst = arith.constant 0xFE00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<128x64xf16> to tensor<128x64xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %3 = tensor.empty() : tensor<128x4096xf16>
  %4 = tensor.empty() : tensor<128x64xf16>
  %5 = linalg.fill ins(%cst_0 : f16) outs(%3 : tensor<128x4096xf16>) -> tensor<128x4096xf16>

  // First generic: Q @ K^T
  %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%0, %1 : tensor<128x64xf16>, tensor<4096x64xf16>) outs(%5 : tensor<128x4096xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %14 = arith.mulf %in, %in_1 : f16
    %15 = arith.addf %out, %14 : f16
    linalg.yield %15 : f16
  } -> tensor<128x4096xf16>

  // === FUSED: Online Softmax (max + sum in single pass) ===
  // Takes: QK^T scores
  // Returns: (max, sum) computed simultaneously
  %7 = tensor.empty() : tensor<128xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<128xf16>) -> tensor<128xf16>
  %9 = linalg.fill ins(%cst_0 : f16) outs(%7 : tensor<128xf16>) -> tensor<128xf16>
  %10:2 = linalg.generic {indexing_maps = [#map3, #map4, #map4], iterator_types = ["parallel", "reduction"]}
    ins(%6 : tensor<128x4096xf16>) outs(%8, %9 : tensor<128xf16>, tensor<128xf16>) {
  ^bb0(%in: f16, %out_max: f16, %out_sum: f16):
    %14 = arith.maxnumf %in, %out_max : f16           // new_max = max(in, old_max)
    %15 = arith.subf %out_max, %14 : f16              // old_max - new_max
    %16 = math.exp %15 : f16                          // correction_factor = exp(old_max - new_max)
    %17 = arith.mulf %out_sum, %16 : f16              // rescaled_sum = old_sum * correction_factor
    %18 = arith.subf %in, %14 : f16                   // in - new_max
    %19 = math.exp %18 : f16                          // exp(in - new_max)
    %20 = arith.addf %17, %19 : f16                   // new_sum = rescaled_sum + exp(in - new_max)
    linalg.yield %14, %20 : f16, f16
  } -> (tensor<128xf16>, tensor<128xf16>)

  // Final generic: compute attention output using max and sum
  %11 = linalg.fill ins(%cst_0 : f16) outs(%4 : tensor<128x64xf16>) -> tensor<128x64xf16>
  %12 = linalg.generic {indexing_maps = [#map, #map5, #map5, #map6, #map2],
                        iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%6, %10#0, %10#1, %2 : tensor<128x4096xf16>, tensor<128xf16>, tensor<128xf16>, tensor<4096x64xf16>)
    outs(%11 : tensor<128x64xf16>) {
  ^bb0(%in: f16, %in_1: f16, %in_2: f16, %in_3: f16, %out: f16):
    %14 = arith.subf %in, %in_1 : f16
    %15 = math.exp %14 : f16
    %16 = arith.divf %15, %in_2 : f16
    %17 = arith.mulf %16, %in_3 : f16
    %18 = arith.addf %out, %17 : f16
    linalg.yield %18 : f16
  } -> tensor<128x64xf16>

  bufferization.materialize_in_destination %12 in restrict writable %arg3 :
    (tensor<128x64xf16>, memref<128x64xf16>) -> ()
  return
}
```

---

## Stage 6: Fully Fused Online Attention (Max-Sum-Matmul Fusion)

The ultimate optimization fuses **all three operations** (max, sum, and final matmul) into a single generic operation. This computes the attention output in one pass over the sequence dimension.

### Key Insight
Not only does the sum need rescaling when max changes, but the **accumulated output** must also be rescaled by the same correction factor to maintain correctness.

```mlir
func.func @attention_final(%arg0: memref<128x64xf16>, %arg1: memref<4096x64xf16>,
                            %arg2: memref<4096x64xf16>, %arg3: memref<128x64xf16>) {
  %cst = arith.constant 0xFE00 : f16
  %cst_0 = arith.constant 0.000000e+00 : f16
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<128x64xf16> to tensor<128x64xf16>
  %1 = bufferization.to_tensor %arg1 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %2 = bufferization.to_tensor %arg2 restrict writable : memref<4096x64xf16> to tensor<4096x64xf16>
  %3 = tensor.empty() : tensor<128x4096xf16>
  %4 = tensor.empty() : tensor<128x64xf16>
  %5 = linalg.fill ins(%cst_0 : f16) outs(%3 : tensor<128x4096xf16>) -> tensor<128x4096xf16>

  // First generic: Q @ K^T
  %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%0, %1 : tensor<128x64xf16>, tensor<4096x64xf16>) outs(%5 : tensor<128x4096xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %14 = arith.mulf %in, %in_1 : f16
    %15 = arith.addf %out, %14 : f16
    linalg.yield %15 : f16
  } -> tensor<128x4096xf16>

  // === FULLY FUSED: Online attention (max + sum + matmul in single pass) ===
  // Takes: QK^T scores and V matrix
  // Returns: (max, sum, attention_output) computed simultaneously
  %7 = tensor.empty() : tensor<128xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<128xf16>) -> tensor<128xf16>
  %9 = linalg.fill ins(%cst_0 : f16) outs(%7 : tensor<128xf16>) -> tensor<128xf16>
  %10 = linalg.fill ins(%cst_0 : f16) outs(%4 : tensor<128x64xf16>) -> tensor<128x64xf16>
  %11:3 = linalg.generic {indexing_maps = [#map, #map6, #map5, #map5, #map2],
                          iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%6, %2 : tensor<128x4096xf16>, tensor<4096x64xf16>)
    outs(%8, %9, %10 : tensor<128xf16>, tensor<128xf16>, tensor<128x64xf16>) {
  ^bb0(%in_qk: f16, %in_v: f16, %out_max: f16, %out_sum: f16, %out_acc: f16):
    // Compute new max
    %14 = arith.maxnumf %in_qk, %out_max : f16
    // Compute correction factor: exp(old_max - new_max)
    %15 = arith.subf %out_max, %14 : f16
    %16 = math.exp %15 : f16
    // Rescale old sum with correction factor
    %17 = arith.mulf %out_sum, %16 : f16
    // Rescale old acc with correction factor (CRITICAL!)
    %18 = arith.mulf %out_acc, %16 : f16
    // Compute exp(in_qk - new_max)
    %19 = arith.subf %in_qk, %14 : f16
    %20 = math.exp %19 : f16
    // Update sum
    %21 = arith.addf %17, %20 : f16
    // Compute weighted value
    %22 = arith.mulf %20, %in_v : f16
    // Update acc
    %23 = arith.addf %18, %22 : f16
    linalg.yield %14, %21, %23 : f16, f16, f16
  } -> (tensor<128xf16>, tensor<128xf16>, tensor<128x64xf16>)

  // Final normalization: divide acc by sum
  %12 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%11#2, %11#1 : tensor<128x64xf16>, tensor<128xf16>) outs(%4 : tensor<128x64xf16>) {
  ^bb0(%in_acc: f16, %in_sum: f16, %out: f16):
    %14 = arith.divf %in_acc, %in_sum : f16
    linalg.yield %14 : f16
  } -> tensor<128x64xf16>

  bufferization.materialize_in_destination %12 in restrict writable %arg3 :
    (tensor<128x64xf16>, memref<128x64xf16>) -> ()
  return
}
```
