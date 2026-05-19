# Fused Attention Kernel Lowering Flow

This document describes the multi-stage lowering process for standard attention kernels in MLIR, showing how high-level operations are progressively transformed into hardware-specific GPU code.

---

## Stage 1: Initial Standard Attention

**Input shape**: `2x8x4096x64xf16` (batch × heads × sequence × head_dim)

### Key Operations

```mlir
// Reshape from 4D to 3D: collapse batch and head dimensions
%q_3d = memref.collapse_shape %arg_q [[0, 1], [2], [3]]
  : memref<2x8x4096x64xf16> into memref<16x4096x64xf16>
%k_3d = memref.collapse_shape %arg_k [[0, 1], [2], [3]]
  : memref<2x8x4096x64xf16> into memref<16x4096x64xf16>
%v_3d = memref.collapse_shape %arg_v [[0, 1], [2], [3]]
  : memref<2x8x4096x64xf16> into memref<16x4096x64xf16>

// Transpose K: [16, 4096, 64] -> [16, 64, 4096]
%k_transposed = linalg.transpose ins(%k_3d : tensor<16x4096x64xf16>)
  outs(%empty_kt : tensor<16x64x4096xf16>) permutation = [0, 2, 1]

// Q @ K^T: [16, 4096, 64] @ [16, 64, 4096] -> [16, 4096, 4096]
%qk_scores = linalg.batch_matmul ins(%q_3d, %k_transposed : ...)
  outs(%qk_init : tensor<16x4096x4096xf16>) -> tensor<16x4096x4096xf16>

// Scale by 1/sqrt(d_k) = 0.125
%qk_scaled = linalg.mul ins(%qk_scores, %scale_factor : tensor<16x4096x4096xf16>, ...)
  -> tensor<16x4096x4096xf16>

// Softmax over last dimension
%attention_weights = linalg.softmax dimension(2) ins(%qk_scaled : ...)
  -> tensor<16x4096x4096xf16>

// Attention @ V: [16, 4096, 4096] @ [16, 4096, 64] -> [16, 4096, 64]
%output = linalg.batch_matmul ins(%attention_weights, %v_3d : ...)
  outs(%output_init : tensor<16x4096x64xf16>) -> tensor<16x4096x64xf16>
```

---

## Stage 2: Tiling and Softmax Decomposition


### Softmax Decomposition
The atomic `linalg.softmax` is decomposed into explicit operations:

**Why decompose so early? :** Softmax operation did not work well tile and fuse.
If last matmul is tiled and then if we try to fuse softmax into the `scf.forall`
It does not **bubble-up** the `tensor.extract_slice`. instead entire softmax is
done first (including parallel dims) and the extracted from the softmax result.

```mlir
// 1. Find max value per row (for numerical stability)
%max_per_row = linalg.generic {
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%qk_scaled) outs(%max_init) {
^bb0(%score: f16, %current_max: f16):
  %new_max = arith.maxnumf %score, %current_max : f16
  linalg.yield %new_max : f16
} -> tensor<16x4096xf16>

// 2. Compute exp(x - max) for each element
%exp_scores = linalg.generic {
  iterator_types = ["parallel", "parallel", "parallel"]
} ins(%qk_scaled, %max_per_row) outs(%exp_init) {
^bb0(%score: f16, %max_val: f16, %out: f16):
  %centered = arith.subf %score, %max_val : f16
  %exp_val = math.exp %centered : f16
  linalg.yield %exp_val : f16
} -> tensor<16x4096x4096xf16>

// 3. Sum exp values per row
%sum_per_row = linalg.generic {
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%exp_scores) outs(%sum_init) {
^bb0(%exp_val: f16, %current_sum: f16):
  %new_sum = arith.addf %exp_val, %current_sum : f16
  linalg.yield %new_sum : f16
} -> tensor<16x4096xf16>

// 4. Normalize: divide each element by sum
%attention_weights = linalg.generic {
  iterator_types = ["parallel", "parallel", "parallel"]
} ins(%exp_scores, %sum_per_row) outs(%norm_init) {
^bb0(%exp_val: f16, %sum: f16, %out: f16):
  %normalized = arith.divf %exp_val, %sum : f16
  linalg.yield %normalized : f16
} -> tensor<16x4096x4096xf16>
```

### Tiling the Output Dimension

The second matmul (attention @ V) is tiled into 32 tiles of size 128:

```mlir
%output = scf.forall (%batch_head_idx, %tile_idx) in (16, 32)
    shared_outs(%out_accumulator = %output_init) -> (tensor<16x4096x64xf16>) {
  %row_offset = affine.apply affine_map<(d0) -> (d0 * 128)>(%tile_idx)

  // Extract 128 rows of attention weights: [1, 128, 4096]
  %attention_tile = tensor.extract_slice %attention_weights[%batch_head_idx, %row_offset, 0]
    [1, 128, 4096] [1, 1, 1] : tensor<16x4096x4096xf16> to tensor<1x128x4096xf16>

  // Extract all of V: [1, 4096, 64]
  %v_tile = tensor.extract_slice %v_3d[%batch_head_idx, 0, 0] [1, 4096, 64] [1, 1, 1]
    : tensor<16x4096x64xf16> to tensor<1x4096x64xf16>

  // Compute partial result: [1, 128, 4096] @ [1, 4096, 64] -> [1, 128, 64]
  %partial_output = linalg.batch_matmul ins(%attention_tile, %v_tile)
    outs(%partial_init : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>

  scf.forall.in_parallel {
    tensor.parallel_insert_slice %partial_output into %out_accumulator[%batch_head_idx, %row_offset, 0]
      [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf16> into tensor<16x4096x64xf16>
  }
}
```

---

## Stage 3: Tiling Batch and Head Dimensions

### Fusion of Operations

The entire attention computation is now fused into a single parallel loop:

```mlir
%output = scf.forall (%batch_head_idx, %tile_idx) in (16, 32)
    shared_outs(%out_accumulator = %output_init) -> (tensor<16x4096x64xf16>) {
  %row_offset = affine.apply affine_map<(d0) -> (d0 * 128)>(%tile_idx)

  // Extract Q tile: [1, 128, 64]
  %q_tile = tensor.extract_slice %q_3d[%batch_head_idx, %row_offset, 0] [1, 128, 64] [1, 1, 1]
    : tensor<16x4096x64xf16> to tensor<1x128x64xf16>

  // Extract full K: [1, 4096, 64]
  %k_tile = tensor.extract_slice %k_3d[%batch_head_idx, 0, 0] [1, 4096, 64] [1, 1, 1]
    : tensor<16x4096x64xf16> to tensor<1x4096x64xf16>

  // Transpose K within tile
  %k_tile_transposed = linalg.transpose ins(%k_tile : tensor<1x4096x64xf16>)
    outs(%kt_init : tensor<1x64x4096xf16>) permutation = [0, 2, 1]

  // Q @ K^T: [1, 128, 64] @ [1, 64, 4096] -> [1, 128, 4096]
  %qk_scores = linalg.batch_matmul ins(%q_tile, %k_tile_transposed : ...)
    outs(%qk_init : tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16>

  // Scale
  %qk_scaled = linalg.mul ins(%qk_scores, %scale_factor : ...)
    -> tensor<1x128x4096xf16>

  // Softmax decomposition (max, exp, sum, normalize)
  %max_per_row = linalg.generic { ... } // max reduction -> [1, 128]
  %exp_scores = linalg.generic { ... }   // exp(x - max)
  %sum_per_row = linalg.generic { ... }  // sum reduction
  %attention_weights = linalg.generic { ... } // normalize

  // Extract V: [1, 4096, 64]
  %v_tile = tensor.extract_slice %v_3d[%batch_head_idx, 0, 0] [1, 4096, 64] [1, 1, 1]
    : tensor<16x4096x64xf16> to tensor<1x4096x64xf16>

  // Attention @ V: [1, 128, 4096] @ [1, 4096, 64] -> [1, 128, 64]
  %partial_output = linalg.batch_matmul ins(%attention_weights, %v_tile : ...)
    outs(%partial_init : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>

  scf.forall.in_parallel {
    tensor.parallel_insert_slice %partial_output into %out_accumulator[%batch_head_idx, %row_offset, 0]
      [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf16> into tensor<16x4096x64xf16>
  }
}
```

Each workgroup now processes:
- 128 rows of Q
- All of K and V (still materializes `128×4096` attention matrix)
- Produces 128 rows of output

16 × 32 = 512 independent workgroups

---

## Stage 4: Vectorization

Linalg operations are converted to vector operations for SIMD execution.

### First Matmul (Q @ K^T)

```mlir
// Read K: [4096, 64]
%k_vec = vector.transfer_read %k_3d[%batch_head_idx, %c0, %c0], %poison {in_bounds = [true, true]}
  : tensor<16x4096x64xf16>, vector<4096x64xf16>

// Read Q tile: [128, 64]
%q_vec = vector.transfer_read %q_3d[%batch_head_idx, %row_offset, %c0], %poison {in_bounds = [true, true]}
  : tensor<16x4096x64xf16>, vector<128x64xf16>

// Contract (matmul): [128, 64] @ [4096, 64]^T -> [128, 4096]
%qk_scores_vec = vector.contract {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,  // Q: reduce over d2
    affine_map<(d0, d1, d2) -> (d1, d2)>,  // K^T: reduce over d2
    affine_map<(d0, d1, d2) -> (d0, d1)>   // Output
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
} %q_vec, %k_vec, %zero_init : vector<128x64xf16>, vector<4096x64xf16> into vector<128x4096xf16>
```

### Softmax with Vector Reductions

```mlir
// Scale by 1/sqrt(d_k)
%qk_scaled_vec = arith.mulf %qk_scores_vec, %scale_factor_vec : vector<128x4096xf16>

// Reshape for reduction: [128, 4096] -> [1, 128, 4096]
%qk_3d = vector.shape_cast %qk_scaled_vec : vector<128x4096xf16> to vector<1x128x4096xf16>

// Max reduction along last dimension (across sequence length)
%max_per_row_vec = vector.multi_reduction <maxnumf>, %qk_3d, %neg_inf_init [2]
  : vector<1x128x4096xf16> to vector<1x128xf16>

// Broadcast max back to full shape for subtraction
%max_broadcast_3d = vector.broadcast %max_per_row_vec : vector<1x128xf16> to vector<4096x1x128xf16>
%max_broadcast_2d = vector.shape_cast %max_broadcast_3d : vector<4096x1x128xf16> to vector<4096x128xf16>
%max_broadcast = vector.transpose %max_broadcast_2d, [1, 0] : vector<4096x128xf16> to vector<128x4096xf16>

// Exp of (x - max) for numerical stability
%centered_scores = arith.subf %qk_scaled_vec, %max_broadcast : vector<128x4096xf16>
%exp_scores_vec = math.exp %centered_scores : vector<128x4096xf16>

// Sum reduction to get denominator
%exp_3d = vector.shape_cast %exp_scores_vec : vector<128x4096xf16> to vector<1x128x4096xf16>
%sum_per_row_vec = vector.multi_reduction <add>, %exp_3d, %zero_init [2]
  : vector<1x128x4096xf16> to vector<1x128xf16>

// Broadcast sum for normalization
%sum_broadcast_3d = vector.broadcast %sum_per_row_vec : vector<1x128xf16> to vector<4096x1x128xf16>
%sum_broadcast_2d = vector.shape_cast %sum_broadcast_3d : vector<4096x1x128xf16> to vector<4096x128xf16>
%sum_broadcast = vector.transpose %sum_broadcast_2d, [1, 0] : vector<4096x128xf16> to vector<128x4096xf16>

// Normalize to get attention weights
%attention_weights_vec = arith.divf %exp_scores_vec, %sum_broadcast : vector<128x4096xf16>
```

### Second Matmul (Attention @ V)

```mlir
// Read V: [4096, 64]
%v_vec = vector.transfer_read %v_3d[%batch_head_idx, %c0, %c0], %poison {in_bounds = [true, true]}
  : tensor<16x4096x64xf16>, vector<4096x64xf16>

// Contract: [128, 4096] @ [4096, 64] -> [128, 64]
%output_vec = vector.contract {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,  // Attention weights: reduce over sequence
    affine_map<(d0, d1, d2) -> (d2, d1)>,  // V: reduce over sequence
    affine_map<(d0, d1, d2) -> (d0, d1)>   // Output
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
} %attention_weights_vec, %v_vec, %zero_output_init :
  vector<128x4096xf16>, vector<4096x64xf16> into vector<128x64xf16>
```

---

## Stage 5: Bufferization

Tensors are converted to memrefs (in-place memory buffers):

```mlir
func.func @payload(%arg_output: memref<2x8x4096x64xf16>,
                   %arg_q: memref<2x8x4096x64xf16>,
                   %arg_k: memref<2x8x4096x64xf16>,
                   %arg_v: memref<2x8x4096x64xf16>) {
  // Constants remain as vectors
  %zero_vec_128x64 = arith.constant dense<0.000000e+00> : vector<128x64xf16>
  %poison = ub.poison : f16

  // Collapse shapes on memrefs
  %q_3d = memref.collapse_shape %arg_q [[0, 1], [2], [3]]
    : memref<2x8x4096x64xf16> into memref<16x4096x64xf16>
  %k_3d = memref.collapse_shape %arg_k [[0, 1], [2], [3]]
    : memref<2x8x4096x64xf16> into memref<16x4096x64xf16>
  %v_3d = memref.collapse_shape %arg_v [[0, 1], [2], [3]]
    : memref<2x8x4096x64xf16> into memref<16x4096x64xf16>
  %output_3d = memref.collapse_shape %arg_output [[0, 1], [2], [3]]
    : memref<2x8x4096x64xf16> into memref<16x4096x64xf16>

  scf.forall (%batch_head_idx, %tile_idx) in (16, 32) {
    %row_offset = affine.apply affine_map<(d0) -> (d0 * 128)>(%tile_idx)

    // Direct reads from memrefs
    %k_vec = vector.transfer_read %k_3d[%batch_head_idx, %c0, %c0], %poison
      : memref<16x4096x64xf16>, vector<4096x64xf16>

    // ... computation ...

    // Create subview for output
    %output_subview = memref.subview %output_3d[%batch_head_idx, %row_offset, 0] [1, 128, 64] [1, 1, 1]
      : memref<16x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>

    // Direct write to memref
    vector.transfer_write %output_vec, %output_subview[%c0, %c0, %c0] {in_bounds = [true, true]}
      : vector<128x64xf16>, memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
  }
}
```

---

## Stage 6: Inner Tiling for Fused Attention (Online Softmax)

1. Implements "online" softmax to avoid materializing the full attention matrix.
2. Tile the K and V loads into `16 x d_head` and interleave with DPAS for lower register preassure.

### The Online Softmax Algorithm

Instead of computing the full `128×4096` attention matrix, we process K/V in chunks of 64 and incrementally update:
- Running maximum `m_i`
- Running sum of exponentials `l_i`
- Partial output `O_i`

```mlir
%final_max:3 = scf.for %kv_chunk_idx = %c0 to %c4096 step %c64
  iter_args(%m_old = %neg_inf_init, %l_old = %zero_init, %O_old = %zero_output_init)
  -> (vector<128xf16>, vector<128xf16>, vector<128x64xf16>) {
  // m_old = running maximum across previous chunks
  // l_old = running sum of exponentials across previous chunks
  // O_old = running partial output across previous chunks
```

#### Process 64 columns of K at a time (4 chunks of 16)

```mlir
  // Chunk 0: columns [0:16] of K
  %k_chunk_0 = vector.transfer_read %k_4d[%batch_idx, %head_idx, %kv_chunk_idx, %c0], %poison
    : memref<2x8x4096x64xf16>, vector<16x64xf16>
  %k_chunk_0_t = vector.transpose %k_chunk_0, [1, 0] : vector<16x64xf16> to vector<64x16xf16>
  %qk_chunk_0 = vector.contract { ... } %q_vec, %k_chunk_0_t, %zero_init :
    vector<128x64xf16>, vector<64x16xf16> -> vector<128x16xf16>

  // Chunk 1: columns [16:32] of K
  %k_offset_16 = arith.addi %kv_chunk_idx, %c16 : index
  %k_chunk_1 = vector.transfer_read %k_4d[%batch_idx, %head_idx, %k_offset_16, %c0], %poison
  %k_chunk_1_t = vector.transpose %k_chunk_1, [1, 0] : vector<16x64xf16> to vector<64x16xf16>
  %qk_chunk_1 = vector.contract { ... } %q_vec, %k_chunk_1_t, %zero_init : ... -> vector<128x16xf16>

  // Chunk 2: columns [32:48] of K
  %k_offset_32 = arith.addi %kv_chunk_idx, %c32 : index
  %k_chunk_2 = vector.transfer_read %k_4d[%batch_idx, %head_idx, %k_offset_32, %c0], %poison
  %k_chunk_2_t = vector.transpose %k_chunk_2, [1, 0] : vector<16x64xf16> to vector<64x16xf16>
  %qk_chunk_2 = vector.contract { ... } %q_vec, %k_chunk_2_t, %zero_init : ... -> vector<128x16xf16>

  // Chunk 3: columns [48:64] of K
  %k_offset_48 = arith.addi %kv_chunk_idx, %c48 : index
  %k_chunk_3 = vector.transfer_read %k_4d[%batch_idx, %head_idx, %k_offset_48, %c0], %poison
  %k_chunk_3_t = vector.transpose %k_chunk_3, [1, 0] : vector<16x64xf16> to vector<64x16xf16>
  %qk_chunk_3 = vector.contract { ... } %q_vec, %k_chunk_3_t, %zero_init : ... -> vector<128x16xf16>
```

#### Find new maximum across all 4 chunks

```mlir
  // Find max across all 4 chunks
  %max_01 = arith.maximumf %qk_chunk_0, %qk_chunk_1 : vector<128x16xf16>
  %max_012 = arith.maximumf %max_01, %qk_chunk_2 : vector<128x16xf16>
  %max_0123 = arith.maximumf %max_012, %qk_chunk_3 : vector<128x16xf16>
  %max_chunk_per_row = vector.multi_reduction <maxnumf>, %max_0123, %neg_inf_init [1] :
    vector<128x16xf16> -> vector<128xf16>

  // Scale and update running maximum
  %max_chunk_scaled = arith.mulf %max_chunk_per_row, %scale_factor_vec : vector<128xf16>
  %m_new = arith.maximumf %m_old, %max_chunk_scaled : vector<128xf16>
```

#### Compute exponentials for each chunk

```mlir
  // Broadcast m_new for subtraction from all chunks
  %m_new_3d = vector.broadcast %m_new : vector<128xf16> to vector<16x128xf16>
  %m_new_broadcast = vector.transpose %m_new_3d, [1, 0] : vector<16x128xf16> to vector<128x16xf16>

  // exp(chunk_0 * scale - m_new)
  %qk_chunk_0_scaled = arith.mulf %qk_chunk_0, %scale_factor_2d : vector<128x16xf16>
  %qk_chunk_0_centered = arith.subf %qk_chunk_0_scaled, %m_new_broadcast : vector<128x16xf16>
  %exp_chunk_0 = math.exp %qk_chunk_0_centered : vector<128x16xf16>

  // exp(chunk_1 * scale - m_new)
  %qk_chunk_1_scaled = arith.mulf %qk_chunk_1, %scale_factor_2d : vector<128x16xf16>
  %qk_chunk_1_centered = arith.subf %qk_chunk_1_scaled, %m_new_broadcast : vector<128x16xf16>
  %exp_chunk_1 = math.exp %qk_chunk_1_centered : vector<128x16xf16>

  // exp(chunk_2 * scale - m_new)
  %qk_chunk_2_scaled = arith.mulf %qk_chunk_2, %scale_factor_2d : vector<128x16xf16>
  %qk_chunk_2_centered = arith.subf %qk_chunk_2_scaled, %m_new_broadcast : vector<128x16xf16>
  %exp_chunk_2 = math.exp %qk_chunk_2_centered : vector<128x16xf16>

  // exp(chunk_3 * scale - m_new)
  %qk_chunk_3_scaled = arith.mulf %qk_chunk_3, %scale_factor_2d : vector<128x16xf16>
  %qk_chunk_3_centered = arith.subf %qk_chunk_3_scaled, %m_new_broadcast : vector<128x16xf16>
  %exp_chunk_3 = math.exp %qk_chunk_3_centered : vector<128x16xf16>
```

#### Update sum of exponentials

```mlir
  // Sum exponentials across the 4 chunks
  %sum_01 = arith.addf %exp_chunk_0, %exp_chunk_1 : vector<128x16xf16>
  %sum_012 = arith.addf %sum_01, %exp_chunk_2 : vector<128x16xf16>
  %sum_0123 = arith.addf %sum_012, %exp_chunk_3 : vector<128x16xf16>
  %l_chunk = vector.multi_reduction <add>, %sum_0123, %zero_init [1] :
    vector<128x16xf16> -> vector<128xf16>

  // Correction factor for previous chunks: exp(m_old - m_new)
  %m_delta = arith.subf %m_old, %m_new : vector<128xf16>
  %correction_factor = math.exp %m_delta : vector<128xf16>

  // Update running sum: l_new = l_old * correction + l_chunk
  %l_old_corrected = arith.mulf %l_old, %correction_factor : vector<128xf16>
  %l_new = arith.addf %l_old_corrected, %l_chunk : vector<128xf16>
```

#### Update partial output

```mlir
  // Rescale old output by correction factor
  %correction_3d = vector.broadcast %correction_factor : vector<128xf16> to vector<64x128xf16>
  %correction_broadcast = vector.transpose %correction_3d, [1, 0] : vector<64x128xf16> to vector<128x64xf16>
  %O_old_corrected = arith.mulf %O_old, %correction_broadcast : vector<128x64xf16>

  // Load corresponding 64 rows of V (chunk 0: rows [0:16])
  %v_chunk_0 = vector.transfer_read %v_4d[%batch_idx, %head_idx, %kv_chunk_idx, %c0], %poison
    : memref<2x8x4096x64xf16>, vector<16x64xf16>

  // Accumulate: O += exp_chunk_0 @ V[0:16, :]
  %O_partial_0 = vector.contract { ... } %exp_chunk_0, %v_chunk_0, %O_old_corrected :
    vector<128x16xf16>, vector<16x64xf16> -> vector<128x64xf16>

  // Accumulate: O += exp_chunk_1 @ V[16:32, :]
  %v_chunk_1 = vector.transfer_read %v_4d[%batch_idx, %head_idx, %k_offset_16, %c0], %poison
  %O_partial_1 = vector.contract { ... } %exp_chunk_1, %v_chunk_1, %O_partial_0 :
    vector<128x16xf16>, vector<16x64xf16> -> vector<128x64xf16>

  // Accumulate: O += exp_chunk_2 @ V[32:48, :]
  %v_chunk_2 = vector.transfer_read %v_4d[%batch_idx, %head_idx, %k_offset_32, %c0], %poison
  %O_partial_2 = vector.contract { ... } %exp_chunk_2, %v_chunk_2, %O_partial_1 :
    vector<128x16xf16>, vector<16x64xf16> -> vector<128x64xf16>

  // Accumulate: O += exp_chunk_3 @ V[48:64, :]
  %v_chunk_3 = vector.transfer_read %v_4d[%batch_idx, %head_idx, %k_offset_48, %c0], %poison
  %O_new = vector.contract { ... } %exp_chunk_3, %v_chunk_3, %O_partial_2 :
    vector<128x16xf16>, vector<16x64xf16> -> vector<128x64xf16>

  scf.yield %m_new, %l_new, %O_new : vector<128xf16>, vector<128xf16>, vector<128x64xf16>
}
```

#### Final normalization

```mlir
// Extract final values from loop
%m_final = %final_max#0 : vector<128xf16>
%l_final = %final_max#1 : vector<128xf16>
%O_accumulated = %final_max#2 : vector<128x64xf16>

// Broadcast sum to full output shape for normalization
%l_final_3d = vector.broadcast %l_final : vector<128xf16> to vector<64x128xf16>
%l_final_broadcast = vector.transpose %l_final_3d, [1, 0] : vector<64x128xf16> to vector<128x64xf16>

// Normalize: O_final = O_accumulated / l_final
%output_normalized = arith.divf %O_accumulated, %l_final_broadcast : vector<128x64xf16>

// Write result back to output buffer
vector.transfer_write %output_normalized, %output_4d[%batch_idx, %head_idx, %row_offset, %c0]
  {in_bounds = [true, true]} : vector<128x64xf16>, memref<2x8x4096x64xf16>
```

---

## Stage 7: GPU Outlining

`scf.forall` loop is distirbuted to workgroups.

```mlir
module attributes {gpu.container_module} {
  func.func @payload(%arg_output: memref<2x8x4096x64xf16>,
                     %arg_q: memref<2x8x4096x64xf16>,
                     %arg_k: memref<2x8x4096x64xf16>,
                     %arg_v: memref<2x8x4096x64xf16>) {
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index

    // Launch GPU kernel
    gpu.launch_func @payload_kernel::@payload_kernel
      blocks in (%c16, %c32, %c1)    // Grid: 16 × 32 × 1 (batch×head, seq_tiles, 1)
      threads in (%c128, %c1, %c1)   // Block: 128 × 1 × 1
      args(%arg_q : memref<2x8x4096x64xf16>,
           %arg_k : memref<2x8x4096x64xf16>,
           %arg_v : memref<2x8x4096x64xf16>,
           %arg_output : memref<2x8x4096x64xf16>)
    return
  }

  gpu.module @payload_kernel {
    gpu.func @payload_kernel(%q: memref<2x8x4096x64xf16>,
                             %k: memref<2x8x4096x64xf16>,
                             %v: memref<2x8x4096x64xf16>,
                             %output: memref<2x8x4096x64xf16>) kernel
      attributes {
        known_block_size = array<i32: 128, 1, 1>,
        known_grid_size = array<i32: 16, 32, 1>
      } {

      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y

      // Compute batch and head indices from block_id_x
      %row_offset = arith.muli %block_id_y, %c128 : index
      %batch_idx = arith.floordivsi %block_id_x, %c8 : index
      %head_idx = arith.remsi %block_id_x, %c8 : index

      // ... computation from Stage 6 ...

      gpu.return
    }
  }
}
```

**Grid Configuration**:
- 16 blocks in X (2 batches × 8 heads)
- 32 blocks in Y (4096 / 128 = 32 tiles)
- Each block has 128 threads (8 subgroups)

---

## Stage 8: Converting to XeGPU Operations

Vector operations are converted to Intel XeGPU-specific operations using tensor descriptors.

### Tensor Descriptor Creation

```mlir
// Create descriptor for reading Q
%q_subview = memref.subview %q[%batch_idx, %head_idx, 0, 0] [1, 1, 4096, 64] [1, 1, 1, 1]
  : memref<2x8x4096x64xf16> to memref<4096x64xf16, strided<[64, 1], offset: ?>>

%q_base_buffer, %q_offset, %q_sizes:2, %q_strides:2 =
  memref.extract_strided_metadata %q_subview : memref<4096x64xf16, strided<[64, 1], offset: ?>>
  -> memref<f16>, index, index, index, index, index

%q_intptr = memref.extract_aligned_pointer_as_index %q_base_buffer : memref<f16> -> index
%q_byte_offset = arith.muli %q_offset, %c2 : index  // Multiply by sizeof(f16)
%q_addr = arith.addi %q_intptr, %q_byte_offset : index
%q_addr_i64 = arith.index_cast %q_addr : index to i64

// Create XeGPU tensor descriptor for Q
%q_tdesc = xegpu.create_nd_tdesc %q_addr_i64, shape : [4096, 64], strides : [64, 1] : i64
  -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
```

### XeGPU Load Operations

```mlir
// Load Q tile using tensor descriptor at row_offset
%q_vec = xegpu.load_nd %q_tdesc[%row_offset, 0]
  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
  -> vector<128x64xf16>

// Load K chunk at kv_chunk_idx
%k_chunk_vec = xegpu.load_nd %k_tdesc[%kv_chunk_idx, 0]
  : !xegpu.tensor_desc<16x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
  -> vector<16x64xf16>
```

### XeGPU DPAS (Dot Product Accumulate Systolic)

The key hardware instruction for matrix multiplication:

```mlir
// Transpose K chunk for matmul
%k_chunk_t = vector.transpose %k_chunk_vec, [1, 0] : vector<16x64xf16> to vector<64x16xf16>

// DPAS: Specialized matmul instruction
// Q: [128, 64], K^T: [64, 16], accumulator: [128, 16]
%qk_chunk = xegpu.dpas %q_vec, %k_chunk_t, %zero_accumulator
  : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf16>
  -> vector<128x16xf16>
```

### XeGPU Store Operations

```mlir
// Create output descriptor (similar to Q descriptor creation)
%output_addr_i64 = arith.index_cast %output_addr : index to i64
%output_tdesc = xegpu.create_nd_tdesc %output_addr_i64, shape : [4096, 64], strides : [64, 1] : i64
  -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>

// Store result at row_offset
xegpu.store_nd %output_normalized, %output_tdesc[%row_offset, 0]
  : vector<128x64xf16>,
    !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
```

---

## Stage 9: Setting XeGPU Layouts

Assign layouts for distributing to subgroups.

### Load Layout for Q (128×64)

```mlir
%q_vec = xegpu.load_nd %q_tdesc[%row_offset, 0] <{
  layout = #xegpu.layout<
    sg_layout = [8, 1],      // 8 subgroups in row, 1 in column
    sg_data = [16, 64],      // Each SG handles 16×64 data
    inst_data = [8, 16]      // Each instruction: 8×16
  >
}> : !xegpu.tensor_desc<128x64xf16, ...> -> vector<128x64xf16>
```

**Breakdown**:
- 8 subgroups handle rows: 8 × 16 = 128 rows ✓
- 1 subgroup handles columns: 1 × 64 = 64 columns ✓
- Each instruction processes 8×16 elements

### DPAS Layout (Q @ K^T → Attention Scores)

```mlir
%qk_chunk = xegpu.dpas %q_vec, %k_chunk_t, %zero_accumulator {
  layout_a = #xegpu.layout<
    sg_layout = [8, 1],      // Q: 8 SGs vertically, 1 horizontally
    sg_data = [16, 64],      // Each SG: 16 rows × 64 cols
    inst_data = [8, 16]      // Each DPAS: 8 rows × 16 cols
  >,
  layout_b = #xegpu.layout<
    sg_layout = [1, 8],      // K^T: 1 SG vertically, 8 horizontally
    sg_data = [64, 16],      // Each SG: 64 rows × 16 cols
    inst_data = [16, 16],    // Each DPAS: 16 × 16
    order = [0, 1]
  >,
  layout_cd = #xegpu.layout<
    sg_layout = [8, 1],      // Output: 8 SGs vertically
    sg_data = [16, 16],      // Each SG: 16 rows × 16 cols
    inst_data = [8, 16]      // Each DPAS output: 8 × 16
  >
} : vector<128x64xf16>, vector<64x16xf16>, vector<128x16xf16>
  -> vector<128x16xf16>
```

**Matrix Multiplication Mapping**:
- Input A (Q): 8 subgroups × (16 rows × 64 cols) = 128×64
- Input B (K^T): 8 subgroups × (64 rows × 16 cols) = 64×128 (transposed)
- Output C (Scores): 8 subgroups × (16 rows × 16 cols) = 128×16

### DPAS Layout (Attention @ V → Output)

```mlir
%O_partial = xegpu.dpas %exp_chunk, %v_chunk, %O_old_corrected {
  layout_a = #xegpu.layout<
    sg_layout = [8, 1],      // Exp attention: 8 SGs vertically
    sg_data = [16, 16],      // Each SG: 16 rows × 16 cols
    inst_data = [8, 16]      // Each DPAS: 8 × 16
  >,
  layout_b = #xegpu.layout<
    sg_layout = [8, 1],      // V: 8 SGs vertically (not transposed)
    sg_data = [16, 64],      // Each SG: 16 rows × 64 cols
    inst_data = [16, 16]     // Each DPAS: 16 × 16
  >,
  layout_cd = #xegpu.layout<
    sg_layout = [8, 1],      // Output: 8 SGs vertically
    sg_data = [16, 64],      // Each SG: 16 rows × 64 cols
    inst_data = [8, 16]      // Each DPAS output: 8 × 16
  >
} : vector<128x16xf16>, vector<16x64xf16>, vector<128x64xf16>
  -> vector<128x64xf16>
```

### Store Layout

```mlir
xegpu.store_nd %output_normalized, %output_tdesc[%row_offset, 0] <{
  layout = #xegpu.layout<
    sg_layout = [8, 1],      // 8 SGs vertically, 1 horizontally
    sg_data = [16, 64],      // Each SG stores: 16 rows × 64 cols
    inst_data = [8, 16]      // Each store instruction: 8 × 16
  >
}> : vector<128x64xf16>, !xegpu.tensor_desc<128x64xf16, ...>
```

### GPU Target Attribute

```mlir
gpu.module @payload_kernel [#xevm.target<O = 3>] {
  // O = 3 specifies optimization level 3
  gpu.func @payload_kernel(...) kernel { ... }
}
```

---

## Conclusion

This lowering flow demonstrates a sophisticated compilation strategy:
1. Start with intuitive high-level operations
2. Progressively expose parallelism through tiling
3. Apply memory-saving algorithms (online softmax)
4. Lower to vector operations for SIMD
5. Map to hardware-specific instructions (DPAS)
6. Optimize data layout for hardware execution

The result is a highly optimized fused attention kernel that maximizes throughput while minimizing memory footprint.
