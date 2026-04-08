# IREE Attention Lowering Pipeline

## 1. ConvertAttentionToOnlineAttentionPass

### Overview

The `ConvertAttentionToOnlineAttentionPass` transforms a standard (offline) attention operation (`iree_linalg_ext.attention`) into an **online attention** operation (`iree_linalg_ext.online_attention`). Online attention computes attention in a **tiled/streaming** fashion, maintaining running max and running sum accumulators to perform numerically stable softmax incrementally — this is the core idea behind **FlashAttention**.

### Why?

Standard attention computes `softmax(Q @ K^T / scale) @ V` in a single monolithic step, requiring the entire attention matrix to be materialized in memory. Online attention tiles the computation over the key/value sequence dimension, updating partial results with running statistics, enabling **O(1) memory** in the sequence length.

### Before the Pass

A standard `iree_linalg_ext.attention` op with Q, K, V, scale, and output:

```mlir
%result = iree_linalg_ext.attention {
    indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1)>,   // Q: (m, k1)
        affine_map<(d0, d1, d2, d3) -> (d2, d1)>,   // K: (k2, k1)
        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,   // V: (k2, n)
        affine_map<(d0, d1, d2, d3) -> ()>,         // scale
        affine_map<(d0, d1, d2, d3) -> (d0, d3)>    // output: (m, n)
    ]}
    ins(%Q, %K, %V, %scale : tensor<16x64xf32>, tensor<4048x64xf32>,
                              tensor<4048x64xf32>, f32)
    outs(%output : tensor<16x64xf32>)
    -> tensor<16x64xf32>
```

### After the Pass

The op is converted to `iree_linalg_ext.online_attention` with two additional accumulator outputs — **running max** and **running sum** — initialized to `-inf` and `0` respectively:

```mlir
// Initialize accumulators
%empty_output = tensor.empty() : tensor<16x64xf32>
%empty_max = tensor.empty() : tensor<16xf32>
%cst_0 = arith.constant 0.000000e+00 : f32  // 0 for output
%cst_neg_inf = arith.constant -3.40282347E+38 : f32  // -inf for max
%cst_zero = arith.constant 0.000000e+00 : f32  // 0 for sum

%output_acc = linalg.fill ins(%cst_0 : f32) outs(%empty_output) -> tensor<16x64xf32>
%max_init = linalg.fill ins(%cst_neg_inf : f32) outs(%empty_max) -> tensor<16xf32>
%sum_init = linalg.fill ins(%cst_zero : f32) outs(%empty_max) -> tensor<16xf32>

// Online attention with streaming accumulators
%result:3 = iree_linalg_ext.online_attention {
    indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1)>,   // Q: (m, k1)
        affine_map<(d0, d1, d2, d3) -> (d2, d1)>,   // K: (k2, k1)
        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,   // V: (k2, n)
        affine_map<(d0, d1, d2, d3) -> ()>,         // scale
        affine_map<(d0, d1, d2, d3) -> (d0, d3)>,   // output: (m, n)
        affine_map<(d0, d1, d2, d3) -> (d0)>,       // running max: (m)
        affine_map<(d0, d1, d2, d3) -> (d0)>        // running sum: (m)
    ]}
    ins(%Q, %K, %V, %scale : tensor<16x64xf32>, tensor<4048x64xf32>,
                              tensor<4048x64xf32>, f32)
    outs(%output_acc, %max_init, %sum_init : tensor<16x64xf32>,
                                              tensor<16xf32>,
                                              tensor<16xf32>) {
  ^bb0(%arg: f32):
    iree_linalg_ext.yield %arg : f32
} -> tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>

// Final normalization: divide accumulated output by the final sum
%final = linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,      // final sum: (m)
        affine_map<(d0, d1) -> (d0, d1)>,  // accumulated output: (m, n)
        affine_map<(d0, d1) -> (d0, d1)>   // normalized output: (m, n)
    ],
    iterator_types = ["parallel", "parallel"]}
    ins(%result#2, %result#0 : tensor<16xf32>, tensor<16x64xf32>)
    outs(%empty_output : tensor<16x64xf32>) {
  ^bb0(%sum: f32, %acc: f32, %out: f32):
    %cst_1 = arith.constant 1.000000e+00 : f32
    %inv_sum = arith.divf %cst_1, %sum : f32
    %normalized = arith.mulf %inv_sum, %acc : f32
    linalg.yield %normalized : f32
} -> tensor<16x64xf32>
```

### Key Transformations

| Aspect | Before | After |
|--------|--------|-------|
| **Op** | `iree_linalg_ext.attention` | `iree_linalg_ext.online_attention` |
| **Q shape** | `tensor<16x64xf32>` | `tensor<16x64xf32>` (unchanged) |
| **K shape** | `tensor<4048x64xf32>` | `tensor<4048x64xf32>` (unchanged) |
| **V shape** | `tensor<4048x64xf32>` | `tensor<4048x64xf32>` (unchanged) |
| **Outputs** | 1 (result: `16x64xf32`) | 3 (result: `16x64xf32`, max: `16xf32`, sum: `16xf32`) |
| **Max accumulator** | N/A | Initialized to `-inf` (`-3.40282347E+38`) |
| **Sum accumulator** | N/A | Initialized to `0.0` |
| **Post-processing** | None | Division by final sum (normalization) |
| **Memory** | Materializes full attention matrix | Streams over K/V tiles |

### Significance in the Pipeline

This pass is a critical step in IREE's attention lowering pipeline. After this conversion, subsequent passes can **tile the online_attention op along the K2 (key sequence) dimension**, processing chunks of keys/values at a time while maintaining numerically stable softmax via the running max/sum — exactly the FlashAttention algorithm.

---

## 2. DecomposeAttentionPass

### Overview

The `DecomposeAttentionPass` (`iree-linalg-ext-decompose-attention`) runs **after** tiling has been applied to the online attention op. It decomposes each tiled `iree_linalg_ext.online_attention` op into a sequence of primitive `linalg.generic` operations that implement the online softmax + attention algorithm explicitly.

This is the pass that eliminates all custom attention ops and produces standard linalg operations that the rest of the compiler knows how to handle (vectorize, bufferize, map to hardware intrinsics, etc.).

### Pipeline Context

By the time `DecomposeAttentionPass` runs, the IR has been through:

1. `ConvertAttentionToOnlineAttention` — introduced online_attention + max/sum accumulators
2. `TileAndDistributeToWorkgroups` — tiled across batch and query-sequence dims
3. `GPUApplyTilingLevel` (multiple times) — tiled the K2 (key-sequence) reduction dimension into chunks (e.g., tiles of 64 or 128)

So the input to this pass is a **tiled** online_attention operating on a slice of K/V.

### Before the Pass (Tiled Online Attention)

After tiling, the online attention operates on a K2-tile (e.g., 16 keys at a time). This example shows a 16x64 Q-tile processing a 16x64 K-tile and V-tile:

```mlir
// Inside an scf.for loop over K2 tiles:
%results:3 = iree_linalg_ext.online_attention {
    indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1)>,   // Q tile: (m, k1)
        affine_map<(d0, d1, d2, d3) -> (d2, d1)>,   // K tile: (k2, k1)
        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,   // V tile: (k2, n)
        affine_map<(d0, d1, d2, d3) -> ()>,         // scale
        affine_map<(d0, d1, d2, d3) -> (d0, d3)>,   // acc output: (m, n)
        affine_map<(d0, d1, d2, d3) -> (d0)>,       // running max: (m)
        affine_map<(d0, d1, d2, d3) -> (d0)>        // running sum: (m)
    ]}
    ins(%q_tile, %k_tile, %v_tile, %scale : tensor<16x64xf32>,
                                             tensor<16x64xf32>,
                                             tensor<16x64xf32>, f32)
    outs(%acc, %old_max, %old_sum : tensor<16x64xf32>,
                                     tensor<16xf32>,
                                     tensor<16xf32>)
    -> tensor<16x64xf32>, tensor<16xf32>, tensor<16xf32>
```

### After the Pass (Decomposed to linalg.generic)

The pass decomposes the single online_attention op into **5 steps**:

#### Step 1: Compute S = Q @ K^T * scale (matmul + scale)

```mlir
// S[m, k2] = sum_k1(Q[m, k1] * K[k2, k1]) * scale
%empty_S = tensor.empty() : tensor<16x16xf32>
%zero_S = linalg.fill ins(%cst_0) outs(%empty_S)
%S = linalg.generic {
    indexing_maps = [
        affine_map<(m, k2, k1) -> (m, k1)>,     // Q
        affine_map<(m, k2, k1) -> (k2, k1)>,     // K
        affine_map<(m, k2, k1) -> ()>,             // scale
        affine_map<(m, k2, k1) -> (m, k2)>        // S (output)
    ],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%q_tile, %k_tile, %scale : ...)
    outs(%zero_S : tensor<16x16xf32>) {
  ^bb0(%q: f32, %k: f32, %s: f32, %out: f32):
    %mul = arith.mulf %q, %k : f32
    %scaled = arith.mulf %mul, %s : f32
    %add = arith.addf %scaled, %out : f32
    linalg.yield %add : f32
} -> tensor<16x16xf32>
```

#### Step 2: Compute new_max = max(old_max, rowmax(S))

```mlir
// Row-wise max of S, then element-wise max with old_max
%new_max = linalg.generic {
    indexing_maps = [
        affine_map<(m, k2) -> (m, k2)>,   // S
        affine_map<(m, k2) -> (m)>          // max accumulator
    ],
    iterator_types = ["parallel", "reduction"]}
    ins(%S : tensor<16x16xf32>)
    outs(%old_max : tensor<16xf32>) {
  ^bb0(%s_val: f32, %cur_max: f32):
    %m = arith.maximumf %s_val, %cur_max : f32
    linalg.yield %m : f32
} -> tensor<16xf32>
```

#### Step 3: Compute P = exp(S - new_max) and correction factor alpha = exp(old_max - new_max)

```mlir
// Subtract new_max from S and exponentiate: P[m, k2] = exp(S[m, k2] - new_max[m])
%P = linalg.generic {
    indexing_maps = [
        affine_map<(m, k2) -> (m, k2)>,   // S
        affine_map<(m, k2) -> (m)>,         // new_max
        affine_map<(m, k2) -> (m, k2)>     // P (output)
    ],
    iterator_types = ["parallel", "parallel"]}
    ins(%S, %new_max : ...)
    outs(%empty_S : tensor<16x16xf32>) {
  ^bb0(%s_val: f32, %max_val: f32, %out: f32):
    %sub = arith.subf %s_val, %max_val : f32
    %exp = math.exp %sub : f32
    linalg.yield %exp : f32
} -> tensor<16x16xf32>

// Correction factor: alpha[m] = exp(old_max[m] - new_max[m])
%alpha = linalg.generic {
    indexing_maps = [
        affine_map<(m) -> (m)>,   // old_max
        affine_map<(m) -> (m)>,   // new_max
        affine_map<(m) -> (m)>    // alpha
    ],
    iterator_types = ["parallel"]}
    ins(%old_max, %new_max : ...)
    outs(%empty_alpha : tensor<16xf32>) {
  ^bb0(%old_m: f32, %new_m: f32, %out: f32):
    %sub = arith.subf %old_m, %new_m : f32
    %exp = math.exp %sub : f32
    linalg.yield %exp : f32
} -> tensor<16xf32>
```

#### Step 4: Update sum = alpha * old_sum + rowsum(P)

```mlir
// Scale old sum by correction factor, then add row sums of P
// new_sum[m] = alpha[m] * old_sum[m] + sum_k2(P[m, k2])
%scaled_sum = linalg.generic {
    indexing_maps = [
        affine_map<(m) -> (m)>,   // old_sum
        affine_map<(m) -> (m)>,   // alpha
        affine_map<(m) -> (m)>    // output
    ],
    iterator_types = ["parallel"]}
    ins(%old_sum, %alpha : ...)
    outs(%empty_sum : tensor<16xf32>) {
  ^bb0(%s: f32, %a: f32, %out: f32):
    %mul = arith.mulf %s, %a : f32
    linalg.yield %mul : f32
} -> tensor<16xf32>

%new_sum = linalg.generic {
    indexing_maps = [
        affine_map<(m, k2) -> (m, k2)>,   // P
        affine_map<(m, k2) -> (m)>          // sum accumulator
    ],
    iterator_types = ["parallel", "reduction"]}
    ins(%P : tensor<16x16xf32>)
    outs(%scaled_sum : tensor<16xf32>) {
  ^bb0(%p_val: f32, %cur_sum: f32):
    %add = arith.addf %p_val, %cur_sum : f32
    linalg.yield %add : f32
} -> tensor<16xf32>
```

#### Step 5: Update output = alpha * old_acc + P @ V

```mlir
// Scale old accumulator by alpha: corrected_acc[m, n] = alpha[m] * old_acc[m, n]
%corrected_acc = linalg.generic {
    indexing_maps = [
        affine_map<(m, n) -> (m)>,     // alpha
        affine_map<(m, n) -> (m, n)>,  // old_acc
        affine_map<(m, n) -> (m, n)>   // output
    ],
    iterator_types = ["parallel", "parallel"]}
    ins(%alpha, %old_acc : ...)
    outs(%empty_acc : tensor<16x64xf32>) {
  ^bb0(%a: f32, %acc: f32, %out: f32):
    %mul = arith.mulf %a, %acc : f32
    linalg.yield %mul : f32
} -> tensor<16x64xf32>

// new_acc[m, n] = corrected_acc[m, n] + sum_k2(P[m, k2] * V[k2, n])
%new_acc = linalg.generic {
    indexing_maps = [
        affine_map<(m, n, k2) -> (m, k2)>,   // P
        affine_map<(m, n, k2) -> (k2, n)>,    // V
        affine_map<(m, n, k2) -> (m, n)>      // acc (output)
    ],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%P, %v_tile : ...)
    outs(%corrected_acc : tensor<16x64xf32>) {
  ^bb0(%p_val: f32, %v_val: f32, %acc: f32):
    %mul = arith.mulf %p_val, %v_val : f32
    %add = arith.addf %mul, %acc : f32
    linalg.yield %add : f32
} -> tensor<16x64xf32>
```

### Summary of Decomposition

The online attention op is decomposed into these primitive operations:

```
┌─────────────────────────────────────────────────┐
│  iree_linalg_ext.online_attention (1 tiled op)  │
└────────────────────┬────────────────────────────┘
                     │ DecomposeAttentionPass
                     ▼
┌─────────────────────────────────────────────────┐
│ 1. S = Q @ K^T * scale        (linalg.generic)  │
│ 2. new_max = max(old_max, rowmax(S))  (generic)  │
│ 3. P = exp(S - new_max)               (generic)  │
│    alpha = exp(old_max - new_max)      (generic)  │
│ 4. new_sum = alpha*old_sum + rowsum(P) (generic)  │
│ 5. new_acc = alpha*old_acc + P @ V     (generic)  │
└─────────────────────────────────────────────────┘
```

| Step | Operation | Type | Dims |
|------|-----------|------|------|
| 1 | `S = Q @ K^T * scale` | Matmul + scale | `[16, 16]` ← `[16, 64] × [16, 64]` |
| 2 | `new_max = max(old_max, rowmax(S))` | Row reduction | `[16]` ← `[16, 16]` |
| 3a | `P = exp(S - new_max)` | Elementwise | `[16, 16]` |
| 3b | `alpha = exp(old_max - new_max)` | Elementwise | `[16]` |
| 4 | `new_sum = alpha * old_sum + Σ P` | Scale + row reduction | `[16]` |
| 5 | `new_acc = alpha * old_acc + P @ V` | Scale + matmul | `[16, 64]` ← `[16, 16] × [16, 64]` |

### Why This Matters

After decomposition, all ops are standard `linalg.generic` operations. This enables:

- **Vectorization** via IREE's vector distribution pipeline
- **Mapping to MMA intrinsics** (e.g., MFMA on MI300X) for the two matmuls (Steps 1 and 5)
- **Register-level tiling** and shared memory promotion for GPU targets
- The `scf.for` loop around these ops implements the streaming/online iteration over K/V chunks
