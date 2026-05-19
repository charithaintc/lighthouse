# Softmax Tiling and Fusion

---

## Stage 0: Initial Softmax

**What we have:** A single high-level `linalg.softmax` operation that computes softmax across dimension 1.

**Softmax formula:** For each row $i$:

$$\text{softmax}(x)_{i,j} = \frac{\exp(x_{i,j} - \max_k x_{i,k})}{\sum_k \exp(x_{i,k} - \max_k x_{i,k})}$$

```mlir
func.func @softmax_v0(%arg0: memref<64x1024xf32>, %arg1: memref<64x1024xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<64x1024xf32> to tensor<64x1024xf32>
  %1 = tensor.empty() : tensor<64x1024xf32>
  %2 = linalg.softmax dimension(1) ins(%0 : tensor<64x1024xf32>) outs(%1 : tensor<64x1024xf32>) -> tensor<64x1024xf32>
  bufferization.materialize_in_destination %2 in restrict writable %arg1 : (tensor<64x1024xf32>, memref<64x1024xf32>) -> ()
  return
}
```

---

## Stage 1: Softmax Decomposition

**Transform applied:** `transform.structured.decompose_interface`

**What happens:** The high-level softmax operation is decomposed into 5 primitive `linalg.generic` operations:

1. **Max reduction** (`%4`): Find the maximum value along each row
2. **Subtract and exp** (`%5`): Compute `exp(x - max)` elementwise
3. **Fill for sum** (`%6`): Initialize accumulator for sum reduction
4. **Sum reduction** (`%7`): Sum the exp values along each row
5. **Final division** (`%8`): Divide each exp value by the sum

```mlir
func.func @softmax_v0(%arg0: memref<64x1024xf32>, %arg1: memref<64x1024xf32>) {
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<64x1024xf32> to tensor<64x1024xf32>
  %1 = tensor.empty() : tensor<64x1024xf32>
  %2 = tensor.empty() : tensor<64xf32>
  %cst = arith.constant 0xFFC00000 : f32
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>

  // 1. Max reduction: compute max along each row
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%0 : tensor<64x1024xf32>) outs(%3 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.maxnumf %in, %out : f32
    linalg.yield %9 : f32
  } -> tensor<64xf32>

  // 2. Subtract max and apply exp: exp(x - max)
  %5 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %4 : tensor<64x1024xf32>, tensor<64xf32>) outs(%1 : tensor<64x1024xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %9 = arith.subf %in, %in_1 : f32
    %10 = math.exp %9 : f32
    linalg.yield %10 : f32
  } -> tensor<64x1024xf32>

  %cst_0 = arith.constant 0.000000e+00 : f32
  %6 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>

  // 3. Sum reduction: sum of exp values
  %7 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%5 : tensor<64x1024xf32>) outs(%6 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.addf %in, %out : f32
    linalg.yield %9 : f32
  } -> tensor<64xf32>

  // 4. Final division: exp(x - max) / sum
  %8 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%5, %7 : tensor<64x1024xf32>, tensor<64xf32>) outs(%1 : tensor<64x1024xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %9 = arith.divf %in, %in_1 : f32
    linalg.yield %9 : f32
  } -> tensor<64x1024xf32>

  bufferization.materialize_in_destination %8 in restrict writable %arg1 : (tensor<64x1024xf32>, memref<64x1024xf32>) -> ()
  return
}
```

---

## Stage 2: Elementwise Operation Fusion

**Transform applied:** `transform.apply_registered_pass "linalg-fuse-elementwise-ops"` with some modifications.

**What happens:** The compiler fuses operations that can be computed together to reduce memory traffic:

1. The elementwise `exp(x - max)` operation (`%5`) is fused into the sum reduction (`%6`)
2. The elementwise `exp(x - max)` and division are fused into the final output (`%7`)

```mlir
func.func @softmax_v0(%arg0: memref<64x1024xf32>, %arg1: memref<64x1024xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFFC00000 : f32
  %0 = bufferization.to_tensor %arg0 restrict writable : memref<64x1024xf32> to tensor<64x1024xf32>
  %1 = tensor.empty() : tensor<64x1024xf32>
  %2 = tensor.empty() : tensor<64xf32>
  %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>

  // 1. Max reduction (unchanged)
  %4 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%0 : tensor<64x1024xf32>) outs(%3 : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %8 = arith.maxnumf %in, %out : f32
    linalg.yield %8 : f32
  } -> tensor<64xf32>

  %5 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>

  // 2. Fused: exp(x - max) and sum reduction
  // This computes exp on-the-fly during the reduction
  %6 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%0, %4 : tensor<64x1024xf32>, tensor<64xf32>) outs(%5 : tensor<64xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %8 = arith.subf %in, %in_1 : f32
    %9 = math.exp %8 : f32
    %10 = arith.addf %9, %out : f32
    linalg.yield %10 : f32
  } -> tensor<64xf32>

  // 3. Fused: exp(x - max) and division
  // This computes exp again and immediately divides by sum
  %7 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%0, %4, %6 : tensor<64x1024xf32>, tensor<64xf32>, tensor<64xf32>) outs(%1 : tensor<64x1024xf32>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
    %8 = arith.subf %in, %in_1 : f32
    %9 = math.exp %8 : f32
    %10 = arith.divf %9, %in_2 : f32
    linalg.yield %10 : f32
  } -> tensor<64x1024xf32>

  bufferization.materialize_in_destination %7 in restrict writable %arg1 : (tensor<64x1024xf32>, memref<64x1024xf32>) -> ()
  return
}
```

---

## Stage 3: Max-Sum Reduction Fusion (Online Softmax)


**What happens:** The max reduction and sum reduction are fused into a single pass over the data using the "online softmax" algorithm. This algorithm maintains running max and running sum values, applying a correction factor when the max changes.

**Why this works?**

Consider the reduction loop at an arbitrary stage $i$. We have access to:

$$m_{i-1} = \max(x_0, x_1, \ldots, x_{i-1})$$

$$s_{i-1} = \sum_{j=0}^{i-1} \exp(x_j - m_{i-1})$$

When we process element $x_i$, we first compute the updated max:

$$m_i = \max(m_{i-1}, x_i)$$

Now we need to compute the updated sum. The challenge is that $s_{i-1}$ was computed with respect to $m_{i-1}$, but we need the sum with respect to the new max $m_i$.

Using the property $\exp(a - b) = \frac{\exp(a)}{\exp(b)} = \exp(a) \cdot \exp(-b)$, we can rewrite each term:

$$\exp(x_j - m_i) = \exp(x_j - m_{i-1} - (m_i - m_{i-1})) = \exp(x_j - m_{i-1}) \cdot \exp(m_{i-1} - m_i)$$

Since multiplication distributes over addition, we can factor out the correction term from the entire sum:

$$\sum_{j=0}^{i-1} \exp(x_j - m_i) = \exp(m_{i-1} - m_i) \cdot \sum_{j=0}^{i-1} \exp(x_j - m_{i-1}) = \exp(m_{i-1} - m_i) \cdot s_{i-1}$$

Therefore, the corrected sum at stage $i$ is:

$$s_i = \exp(m_{i-1} - m_i) \cdot s_{i-1} + \exp(x_i - m_i)$$

This shows that we can maintain running max and sum values in a single pass, applying a multiplicative correction factor $\exp(m_{i-1} - m_i)$ whenever the max changes.

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @softmax_v2(%arg0: memref<64x1024xf32>, %arg1: memref<64x1024xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFFC00000 : f32
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<64x1024xf32> to tensor<64x1024xf32>
    %1 = tensor.empty() : tensor<64x1024xf32>
    %2 = tensor.empty() : tensor<64xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>

    // Fused max and sum reduction with online correction
    %5:2 = linalg.generic {
      indexing_maps = [#map, #map1, #map1],
      iterator_types = ["parallel", "reduction"]
    } ins(%0 : tensor<64x1024xf32>) outs(%3, %4 : tensor<64xf32>, tensor<64xf32>) {
    ^bb0(%in: f32, %out_max: f32, %out_sum: f32):
      // Update max
      %new_max = arith.maxnumf %in, %out_max : f32

      // Compute correction factor for sum when max changes
      // correction = exp(old_max - new_max)
      %max_diff = arith.subf %out_max, %new_max : f32
      %correction = math.exp %max_diff : f32
      %corrected_sum = arith.mulf %out_sum, %correction : f32

      // Add new contribution: exp(value - new_max)
      %val_diff = arith.subf %in, %new_max : f32
      %exp_val = math.exp %val_diff : f32
      %new_sum = arith.addf %corrected_sum, %exp_val : f32

      linalg.yield %new_max, %new_sum : f32, f32
    } -> (tensor<64xf32>, tensor<64xf32>)

    // Final division to compute softmax
    %6 = linalg.generic {
      indexing_maps = [#map, #map1, #map1, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%0, %5#0, %5#1 : tensor<64x1024xf32>, tensor<64xf32>, tensor<64xf32>) outs(%1 : tensor<64x1024xf32>) {
    ^bb0(%in: f32, %in_max: f32, %in_sum: f32, %out: f32):
      %7 = arith.subf %in, %in_max : f32
      %8 = math.exp %7 : f32
      %9 = arith.divf %8, %in_sum : f32
      linalg.yield %9 : f32
    } -> tensor<64x1024xf32>

    bufferization.materialize_in_destination %6 in restrict writable %arg1 : (tensor<64x1024xf32>, memref<64x1024xf32>) -> ()
    return
}
```
