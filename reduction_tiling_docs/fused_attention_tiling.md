# Attention Tiling

## Linalg level implementation

Input sizes:
- Q: 4096x64
- K: 4096x64
- V: 4096x64

```mlir
func.func @attention(%Q : memref<4096x64xf32>, %K: memref<4096x64xf32>,
%V: memref<4096x64xf32>, %out: memref<4096x64xf32>) {
  ...
  // Transpose K
  %k_transpose = linalg.transpose ... -> tensor<64x4096xf32>

  // QK^T
  %QKT = linalg.matmul ins(%q, %k_transpose : tensor<4096x64xf32>, tensor<64x4096xf32>)
                       outs(%empty_NxN : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>

  // Fill with -inf
  %t_minf = linalg.fill ins(%cst_minus_inf : f32) outs(%empty_N : tensor<4096xf32>) -> tensor<4096xf32>

  // Max reduce along rows
  %max = linalg.reduce ins(%QKT : tensor<4096x4096xf32>) ... %m = arith.maximumf %in, %init : f32 -> tensor<4096xf32>

  // Broadcast max
  %maxb = linalg.broadcast ins(%max: tensor<4096xf32>) outs(%empty_NxN : tensor<4096x4096xf32>) dimensions = [1] -> tensor<4096x4096xf32>

  // Subtract
  %sub = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ... -> tensor<4096x4096xf32>

  // Exp
  %exp = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ... -> tensor<4096x4096xf32>

  // Fill with zeros
  %t_zeros = linalg.fill ins(%c0f : f32) outs(%empty_N : tensor<4096xf32>) -> tensor<4096xf32>

  // Sum reduce along rows
  %sum = linalg.reduce ... %s = arith.addf %in, %init : f32 ... -> tensor<4096xf32>

  // Broadcast sum and div
  %sums = linalg.broadcast ... -> tensor<4096x4096xf32>
  %p = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
       ins(%exp, %sums : tensor<4096x4096xf32>, tensor<4096x4096xf32>) ... -> tensor<4096x4096xf32>

  // Final matmul
  %o = linalg.matmul ins(%p, %v : tensor<4096x4096xf32>, tensor<4096x64xf32>) ... -> tensor<4096x64xf32>
  ...
}
```

---

## Stage 1: Tile the last matmul in K dim (tile size = 16)

After tiling the final matmul `%o = linalg.matmul ins(%p, %v)` along the K dimension with tile size 16:

```mlir
func.func @attention(%Q : memref<4096x64xf32>, %K: memref<4096x64xf32>,
%V: memref<4096x64xf32>, %out: memref<4096x64xf32>) {
  ...
  // Compute p = Softmax(Q @ K^T)
  // ...

  // Final matmul TILED in K dimension (4096 / 16 = 256 tiles)
  // Loop over K dimension: k = 0 to 4096 step 16
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c16 = arith.constant 16 : index

  // Initialize output with zeros: 4096x64
  %o_init = linalg.fill ins(%c0f : f32) outs(%empty_out : tensor<4096x64xf32>) -> tensor<4096x64xf32>

  %o = scf.for %k = %c0 to %c4096 step %c16 iter_args(%o_acc = %o_init) -> (tensor<4096x64xf32>) {
    // Extract slice from %p: 4096x16 (from columns [k:k+16])
    %p_slice = tensor.extract_slice %p[0, %k][4096, 16][1, 1] -> tensor<4096x16xf32>

    // Extract slice from %v: 16x64 (from rows [k:k+16])
    %v_slice = tensor.extract_slice %v[%k, 0][16, 64][1, 1] -> tensor<16x64xf32>

    // Partial matmul: (4096x16) @ (16x64) -> 4096x64
    %partial = linalg.matmul ins(%p_slice, %v_slice : tensor<4096x16xf32>, tensor<16x64xf32>)
                             outs(%empty_partial : tensor<4096x64xf32>) -> tensor<4096x64xf32>

    // Accumulate: 4096x64 + 4096x64 -> 4096x64
    %o_new = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
             ins(%o_acc, %partial : tensor<4096x64xf32>, tensor<4096x64xf32>) ... -> tensor<4096x64xf32>

    scf.yield %o_new : tensor<4096x64xf32>
  }
  ...
}
```

## Stage 2: Tile and fuse the softmax computation (tile size = 16)

After tiling the softmax computation in the reduction dimension with tile size 16 and fusing operations, following the pattern from the softmax lowering flow:

```mlir
func.func @attention(%Q : memref<4096x64xf32>, %K: memref<4096x64xf32>,
%V: memref<4096x64xf32>, %out: memref<4096x64xf32>) {
  ...
  // === First matmul: Q @ K^T ===
  // Transpose K: 4096x64 -> 64x4096
  %k_transpose = linalg.transpose ... -> tensor<64x4096xf32>

  // QK^T: (4096x64) @ (64x4096) -> 4096x4096
  %QKT = linalg.matmul ins(%q, %k_transpose : tensor<4096x64xf32>, tensor<64x4096xf32>)
                       outs(%empty_NxN : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>


  // === Tiled and fused softmax computation ===
  // Tile size = 16, number of tiles = 4096 / 16 = 256
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c16 = arith.constant 16 : index

  // Initialize max buffer with -inf: 4096x16
  %max_buffer_init = linalg.fill ins(%cst_minus_inf : f32) outs(%empty_max_buf : tensor<4096x16xf32>) -> tensor<4096x16xf32>

  // Loop 1: Max reduction (4096 / 16 = 256 iterations)
  %max_buffer = scf.for %k = %c0 to %c4096 step %c16 iter_args(%max_acc = %max_buffer_init) -> (tensor<4096x16xf32>) {
    // Extract slice from QKT: 4096x16
    %QKT_slice = tensor.extract_slice %QKT[0, %k][4096, 16][1, 1] -> tensor<4096x16xf32>

    // Max accumulation: 4096x16
    %max_new = linalg.generic {iterator_types = ["parallel", "parallel"]}
               ins(%QKT_slice : tensor<4096x16xf32>) outs(%max_acc : tensor<4096x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %max_val = arith.maxnumf %in, %out : f32
        linalg.yield %max_val : f32
    } -> tensor<4096x16xf32>

    scf.yield %max_new : tensor<4096x16xf32>
  }

  // Final max reduction: 4096x16 -> 4096
  %max = linalg.reduce ins(%max_buffer : tensor<4096x16xf32>) outs(%empty_N : tensor<4096xf32>) dimensions = [1] {
    (%in: f32, %init: f32) {
      %m = arith.maxnumf %in, %init : f32
      linalg.yield %m : f32
    }
  } -> tensor<4096xf32>


  // Initialize sum buffer with zeros: 4096x16
  %sum_buffer_init = linalg.fill ins(%c0f : f32) outs(%empty_sum_buf : tensor<4096x16xf32>) -> tensor<4096x16xf32>

  // Loop 2: Sum reduction with fused center+exp (256 iterations)
  %sum_buffer = scf.for %k = %c0 to %c4096 step %c16 iter_args(%sum_acc = %sum_buffer_init) -> (tensor<4096x16xf32>) {
    // Extract slice from QKT: 4096x16
    %QKT_slice = tensor.extract_slice %QKT[0, %k][4096, 16][1, 1] -> tensor<4096x16xf32>

    // Fused center+exp: 4096x16
    %exp_slice = linalg.generic {iterator_types = ["parallel", "parallel"]}
                 ins(%QKT_slice, %max : tensor<4096x16xf32>, tensor<4096xf32>) outs(%empty_slice : tensor<4096x16xf32>) {
      ^bb0(%in: f32, %max_val: f32, %out: f32):
        %centered = arith.subf %in, %max_val : f32
        %exp_val = math.exp %centered : f32
        linalg.yield %exp_val : f32
    } -> tensor<4096x16xf32>

    // Sum accumulation: 4096x16
    %sum_new = linalg.generic {iterator_types = ["parallel", "parallel"]}
               ins(%exp_slice : tensor<4096x16xf32>) outs(%sum_acc : tensor<4096x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %sum_val = arith.addf %in, %out : f32
        linalg.yield %sum_val : f32
    } -> tensor<4096x16xf32>

    scf.yield %sum_new : tensor<4096x16xf32>
  }

  // Final sum reduction: 4096x16 -> 4096
  %sum = linalg.reduce ins(%sum_buffer : tensor<4096x16xf32>) outs(%empty_N : tensor<4096xf32>) dimensions = [1] {
    (%in: f32, %init: f32) {
      %s = arith.addf %in, %init : f32
      linalg.yield %s : f32
    }
  } -> tensor<4096xf32>


  // Initialize output buffer for softmax: 4096x4096
  %p_init = linalg.fill ins(%c0f : f32) outs(%empty_NxN : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>

  // Loop 3: Division with fused center+exp+div (256 iterations)
  %p = scf.for %k = %c0 to %c4096 step %c16 iter_args(%p_acc = %p_init) -> (tensor<4096x4096xf32>) {
    // Extract slice from QKT: 4096x16
    %QKT_slice = tensor.extract_slice %QKT[0, %k][4096, 16][1, 1] -> tensor<4096x16xf32>

    // Fused center+exp: 4096x16
    %exp_slice = linalg.generic {iterator_types = ["parallel", "parallel"]}
                 ins(%QKT_slice, %max : tensor<4096x16xf32>, tensor<4096xf32>) outs(%empty_slice : tensor<4096x16xf32>) {
      ^bb0(%in: f32, %max_val: f32, %out: f32):
        %centered = arith.subf %in, %max_val : f32
        %exp_val = math.exp %centered : f32
        linalg.yield %exp_val : f32
    } -> tensor<4096x16xf32>

    // Division: 4096x16
    %p_slice = linalg.generic {iterator_types = ["parallel", "parallel"]}
               ins(%exp_slice, %sum : tensor<4096x16xf32>, tensor<4096xf32>) outs(%empty_slice : tensor<4096x16xf32>) {
      ^bb0(%exp_val: f32, %sum_val: f32, %out: f32):
        %result = arith.divf %exp_val, %sum_val : f32
        linalg.yield %result : f32
    } -> tensor<4096x16xf32>

    // Insert slice back: 4096x16 -> 4096x4096
    %p_new = tensor.insert_slice %p_slice into %p_acc[0, %k][4096, 16][1, 1] -> tensor<4096x4096xf32>

    scf.yield %p_new : tensor<4096x4096xf32>
  }
  // Result: %p contains softmax(Q @ K^T) with shape 4096x4096


  // === Final matmul TILED in K dimension (256 iterations) ===
  // Initialize output with zeros: 4096x64
  %o_init = linalg.fill ins(%c0f : f32) outs(%empty_out : tensor<4096x64xf32>) -> tensor<4096x64xf32>

  %o = scf.for %k = %c0 to %c4096 step %c16 iter_args(%o_acc = %o_init) -> (tensor<4096x64xf32>) {
    // Extract slice from %p: 4096x16
    %p_slice = tensor.extract_slice %p[0, %k][4096, 16][1, 1] -> tensor<4096x16xf32>

    // Extract slice from %v: 16x64
    %v_slice = tensor.extract_slice %v[%k, 0][16, 64][1, 1] -> tensor<16x64xf32>

    // Partial matmul: (4096x16) @ (16x64) -> 4096x64
    %partial = linalg.matmul ins(%p_slice, %v_slice : tensor<4096x16xf32>, tensor<16x64xf32>)
                             outs(%empty_partial : tensor<4096x64xf32>) -> tensor<4096x64xf32>

    // Accumulate: 4096x64 + 4096x64 -> 4096x64
    %o_new = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
             ins(%o_acc, %partial : tensor<4096x64xf32>, tensor<4096x64xf32>) ... -> tensor<4096x64xf32>

    scf.yield %o_new : tensor<4096x64xf32>
  }
  ...
}
```
