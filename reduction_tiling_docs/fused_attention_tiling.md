## Attention Tiling

# Linalg level implementation

Input sizes:
- Q: 4096x64
- K: 4096x64
- V: 4096x64

```
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

# Stage 1: Tile the last matmul in K dim.
