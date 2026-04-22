# Softmax Lowering Flow: IR Transformation Stages

**Input Shape**: `1024x512xf32` (1024 rows, 512 columns)
**Softmax Dimension**: dim=1 (along the 512-element rows)

---

## Stage 1: Initial IR

Single high-level `linalg.softmax` operation on the full tensor.

```mlir
func.func @payload(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) {
  %1 = bufferization.to_tensor %arg1 : tensor<1024x512xf32>

  // Single softmax op over entire tensor
  %3 = linalg.softmax dimension(1) ins(%1 : tensor<1024x512xf32>)
                                   outs(%2 : tensor<1024x512xf32>) -> tensor<1024x512xf32>

  bufferization.materialize_in_destination %3 in %arg0
}
```

---

## Stage 2: After Tiling Parallel Dim

Parallel dimension (rows) tiled into 16 chunks of 64 rows each. Introduces `scf.forall` for parallel execution.

```mlir
func.func @payload(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) {
  // Parallel loop over 16 tiles (1024 / 64 = 16)
  %3 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %2) -> (tensor<1024x512xf32>) {
    %4 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg2)
    %slice = tensor.extract_slice %1[%4, 0] [64, 512] [1, 1]

    // Softmax on 64x512 slice
    %5 = linalg.softmax dimension(1) ins(%slice : tensor<64x512xf32>)
                                     outs(%slice_0 : tensor<64x512xf32>) -> tensor<64x512xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg3[%4, 0] [64, 512] [1, 1]
    }
  }
}
```

---

## Stage 3: After Decomposing Softmax

Softmax decomposed into 4 operations: max reduction → center+exp → sum reduction → division.

```mlir
func.func @payload(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) {
  %3 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %2) -> (tensor<1024x512xf32>) {
    %slice = tensor.extract_slice %1[%4, 0] [64, 512] [1, 1]

    // 1. Max reduction: (64,512) -> (64,)
    %7 = linalg.generic {indexing_maps = [map<(d0,d1) -> (d0,d1)>, map<(d0,d1) -> (d0)>],
                         iterator_types = ["parallel", "reduction"]}
         ins(%slice : tensor<64x512xf32>) outs(%6 : tensor<64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = arith.maxnumf %in, %out : f32
        linalg.yield %12 : f32
    } -> tensor<64xf32>

    // 2. Center and exp: (64,512) -> (64,512)
    %8 = linalg.generic {indexing_maps = [map<(d0,d1) -> (d0,d1)>, map<(d0,d1) -> (d0)>, map<(d0,d1) -> (d0,d1)>],
                         iterator_types = ["parallel", "parallel"]}
         ins(%slice, %7 : tensor<64x512xf32>, tensor<64xf32>) outs(%slice_0 : tensor<64x512xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %12 = arith.subf %in, %in_2 : f32
        %13 = math.exp %12 : f32
        linalg.yield %13 : f32
    } -> tensor<64x512xf32>

    // 3. Sum reduction: (64,512) -> (64,)
    %10 = linalg.generic {indexing_maps = [map<(d0,d1) -> (d0,d1)>, map<(d0,d1) -> (d0)>],
                          iterator_types = ["parallel", "reduction"]}
          ins(%8 : tensor<64x512xf32>) outs(%9 : tensor<64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %12 = arith.addf %in, %out : f32
        linalg.yield %12 : f32
    } -> tensor<64xf32>

    // 4. Division: (64,512) -> (64,512)
    %11 = linalg.generic {indexing_maps = [map<(d0,d1) -> (d0,d1)>, map<(d0,d1) -> (d0)>, map<(d0,d1) -> (d0,d1)>],
                          iterator_types = ["parallel", "parallel"]}
          ins(%8, %10 : tensor<64x512xf32>, tensor<64xf32>) outs(%slice_0 : tensor<64x512xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %12 = arith.divf %in, %in_2 : f32
        linalg.yield %12 : f32
    } -> tensor<64x512xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg3[%4, 0] [64, 512] [1, 1]
    }
  }
}
```

**Loop structure:**
- **Outer**: `scf.forall` with 16 parallel iterations
- Each iteration performs 4 sequential linalg ops

**Key operations:**
1. **Max reduction**: Reduces from `(64, 512)` to `(64,)` using `maxnumf`
2. **Center+exp**: Element-wise subtract max and apply exp
3. **Sum reduction**: Reduces from `(64, 512)` to `(64,)` using `addf`
4. **Division**: Element-wise divide by sum

---

## Stage 4: After Tiling Division

Division operation tiled along dimension 1 into chunks of 16 columns.

```mlir
func.func @payload(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) {
  %2 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %1) -> (tensor<1024x512xf32>) {
    %slice = tensor.extract_slice %0[%3, 0] [64, 512] [1, 1]

    // Max reduction (64,512) -> (64,)
    %6 = linalg.generic {iterator_types = ["parallel", "reduction"]}
         ins(%slice) outs(%5) { maxnumf } -> tensor<64xf32>

    // Center and exp (64,512) -> (64,512)
    %7 = linalg.generic {iterator_types = ["parallel", "parallel"]}
         ins(%slice, %6) outs(%slice_1) { subf, exp } -> tensor<64x512xf32>

    // Sum reduction (64,512) -> (64,)
    %9 = linalg.generic {iterator_types = ["parallel", "reduction"]}
         ins(%7) outs(%8) { addf } -> tensor<64xf32>

    // Division tiled over columns: loop from 0 to 512 step 16
    %10 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %slice_1) -> (tensor<64x512xf32>) {
      %slice_2 = tensor.extract_slice %7[0, %arg4] [64, 16] [1, 1]
      %slice_3 = tensor.extract_slice %arg5[0, %arg4] [64, 16] [1, 1]

      // Division on 64x16 tile
      %11 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%slice_2, %9 : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_3 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %12 = arith.divf %in, %in_4 : f32
          linalg.yield %12 : f32
      } -> tensor<64x16xf32>

      %inserted = tensor.insert_slice %11 into %arg5[0, %arg4] [64, 16] [1, 1]
      scf.yield %inserted : tensor<64x512xf32>
    }

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg3[%3, 0] [64, 512] [1, 1]
    }
  }
}
```

**Loop structure:**
- **Outer**: `scf.forall` with 16 parallel iterations (64-row tiles)
- **Inner**: `scf.for` with 32 sequential iterations (512/16 = 32 column tiles)

**Key change:** Division now operates on `64x16` tiles instead of full `64x512`

---

## Stage 5: After Fusing Max+Center+Exp into Division Loop

The center-and-exp computation is fused into the division loop to recompute values on-the-fly.

```mlir
func.func @payload(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) {
  %2 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %1) -> (tensor<1024x512xf32>) {
    %slice = tensor.extract_slice %0[%3, 0] [64, 512] [1, 1]

    // Max reduction (64,512) -> (64,)
    %6 = linalg.generic {iterator_types = ["parallel", "reduction"]}
         ins(%slice) outs(%5) { maxnumf } -> tensor<64xf32>

    // Center and exp (still materialized for sum reduction)
    %7 = linalg.generic {iterator_types = ["parallel", "parallel"]}
         ins(%slice, %6) outs(%slice_1) { subf, exp } -> tensor<64x512xf32>

    // Sum reduction (64,512) -> (64,)
    %9 = linalg.generic {iterator_types = ["parallel", "reduction"]}
         ins(%7) outs(%8) { addf } -> tensor<64xf32>

    // Division loop with fused center+exp+div
    %10 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %slice_1) -> (tensor<64x512xf32>) {
      %slice_2 = tensor.extract_slice %slice[0, %arg4] [64, 16] [1, 1]  // from original input
      %slice_3 = tensor.extract_slice %arg5[0, %arg4] [64, 16] [1, 1]

      // Fused center+exp on 64x16 tile
      %11 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%slice_2, %6 : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_3 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %13 = arith.subf %in, %in_4 : f32
          %14 = math.exp %13 : f32
          linalg.yield %14 : f32
      } -> tensor<64x16xf32>

      // Division on 64x16 tile
      %12 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%11, %9 : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_3 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %13 = arith.divf %in, %in_4 : f32
          linalg.yield %13 : f32
      } -> tensor<64x16xf32>

      %inserted = tensor.insert_slice %12 into %arg5[0, %arg4] [64, 16] [1, 1]
      scf.yield %inserted : tensor<64x512xf32>
    }

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg3[%3, 0] [64, 512] [1, 1]
    }
  }
}
```

**Loop structure:**
- **Outer**: `scf.forall` with 16 parallel iterations
- **Inner**: `scf.for` with 32 sequential iterations

**Key change:** Inside the division loop, center+exp is recomputed on `64x16` tiles from the original input, avoiding the need to store the full `64x512` exp tensor.

---

## Stage 6: After Tiling Sum Reduction

Sum reduction tiled into chunks of 16 columns, introducing partial sums followed by a final reduction.

```mlir
func.func @payload(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) {
  %2 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %1) -> (tensor<1024x512xf32>) {
    %slice = tensor.extract_slice %0[%3, 0] [64, 512] [1, 1]

    // Max reduction (64,512) -> (64,)
    %6 = linalg.generic {iterator_types = ["parallel", "reduction"]}
         ins(%slice) outs(%5) { maxnumf } -> tensor<64xf32>

    // Center and exp (64,512) -> (64,512)
    %7 = linalg.generic {iterator_types = ["parallel", "parallel"]}
         ins(%slice, %6) outs(%slice_1) { subf, exp } -> tensor<64x512xf32>

    // Tiled sum reduction: accumulate into 64x16 buffer
    %11 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %10) -> (tensor<64x16xf32>) {
      %slice_2 = tensor.extract_slice %7[0, %arg4] [64, 16] [1, 1]

      // Accumulate sums
      %13 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%slice_2 : tensor<64x16xf32>) outs(%arg5 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %out: f32):
          %14 = arith.addf %in, %out : f32
          linalg.yield %14 : f32
      } -> tensor<64x16xf32>

      scf.yield %13 : tensor<64x16xf32>
    }

    // Final reduction: (64,16) -> (64,)
    %reduced = linalg.reduce ins(%11 : tensor<64x16xf32>) outs(%8 : tensor<64xf32>) dimensions = [1] {
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    }

    // Division loop (same as before)
    %12 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %slice_1) -> (tensor<64x512xf32>) {
      // ... fused center+exp+div ...
    }

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %12 into %arg3[%3, 0] [64, 512] [1, 1]
    }
  }
}
```

**Loop structure:**
- **Outer**: `scf.forall` with 16 parallel iterations
- **Sum reduction loop**: `scf.for` with 32 iterations, accumulating into `64x16`
- **Final reduction**: `linalg.reduce` from `(64, 16)` to `(64,)`
- **Division loop**: `scf.for` with 32 iterations

**Key change:** Sum reduction split into partial accumulation (loop) + final reduction (linalg.reduce)

---

## Stage 7: After Fusing Max+Center+Exp into Sum Reduction Loop

The center-and-exp computation is now fused into the sum reduction loop as well.

```mlir
func.func @payload(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) {
  %2 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %1) -> (tensor<1024x512xf32>) {
    %slice = tensor.extract_slice %0[%3, 0] [64, 512] [1, 1]

    // Max reduction (64,512) -> (64,)
    %6 = linalg.generic {iterator_types = ["parallel", "reduction"]}
         ins(%slice) outs(%5) { maxnumf } -> tensor<64xf32>

    // Sum reduction loop with fused center+exp
    %10 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %9) -> (tensor<64x16xf32>) {
      %slice_2 = tensor.extract_slice %slice[0, %arg4] [64, 16] [1, 1]

      // Fused center+exp
      %12 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%slice_2, %6 : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_3 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %14 = arith.subf %in, %in_4 : f32
          %15 = math.exp %14 : f32
          linalg.yield %15 : f32
      } -> tensor<64x16xf32>

      // Accumulate into sum buffer
      %13 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%12 : tensor<64x16xf32>) outs(%arg5 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %out: f32):
          %14 = arith.addf %in, %out : f32
          linalg.yield %14 : f32
      } -> tensor<64x16xf32>

      scf.yield %13 : tensor<64x16xf32>
    }

    // Final reduction: (64,16) -> (64,)
    %reduced = linalg.reduce ins(%10 : tensor<64x16xf32>) outs(%7 : tensor<64xf32>) dimensions = [1] {
      (%in: f32, %init: f32) {
        %12 = arith.addf %in, %init : f32
        linalg.yield %12 : f32
      }
    }

    // Division loop with fused center+exp+div
    %11 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %slice_1) -> (tensor<64x512xf32>) {
      %slice_2 = tensor.extract_slice %slice[0, %arg4] [64, 16] [1, 1]

      // Fused center+exp
      %12 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%slice_2, %6 : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_3 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %14 = arith.subf %in, %in_4 : f32
          %15 = math.exp %14 : f32
          linalg.yield %15 : f32
      } -> tensor<64x16xf32>

      // Division
      %13 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%12, %reduced : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_3 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %14 = arith.divf %in, %in_4 : f32
          linalg.yield %14 : f32
      } -> tensor<64x16xf32>

      %inserted = tensor.insert_slice %13 into %arg5[0, %arg4] [64, 16] [1, 1]
      scf.yield %inserted : tensor<64x512xf32>
    }

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg3[%3, 0] [64, 512] [1, 1]
    }
  }
}
```

**Loop structure:**
- **Outer**: `scf.forall` with 16 parallel iterations
- **Sum reduction loop**: `scf.for` with 32 iterations (fused center+exp+accumulate)
- **Final reduction**: `linalg.reduce`
- **Division loop**: `scf.for` with 32 iterations (fused center+exp+div)

**Key change:** Center+exp is recomputed twice (once for sum, once for division) to avoid storing intermediate `64x512` tensor.

---

## Stage 8: After Tiling Max Reduction

Max reduction also tiled into 16-column chunks with partial max followed by final reduction.

```mlir
func.func @payload(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) {
  %2 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %1) -> (tensor<1024x512xf32>) {
    %slice = tensor.extract_slice %0[%3, 0] [64, 512] [1, 1]

    // Tiled max reduction: accumulate into 64x16 buffer
    %8 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %7) -> (tensor<64x16xf32>) {
      %slice_7 = tensor.extract_slice %slice[0, %arg4] [64, 16] [1, 1]

      // Max accumulation
      %14 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%slice_7 : tensor<64x16xf32>) outs(%slice_8 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.maxnumf %in, %out : f32
          linalg.yield %15 : f32
      } -> tensor<64x16xf32>

      %inserted = tensor.insert_slice %14 into %arg5[0, 0] [64, 16] [1, 1]
      scf.yield %inserted : tensor<64x16xf32>
    }

    // Final max reduction: (64,16) -> (64,)
    %reduced = linalg.reduce ins(%8 : tensor<64x16xf32>) outs(%5 : tensor<64xf32>) dimensions = [1] {
      (%in: f32, %init: f32) {
        %14 = arith.maxnumf %in, %init : f32
        linalg.yield %14 : f32
      }
    }

    // Sum reduction loop with fused center+exp
    %12 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %11) -> (tensor<64x16xf32>) {
      %slice_7 = tensor.extract_slice %slice[0, %arg4] [64, 16] [1, 1]

      // Fused center+exp using reduced max
      %14 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%slice_7, %reduced : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_8 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_9: f32, %out: f32):
          %16 = arith.subf %in, %in_9 : f32
          %17 = math.exp %16 : f32
          linalg.yield %17 : f32
      } -> tensor<64x16xf32>

      // Sum accumulation
      %15 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%14 : tensor<64x16xf32>) outs(%arg5 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %out: f32):
          %16 = arith.addf %in, %out : f32
          linalg.yield %16 : f32
      } -> tensor<64x16xf32>

      scf.yield %15 : tensor<64x16xf32>
    }

    // Final sum reduction: (64,16) -> (64,)
    %reduced_6 = linalg.reduce ins(%12 : tensor<64x16xf32>) outs(%9 : tensor<64xf32>) dimensions = [1] {
      (%in: f32, %init: f32) {
        %14 = arith.addf %in, %init : f32
        linalg.yield %14 : f32
      }
    }

    // Division loop with fused center+exp+div
    %13 = scf.for %arg4 = %c0 to %c512 step %c16 iter_args(%arg5 = %slice_1) -> (tensor<64x512xf32>) {
      %slice_7 = tensor.extract_slice %slice[0, %arg4] [64, 16] [1, 1]

      // Fused center+exp
      %14 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%slice_7, %reduced : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_8 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_9: f32, %out: f32):
          %16 = arith.subf %in, %in_9 : f32
          %17 = math.exp %16 : f32
          linalg.yield %17 : f32
      } -> tensor<64x16xf32>

      // Division
      %15 = linalg.generic {iterator_types = ["parallel", "parallel"]}
            ins(%14, %reduced_6 : tensor<64x16xf32>, tensor<64xf32>) outs(%slice_8 : tensor<64x16xf32>) {
        ^bb0(%in: f32, %in_9: f32, %out: f32):
          %16 = arith.divf %in, %in_9 : f32
          linalg.yield %16 : f32
      } -> tensor<64x16xf32>

      %inserted = tensor.insert_slice %15 into %arg5[0, %arg4] [64, 16] [1, 1]
      scf.yield %inserted : tensor<64x512xf32>
    }

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg3[%3, 0] [64, 512] [1, 1]
    }
  }
}
```

**Loop structure:**
- **Outer**: `scf.forall` with 16 parallel iterations (64-row tiles)
- **Max reduction loop**: `scf.for` with 32 iterations → `linalg.reduce`
- **Sum reduction loop**: `scf.for` with 32 iterations → `linalg.reduce`
- **Division loop**: `scf.for` with 32 iterations

**Key change:** All three stages (max, sum, div) now use tiled loops operating on `64x16` chunks.

---

## Stage 9: Final Vectorized XeGPU Version

After vectorization, bufferization, and conversion to XeGPU operations. Uses shared local memory (SLM) for partial reductions.

```mlir
gpu.module @payload_kernel {
  gpu.func @payload_kernel(%arg0: memref<1024x512xf32>, %arg1: memref<1024x512xf32>) kernel {
    %block_id_x = gpu.block_id x
    %0 = arith.muli %block_id_x, %c64 : index
    %subview = memref.subview %arg0[%0, 0] [64, 512] [1, 1]

    // Allocate SLM buffer for partial reductions
    %alloca = memref.alloca() : memref<64x16xf32, 3>
    %1 = xegpu.create_mem_desc %alloca : !xegpu.mem_desc<64x16xf32>

    // Max reduction loop
    xegpu.store_matrix %cst_2, %1[0, 0]  // init with -inf
    scf.for %arg2 = %c0 to %c512 step %c16 {
      // Load 64x16 tile from global memory
      %6 = xegpu.create_nd_tdesc %arg1 : !xegpu.tensor_desc<64x16xf32>
      %7 = xegpu.load_nd %6[%0, %arg2] : vector<64x16xf32>

      // Load partial max from SLM, compute max, store back
      %8 = xegpu.load_matrix %1[0, 0] : vector<64x16xf32>
      %9 = arith.maxnumf %7, %8 : vector<64x16xf32>
      xegpu.store_matrix %9, %1[0, 0]
    }

    // Final max reduction across 16 columns
    %2 = xegpu.load_matrix %1[0, 0] : vector<64x16xf32>
    %3 = vector.multi_reduction <maxnumf>, %2, %cst_1 [1] : vector<64x16xf32> to vector<64xf32>

    // Sum reduction loop
    xegpu.store_matrix %cst_0, %1[0, 0]  // init with 0.0
    scf.for %arg2 = %c0 to %c512 step %c16 {
      // Load 64x16 tile
      %6 = xegpu.create_nd_tdesc %arg1 : !xegpu.tensor_desc<64x16xf32>
      %7 = xegpu.load_nd %6[%0, %arg2] : vector<64x16xf32>

      // Fused center+exp
      %8 = vector.broadcast %3 : vector<64xf32> to vector<16x64xf32>
      %9 = vector.transpose %8, [1, 0] : vector<64x16xf32>
      %10 = arith.subf %7, %9 : vector<64x16xf32>
      %11 = math.exp %10 : vector<64x16xf32>

      // Accumulate sum in SLM
      %12 = xegpu.load_matrix %1[0, 0] : vector<64x16xf32>
      %13 = arith.addf %11, %12 : vector<64x16xf32>
      xegpu.store_matrix %13, %1[0, 0]
    }

    // Final sum reduction across 16 columns
    %4 = xegpu.load_matrix %1[0, 0] : vector<64x16xf32>
    %5 = vector.multi_reduction <add>, %4, %cst [1] : vector<64x16xf32> to vector<64xf32>

    // Division loop
    scf.for %arg2 = %c0 to %c512 step %c16 {
      // Load 64x16 tile
      %6 = xegpu.create_nd_tdesc %arg1 : !xegpu.tensor_desc<64x16xf32>
      %7 = xegpu.load_nd %6[%0, %arg2] : vector<64x16xf32>

      // Fused center+exp
      %8 = vector.broadcast %3 : vector<64xf32> to vector<16x64xf32>
      %9 = vector.transpose %8, [1, 0] : vector<64x16xf32>
      %10 = arith.subf %7, %9 : vector<64x16xf32>
      %11 = math.exp %10 : vector<64x16xf32>

      // Division
      %12 = vector.broadcast %5 : vector<64xf32> to vector<16x64xf32>
      %13 = vector.transpose %12, [1, 0] : vector<64x16xf32>
      %14 = arith.divf %11, %13 : vector<64x16xf32>

      // Store result to global memory
      %18 = xegpu.create_nd_tdesc %intptr : !xegpu.tensor_desc<64x16xf32>
      xegpu.store_nd %14, %18[0, %arg2]
    }

    gpu.return
  }
}
```

**Loop structure:**
- **Grid**: 16 blocks (one per 64-row tile)
- **Per block**:
  - **Max reduction loop**: 32 iterations (512/16)
  - **Final max reduction**: `vector.multi_reduction`
  - **Sum reduction loop**: 32 iterations (512/16)
  - **Final sum reduction**: `vector.multi_reduction`
  - **Division loop**: 32 iterations (512/16)

**Key transformations:**
- Linalg operations → vectorized operations on `vector<64x16xf32>`
- Tensor buffers → SLM allocation (`memref<64x16xf32, 3>`)
- Memory operations → XeGPU load/store operations (`xegpu.load_nd`, `xegpu.store_nd`, `xegpu.load_matrix`, `xegpu.store_matrix`)
- GPU kernel launch with 16 blocks × 128 threads

---

## Summary of Transformations

| Stage | Key Transformation | Loop Structure |
|-------|-------------------|----------------|
| 1 | Initial high-level softmax | No loops |
| 2 | Tile parallel dimension | `scf.forall(16)` |
| 3 | Decompose softmax | `scf.forall(16)` + 4 sequential ops |
| 4 | Tile division | `scf.forall(16)` → `scf.for(32)` |
| 5 | Fuse into division loop | Recompute center+exp in div loop |
| 6 | Tile sum reduction | Add sum loop + final reduction |
| 7 | Fuse into sum loop | Recompute center+exp in sum loop |
| 8 | Tile max reduction | Add max loop + final reduction |
| 9 | Vectorize + XeGPU | GPU kernel with SLM and vector ops |

**Final computation pattern per GPU block:**
1. **Max reduction**: 32-iteration loop with SLM accumulation → final reduction
2. **Sum reduction**: 32-iteration loop (fused center+exp) with SLM accumulation → final reduction
3. **Division**: 32-iteration loop (fused center+exp+div) writing to global memory

This progressive lowering enables efficient GPU execution with:
- Parallelism across 64-row tiles
- SLM for partial reduction storage
- Recomputation of center+exp to reduce memory traffic
- Vectorized 64x16 tile operations
