# Linalg softmax lowering to XeGPU (Currently supported in lighthouse)

## Overview

**Assumptions:**
Softmax dimension size is small (64 in this example). 

The lowering process consists of seven stages:
1. **initial** - High-level tensor operations
2. **tiled-softmax** - Tiled softmax operations
3. **decomposed** - Decomposition into constituent operations
4. **vectorized** - Vector operations
5. **bufferized** - Memory-based representation
6. **xegpu-initial** - GPU kernel with XeGPU operations
7. **xegpu-wg** - Work-group optimized XeGPU

---

## Stage 1: Initial

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  // ...
  %2 = tensor.empty() : tensor<1024x64xf32>
  %3 = linalg.softmax dimension(1) ins(%1 : tensor<1024x64xf32>) 
                                  outs(%2 : tensor<1024x64xf32>) -> tensor<1024x64xf32>
  // ...
  return
}
```
---

## Stage 2: Tiled Softmax

**Notes**
- Work distribution via `scf.forall` (16 parallel iterations)
- Each tile processes 64x64 elements

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  // ...
  %3 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %2) -> (tensor<1024x64xf32>) {
    %4 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg2)
    // Extract 64x64 input slice
    %extracted_slice = tensor.extract_slice ...
    // Extract 64x64 output slice
    %extracted_slice_0 = tensor.extract_slice ...
    // Apply softmax to the tile
    %5 = linalg.softmax dimension(1) ins(%extracted_slice : tensor<64x64xf32>) 
                                     outs(%extracted_slice_0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg3[%4, %c0] [64, 64] [1, 1] : 
        tensor<64x64xf32> into tensor<1024x64xf32>
    }
  }
  // ...  
  return
}
```

---

## Stage 3: Decomposed

**Notes**
- Softmax decomposed into 4 constituent `linalg.generic` ops : max, sub+exp, sum, divide
- Uses `structured.structured_decompose_interface` implemented by `linalg.softmax`

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  // ...
  
  %2 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %1) -> (tensor<1024x64xf32>) {
    %3 = affine.apply #map(%arg2)  // %3 = %arg2 * 64
    %extracted_slice = tensor.extract_slice ...
    
    // Step 1: Find max along dimension 1
    %4 = tensor.empty() : tensor<64xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
    %6 = linalg.generic // ...
      %11 = arith.maxnumf %in, %out : f32
      // ...
    } -> tensor<64xf32>
    
    // Step 2: Subtract max and exponentiate
    %7 = linalg.generic // ...
      %11 = arith.subf %in, %in_2 : f32
      %12 = math.exp %11 : f32
      // ...
    } -> tensor<64x64xf32>
    
    // Step 3: Sum exponentials
    %8 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
    %9 = linalg.generic // ...
      %11 = arith.addf %in, %out : f32
      // ...
    } -> tensor<64xf32>
    
    // Step 4: Normalize by sum
    %10 = linalg.generic // ...
      %11 = arith.divf %in, %in_2 : f32
      // ...
    } -> tensor<64x64xf32>
    
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg3[%3, 0] [64, 64] [1, 1] : 
        tensor<64x64xf32> into tensor<1024x64xf32>
    }
  }
  return
}
```

---

## Stage 4: Vectorized

**Notes**
- `linalg.generic` operations replaced with vector operations
- Vector transfers for reading/writing data

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  // ...  
  %3 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %2) -> (tensor<1024x64xf32>) {
    %4 = affine.apply #map(%arg2)  // %4 = %arg2 * 64
    %extracted_slice = tensor.extract_slice ..
    
    // Vector read: Load 64x64 tile
    %5 = vector.transfer_read %1[%4, %c0], %0 {in_bounds = [true, true]} : 
      tensor<1024x64xf32>, vector<64x64xf32>
    
    // Max reduction: Reduce dimension 1 -> vector<64xf32>
    %6 = vector.multi_reduction <maxnumf>, %5, %cst_0 [1] : 
      vector<64x64xf32> to vector<64xf32>
    
    // Broadcast max values back to 64x64 and transpose
    %7 = vector.broadcast %6 : vector<64xf32> to vector<64x64xf32>
    %8 = vector.transpose %7, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
    
    // Subtract max and exponentiate
    %9 = arith.subf %5, %8 : vector<64x64xf32>
    %10 = math.exp %9 : vector<64x64xf32>
    
    // Sum reduction: Reduce dimension 1 -> vector<64xf32>
    %11 = vector.multi_reduction <add>, %10, %cst [1] : 
      vector<64x64xf32> to vector<64xf32>
    
    // Broadcast sums back to 64x64 and transpose
    %12 = vector.broadcast %11 : vector<64xf32> to vector<64x64xf32>
    %13 = vector.transpose %12, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
    
    // Normalize
    %14 = arith.divf %10, %13 : vector<64x64xf32>
    
    // Vector write
    %15 = vector.transfer_write %14, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : 
      vector<64x64xf32>, tensor<64x64xf32>
    
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %15 into %arg3[%4, 0] [64, 64] [1, 1]
    }
  }
  return
}
```
---

## Stage 5: Bufferized

**Notes**
- Tensors eliminated, working directly with memrefs

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  // ...
  
  scf.forall (%arg2) in (16) {
    %1 = affine.apply #map(%arg2)  // %1 = %arg2 * 64
    
    // Direct memref read
    %2 = vector.transfer_read %arg1[%1, %c0], %0 {in_bounds = [true, true]} : 
      memref<1024x64xf32>, vector<64x64xf32>
    
    // Max reduction
    %3 = vector.multi_reduction <maxnumf>, %2, %cst_0 [1] : 
      vector<64x64xf32> to vector<64xf32>
    %4 = vector.broadcast %3 : vector<64xf32> to vector<64x64xf32>
    %5 = vector.transpose %4, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
    
    // Subtract and exp
    %6 = arith.subf %2, %5 : vector<64x64xf32>
    %7 = math.exp %6 : vector<64x64xf32>
    
    // Sum reduction
    %8 = vector.multi_reduction <add>, %7, %cst [1] : 
      vector<64x64xf32> to vector<64xf32>
    %9 = vector.broadcast %8 : vector<64xf32> to vector<64x64xf32>
    %10 = vector.transpose %9, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
    
    // Normalize
    %11 = arith.divf %7, %10 : vector<64x64xf32>
    
    // Direct memref write
    vector.transfer_write %11, %arg0[%1, %c0] {in_bounds = [true, true]} : 
      vector<64x64xf32>, memref<1024x64xf32>
  }
  return
}
```

---

## Stage 6: XeGPU-Initial

**Notes**
- GPU kernel separated from host code (Gpu Outlining)
- `gpu.launch_func` invocation with grid/block dimensions
- Use `vector-to-xegpu`

**Code:**

**Host Side:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  // ...
  gpu.launch_func @payload_kernel::@payload_kernel 
    blocks in (%c16, %c1, %c1) 
    threads in (%c128, %c1, %c1)
    args(%arg1 : memref<1024x64xf32>, %arg0 : memref<1024x64xf32>)
  return
}
```

**GPU Kernel:**
```mlir
gpu.module @payload_kernel [#xevm.target<O = 3>] {
  gpu.func @payload_kernel(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) kernel 
    attributes {known_block_size = array<i32: 128, 1, 1>, 
                known_grid_size = array<i32: 16, 1, 1>} {
    // ...
    %block_id_x = gpu.block_id x
    %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
    
    // Create XeGPU tensor descriptor for load
    %1 = xegpu.create_nd_tdesc %arg0 : memref<1024x64xf32> -> 
      !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
    
    // XeGPU block load
    %2 = xegpu.load_nd %1[%0, 0] : 
      !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> 
      vector<64x64xf32>
    
    // Same compute operations as before
    %3 = vector.multi_reduction <maxnumf>, %2, %cst_0 [1] : 
      vector<64x64xf32> to vector<64xf32>
    %4 = vector.broadcast %3 : vector<64xf32> to vector<64x64xf32>
    %5 = vector.transpose %4, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
    %6 = arith.subf %2, %5 : vector<64x64xf32>
    %7 = math.exp %6 : vector<64x64xf32>
    %8 = vector.multi_reduction <add>, %7, %cst [1] : 
      vector<64x64xf32> to vector<64xf32>
    %9 = vector.broadcast %8 : vector<64xf32> to vector<64x64xf32>
    %10 = vector.transpose %9, [1, 0] : vector<64x64xf32> to vector<64x64xf32>
    %11 = arith.divf %7, %10 : vector<64x64xf32>
    
    // Create XeGPU tensor descriptor for store
    %12 = xegpu.create_nd_tdesc %arg1 : memref<1024x64xf32> -> 
      !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
    
    // XeGPU block store
    xegpu.store_nd %11, %12[%0, 0] : 
      vector<64x64xf32>, 
      !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
    
    gpu.return
  }
}
```

---

## Stage 7: XeGPU-WG (Work-Group Optimized)

**Notes**
- Sets the layout for anchor xegpu ops. Each Wg consistes of [8, 1] subgroups
  doing 8x64 softmax slice. 
- Only sets the layotu for `store_nd`. Layout propagation does the rest.  

**Code (differences from xegpu-initial):**
```mlir
// Store operation now includes layout hints
xegpu.store_nd %11, %12[%0, 0] 
  <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 64]>}> : 
  vector<64x64xf32>, 
  !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
```

---

# Supporting larger Softmax dimension sizes

Previsouly we tiled all ops in the parallel dimension only (i.e. non softmax dim). Handling a larger softmax dimension require tiling the softmax contituent ops in that dimension. 

**Approach:** Tile reductions along dimension 1 (using appropriate step size) and fuse producers into consumers to enable streaming computation.

---

## Decomposed → Tiled: Stage A - Tile div op

**Notes:**
- Tile the division operation with step size 16 along dimension 1
- Creates `scf.for` loop iterating over 64 elements in chunks of 16

**Key Changes:**
```mlir
// Before: Single division linalg.generic over 64x64
scf.forall ... {
  // Max, Center+Exp, Sum ops ...
  %11 = linalg.generic {...} ins(%8, %10 : tensor<64x64xf32>, tensor<64xf32>) outs(%extracted_slice_0 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %12 = arith.divf %in, %in_2 : f32
      linalg.yield %12 : f32
    } -> tensor<64x64xf32>
}

// After: Division tiled into 64x16 chunks
scf.forall ... {
  %11 = scf.for %arg4 = %c0_2 to %c64 step %c16 iter_args(%arg5 = %extracted_slice_0) -> (tensor<64x64xf32>) {
      // Max, Center+Exp, Sum ops ...
      %12 = linalg.generic {...} ins(%extracted_slice_3, %extracted_slice_4 : tensor<64x16xf32>, tensor<64xf32>) outs(%extracted_slice_5 : tensor<64x16xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %13 = arith.divf %in, %in_6 : f32
        linalg.yield %13 : f32
      } -> tensor<64x16xf32>
      %inserted_slice = tensor.insert_slice %12 into %arg5[0, %arg4] [64, 16] [1, 1] : tensor<64x16xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
}
```

---

## Stage B - Fuse sub+exp into div loop

**Notes:**
- Fuse the `sub+exp` producer (max_center_and_exp_op) into the div loop
- Recomputes exp values on-the-fly instead of materializing full 64x64 tensor

**Key Changes:**
```mlir
%11 = scf.for %arg4 = %c0_2 to %c64 step %c16 iter_args(%arg5 = %extracted_slice_0) -> (tensor<64x64xf32>) {
  %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, %arg4] [64, 16] [1, 1]
  
  // Fused: sub+exp computed per 16-element chunk
  %12 = linalg.generic {...} ins(%extracted_slice_3, %extracted_slice_4 : tensor<64x16xf32>, tensor<64xf32>) 
        outs(%extracted_slice_5 : tensor<64x16xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %14 = arith.subf %in, %in_8 : f32
      %15 = math.exp %14 : f32
      linalg.yield %15 : f32
  } -> tensor<64x16xf32>
  
  // Division operation
  %13 = linalg.generic {...} ins(%12, %extracted_slice_6 : tensor<64x16xf32>, tensor<64xf32>) 
        outs(%extracted_slice_7 : tensor<64x16xf32>) { ... } -> tensor<64x16xf32>
  // ...
}
```

---

## Stage C - Tile sum reduction

**Notes:**
- Tile the sum reduction using `structured_tile_reduction_using_for`
- Creates intermediate accumulator tensor (64x16)
- Final reduction via `linalg.reduce` over dimension 1

**Key Changes:**
```mlir
// Tiled sum reduction with intermediate accumulator
%10 = tensor.empty() : tensor<64x16xf32>
%11 = linalg.fill ins(%cst_2 : f32) outs(%10 : tensor<64x16xf32>) -> tensor<64x16xf32>

%12 = scf.for %arg4 = %c0_3 to %c64 step %c16 iter_args(%arg5 = %11) -> (tensor<64x16xf32>) {
  %extracted_slice_7 = tensor.extract_slice %8[0, %arg4] [64, 16] [1, 1]
  %14 = linalg.generic {...} ins(%extracted_slice_7 : tensor<64x16xf32>) 
        outs(%extracted_slice_8 : tensor<64x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.addf %in, %out : f32
      linalg.yield %15 : f32
  } -> tensor<64x16xf32>
  // ...
}

// Final reduction to 64xf32
%reduced = linalg.reduce ins(%12 : tensor<64x16xf32>) outs(%9 : tensor<64xf32>) dimensions = [1] 
  (%in: f32, %init: f32) {
    %14 = arith.addf %in, %init : f32
    linalg.yield %14 : f32
  }
```

---

## Stage D - Fuse sub+exp into sum loop

**Notes:**
- Fuse `sub+exp` into the sum reduction loop
- Stream computation: compute exp and accumulate in same loop

**Key Changes:**
```mlir
%12 = scf.for %arg4 = %c0_3 to %c64 step %c16 iter_args(%arg5 = %11) -> (tensor<64x16xf32>) {
  %extracted_slice_7 = tensor.extract_slice %extracted_slice[0, %arg4] [64, 16] [1, 1]
  
  // Fused: sub+exp
  %14 = linalg.generic {...} ins(%extracted_slice_7, %extracted_slice_8 : tensor<64x16xf32>, tensor<64xf32>) 
        outs(%extracted_slice_9 : tensor<64x16xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %16 = arith.subf %in, %in_11 : f32
      %17 = math.exp %16 : f32
      linalg.yield %17 : f32
  } -> tensor<64x16xf32>
  
  // Accumulate sum
  %15 = linalg.generic {...} ins(%14 : tensor<64x16xf32>) 
        outs(%extracted_slice_10 : tensor<64x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.addf %in, %out : f32
      linalg.yield %16 : f32
  } -> tensor<64x16xf32>
  // ...
}
```

---

## Stage E - Tile max reduction

**Notes:**
- Tile max reduction similar to sum reduction
- Creates 64x16 intermediate accumulator
- Final reduction via `linalg.reduce` with maxnumf

**Key Changes:**
```mlir
// Tiled max reduction
%7 = tensor.empty() : tensor<64x16xf32>
%8 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<64x16xf32>) -> tensor<64x16xf32>

%9 = scf.for %arg4 = %c0_2 to %c64 step %c16 iter_args(%arg5 = %8) -> (tensor<64x16xf32>) {
  %extracted_slice_12 = tensor.extract_slice %extracted_slice[0, %arg4] [64, 16] [1, 1]
  %16 = linalg.generic {...} ins(%extracted_slice_12 : tensor<64x16xf32>) 
        outs(%extracted_slice_13 : tensor<64x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = arith.maxnumf %in, %out : f32
      linalg.yield %17 : f32
  } -> tensor<64x16xf32>
  // ...
}

// Final max reduction
%reduced = linalg.reduce ins(%9 : tensor<64x16xf32>) outs(%6 : tensor<64xf32>) dimensions = [1] 
  (%in: f32, %init: f32) {
    %16 = arith.maxnumf %in, %init : f32
    linalg.yield %16 : f32
  }
```

**Result:** Now all three major computations (max, sum, div) are tiled and operate on 64x16 chunks, with exp computation fused into both sum and div loops.

---

## Stage F - Vectorization

**Notes:**
- Convert tiled linalg operations to vector operations
- `scf.for` loops remain but operate on vectors
- Vector size: 64x16 for tiled operations

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  // ...
  %3 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %2) -> (tensor<1024x64xf32>) {
    // ...
    
    // Vectorized max reduction loop
    %6 = vector.transfer_write %cst_1, %5[%c0, %c0] : vector<64x16xf32>, tensor<64x16xf32>
    %7 = scf.for %arg4 = %c0 to %c64 step %c16 iter_args(%arg5 = %6) -> (tensor<64x16xf32>) {
      %15 = vector.transfer_read %1[%4, %arg4], %0 : tensor<1024x64xf32>, vector<64x16xf32>
      %16 = vector.transfer_read %arg5[%c0, %c0], %0 : tensor<64x16xf32>, vector<64x16xf32>
      %17 = arith.maxnumf %15, %16 : vector<64x16xf32>
      %18 = vector.transfer_write %17, %arg5[%c0, %c0] : vector<64x16xf32>, tensor<64x16xf32>
      scf.yield %18 : tensor<64x16xf32>
    }
    %8 = vector.transfer_read %7[%c0, %c0], %0 : tensor<64x16xf32>, vector<64x16xf32>
    %9 = vector.multi_reduction <maxnumf>, %8, %cst_2 [1] : vector<64x16xf32> to vector<64xf32>
    
    // Vectorized sum reduction loop with fused sub+exp
    %11 = scf.for %arg4 = %c0 to %c64 step %c16 iter_args(%arg5 = %10) -> (tensor<64x16xf32>) {
      %15 = vector.transfer_read %1[%4, %arg4], %0 : tensor<1024x64xf32>, vector<64x16xf32>
      %16 = vector.broadcast %9 : vector<64xf32> to vector<16x64xf32>
      %17 = vector.transpose %16, [1, 0] : vector<16x64xf32> to vector<64x16xf32>
      %18 = arith.subf %15, %17 : vector<64x16xf32>
      %19 = math.exp %18 : vector<64x16xf32>
      %20 = vector.transfer_read %arg5[%c0, %c0], %0 : tensor<64x16xf32>, vector<64x16xf32>
      %21 = arith.addf %19, %20 : vector<64x16xf32>
      %22 = vector.transfer_write %21, %arg5[%c0, %c0] : vector<64x16xf32>, tensor<64x16xf32>
      scf.yield %22 : tensor<64x16xf32>
    }
    %12 = vector.transfer_read %11[%c0, %c0], %0 : tensor<64x16xf32>, vector<64x16xf32>
    %13 = vector.multi_reduction <add>, %12, %cst_0 [1] : vector<64x16xf32> to vector<64xf32>
    
    // Vectorized div loop with fused sub+exp
    %14 = scf.for %arg4 = %c0 to %c64 step %c16 iter_args(%arg5 = %extracted_slice) -> (tensor<64x64xf32>) {
      %15 = vector.transfer_read %1[%4, %arg4], %0 : tensor<1024x64xf32>, vector<64x16xf32>
      %16 = vector.broadcast %9 : vector<64xf32> to vector<16x64xf32>
      %17 = vector.transpose %16, [1, 0] : vector<16x64xf32> to vector<64x16xf32>
      %18 = arith.subf %15, %17 : vector<64x16xf32>
      %19 = math.exp %18 : vector<64x16xf32>
      %20 = vector.broadcast %13 : vector<64xf32> to vector<16x64xf32>
      %21 = vector.transpose %20, [1, 0] : vector<16x64xf32> to vector<64x16xf32>
      %22 = arith.divf %19, %21 : vector<64x16xf32>
      %23 = vector.transfer_write %22, %arg5[%c0, %arg4] : vector<64x16xf32>, tensor<64x64xf32>
      scf.yield %23 : tensor<64x64xf32>
    }
  }
  // ...
}
```

---

## Stage G - Bufferization

**Notes:**
- Convert tensors to memrefs
- Allocate stack buffer for 64x16 accumulator: `memref.alloc()`

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  // ...
  scf.forall (%arg2) in (16) {
    %1 = affine.apply #map(%arg2)
    %subview = memref.subview %arg0[%1, 0] [64, 64] [1, 1]
    
    // Allocate accumulator buffer
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x16xf32>
    
    // Max reduction loop
    vector.transfer_write %cst_1, %alloc[%c0, %c0] : vector<64x16xf32>, memref<64x16xf32>
    scf.for %arg3 = %c0 to %c64 step %c16 {
      %6 = vector.transfer_read %arg1[%1, %arg3], %0 : memref<1024x64xf32>, vector<64x16xf32>
      %7 = vector.transfer_read %alloc[%c0, %c0], %0 : memref<64x16xf32>, vector<64x16xf32>
      %8 = arith.maxnumf %6, %7 : vector<64x16xf32>
      vector.transfer_write %8, %alloc[%c0, %c0] : vector<64x16xf32>, memref<64x16xf32>
    }
    %2 = vector.transfer_read %alloc[%c0, %c0], %0 : memref<64x16xf32>, vector<64x16xf32>
    %3 = vector.multi_reduction <maxnumf>, %2, %cst_2 [1] : vector<64x16xf32> to vector<64xf32>
    
    // Sum reduction loop (reuses %alloc)
    // ...
    
    // Div loop (writes to %subview)
    // ...
  }
}
```

---

## Stage H - Promote buffers to stack

**Notes:**
- Convert `memref.alloc()` to `memref.alloca()` for stack allocation
- Reduces memory allocation overhead

**Code:**
```mlir
scf.forall (%arg2) in (16) {
  %1 = affine.apply #map(%arg2)
  %subview = memref.subview %arg0[%1, 0] [64, 64] [1, 1]
  
  // Stack allocation instead of heap
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<64x16xf32>
  
  // ... same operations using %alloca ...
}
```

---

## Stage I - GPU outlining

**Notes:**
- Convert `scf.forall` to `scf.parallel`, then to `gpu.launch`
- Extract GPU kernel into separate `gpu.module`
- Set thread count: 128 threads = (64 rows / 8 sg_rows) × 16 subgroup_size

**Host Side:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  gpu.launch_func @payload_kernel::@payload_kernel 
    blocks in (%c16, %c1, %c1) 
    threads in (%c128, %c1, %c1)
    args(%arg0 : memref<1024x64xf32>, %arg1 : memref<1024x64xf32>)
  return
}
```

**GPU Kernel:**
```mlir
gpu.module @payload_kernel {
  gpu.func @payload_kernel(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) kernel 
    attributes {known_block_size = array<i32: 128, 1, 1>, 
                known_grid_size = array<i32: 16, 1, 1>} {
    %block_id_x = gpu.block_id x
    %1 = arith.muli %block_id_x, %c64 overflow<nsw> : index
    %subview = memref.subview %arg0[%1, 0] [64, 64] [1, 1]
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<64x16xf32>
    
    // Three reduction loops (max, sum, div) with same structure
    scf.for %arg2 = %c0 to %c64 step %c16 {
      // Max: accumulate max values
      // Sum: compute & accumulate exp(x - max)
      // Div: compute exp(x - max) / sum
    }
    
    gpu.return
  }
}
```

**Summary:** At this stage, the kernel processes 64x16 chunks in streaming fashion through three sequential loops, minimizing memory footprint.
