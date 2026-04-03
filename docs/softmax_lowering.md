# Linalg softmax lowering in XeGPU pipeline

## Overview

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

**Key Characteristics:**
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

**Key Characteristics:**
- Softmax decomposed into 4 constituent `linalg.generic` ops : max, sub+exp, sum, divide
- Uses `structured.structured_decompose_interface` implemented by `linalg.softmax`

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFFC00000 : f32  // -inf for max reduction
  %0 = bufferization.to_tensor %arg1 restrict : memref<1024x64xf32> to tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024x64xf32>
  
  %2 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %1) -> (tensor<1024x64xf32>) {
    %3 = affine.apply #map(%arg2)  // %3 = %arg2 * 64
    %extracted_slice = tensor.extract_slice %0[%3, 0] [64, 64] [1, 1] : 
      tensor<1024x64xf32> to tensor<64x64xf32>
    
    // Step 1: Find max along dimension 1
    %4 = tensor.empty() : tensor<64xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], 
                         iterator_types = ["parallel", "reduction"]} 
         ins(%extracted_slice : tensor<64x64xf32>) outs(%5 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.maxnumf %in, %out : f32
      linalg.yield %11 : f32
    } -> tensor<64xf32>
    
    // Step 2: Subtract max and exponentiate
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], 
                         iterator_types = ["parallel", "parallel"]} 
         ins(%extracted_slice, %6 : tensor<64x64xf32>, tensor<64xf32>) 
         outs(%extracted_slice_1 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.subf %in, %in_2 : f32
      %12 = math.exp %11 : f32
      linalg.yield %12 : f32
    } -> tensor<64x64xf32>
    
    // Step 3: Sum exponentials
    %8 = linalg.fill ins(%cst : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map2], 
                         iterator_types = ["parallel", "reduction"]} 
         ins(%7 : tensor<64x64xf32>) outs(%8 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.addf %in, %out : f32
      linalg.yield %11 : f32
    } -> tensor<64xf32>
    
    // Step 4: Normalize by sum
    %10 = linalg.generic {indexing_maps = [#map1, #map2, #map1], 
                          iterator_types = ["parallel", "parallel"]} 
          ins(%7, %9 : tensor<64x64xf32>, tensor<64xf32>) 
          outs(%extracted_slice_1 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.divf %in, %in_2 : f32
      linalg.yield %11 : f32
    } -> tensor<64x64xf32>
    
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg3[%3, 0] [64, 64] [1, 1] : 
        tensor<64x64xf32> into tensor<1024x64xf32>
    }
  }
  return
}
```

**What Happens:**
- The 1024x64 input is divided into 16 tiles of 64x64 each
- Softmax algorithm made explicit:
  1. **Max reduction**: Find maximum value per row (for numerical stability)
  2. **Exp**: Compute exp(x - max) for each element
  3. **Sum reduction**: Sum exponentials per row
  4. **Normalize**: Divide each element by its row sum
- Each tile is processed independently, enabling parallelization
- Results are inserted back into the output tensor

---

## Stage 4: Vectorized

**Key Characteristics:**
- `linalg.generic` operations replaced with vector operations
- SIMD-friendly representation using `vector<64x64xf32>`
- Explicit vector multi-reductions
- Vector transfers for reading/writing data

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  %0 = ub.poison : f32
  %cst_0 = arith.constant dense<0xFFC00000> : vector<64xf32>
  %c0 = arith.constant 0 : index
  %1 = bufferization.to_tensor %arg1 restrict : memref<1024x64xf32> to tensor<1024x64xf32>
  %2 = tensor.empty() : tensor<1024x64xf32>
  
  %3 = scf.forall (%arg2) in (16) shared_outs(%arg3 = %2) -> (tensor<1024x64xf32>) {
    %4 = affine.apply #map(%arg2)  // %4 = %arg2 * 64
    %extracted_slice = tensor.extract_slice %arg3[%4, 0] [64, 64] [1, 1]
    
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

**What Happens:**
- Linalg operations converted to vector dialect operations
- `vector.transfer_read` loads entire 64x64 tile at once
- `vector.multi_reduction` performs SIMD reductions (max and sum)
- `vector.broadcast` and `vector.transpose` handle dimension alignment
- All arithmetic operations work on vectors, enabling SIMD execution
- `vector.transfer_write` stores results back

---

## Stage 5: Bufferized

**Key Characteristics:**
- Tensors eliminated, working directly with memrefs
- Vector operations read/write directly from/to memory
- No more tensor extract/insert operations
- Simplified control flow

**Code:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
  %0 = ub.poison : f32
  %cst_0 = arith.constant dense<0xFFC00000> : vector<64xf32>
  %c0 = arith.constant 0 : index
  
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

**What Happens:**
- All tensor operations converted to memref-based operations
- `scf.forall` no longer uses `shared_outs`, simplified to pure side effects
- Vector transfers work directly on input/output memrefs
- Memory layout is now explicit
- This representation is ready for GPU kernel extraction

---

## Stage 6: XeGPU-Initial

**Key Characteristics:**
- GPU kernel separated from host code
- `gpu.launch_func` invocation with grid/block dimensions
- XeGPU tensor descriptors for memory access
- Block-based load/store operations

**Code:**

**Host Side:**
```mlir
func.func @payload(%arg0: memref<1024x64xf32>, %arg1: memref<1024x64xf32>) {
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
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
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<0xFFC00000> : vector<64xf32>
    %c64 = arith.constant 64 : index
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

**What Happens:**
- Host function now calls `gpu.launch_func` with 16 blocks, 128 threads per block
- Separate `gpu.module` contains the kernel code
- `gpu.block_id x` replaces the loop iterator
- `xegpu.create_nd_tdesc` creates tensor descriptors for memory regions
- `xegpu.load_nd` performs hardware-optimized block loads
- `xegpu.store_nd` performs hardware-optimized block stores
- XeGPU operations map directly to Intel GPU instructions
- Boundary checking disabled for performance (sizes known at compile time)

---

## Stage 7: XeGPU-WG (Work-Group Optimized)

**Key Characteristics:**
- Additional layout hints for work-group optimization
- Sub-group layout specification: `sg_layout` and `sg_data`
- Optimized memory access patterns for Intel XeGPU

**Code (differences from xegpu-initial):**
```mlir
// Store operation now includes layout hints
xegpu.store_nd %11, %12[%0, 0] 
  <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [8, 64]>}> : 
  vector<64x64xf32>, 
  !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
```

**What Happens:**
- The `layout` attribute provides explicit sub-group (SG) tiling information:
  - `sg_layout = [8, 1]`: Data is distributed across 8 sub-groups in the first dimension
  - `sg_data = [8, 64]`: Each sub-group handles an 8x64 slice of data
- This layout specification:
  - Matches the 64x64 total data size (8 sub-groups × 8 rows = 64 rows, 64 columns)
  - Optimizes coalesced memory accesses
  - Enables efficient SIMD execution within each sub-group
  - Aligns with Intel GPU hardware execution model (128 threads = 8 sub-groups of 16 threads)
- The layout is applied to the store operation to optimize write patterns
- Load operation remains unchanged as reads are typically more flexible

---

## Summary

The lowering pipeline progressively transforms abstract operations into hardware-specific instructions:

| Stage | Abstraction Level | Key Operations |
|-------|------------------|----------------|
| **initial** | High-level ML | `linalg.softmax` on full tensor |
| **tiled-softmax** | Tiled high-level | `linalg.softmax` on tiles, `scf.forall` |
| **decomposed** | Tiled computation | `linalg.generic` with reductions |
| **vectorized** | Vector operations | `vector.multi_reduction`, `vector.transfer_read/write` |
| **bufferized** | Memory-based | Direct memref operations with vectors |
| **xegpu-initial** | GPU-specific | `xegpu.load_nd`, `xegpu.store_nd`, `gpu.launch_func` |
| **xegpu-wg** | Hardware-optimized | Layout hints for sub-group optimization |

Each stage maintains the same computational semantics while providing increasingly detailed control over execution and memory access patterns, ultimately targeting efficient execution on Intel XeGPU hardware.
