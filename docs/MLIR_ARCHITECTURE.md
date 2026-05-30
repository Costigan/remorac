# Remora on MLIR + Linalg + Mojo: Architectural Analysis

## Overview

This document describes the option of implementing a GPU-targeting Remora compiler using **LLVM's MLIR** infrastructure — specifically the `linalg`, `affine`, `tensor`, and `gpu` dialects — optionally with **Mojo** as either the implementation language or the host runtime. It then compares this approach in detail with the ILGPU-based approach described in `ILGPU_ARCHITECTURE.md`.

---

## Background: What Is MLIR?

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure project under the LLVM umbrella. Its central idea is that a compiler should be able to work at *multiple abstraction levels simultaneously*, each represented by a **dialect** — a named extension of the core IR with its own operations, types, attributes, and transformation passes.

Key properties:

- **Dialect composition**: dialects coexist in the same IR module and can be progressively lowered from high-level to machine-level.
- **Reusable passes**: transformations written against one dialect (e.g., loop fusion in `affine`) apply to any code that uses that dialect, regardless of what higher-level source language produced it.
- **No fixed pipeline**: the lowering order and pass selection are fully configurable per use case.
- **First-class extensibility**: adding a new dialect with new operations is a supported, documented workflow.

MLIR is the compilation backbone of TensorFlow (via MHLO), PyTorch (via Torch-MLIR), IREE, and several ML hardware compilers. It is production-grade, actively maintained, and has a large community.

---

## The Dialect Stack Relevant to Remora

A Remora GPU implementation would use the following dialects, in descending order of abstraction:

### `tensor` Dialect — Immutable Abstract Tensors

- `tensor<DxDxf32>` — an *immutable*, SSA-valued, rank-static tensor.
- `tensor<?x?xf32>` — a tensor with *dynamic* dimension sizes (known at runtime).
- Operations: `tensor.extract`, `tensor.insert`, `tensor.empty`, `tensor.reshape`, `tensor.expand_shape`, `tensor.collapse_shape`.
- This is the natural type for Remora arrays: immutable, shaped, rank-typed.
- Dynamic dimensions (`?`) are the mechanism for shape-polymorphic programs where extents are not known at compile time.

### `linalg` Dialect — Structured Array Computation

This is the most important dialect for a Remora implementation.

`linalg.generic` is the universal operation for any computation that can be expressed as a loop nest with affine index expressions. Its key fields:

```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(i, j, k) -> (i, k)>,   // input A: reads row i, col k
    affine_map<(i, j, k) -> (k, j)>,   // input B: reads row k, col j
    affine_map<(i, j, k) -> (i, j)>    // output C: writes row i, col j
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
  outs(%C : tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
}
```

- **Indexing maps**: affine maps from a shared *iteration space* to each operand's index space. Expresses reductions, transposes, broadcasts, and arbitrary strided access in one framework.
- **Iterator types**: `parallel` (can be parallelized freely) or `reduction` (must be serialized or handled with parallel reduction protocols). This is the axis/rank information for Remora's lifting.
- **Region body**: the per-element scalar computation — a pure, side-effect-free block of `arith`, `math`, or custom operations.

**Why this maps perfectly to Remora lifting**: A Remora `map f arr` where `arr` has frame shape `[d0, d1]` and `f` operates on cells of some shape `[c0, c1]` becomes a `linalg.generic` with:
- Parallel iterators for the frame dimensions `d0 × d1`.
- The body of `f` as the region.
- Indexing maps extracting the cell slice from the full array.

Reductions (`fold`/`reduce`) add reduction iterators over the cell dimensions.

### `affine` Dialect — Polyhedral Loop Representation

Once `linalg.generic` is lowered to explicit loops, the result lands in the `affine` dialect:

- `affine.for %i = 0 to %N` — a loop with affine bounds.
- `affine.load %A[%i, %j]` — a memory access with an affine index expression.
- `affine.if` — a conditional with a polyhedral predicate.

The `affine` dialect supports a rich set of transformations grounded in polyhedral theory:
- **Loop fusion** (`-affine-loop-fusion`): merge adjacent loop nests that access the same data.
- **Loop tiling** (`-affine-loop-tile`): tile loop nests for cache blocking, vectorization, or GPU block decomposition.
- **Loop interchange** (`-affine-loop-permute`): reorder loop dimensions to improve access locality.
- **Parallel marking** (`affine.parallel`): mark loops as parallel for subsequent GPU mapping.

### `scf` Dialect — Structured Control Flow

After `affine` loops are no longer amenable to polyhedral analysis (e.g., because bounds are non-affine), they are lowered to `scf.for`, `scf.while`, `scf.if`. This is essentially a structured loop IR without polyhedral constraints.

### `gpu` Dialect — GPU Thread Hierarchy

The `gpu` dialect abstracts the GPU execution model:

- `gpu.launch` / `gpu.launch_func` — launch a kernel with explicit grid and block dimensions.
- `gpu.thread_id`, `gpu.block_id`, `gpu.grid_dim`, `gpu.block_dim` — thread hierarchy intrinsics.
- `gpu.barrier` — thread-group synchronization.
- `gpu.shuffle` — warp shuffle operations.
- `gpu.alloc` / `gpu.dealloc` — device memory management.
- `gpu.memcpy` — host-device transfers.

Parallel `affine` or `scf` loops are mapped to GPU threads/blocks by passes such as `-gpu-map-parallel-loops` and `-convert-parallel-loops-to-gpu`.

### `nvvm` and `rocdl` Dialects — Hardware Intrinsics

The final lowering step translates `gpu` dialect ops to hardware-specific LLVM IR:

- `nvvm` — NVIDIA's LLVM IR dialect; compiles to PTX via LLVM's NVPTX backend.
- `rocdl` — AMD's ROCm dialect; compiles to AMDGPU ISA.

Both backends ultimately go through LLVM IR and out to GPU machine code.

### Supporting Dialects

| Dialect | Purpose |
|---|---|
| `arith` | Integer and floating-point arithmetic |
| `math` | Transcendental functions (sin, cos, exp, sqrt, etc.) |
| `memref` | Mutable memory references (buffers) with static/dynamic shapes and strides |
| `vector` | SIMD vector operations, mapped to GPU warp/subgroup ops or SIMD lanes |
| `func` | Function definitions, calls, returns |
| `cf` | Low-level conditional branches (CFG-level) |
| `llvm` | Direct LLVM IR ops for final lowering |

---

## Mojo's Role

**Mojo** (from Modular) is a programming language that compiles through MLIR and is designed for high-performance systems and AI/ML programming. It sits above MLIR in the same way that C++ sits above LLVM IR.

Mojo is relevant to a Remora implementation in two distinct ways:

### Option A: Mojo as the Host/Runtime Language

Mojo can serve as the host language in which the Remora runtime, interpreter, and GPU dispatch layer are written, instead of C++, Rust, or Python.

Mojo's relevant features for this role:

- **Parameterized types and functions** (`fn foo[rank: Int](...)`): compile-time integer and type parameters, suitable for encoding array rank statically. This is the equivalent of C++ template parameters — each concrete `rank` value produces a separately compiled code path.
- **`alias` and `parameter_if`**: compile-time conditionals and constants for generating rank-specific code without runtime overhead.
- **`SIMD[DType, width]`**: first-class SIMD types that lower to GPU vector ops.
- **Manual memory control**: Mojo has both managed and unmanaged memory, suitable for implementing a GPU buffer pool.
- **Python interoperability**: Mojo can call Python and vice versa, useful during prototyping.
- **`@parameter` for`fn`**: marks a function as a compile-time-evaluated metafunction, useful for shape-specialized code generation.

### Option B: Mojo as a Compilation Target

The Remora compiler (written in any language) generates Mojo source code as output. Mojo's compiler then handles:
- MLIR lowering through its built-in pipeline.
- GPU kernel generation and launch.
- Optimization passes.

This is analogous to targeting C++ for CUDA — Mojo would be the "portable GPU C++" of the pipeline. This option is less mature (Mojo's GPU API surface is still evolving) and adds a dependency on Mojo's compiler correctness.

### Option C: Mojo Bypassed — Pure MLIR from C++ or Python

The Remora compiler is written in C++ or Python and generates MLIR directly using the MLIR C++ API or the `mlir-python-bindings` (the official Python API for constructing and transforming MLIR). This is the most battle-tested approach and avoids Mojo's current immaturity.

`mlir-python-bindings` exposes the full MLIR op-builder, pass-manager, and dialect APIs from Python, making it possible to write a Remora-to-MLIR compiler entirely in Python without any C++. This is how projects like JAX's StableHLO and some IREE frontends work.

**For a prototype, Option C (pure MLIR via Python bindings) or a hybrid of A and C is recommended.** Mojo's GPU and MLIR-integration APIs are valuable but not yet production-stable.

---

## Remora → MLIR Mapping

### Arrays

A Remora array of element type `τ` and shape `[d₀, d₁, ..., dₙ₋₁]` maps to:

- `tensor<d₀ x d₁ x ... x dₙ₋₁ x τ>` if dimensions are statically known.
- `tensor<? x ? x ... x τ>` if dimensions are runtime values.
- `memref<...>` after bufferization (the mutable version, used inside kernels).

Shape descriptors (the `dᵢ` values) are passed as additional `index`-typed function arguments when dynamic.

### Lifted Application (`map f arr`)

A Remora lifted application of a rank-*k* function `f` over the frame dimensions of a rank-*k+n* array `arr` maps to a `linalg.generic` with:

- `n` parallel iterator dimensions (the frame, one per excess rank dimension).
- The body of `f` inlined into the region (possibly with additional iterators for `f`'s own internal structure).
- An indexing map that slices the frame position out of `arr` and routes it to `f`'s input.

Example: `map (λ x → x * 2.0) arr` where `arr : float[m, n]` (applying scalar double to each element):

```mlir
%result = linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,   // input arr
    affine_map<(i, j) -> (i, j)>    // output
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%arr : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {
^bb0(%elem: f32, %out_elem: f32):
  %two = arith.constant 2.0 : f32
  %res = arith.mulf %elem, %two : f32
  linalg.yield %res : f32
}
```

### Reduction / Fold

`fold f init arr` where `arr : float[n]`:

```mlir
%result = linalg.generic {
  indexing_maps = [
    affine_map<(i) -> (i)>,   // input arr
    affine_map<(i) -> ()>     // scalar output (0D)
  ],
  iterator_types = ["reduction"]
} ins(%arr : tensor<?xf32>) outs(%init : tensor<f32>) {
^bb0(%elem: f32, %acc: f32):
  %res = arith.addf %acc, %elem : f32
  linalg.yield %res : f32
}
```

### Rank Polymorphism

Rank polymorphism requires generating different `linalg.generic` operations for different concrete ranks. The options are:

1. **Static rank (monomorphization)**: generate separate MLIR code per concrete rank at Remora compile time. The MLIR type `tensor<f32>`, `tensor<?xf32>`, `tensor<?x?xf32>`, etc. are different types; each rank yields a different `linalg.generic`. This is the simplest and most performant approach.

2. **Dynamic rank with runtime shape packing**: represent arrays uniformly as a 1D `memref<?xτ>` plus a shape struct (a small `memref<?xindex>` holding the dimension extents). Generate a single runtime-polymorphic kernel that decodes the shape at runtime to compute linear indices. Simpler implementation, slower execution (no affine analysis possible on dynamic index expressions).

A practical Remora prototype would use monomorphization: the type checker resolves all ranks statically, and the code generator emits specialized MLIR per concrete rank.

### Higher-Order Functions

As with ILGPU, the GPU execution model does not support runtime function pointers cheaply. The MLIR approach uses the same fundamental techniques:

1. **Defunctionalization**: convert function values to tagged structs + dispatch switch. In MLIR this is a `scf.switch` or a chain of `scf.if` operations over an `i32` tag. The `arith.select` instruction can convert short switches to predicated selection for scalar values.

2. **Monomorphization via generic code generation**: at Remora compile time, for each distinct function passed to a higher-order operation, generate a specialized `linalg.generic` with the appropriate body. No runtime dispatch; the function is baked into the MLIR region at generation time. This is the preferred approach for an AOT compiler.

3. **MLIR interfaces / op interfaces**: define a custom MLIR op interface `RemoraFunctionOp` and generate a distinct op type per Remora function value. The lowering of `apply(f, x)` dispatches on the concrete op type at lowering time. This is elegant but requires more MLIR dialect engineering.

For a prototype, option 2 (monomorphization) is simplest and requires no MLIR modifications.

### Shape-Dependent Control Flow

Remora programs may include conditionals on shape (e.g., `if rank arr > 1 then ... else ...`). In MLIR this becomes `scf.if` with shape-derived conditions, or `affine.if` if the condition is an affine predicate. These work naturally and can be folded away at compile time if shapes are statically known.

---

## The Compilation Pipeline

A full Remora → GPU compilation pipeline using MLIR:

```
Remora source
     │
     ▼
[1] Remora frontend
     │   Parsing, type checking, shape inference, rank resolution
     │   Produces: Remora IR (rank-resolved, typed, functional)
     ▼
[2] Remora → Linalg lowering
     │   Each lifted application → linalg.generic on tensors
     │   Each fold/scan → linalg.generic with reduction iterators
     │   Dense Core shapes → static tensor dimensions
     │   Future dynamic shapes → tensor dynamic dimensions or index args
     │   HOFs → monomorphized linalg.generic bodies
     ▼
[3] Linalg-level optimization
     │   -linalg-fuse-elementwise-ops    (map-chain fusion)
     │   -linalg-generalize-named-ops    (normalize to generic)
     │   -linalg-fold-reshape-by-linear-layout
     ▼
[4] One-shot bufferization
     │   Converts immutable tensor SSA values to mutable memref buffers
     │   Inserts alloc/dealloc automatically
     │   Eliminates copies where possible (in-place analysis)
     ▼
[5] Affine-level optimization
     │   -affine-loop-fusion             (remaining fusible loop pairs)
     │   -affine-loop-tile               (cache blocking, GPU tile sizes)
     │   -affine-loop-unroll             (small fixed-bound loops)
     │   -affine-parallelize             (mark parallel loops)
     ▼
[6] GPU mapping
     │   -gpu-map-parallel-loops         (parallel loops → GPU thread/block grid)
     │   -convert-parallel-loops-to-gpu  (generate gpu.launch regions)
     │   Tiled loops → thread blocks; outer loops → grid dimensions
     ▼
[7] GPU dialect lowering
     │   -gpu-kernel-outlining           (extract gpu.launch bodies to module)
     │   -lower-affine                   (affine.for → scf.for)
     │   -convert-scf-to-cf             (scf → basic block CFG)
     │   -convert-gpu-to-nvvm            (gpu ops → nvvm intrinsics)
     │   or -convert-gpu-to-rocdl       (for AMD)
     ▼
[8] LLVM IR lowering
     │   -convert-nvvm-to-llvmir
     │   Standard LLVM passes (DCE, GVN, loop vectorization, etc.)
     ▼
[9] PTX / AMDGPU machine code
     │   LLVM NVPTX / AMDGPU backend
     ▼
[10] Runtime launch
     │   Host code (C++, Python, or Mojo) calls CUDA/ROCm driver
     │   to load and execute the compiled kernel module
```

---

## Fusion in MLIR/Linalg

Fusion is MLIR's strongest advantage over ILGPU for this use case. It operates at two levels:

### Level 1: Linalg Elementwise Fusion (`-linalg-fuse-elementwise-ops`)

The pass identifies producer-consumer pairs of `linalg.generic` operations where:
- The producer's output is consumed only by the consumer.
- Both operations have the same iteration space (same parallel dimensions).

It merges them into a single `linalg.generic` whose region contains both bodies inlined sequentially, with the intermediate tensor replaced by a scalar SSA value. The result:
- The intermediate array never exists in memory; it is computed and consumed in registers.
- The fused kernel makes a single pass over the data.

This handles `map f ∘ map g`, `map f (map g arr)`, and chains of any length.

### Level 2: Affine Loop Fusion (`-affine-loop-fusion`)

After lowering to `affine.for` loops, the affine fusion pass uses polyhedral dependence analysis to fuse loop nests that share data:
- Detects producer-consumer pairs based on memory access patterns.
- Checks legality (no dependence cycle introduced by fusion).
- Applies a cost model (iteration count, memory footprint) to decide whether fusion is profitable.
- Handles imperfect nests by inserting `affine.if` guards where needed.

This catches fusions that the Linalg-level pass missed (e.g., operations with incompatible iteration spaces that become compatible after tiling).

### Level 3: Tile-and-Fuse (GPU-Specific)

The `linalg` dialect's tiling infrastructure supports **tile-and-fuse**: tile the outer (parallel) dimensions of a consumer, then fuse the producer into the tiled consumer loop. The result is:
- Each GPU thread block computes a tile of the output.
- All input data for that tile is fetched and transformed locally.
- Shared memory is used for the tile of the inner operand.

This is the standard pattern for efficient GPU matrix multiplication and convolution, and it generalizes to arbitrary `linalg.generic` pairs.

---

## Memory Management

MLIR's **one-shot bufferization** pass (`-one-shot-bufferize`) handles memory management automatically:

1. **In-place analysis**: determines whether a `tensor` SSA value can be bufferized to the same memory as one of its operands (avoiding a copy).
2. **Allocation insertion**: inserts `memref.alloc` and `memref.dealloc` for tensors that cannot be bufferized in-place.
3. **Copy insertion**: inserts `memref.copy` only where aliasing analysis requires it.
4. **Buffer deallocation**: the `-buffer-deallocation` pass inserts `memref.dealloc` at the correct post-dominator points, effectively implementing region-scoped memory management without a runtime GC.

For a Remora program, each intermediate array in a `tensor`-level computation becomes a `memref.alloc`/`memref.dealloc` pair scoped to its liveness region. After fusion, most intermediate tensors are eliminated entirely by in-place analysis before any allocation is even inserted. The result: the compiler manages memory automatically, correctly, and with no runtime GC overhead, through a combination of fusion and static lifetime analysis — exactly what a Remora implementation needs.

---

## Strengths of the MLIR/Linalg Approach

### 1. Fusion is Built In and Mature

MLIR's linalg elementwise fusion and affine loop fusion are production-quality passes used in TensorFlow, PyTorch, and IREE. For the common Remora patterns (map chains, map-reduce), fusion works out of the box with no implementation work from the Remora compiler author.

### 2. Memory Management is Automatic

One-shot bufferization plus buffer deallocation handles the entire allocation/GC problem automatically and correctly. This removes the largest implementation burden from the ILGPU comparison.

### 3. `linalg.generic` Was Designed for This Problem

The `linalg.generic` abstraction — iteration space, indexing maps, iterator types, region body — is essentially a formalization of rank-polymorphic array operations. The mapping from Remora lifting to `linalg.generic` is direct and semantics-preserving.

### 4. Multiple GPU Targets

One pipeline targets NVIDIA (PTX via NVVM), AMD (ROCm via ROCDL), and potentially Intel (SPIRV via SPIR-V dialect), with no per-backend work required at the Remora level.

### 5. Loop Tiling and Shared Memory Optimization

MLIR's tiling infrastructure automatically generates tiled loop nests that exploit GPU shared memory. For Remora programs performing reductions or matrix operations over large arrays, this is critical for reaching peak GPU performance.

### 6. Dynamic Shapes Are First Class

`tensor<?x?xf32>` and `memref<?x?xf32>` with dynamic sizes are fully supported throughout the pipeline. Shape-polymorphic Remora programs compile without the need for per-shape JIT compilation.

### 7. Polyhedral Analysis

The `affine` dialect and its associated analyses (dependence checking, loop transformations) are grounded in decades of polyhedral compilation research. For regular, dense Remora array programs, the polyhedral model is exactly the right framework.

### 8. Python Bindings Make Prototyping Fast

`mlir-python-bindings` provides a complete Python API for constructing MLIR modules, running passes, and emitting code. A Remora-to-MLIR compiler can be prototyped entirely in Python, with no C++ required.

---

## Weaknesses and Challenges

### 1. Ecosystem Complexity

MLIR has a steep learning curve. The dialect hierarchy, the lowering pipeline, the pass infrastructure, and the bufferization model are sophisticated and not always well-documented for newcomers. Getting a full pipeline from `linalg.generic` to running PTX requires understanding ~6 dialect layers and their interactions.

### 2. No .NET / C# Ecosystem

MLIR is primarily a C++ and Python ecosystem. Interfacing with .NET requires C interop (via a C wrapper over the MLIR C API) or abandoning the .NET host entirely. This is a significant disadvantage if the Remora runtime or application layer is in C#.

### 3. Mojo Is Not Production-Ready for This Use Case (yet)

Mojo's GPU API and MLIR-integration features are still evolving rapidly. Using Mojo as the implementation language introduces risk around API stability and capability gaps (e.g., warp-level intrinsics, shared memory layout control). For a 2025 prototype, Mojo is plausible but the C++ MLIR API or `mlir-python-bindings` are safer foundations.

### 4. Custom Dialect Engineering for Remora Semantics

If you want to represent Remora's rank-polymorphic types in the MLIR type system (rather than erasing to `tensor<?x...xf32>`), you need to implement a custom Remora dialect in C++. This is powerful but is a substantial engineering task, requiring familiarity with MLIR's `TableGen`-based op definition system, type system, and dialect registration.

### 5. Debugging GPU Code Is Still Hard

MLIR does not substantially improve the GPU debugging experience compared to raw CUDA. Errors in GPU kernels still manifest as cryptic PTX crashes or silent wrong results. The CPU fallback path (lowering to LLVM CPU instead of NVVM) is the primary debugging tool, similar to ILGPU's CPU accelerator — but less seamlessly integrated into the development loop.

### 6. Higher-Order Functions Still Require Engineering

MLIR has no native support for passing functions as values at runtime. Defunctionalization or monomorphization must be implemented at the Remora IR level, before generating MLIR — exactly as in the ILGPU case. MLIR does not make this problem easier.

### 7. Dynamic Parallelism Not Supported

Like ILGPU, MLIR's `gpu` dialect does not model device-side kernel launches. Ragged/irregular Remora arrays are out of scope for the same reason.

---

## Comparison: MLIR/Linalg vs. ILGPU

| Dimension | MLIR + Linalg + Mojo | ILGPU |
|---|---|---|
| **Kernel fusion** | ✓ Built-in, mature, production-quality | ✗ Not present; must be built |
| **Memory management** | ✓ Automatic via one-shot bufferization | ✗ Manual; must build pool/GC layer |
| **Higher-order functions** | ✗ Must implement (defunctionalization or monomorphization) | ✗ Same requirement |
| **Dynamic shapes** | ✓ First-class (`tensor<?x?xf32>`) | Partial (shapes as runtime struct args) |
| **Rank polymorphism** | ✓ `linalg.generic` models it directly | ✗ Must generate per-rank kernels |
| **GPU targets** | NVIDIA + AMD + (Intel SPIRV) | NVIDIA + OpenCL |
| **CPU debugging** | ✓ Lower to LLVM CPU target | ✓ Excellent (native .NET debugger) |
| **Loop tiling / shared memory** | ✓ Automatic via tile-and-fuse pass | ✗ Manual kernel design required |
| **Polyhedral analysis** | ✓ Full affine analysis | ✗ None |
| **Python ecosystem** | ✓ (`mlir-python-bindings`) | ✗ |
| **.NET / C# ecosystem** | ✗ C interop required | ✓ Native |
| **Learning curve** | High (6+ dialect layers, complex pipeline) | Moderate (clean C# API) |
| **Maturity** | High (production-used in TF/PyTorch/IREE) | Moderate (production-quality core) |
| **Mojo integration** | Natural (Mojo compiles through MLIR) | N/A |
| **Prototype speed** | Moderate (Python bindings ease prototyping, but pipeline setup is complex) | Fast (C# kernel code runs immediately) |
| **Performance ceiling** | Very high (polyhedral transforms, tile-and-fuse, warp vectorization) | High (solid PTX/OpenCL codegen, but no tiling) |
| **HOF values at runtime** | ✗ Must build (same as ILGPU) | ✗ Must build (same) |
| **Dynamic parallelism** | ✗ Not modeled | ✗ Not modeled |

### When to Choose MLIR

- You need **fusion and automatic memory management without building them yourself** — these are the decisive factors.
- Your team is comfortable with Python or C++ and is not invested in .NET.
- You need to target **AMD GPUs** as well as NVIDIA.
- You want **polyhedral loop optimization** (tiling, interchange) for Remora programs that map to dense linear algebra (matrix operations, convolution-like array ops).
- You are building for the **long term** and want to leverage a large, growing infrastructure investment.
- You do not need a working end-to-end system in the first few weeks — MLIR's pipeline requires significant setup before the first kernel runs.

### When to Choose ILGPU

- Your application is in **.NET / C#** and you want seamless interoperability.
- You value **rapid prototyping and debuggability** above all — ILGPU's CPU backend with .NET debugging is unmatched.
- You are targeting **NVIDIA only** and are comfortable implementing fusion as a Remora-level pass.
- Your Remora programs are **rank-monomorphic and shape-static** — the gaps in ILGPU's type system and optimization are non-issues.
- You want to **own the full stack** and understand every layer clearly, rather than depending on a complex multi-layer infrastructure.
- The team knows C# better than C++ or Python.

### The Hybrid Option

It is architecturally possible to use MLIR as a backend for the ILGPU approach: the Remora compiler generates MLIR rather than ILGPU IR, while the host application, runtime, and Remora standard library remain in C#/ILGPU. The MLIR C API is callable from .NET via P/Invoke. This hybrid preserves .NET interoperability while gaining MLIR's fusion and bufferization passes. The cost is increased integration complexity — two distinct IR systems must coexist — but it is a viable long-term architecture if the .NET constraint is non-negotiable.

---

## Recommended Prototype Architecture (MLIR Path)

For a Remora prototype targeting the MLIR path:

```
Remora front-end (Python)
  │  Parser, type checker, shape inference
  │  Produces: typed Remora AST with resolved ranks and static shapes
  ▼
Remora → Linalg lowering (Python, using mlir-python-bindings)
  │  Map each Remora lifted op → linalg.generic
  │  Preserve Remora semantics in Python HIR, not a custom MLIR dialect
  │  Defunctionalize static higher-order functions before this step
  │  Shapes → tensor static dimensions for Dense Core rank 0..3
  ▼
MLIR pass pipeline (Python pass manager)
  │  -linalg-fuse-elementwise-ops
  │  -one-shot-bufferize
  │  -affine-loop-fusion
  │  -affine-parallelize
  │  -gpu-map-parallel-loops
  │  -convert-parallel-loops-to-gpu
  │  -gpu-kernel-outlining
  │  -convert-gpu-to-nvvm (or -convert-gpu-to-rocdl)
  │  -convert-nvvm-to-llvmir
  ▼
LLVM NVPTX backend → PTX
  ▼
CUDA runtime launch (Python ctypes wrapper, using docs/ABI.md descriptors)
```

Development phases:
1. **Phase 1**: Implement Remora Dense Core front-end and Linalg lowering for static dense rank-0 through rank-3 arrays, scalar maps, static frame/cell maps, and outermost-dimension folds. Verify correctness by lowering to CPU LLVM instead of NVVM.
2. **Phase 2**: Add a CPU-first `remora` REPL that reuses the full compiler path by recompiling each expression as a self-contained temporary program with accumulated definitions. This gives early interactive feedback without requiring incremental MLIR compilation.
3. **Phase 3**: Validate the rank-0..3 descriptor ABI in `docs/ABI.md`; then enable GPU lowering.
4. **Phase 4**: Add fusion verification — confirm intermediate arrays are eliminated.
5. **Phase 5**: Extend beyond Dense Core with stdlib, transpose/slice syntax over strided views, and broader rank-polymorphic examples.
6. **Phase 6** (if needed): Add tiling for performance-critical operations.

Incremental REPL compilation is deliberately out of the initial architecture. If full-program recompilation becomes too slow in real sessions, add content-hash caching for accumulated definitions and generated artifacts after the direct pipeline is stable.

---

*Document generated for engineering analysis of GPU compilation infrastructure options for the Remora rank-polymorphic array language.*
