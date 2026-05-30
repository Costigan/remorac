# ILGPU: Detailed Architectural Description

## Overview

ILGPU (Intermediate Language GPU) is a JIT (just-in-time) compiler for high-performance GPU programs written in .NET-based languages (C#, F#, VB.NET). It compiles standard .NET methods — decorated with no special attributes — directly into GPU code at runtime, targeting NVIDIA CUDA (via PTX), OpenCL, and a multi-threaded CPU fallback.

ILGPU is written entirely in C# with no native dependencies in its core. It leverages .NET's `System.Reflection` and `System.Reflection.Emit` to disassemble MSIL bytecode and re-emit it as GPU-native code. The project spans two main libraries:

- **ILGPU** — the core compiler, runtime, and accelerator abstractions.
- **ILGPU.Algorithms** — a standard library of high-level parallel algorithms (scan, reduce, sort, etc.) that run on all backends.

---

## Repository Layout

```
ILGPU/
├── Src/
│   ├── ILGPU/                  # Core compiler and runtime
│   │   ├── Frontend/           # MSIL disassembler and IL-to-IR lifting
│   │   ├── IR/                 # Intermediate Representation (SSA-based)
│   │   │   ├── Values/         # IR node types (arithmetic, memory, threads, etc.)
│   │   │   ├── Types/          # IR type system
│   │   │   ├── Analyses/       # CFG, dominators, loop info, uniform analysis, etc.
│   │   │   ├── Construction/   # IRBuilder, SSABuilder, IRRebuilder
│   │   │   ├── Transformations/# Optimization and lowering passes
│   │   │   ├── Intrinsics/     # Intrinsic implementation registry
│   │   │   └── Rewriting/      # Rewriter infrastructure (visitor + replacement)
│   │   ├── Backends/           # Code generation backends
│   │   │   ├── PTX/            # NVIDIA PTX (Cuda) backend
│   │   │   ├── OpenCL/         # OpenCL C source backend
│   │   │   ├── IL/             # CPU IL-emit backend
│   │   │   └── EntryPoints/    # Kernel entry-point descriptions and argument mapping
│   │   └── Runtime/            # Accelerator abstractions, memory, kernel launch
│   │       ├── Cuda/           # Cuda driver API wrappers
│   │       ├── OpenCL/         # OpenCL API wrappers
│   │       └── CPU/            # Multi-threaded CPU accelerator
│   └── ILGPU.Algorithms/       # Standard parallel algorithms
├── Samples/                    # Usage examples (SimpleKernel, MatrixMultiply, etc.)
└── Tools/                      # Code-generation T4 templates and utilities
```

---

## Key Abstractions

### `Context`

`Context` (in `ILGPU/Context.cs`) is the root object of the entire ILGPU system. It owns:

- The global IR type context (`IRTypeContext`).
- The intrinsic implementation manager.
- The IL frontend (`ILFrontend`).
- All registered devices (`Device` objects for Cuda, OpenCL, CPU).
- Configuration properties (`ContextProperties`) — optimization level, inlining mode, debug settings.

A `Context` is created once per application and shared across accelerators:

```csharp
using var context = Context.CreateDefault();
```

`Context.Builder` (a nested fluent builder type) allows fine-grained configuration before construction.

### `Device` and `Accelerator`

`Device` is a description of a hardware unit (returned by the context's device enumeration). Each device produces an `Accelerator`:

```csharp
using var accelerator = device.CreateAccelerator(context);
```

`Accelerator` (abstract base in `Runtime/Accelerator.cs`) is the handle through which all GPU work is done:

- **Memory allocation**: `Allocate1D<T>`, `Allocate2D<T>`, `Allocate3D<T>` → `MemoryBuffer` objects.
- **Kernel loading**: `LoadAutoGroupedStreamKernel<TIndex, ...>`, `LoadKernel<...>` → compiled, cached kernel launchers.
- **Streams**: `CreateStream()` → `AcceleratorStream` for command sequencing and synchronization.

Concrete implementations:
- `CudaAccelerator` — wraps the CUDA driver API.
- `CLAccelerator` — wraps the OpenCL API.
- `CPUAccelerator` — simulates GPU execution on the host using .NET threads.

### `ArrayView<T>` and `MemoryBuffer`

Memory on an accelerator is represented as a `MemoryBuffer` (opaque handle) with typed views over it:

- `ArrayView<T>` — a 1D view (pointer + length).
- `ArrayView<T, TIndex>` — a strided, multi-dimensional view.
- `VariableView<T>` — a view of exactly one element.

These are value types (structs) that are safe to pass into GPU kernels as parameters. Inside kernels, element access `view[index]` compiles to a raw pointer dereference in PTX/OpenCL.

The `Stride` system allows non-contiguous views (row-major, column-major, custom stride types) to be expressed at the type level.

### Kernel Index Types

Every kernel receives an index as its first parameter, determining how ILGPU maps threads:

- `Index1D`, `Index2D`, `Index3D` — implicit grouping; ILGPU automatically selects a group (block) size and maps the flat thread index to the user's index type.
- `KernelConfig` — explicit grouping; the user specifies the grid and block dimensions directly. Thread-group intrinsics (`Group.IdxX`, `Warp.LaneIdx`, etc.) are then available.

Under the hood all index types implement `IIndex`, and the `IndexType` enum drives backend code generation.

---

## Compilation Pipeline

The compilation of a C# kernel method into GPU machine code follows this pipeline:

```
C# Method (MethodInfo)
        │
        ▼
[1] IL Frontend (Disassembler + CodeGenerator)
        │   Reads MSIL bytes via Reflection, produces DisassembledMethod
        │   Lifts MSIL into ILGPU IR (SSA, BasicBlocks, Values)
        ▼
[2] IR: Method (SSA-based, typed, graph-structured)
        │
        ▼
[3] Generic Optimization Passes (Transformer pipeline)
        │   Inlining, SSA construction, DCE, UCE, SimplifyControlFlow,
        │   LoopInvariantCodeMotion, IfConversion, LowerStructures, etc.
        ▼
[4] Backend-Specific Lowering (via Backend.Compile)
        │   LowerArrays, LowerPointerViews, AcceleratorSpecializer,
        │   InferAddressSpaces, IntrinsicSpecializer, etc.
        ▼
[5] Code Generation (PTXCodeGenerator / CLCodeGenerator / ILEmitter)
        │   Visitor over IR values → emits PTX text / OpenCL C / .NET IL
        ▼
[6] Machine Code / Compiled Kernel
        │   PTX → NVRTC/NVVM → cubin loaded via CUDA driver
        │   OpenCL C → clBuildProgram → CL kernel object
        │   .NET IL → AssemblyBuilder → JIT-compiled by CLR
        ▼
[7] Kernel Launcher (generated delegate)
        │   A strongly-typed, boxing-free delegate wrapping the GPU launch
        ▼
Kernel invocation: kernel(count, buffer.View, constant)
```

Each stage is described in detail below.

---

## Stage 1: IL Frontend

### Disassembler (`Frontend/Disassembler.cs`)

The `Disassembler` class reads a `MethodBase`'s raw MSIL byte array via `MethodBase.GetMethodBody()` and decodes it instruction-by-instruction into a sequence of `ILInstruction` records. Each `ILInstruction` has:
- An `ILOpCode` (the MSIL opcode).
- An optional argument (operand: integer, float, MethodInfo, FieldInfo, Type, etc.).
- Optional debug sequence points (source locations from PDB/embedded PDB).

The result is a `DisassembledMethod` — an immutable list of decoded instructions.

### ILFrontend and CodeGenerator (`Frontend/ILFrontend.cs`, `Frontend/CodeGenerator/`)

`ILFrontend` manages a worker thread that processes compilation requests asynchronously. When a kernel is compiled, `ILFrontend` takes the `DisassembledMethod` and runs a `CodeGenerator` over it.

`CodeGenerator` performs a single-pass conversion from MSIL to ILGPU IR:

1. **Control-flow reconstruction**: The MSIL instruction stream is divided into basic blocks by identifying branch targets and exception handlers.
2. **Stack simulation**: MSIL is stack-based; the code generator maintains a virtual stack and maps each stack slot to an SSA `Value`.
3. **Value lifting**: Each MSIL opcode is translated to one or more IR `Value` nodes. For example:
   - `ldarg` → a `Parameter` value reference.
   - `add` → a `BinaryArithmetic` value.
   - `call` → a `MethodCall` value (which recursively triggers compilation of the callee).
   - `ldfld` → a `LoadField` value.
   - `newobj` on a struct → a `StructureValue`.
4. **Intrinsic detection**: Calls to methods decorated with `[GridIntrinsic]`, `[GroupIntrinsic]`, `[WarpIntrinsic]`, etc. are recognized and replaced by specialized IR `DeviceConstant` or `Threads` values during code generation.
5. **Exception handling**: try/catch/finally are partially supported for CPU emulation; GPU paths typically require exception-free code.

The output is a fully populated `Method` in the `IRContext`.

---

## Stage 2: Intermediate Representation (IR)

ILGPU's IR is a **Static Single Assignment (SSA)**, **typed**, **graph-based** intermediate representation inspired by sea-of-nodes and classical LLVM-style IR.

### Method and BasicBlock

An IR `Method` consists of:
- A list of `Parameter` values (entry parameters).
- A collection of `BasicBlock` objects connected by `Terminator` values (branches, returns).
- Metadata: `MethodFlags` (Inline, External, Intrinsic, EntryPoint), `MethodTransformationFlags` (Dirty, Transformed).

Each `BasicBlock` holds a sequence of `Value` nodes in execution order, terminated by a `Terminator`:
- `UnconditionalBranch` — unconditional jump.
- `ConditionalBranch` — two-target conditional.
- `SwitchBranch` — multi-target jump table.
- `ReturnTerminator` — function return.

### Value Hierarchy

`Value` (in `IR/Values/Value.cs`) is the base of all IR nodes. Every value has:
- A `TypeNode` — its type in the IR type system.
- A list of `ValueReference` operands (inputs from other values, forming the use-def graph).
- A list of `Use` records (def-use edges, maintained for efficient rewriting).
- A `ValueKind` discriminator enum for efficient dispatch.

Values are visited via the `IValueVisitor` interface (visitor pattern), which code generators and transformations implement.

Key value categories:

| Category | Examples |
|---|---|
| Constants | `PrimitiveValue` (int/float/bool), `NullValue`, `UndefinedValue` |
| Arithmetic | `BinaryArithmeticValue` (add/mul/and/or/...), `UnaryArithmeticValue` (neg/not/...) |
| Memory | `Alloca`, `Load`, `Store`, `AddressSpaceCast`, `SubViewValue` |
| Structures | `StructureValue` (construct), `GetField`, `SetField` |
| Arrays | `NewArray`, `GetArrayElement`, `SetArrayElement` (before lowering) |
| Pointers/Views | `LoadElementAddress`, `SubViewValue`, `NewView` |
| Threads | `PredicateBarrier`, `Barrier`, `WarpShuffle`, `SubGroupOperation` |
| Atomics | `AtomicCAS`, `AtomicOperations` |
| Device constants | `GridIndexValue`, `GroupIndexValue`, `GridDimensionValue`, `WarpSizeValue` |
| Method calls | `MethodCall` |
| Control flow | Terminators: `UnconditionalBranch`, `ConditionalBranch`, `ReturnTerminator` |
| Phi nodes | `PhiValue` (SSA merge points) |
| Conversions | `Convert`, `IntAsFloat`, `FloatAsInt`, `AddressSpaceCast` |
| Comparisons | `Compare` |
| Intrinsics | `LanguageEmitValue` (inline PTX/OpenCL), `DebugAssertOperation`, `WriteToOutput` |

### IR Type System

The IR type system (`IR/Types/`) is separate from .NET's `System.Type`. It includes:

- `PrimitiveType` — `Int1`, `Int8`, `Int16`, `Int32`, `Int64`, `Float16`, `Float32`, `Float64`.
- `PointerType` — typed pointer with an `AddressSpace` tag (Generic, Global, Local, Shared, Private).
- `ViewType` — typed array view (pointer + length), used to represent `ArrayView<T>`.
- `StructureType` — named sequence of field types (lowers to C structs or PTX `.param`).
- `ArrayType` — N-dimensional array (later lowered to a `StructureType` + `ViewType`).
- `HandleType` — opaque handle (for .NET object handles, CPU only).
- `VoidType` — void return.

Address spaces are a critical feature: `Global` (device DRAM), `Shared` (per-block scratchpad), `Local` (per-thread stack), `Private` (registers). The `InferAddressSpaces` and `InferKernelAddressSpaces` passes propagate address-space information through the IR to allow the backend to generate more efficient load/store instructions.

### IRContext and IRTypeContext

`IRContext` is a mutable container holding a set of `Method` objects. It provides:
- Lookup by `MethodHandle` or `MethodBase`.
- Methods for GC (removing unreferenced IR methods).
- Thread-safe access patterns (reads under a read-lock, writes under a write-lock).

`IRTypeContext` is the flyweight factory for type nodes — the same type (e.g., `Int32`) is always the same object instance, enabling pointer equality for type comparison.

---

## Stage 3: Generic Optimization Passes

Optimizations are organized into a `Transformer` — an ordered pipeline of `Transformation` passes applied to an `IRContext`. The `Optimizer` static class provides preset pipelines:

### O0 (Debug)
- `Inliner` (if enabled) — inlines small callees (≤32 MSIL instructions).
- `SimplifyControlFlow` — merge sequential basic blocks, remove dead edges.
- `SSAConstruction` — promote allocas to SSA phi nodes (Braun et al. algorithm).
- `DeadCodeElimination` — remove values with no uses.

### O1 (Release)
All O0 passes, plus:
- `LowerArrays`, `LowerPointerViews`, `AcceleratorSpecializer`.
- `LowerStructures` — decompose aggregate structure values into scalar fields.
- `SSAStructureConstruction` — SSA-promote structure allocas.
- `InferAddressSpaces`.
- `CleanupBlocks`, `SimplifyControlFlow` (additional rounds).

### O2 (Aggressive Release)
All O1 passes, plus:
- `InferKernelAddressSpaces` (global memory propagation).
- `InferLocalAddressSpaces`.
- `LoopInvariantCodeMotion` — hoist loop-invariant computations out of loops.
- `CodePlacement` — reduce register pressure by sinking/hoisting values.
- `IfConversion` — convert short if-else chains into predicated instructions.

### Key Transformation Details

**Inliner**: Identifies `MethodCall` values targeting small or `[MethodImpl(AggressiveInlining)]` methods and replaces the call with an inlined copy of the callee's body (using `IRRebuilder`).

**SSAConstruction**: Converts `Alloca`/`Load`/`Store` patterns (stack variables) into SSA phi nodes using the Braun/Buchwald/Hack algorithm — phi nodes are inserted lazily at dominance frontiers.

**LowerArrays**: Replaces `ArrayType` with an equivalent `StructureType` containing a `ViewType` + N `Int32` dimension fields. `NewArray`, `GetArrayElement`, `SetArrayElement` values are rewritten into pointer arithmetic on the underlying view.

**LowerStructures**: Scalar-replaces aggregate structure loads and stores into individual field accesses, reducing register pressure.

**AcceleratorSpecializer**: Replaces `Accelerator.CurrentType` queries and other runtime-polymorphic properties with compile-time constants for the specific target accelerator.

**InferAddressSpaces**: A data-flow analysis that tracks pointer provenance to assign the most specific address space (instead of `Generic`) to each pointer value, enabling backend-specific addressing modes.

**IfConversion**: Converts diamond-shaped CFG patterns (`if (cond) A else B; join`) into a flat sequence with `Predicate` values, eliminating branches that cause warp divergence on GPUs.

---

## Stage 4 & 5: Backend Compilation and Code Generation

### Backend Base (`Backends/Backend.cs`)

`Backend` is an abstract class that:
1. Applies backend-specific transformation passes (via `AddBackendOptimizations`).
2. Runs the `IntrinsicSpecializer` — replaces calls to methods with `[IntrinsicImplementation]` by either redirecting to a different IR method or invoking a `GenerateCode` handler.
3. Iterates over all methods in topological order, calling `CreateCodeGenerator` to get a backend-specific code generator.

`CodeGeneratorBackend<THandler, TGeneratorArgs, TCodeGenerator, TOutput>` is a generic subclass that parameterizes the entire pipeline by the code generator type and output type.

### PTX Backend (`Backends/PTX/PTXBackend.cs`)

Targets NVIDIA GPUs via CUDA's PTX (Parallel Thread Execution) ISA.

**PTXBackend** applies additional PTX-specific transformations:
- `LowerAtomics` — lowers IR atomic operations to PTX atomic variants.
- `LowerWarpShuffles` — maps IR warp shuffle values to PTX `shfl` instructions.
- `LowerThreadIntrinsics` — maps thread synchronization primitives.

**PTXRegisterAllocator** assigns virtual registers to IR values:
- Each value gets a PTX register of the appropriate type (`.b8`, `.b16`, `.b32`, `.b64`, `.f32`, `.f64`, `.pred`).
- A separate allocator handles predicate registers.
- Phi nodes and parameters are handled as register assignments without moves.

**PTXCodeGenerator** (a partial class spanning multiple files) emits PTX text:

- `PTXCodeGenerator.cs` — manages the `StringBuilder` output, register allocator, and general infrastructure.
- `PTXCodeGenerator.Values.cs` — generates code for each IR value kind via `IValueVisitor` methods:
  - Arithmetic: `add.s32 %r1, %r2, %r3;`
  - Memory: `ld.global.f32 %f1, [%rd1];` / `st.shared.b64 [%rd2], %rd3;`
  - Control flow: `@%p1 bra label_1;`
  - Method calls: PTX `call (ret), funcname, (arg0, arg1, ...);`
  - Atomic operations: `atom.global.add.f32 %f1, [%rd1], %f2;`
  - Warp shuffles: `shfl.sync.bfly.b32 %r1, %r2, %r3, 0x1f;`
- `PTXCodeGenerator.Terminators.cs` — generates branch/return PTX instructions.
- `PTXCodeGenerator.Views.cs` — generates view pointer arithmetic.
- `PTXCodeGenerator.Emitter.cs` — low-level PTX text emission helpers.

**PTXKernelFunctionGenerator** handles the kernel entry-point specifically:
- Emits the `.entry` PTX declaration with `.param` arguments.
- Loads kernel parameters from `.param` space into registers.
- Handles dynamic shared memory allocation (`.extern .shared .align 16 .b8 __dyn_shared_alloca[];`).

**PTXFunctionGenerator** handles device (non-kernel) functions — emits `.func` PTX declarations.

The generated PTX text is passed to either NVVM (LLVM-based PTX → CUBIN) or NVRTC for JIT compilation into a cubin module, which is then loaded via the CUDA driver API.

### OpenCL Backend (`Backends/OpenCL/CLBackend.cs`)

Targets GPUs and CPUs via OpenCL. Generates OpenCL C source code.

The architecture mirrors the PTX backend:
- `CLCodeGenerator` emits OpenCL C text (`__kernel void`, `__global`, `__local`, `barrier(CLK_LOCAL_MEM_FENCE)`, etc.).
- `CLTypeGenerator` maps IR types to OpenCL C type names.
- `CLVariableAllocator` assigns named variables to IR values.
- `CLKernelTypeGenerator` generates the kernel argument struct and accessor code.
- The generated OpenCL C source is compiled at runtime using `clBuildProgram`.

### CPU / IL Backend (`Backends/IL/ILBackend.cs`)

The CPU backend re-emits the IR as .NET IL using `System.Reflection.Emit.ILGenerator`. This is not just an interpreter — it generates native-calling-convention .NET delegates that are JIT-compiled by the CLR.

`CPUAccelerator` (in `Runtime/CPU/`) uses .NET threads to simulate GPU grid/group/warp execution:
- Each kernel invocation spawns or reuses a thread pool.
- `CPURuntimeThreadContext` (thread-local) tracks current `GridIndex`, `GroupIndex`, `WarpIndex`, lane ID.
- `CPURuntimeGroupContext` manages per-group shared memory and barriers using .NET synchronization primitives.
- `CPURuntimeWarpContext` manages warp-level operations.

The CPU backend is extremely valuable for debugging — all GPU kernel code can be single-stepped through in a regular debugger.

### Intrinsic System

The intrinsic system bridges the gap between GPU hardware features and C# user code.

**Frontend intrinsics** (`Frontend/Intrinsic/`): C# methods are decorated with custom attributes:
- `[GridIntrinsic(GridIntrinsicKind.GetGridIndex, DeviceConstantDimension3D.X)]` → `Grid.IdxX` is compiled directly to a `GridIndexValue` IR node.
- `[GroupIntrinsic(...)]` → `Group.IdxX`, `Group.Barrier()`, etc.
- `[WarpIntrinsic(...)]` → `Warp.LaneIdx`, `Warp.Shuffle(...)`, etc.
- `[AtomicIntrinsic(...)]` → `Atomic.Add(ref T, T)`, etc.

**Backend intrinsics** (`IR/Intrinsics/`): The `IntrinsicImplementationManager` maps method handles to `IntrinsicImplementation` records, which have two modes:
- `Redirect`: Replace the method call in the IR with a call to a different (e.g., PTX-specific) method.
- `GenerateCode`: Invoke a custom `IBackendCodeGenerator` handler that emits backend-specific instructions directly.

PTX-specific intrinsics include `PTXIntrinsics` which maps math operations to `libdevice` NVVM functions, inline PTX assembly (`Cuda.Ptx.Asm`), and CuBLAS/cuFFT interop.

---

## Stage 6: Kernel Launcher Generation

When `accelerator.LoadAutoGroupedStreamKernel<TIndex, T1, T2>` is called, ILGPU:

1. Compiles the target method (or retrieves it from the `KernelCache`).
2. Creates an `EntryPointDescription` identifying the kernel, index type, and parameter types.
3. Generates a **kernel launcher delegate** — a strongly-typed `Action<TIndex, T1, T2>` (or similar) that, when invoked, performs the GPU launch with zero boxing.

The launcher is generated by `KernelLauncherBuilder` using `ILEmitter` (`Reflection.Emit`). It:
- Maps typed parameters to backend-specific argument buffers.
- Computes the grid/block dimensions (for auto-grouped kernels, based on device occupancy hints).
- Calls the native CUDA/OpenCL API to launch the compiled kernel.

`ArgumentMapper` (per-backend in `Backends/EntryPoints/`) handles the conversion of .NET value-type parameters (structs, `ArrayView<T>`, etc.) into the binary layout expected by the kernel ABI.

---

## The Algorithms Library (ILGPU.Algorithms)

`ILGPU.Algorithms` builds on top of the core ILGPU runtime to provide reusable, portable parallel primitives.

### Parallel Primitives

| Primitive | Description |
|---|---|
| **Reduce** | Tree reduction across all threads: sum, min, max, logical-and/or |
| **Scan** | Prefix sum (inclusive/exclusive) across all threads |
| **RadixSort** | GPU radix sort using scan as a subroutine |
| **Histogram** | Multi-bin histogram with atomics |
| **Transform** | Element-wise map over an array |
| **Sequence** | Fill an array with a user-defined sequencer |
| **Initialize** | Fill with a constant value |
| **Unique** | Remove adjacent duplicates |
| **Reorder** | Scatter/gather |
| **WarpExtensions** | Warp-level reduce/scan using shuffle instructions |
| **GroupExtensions** | Thread-group-level reduce/scan using shared memory |
| **XMath** | GPU-optimized math functions (sin, cos, exp, sqrt, etc.) |
| **Random** | GPU-side PRNG (XorShift, etc.) |
| **MatrixOperations** | Dense matrix multiply using tiled shared memory |
| **Optimization** | Gradient descent, etc. |

Operations are expressed using generic type parameters constrained to operator interfaces (`IScanReduceOperation<T>`, `IRadixSortOperation<T>`), enabling monomorphization per type without virtual dispatch overhead.

### Algorithm Context

`AlgorithmContext` associates algorithm state (temp buffers, etc.) with an `Accelerator`. The `AlgorithmContextMappings` (T4-generated) register algorithm implementations per accelerator type.

---

## Memory Model

ILGPU exposes GPU memory spaces directly:

- **Global memory** (`MemoryBuffer`, `ArrayView<T>`): Device DRAM, accessible from all threads; allocated with `accelerator.Allocate1D<T>`.
- **Shared memory** (`SharedMemory.Allocate<T>`, `SharedMemory.AllocateDynamic<T>`): Per-thread-group scratchpad; declared statically (fixed size) or dynamically (runtime size).
- **Local memory** (`LocalMemory.Allocate<T>`): Per-thread private stack; stored in registers or local DRAM.
- **Page-locked (pinned) memory** (`PageLockedArrays`): Host-side pinned memory for async DMA transfers.

`AcceleratorStream` wraps a CUDA stream or OpenCL command queue, allowing asynchronous memory copies and kernel launches. Multiple streams enable overlap of computation and data transfer.

---

## Control Flow and Thread Hierarchy

ILGPU models the standard GPU thread hierarchy:

| Concept | ILGPU API | PTX equivalent |
|---|---|---|
| Grid | `Grid.IdxX/Y/Z`, `Grid.DimX/Y/Z` | `%ctaid.x`, `%nctaid.x` |
| Thread group (block) | `Group.IdxX/Y/Z`, `Group.DimX/Y/Z` | `%tid.x`, `%ntid.x` |
| Warp | `Warp.LaneIdx`, `Warp.WarpSize` | lane within warp |
| Group barrier | `Group.Barrier()` | `bar.sync 0;` |
| Predicate barrier | `Group.Barrier(predicate)` | `bar.red.and.pred` |
| Warp shuffle | `Warp.Shuffle(value, srcLane)` | `shfl.sync.idx.b32` |
| Warp ballot | `Warp.Ballot(predicate)` | `vote.sync.ballot.b32` |
| Memory fence | `MemoryFence.SystemLevel()` etc. | `membar.sys` |

The `IfConversion` pass optimizes warp divergence by converting short conditional branches to predicated execution where profitable.

---

## Code-Generation T4 Templates

Many repetitive parts of the codebase are generated by T4 (`.tt`) template files:

- `IndexTypes.tt` → `Index1D`, `Index2D`, `Index3D`, `LongIndex1D`, etc.
- `StrideTypes.tt` → stride type hierarchy for multi-dimensional views.
- `MemoryBuffers.tt` → `Allocate1D/2D/3D` overloads on `Accelerator`.
- `AtomicFunctions.tt` → `Atomic.Add`, `Atomic.Exchange`, etc.
- `PTXIntrinsics.Generated.tt` → PTX math intrinsic mappings.
- `PTXLibDeviceMethods.tt` → NVVM libdevice math function signatures.
- `ArithmeticOperations.tt` → Binary/unary arithmetic IR values.
- `ScanReduceOperations.tt` → Type-specialized scan/reduce implementations.

This T4 approach avoids code duplication while keeping the generated code fully visible and inspectable.

---

## Extension and Plugin Points

ILGPU provides several extension points for library authors and language implementers:

### Custom Intrinsics

Any .NET method can be given a backend-specific implementation by registering an `IntrinsicImplementation` with the `IntrinsicImplementationManager`. The `[IntrinsicImplementation]` attribute (along with backend-specific subclasses `[PTXIntrinsic]`, `[CLIntrinsic]`) enable this at the class level.

Example pattern:
```csharp
[PTXIntrinsic(PTXIntrinsicKind.Cos)]
public static float Cos(float x) => MathF.Cos(x); // CPU fallback
```

### Backend Extensions and Context Extensions

Both `Backend` and `Context` support `CachedExtension` objects — arbitrary data attached to these objects by key type, enabling third-party libraries (e.g., cuBLAS, cuFFT wrappers) to piggyback on the lifetime of core objects.

### Custom Accelerator Builders

Implementing `IAcceleratorBuilder` and registering it with the context allows new accelerator types to be plugged in.

---

## Debugging and Diagnostics

- **CPU Accelerator**: All kernel code is executed on the CPU with full debugger support (breakpoints, watches, stack traces).
- **Debug assertions**: `Interop.WriteLine` and `Interop.WriteLineError` emit printf-like output from GPU kernels (mapped to `cuPrintf` / OpenCL `printf`).
- **`DebugAssert`**: Kernel-side assertions that trigger at runtime.
- **Verification**: The `Verifier` class checks IR invariants (type consistency, SSA correctness, terminator presence) in debug builds.
- **IR Dump**: `Method.Dump(TextWriter)` emits a human-readable textual representation of the IR for inspection.
- **Source mapping**: Sequence points from `.pdb` files are propagated through the IR and optionally emitted as PTX `.loc` directives.
- **Profiling markers**: `ProfilingMarker` objects wrap CUDA events / OpenCL profiling events for kernel timing.

---

## Relevance to Remora Language Implementation

Remora (arXiv:1912.13451; Shivers et al.) is a rank-polymorphic, functional array language in the APL/J tradition. Its key semantic features are:

1. **Rank polymorphism**: Functions lift automatically over arrays of higher rank than expected. A scalar add lifts to elementwise addition over arrays of any shape.
2. **Shape types**: Array types carry their shape (a tuple of extents) as part of the type. Shape variables enable generic, rank-polymorphic functions.
3. **Lifting semantics**: Given a function expecting a frame of rank `k`, applying it to an array of rank `k+n` lifts the function to operate on each rank-`k` cell, producing a rank-`n` result frame.
4. **First-class functions and closures**: Higher-order array operations (map, fold, scan over rank-polymorphic functions).
5. **Dependent array types**: The type system tracks shapes as dependent types.

### ILGPU as a Backend for Remora

ILGPU provides the following foundations relevant to a Remora GPU backend:

#### 1. Multi-Dimensional Memory and Views

`ArrayView<T, TIndex>` with strided access (`Stride`) and multi-dimensional buffers (`Allocate2D`, `Allocate3D`) map naturally to Remora's array cells of varying rank. The `StrideTypes.tt`-generated types (`XStride`, `YStride`, `GeneralArrayView`) give precise control over in-memory layout.

For a Remora implementation, each runtime array would be represented as an `ArrayView` plus a shape descriptor (a small struct of dimension extents), analogous to how `LowerArrays.cs` represents `ArrayType` as a `ViewType` + dimension integers.

#### 2. IR as the Target for Remora Lowering

A Remora compiler can target ILGPU's IR directly, bypassing the MSIL frontend entirely. `IRBuilder` provides a programmatic API for constructing IR values:
- `builder.CreateArithmetic(loc, left, right, BinaryArithmeticKind.Add)` for element-wise arithmetic.
- `builder.CreateLoad(loc, pointer, alignment)` / `builder.CreateStore(...)` for array element access.
- `builder.CreatePhi(loc, type)` + `AddArgument(block, value)` for rank-polymorphic dispatch merge points.
- `builder.CreateCall(loc, method, args)` for lifted function application.

Each Remora "cell operation" (applying a rank-`k` function to each cell of a rank-`k+n` array) maps to a GPU kernel where the thread index encodes the cell position in the frame dimensions.

#### 3. Shape-Driven Kernel Launch

The Remora lifting semantics require iterating over the frame dimensions (the "excess" dimensions beyond the function's expected rank). On a GPU, this maps naturally to the grid/group thread hierarchy:
- The frame shape determines the kernel grid dimensions (or 1D flattened index into the frame).
- The cell shape determines per-thread work (or a nested inner loop).

ILGPU's `KernelConfig` (explicit grouping) allows full control over grid and block sizes, making it straightforward to dispatch over arbitrary frame shapes.

#### 4. Specialization and Monomorphization

Remora's dependent shape types require knowing shapes at compile time (or specializing per shape). ILGPU's `KernelSpecialization` and the `SpecializationCache` support creating specialized kernel variants for different parameter values — directly applicable to specializing kernels per concrete shape.

`AcceleratorSpecializer` demonstrates the pattern of replacing abstract queries with concrete values during compilation, which is exactly what a shape-specialized Remora kernel would need.

#### 5. Rank-0 Kernels as Scalar Primitives

In Remora, rank-0 functions (scalar primitives) are the leaves of the lifting hierarchy. These map directly to ILGPU kernel device functions — `[MethodFlags.Intrinsic]` or simple inlineable device functions emitting, e.g., a single PTX `add.f32`.

#### 6. Scan, Reduce, and Fold over Arrays

Remora's `fold` (reduction) and similar combinators map to ILGPU.Algorithms' `ReductionExtensions` and `ScanExtensions`, which already implement parallel prefix sum and reduction with generic operator types — covering common Remora reduction patterns.

#### 7. Higher-Order Functions and Closures

ILGPU kernels are C# generic methods, and ILGPU performs full monomorphization (each type specialization is a distinct compiled kernel). A Remora function value passed to a higher-order kernel (e.g., `map f arr`) would be represented as a generic type parameter constrained by an interface, with ILGPU generating a specialized kernel per concrete `f`. This is the same pattern used by `IScanReduceOperation<T>` in ILGPU.Algorithms.

#### 8. Shape Polymorphism at the IR Level

For a full Remora implementation, the key extension point is augmenting the IR type system (adding a shape-indexed array type to `IRTypeContext`), adding shape-aware lowering passes (analogous to `LowerArrays.cs`), and generating kernel launches that compute from runtime shape descriptors. Because ILGPU's IR types, transformations, and backends are all generic/pluggable, these extensions slot in at well-defined interfaces without requiring modification of the core backends.

### Recommended Integration Architecture

A Remora→ILGPU implementation would likely follow this structure:

```
Remora source
     │
     ▼
Remora parser & type checker (shape inference, rank lifting)
     │
     ▼
Remora IR (rank-polymorphic array expressions + shape metadata)
     │
     ▼
Remora→ILGPU lowering pass
     │  For each lifted application over frame dims → new ILGPU Method
     │  Shape descriptor = small ILGPU StructureValue (Int32 × rank)
     │  Cell iteration → GPU thread index decoding
     │  Scalar primitives → inline ILGPU IR arithmetic
     ▼
ILGPU IRContext (populated programmatically via IRBuilder)
     │
     ▼
ILGPU standard optimization + backend pipeline
     │  (all existing PTX/OpenCL/CPU backends unchanged)
     ▼
GPU execution (CUDA / OpenCL / CPU)
```

The main new components needed:
1. A shape type representation in ILGPU IR (or as a convention using existing `StructureType`).
2. A Remora "lifting code generator" that emits ILGPU IR for the lifted-application loop pattern.
3. A runtime shape-descriptor protocol for passing shapes to kernels.
4. Potentially: a shape-specialized `KernelCache` variant that caches per concrete shape tuple.

---

## Summary of Key Design Principles

| Principle | How ILGPU implements it |
|---|---|
| **Zero-overhead abstractions** | `ArrayView<T>` and index types are value types; kernel launchers are generated delegates — no boxing |
| **No native dependencies** | Entire compiler in C#; PTX/OpenCL are text strings passed to vendor JIT compilers |
| **Backend portability** | Abstract IR + per-backend code generators; same kernel C# code runs on Cuda/OpenCL/CPU |
| **Transparent compilation** | C# methods compiled as-is via MSIL disassembly; no special markup needed |
| **Intrinsic extensibility** | Per-backend intrinsic registry allows hardware features without IR changes |
| **SSA-based IR** | Simplifies optimization (DCE, CSE, SSA-based analyses); explicit phi nodes |
| **Layered optimization** | Generic passes first, then backend-specific lowering, then final DCE/CF cleanup |
| **Debuggability** | CPU backend provides full .NET debugging experience for all GPU kernels |

---

*Document generated from source analysis of the ILGPU repository (m4rs-mt/ILGPU), targeting audience: language implementers considering ILGPU as a GPU compilation backend, particularly for the Remora rank-polymorphic array language.*

---

## Suitability Assessment: ILGPU as a Remora GPU Backend

### What Fits Well

#### 1. The Lifting → Kernel-Launch Mapping is Natural

Remora's core operation — apply a rank-*k* function to each rank-*k* cell of a rank-*k+n* array — maps cleanly onto a GPU kernel where the thread index encodes position in the *n*-dimensional frame. ILGPU's `KernelConfig` gives full control over grid/block dimensions, and the `IRBuilder` API lets you construct the cell-indexing arithmetic programmatically. This is the best-aligned part of the two systems.

#### 2. Multiple Backends for Free

A Remora implementation targeting ILGPU gets CUDA, OpenCL, and a multi-threaded CPU backend immediately. The CPU backend is particularly valuable during development — Remora kernels can be single-stepped in a standard .NET debugger, which is otherwise very painful on GPU hardware.

#### 3. Programmatic IR Construction

`IRBuilder` is a complete API for constructing IR without going through the MSIL frontend. A Remora compiler can target it directly, treating ILGPU's IR as a typed assembly language. The SSA form, type system, and optimization passes are all available without modification.

#### 4. Generic Monomorphization

ILGPU monomorphizes fully: each C# generic instantiation produces a distinct compiled kernel. For Remora programs where rank and element type are statically known, this is exactly right — a dedicated, optimized kernel is generated per concrete array type.

#### 5. Algorithms Library

Parallel scan, reduce, radix sort, and histogram are already implemented correctly and portably. Remora's `fold` and related combinators map directly onto these primitives.

---

### Significant Gaps

#### 1. No Kernel Fusion / Deforestation

This is the most serious performance problem. Remora's functional style naturally produces chains of lifted operations:

```
map f (map g arr)  →  ideally one single kernel pass
```

ILGPU has no fusion mechanism. Every lifted operation becomes a separate kernel launch, and every intermediate array must be materialized in global memory. For Remora programs that compose many operations, this produces severe memory bandwidth pressure. Systems like Futhark and JAX's XLA backend invest heavily in fusion precisely for this reason. This gap would require building a fusion pass over ILGPU IR before entering the standard backend pipeline.

#### 2. No First-Class Function Values in Kernels

ILGPU monomorphizes at *compile time* via C# generics. It cannot accept a function as a *runtime value* passed into a kernel. Higher-order Remora patterns like `(λ f → map f arr)` where `f` is unknown until runtime have no direct representation. The available options are:

- **Compile-time specialization only**: works for statically-known function arguments but limits expressiveness.
- **Church-encode as a sum type with a switch**: exhaustively enumerate all possible `f` values; kills performance through branch divergence and requires a closed universe of functions.
- **Re-JIT per value of `f`**: adds per-call compilation latency and requires a caching layer on top of `KernelCache`.

#### 3. Memory Management Mismatch

Remora is functional; intermediate arrays are conceptually immutable and garbage-collected. ILGPU requires explicit `Allocate`/`Dispose` for every buffer. A GPU memory allocator and pool/GC layer must be built on top of the raw ILGPU allocation API. This is non-trivial and is a source of fragmentation when intermediate arrays are frequent and varied in size.

#### 4. Dynamic Shapes Require Runtime Dispatch or Per-Shape Compilation

ILGPU's IR type system is static — there are no dependent types or shape variables. A Remora array of type `int[n, m]` where `n` and `m` are runtime values must be represented as an `ArrayView<int>` plus a shape descriptor struct. This works at runtime but means:

- Shape-polymorphic kernels must be written uniformly (one kernel, shapes passed as arguments), forgoing optimizations that come from knowing shapes at compile time.
- Alternatively, a new kernel can be JIT-compiled per concrete shape, adding latency. ILGPU's `KernelCache` and `KernelSpecialization` partially support this, but were not designed for arbitrary shape tuples.

#### 5. No Device-Side Kernel Launches (No Dynamic Parallelism)

Remora's nested array structure can produce *irregular* parallelism — cells in the frame may have different shapes in ragged arrays. ILGPU does not support CUDA dynamic parallelism (launching kernels from within kernels). Handling ragged arrays requires either padding-plus-masking or a sequential outer loop on the host. The latter eliminates GPU parallelism across the outer dimensions.

#### 6. Rank Polymorphism Requires a Metaprogramming Layer

ILGPU has no notion of rank. A rank-*k*-generic kernel requires either:
- A fixed upper bound on rank with a runtime rank variable (awkward, generates dead code for unused dimensions), or
- Generating a separate kernel per concrete rank via JIT compilation (suitable for fully monomorphic programs; adds latency otherwise).

---

### Summary Verdict

| Concern | Severity | Notes |
|---|---|---|
| No kernel fusion | **Critical** for performance | Requires a custom fusion pass over ILGPU IR |
| No first-class function values | **High** for dynamic higher-order programs | Compile-time specialization only; limits expressiveness |
| Manual memory management | **High** | GPU allocator/pool layer must be built |
| No dependent/shape types in IR | **Medium** | Shapes as runtime structs; accept uniform kernels |
| No dynamic parallelism | **Medium** | Restrict to regular arrays, or use host-side outer loops |
| Rank polymorphism | **Medium** | Per-rank monomorphization at JIT time |
| Per-variant compilation latency | **Low–Medium** | `KernelCache` + warm-up phase mitigates this |

**Bottom line**: ILGPU is a *workable but incomplete* foundation. It handles the mechanics of GPU code generation (PTX/OpenCL emission, memory layout, thread-index arithmetic) and provides a multi-backend IR with solid optimization passes. The gaps all sit *above* the backend level — fusion, higher-order function values, and memory management — meaning a significant layer must be built between a Remora front-end and ILGPU's APIs. That layer is essentially what systems like Futhark or Dex implement internally.

### Comparison to Alternative Backends

| Backend | Fusion | HOF values | Shape types | C# ecosystem | Notes |
|---|---|---|---|---|---|
| **ILGPU** | None | Compile-time only | None | ✓ | Easiest to integrate; CPU backend great for debugging |
| **LLVM (LLVMSharp)** | Via loop passes | None | None | Partial | Lower level; NVPTX target; full IR extensibility |
| **Futhark** | ✓ (aggressive) | Limited | ✓ | External | Solves most hard problems; call from C# via FFI |
| **MLIR (linalg/affine)** | ✓ (excellent) | Limited | ✓ | None | Most principled; `linalg` designed for rank-polymorphic ops |
| **JAX/XLA** | ✓ (XLA HLO) | ✓ (Python) | ✓ | None | Python ecosystem only |

For a *research prototype* or *exploratory implementation* — especially one that values .NET interoperability and debuggability — ILGPU is a reasonable choice. The CPU backend makes correctness verification tractable, the codebase is clean and well-structured C#, and GPU kernels can be running quickly.

For a *production system* targeting competitive performance with Futhark or JAX, the absence of kernel fusion is the dominant blocker and would require substantial engineering investment. In that case, MLIR's `linalg` dialect (designed explicitly for rank-polymorphic array operation lowering with fusion) or Futhark as a backend would be a more principled long-term foundation.

---

## Implementing the Missing Features on Top of ILGPU

The three critical gaps — kernel fusion, higher-order function values, and memory management — are all implementable *above* the ILGPU layer in a Remora→ILGPU lowering pass, with no modifications to ILGPU itself. ILGPU is used as a typed assembly language: the Remora compiler generates its IR programmatically via `IRBuilder`, and the standard ILGPU backends take it from there. The sections below assume regular (non-ragged) arrays; device-side kernel launches for irregular structures are out of scope.

---

### Kernel Fusion

#### The Right Level to Implement It

Fusion should be implemented at the **Remora IR level**, before any ILGPU IR is generated. This is both simpler and more powerful than trying to merge ILGPU `Method` objects after the fact:

```
Remora operation graph  →  fusion pass  →  fused Remora ops  →  ILGPU IR generation
```

Each node in the operation graph is a lifted application (a potential kernel). The fusion pass merges adjacent nodes whose shapes are compatible and whose dependency relationship is producer→consumer.

#### Fusible Patterns

**Map-chain fusion** (`map f ∘ map g`): the simplest and most common case. Compose `f` and `g` in the Remora IR and generate a single ILGPU kernel for the composition. No synchronization boundary is required, and every intermediate array element is consumed immediately from a register.

**Map-reduce fusion** (`reduce f (map g arr)`): the reduce already reads every element once; fusing the map eliminates a global memory roundtrip entirely. Requires recognizing the pattern and generating a single ILGPU kernel that applies `g` per element before accumulating. Standard technique.

**Horizontal fusion** (two independent `map`s over the same array): fuse into one kernel with two outputs. Less critical for correctness, meaningful for performance when memory bandwidth is the bottleneck.

**Scan fusion**: harder because scans have synchronization barriers at their boundaries. Best left for a later pass.

#### How It Works in ILGPU IR

After the Remora-level rewrite, the fused operation is a single Remora node that the lowering pass converts to a single ILGPU `Method`. The method body contains the inlined composition of the constituent operations. ILGPU's existing DCE and inliner passes then clean up any residual dead code. No new ILGPU infrastructure is required.

#### Difficulty

**Moderate.** The core algorithm is: build a DAG of Remora operations, apply rewrite rules that merge fusible pairs (map-map, map-reduce), and emit one ILGPU method per fused group. Estimated effort for map-chain and map-reduce fusion covering the common cases: **3–6 weeks**.

---

### Higher-Order Function Values

#### The Right Technique: Defunctionalization

Defunctionalization (Reynolds 1972) is the standard, correct approach for higher-order functions in a compiled language targeting hardware with no cheap indirect calls:

1. Enumerate all function values in the program (closed-world assumption — appropriate for an AOT compiler).
2. Replace each function value with a tagged integer (its "closure tag"), plus a struct of captured variables.
3. Replace each application site `apply(f, x)` with a `switch` on the tag dispatching to the concrete function bodies.

In ILGPU IR this becomes a `StructureValue` (tag field + captured-variable fields) and a chain of conditional branches. ILGPU's `IfConversion` pass then converts short dispatch chains to predicated instructions, eliminating branches where profitable.

#### The Warp Divergence Concern

If threads in the same warp take different branches of the dispatch (i.e., apply different functions to different elements), warp divergence occurs. In practice for Remora programs this is rarely a problem: a `map f arr` applies the *same* `f` uniformly to every element. Divergence only arises if `f` is itself a conditional over data (which `IfConversion` already handles) or if per-element function selection is explicit — a rare pattern that can be handled with lane-masking.

#### Implementation

The lowering pass requires:
- A pass over the Remora IR that collects all lambda values and assigns integer tags.
- A `ClosureType` struct generator that maps each lambda to a `StructureType` containing its free variables.
- Rewriting all application sites to construct and dispatch the closure struct.
- Generating the ILGPU IR switch sequence.

#### Difficulty

**Moderate, and well-bounded.** Defunctionalization is a textbook transformation with no GPU-specific complications. Estimated effort: **2–3 weeks**. It integrates cleanly with the rest of the lowering pipeline.

---

### Memory Management

#### The Key Insight: Fusion Simplifies GC Considerably

After a fusion pass, most intermediate arrays disappear entirely — they live in registers within the fused kernel. The remaining intermediate arrays are exactly those that cross fusion boundaries: one kernel produces them, a later kernel consumes them, and they have no other uses. This reduces memory management to a **liveness problem on buffers**, not a full GC problem.

#### Static Lifetime Analysis + Buffer Pool

For regular Remora arrays, the right approach is a compile-time allocator:

1. **Liveness analysis** on the Remora operation graph: for each intermediate array, determine the last operation that reads it (its death point).
2. **Buffer pool**: pre-allocate a set of GPU memory blocks. Assign arrays to blocks using a linear-scan or graph-coloring allocator — exactly like register allocation, but for memory. Arrays whose lifetimes do not overlap can share the same block.
3. **Reuse**: if array A's lifetime ends before array B is allocated, and both have the same element type and total size, B can reuse A's memory region with zero allocation cost.

This is a purely compile-time assignment — no runtime GC mechanism is needed. All `Allocate`/`Dispose` calls to ILGPU are made at program initialization and teardown, not during execution.

#### Dynamically-Sized Arrays

When shapes are not known at compile time, static buffer assignment is not possible. The fallback is a **bump allocator arena** per top-level program execution:

- At launch time, allocate a large arena via `accelerator.Allocate1D<byte>(estimatedSize)`.
- Sub-allocate intermediate arrays from the arena at runtime using shape-product calculations (runtime arithmetic in the host wrapper, not inside GPU kernels).
- Free the entire arena when the top-level operation completes.

This gives O(1) allocation cost, no fragmentation within a single program run, and requires no on-GPU GC logic. The arena size can be estimated conservatively using a shape-propagation pass over the Remora IR, or bounded by a user-configurable limit.

#### Implementation Parts

| Part | Difficulty |
|---|---|
| Liveness analysis on operation graph | Low |
| Buffer pool + linear-scan allocator (static shapes) | Moderate |
| Arena allocator (dynamic shapes) | Low |
| Shape propagation for static arena sizing | Moderate |

#### Difficulty

**Moderate.** A first implementation using an arena-per-execution is straightforward: **2–3 weeks**. The full static buffer-reuse allocator for known shapes adds another **3–4 weeks** and delivers meaningfully better memory utilization for programs with many intermediate arrays.

---

### Combined Effort Summary

| Feature | Approach | Estimated Effort | ILGPU modifications? |
|---|---|---|---|
| Kernel fusion | Dataflow rewrite above ILGPU IR | 3–6 weeks | None |
| Higher-order functions | Defunctionalization | 2–3 weeks | None |
| Memory management | Liveness analysis + buffer pool / arena | 4–6 weeks | None |
| **Total** | | **~2–4 months** | **None** |

All three features are implemented entirely in the Remora→ILGPU lowering layer. ILGPU itself requires no modification, which means the full CUDA/OpenCL/CPU backend machinery remains available and stable throughout.

### Recommended Implementation Order

1. **Correct execution first**: no fusion, arena GC, defunctionalization for HOFs. This gives a working but potentially slow end-to-end system and makes it possible to validate correctness on the CPU backend.
2. **Add fusion**: eliminates most intermediate arrays, substantially reduces memory pressure, and simplifies the GC problem (fewer and longer-lived buffers remain).
3. **Refine memory management**: replace the arena with static lifetime analysis and buffer reuse on the now-simpler post-fusion allocation graph.
