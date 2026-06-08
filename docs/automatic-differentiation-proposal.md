# Architectural Proposal: Natively Differentiable Remora via LLVM

This document outlines the design, rationale, and implementation strategy for integrating Reverse-Mode Automatic Differentiation (AD) directly into a new, ahead-of-time (AOT) compiled implementation of the Remora programming language targeting the LLVM compiler infrastructure.Unlike runtime-graph ML frameworks (e.g., PyTorch) or dynamically typed prototypes, this architecture leverages Remora’s restricted dependent type system to perform type-safe, ahead-of-time symbolic differentiation before type erasure, emitting high-performance, statically allocated vector instructions via LLVM.

## 1. The Core Idea

The core idea is to transform Remora from a shape-safe array language into a native Differentiable Programming Language.

By introducing a first-class language primitive—the gradient operator grad—the compiler natively understands the mathematics of the chain rule.racket

```
;; Explicitly Typed Remora with Integrated AD

(define train-step
  (grad (λ ([weights (Arr Float (Shp 128 64))] 
            [inputs  (Arr Float (Shp 64))])
          (loss-fn weights inputs))))

```

When the compiler encounters grad, it executes a Source-to-Source Typed Transformation Pass on the Abstract Syntax Tree (AST). Instead of generating a dynamic tape at runtime, it evaluates the exact geometric shape transformations of the neural network at compile time. It then synthesizes a perfectly optimized, statically bound backward-pass expression that maps directly to parallelized LLVM Intermediate Representation (IR).

## 2. Rationale & Structural Synergy

Integrating Reverse-Mode AD into an LLVM-backed, explicitly typed Remora compiler resolves the most critical performance and safety engineering bottlenecks found in modern machine learning infrastructure.

### A. Zero-Overhead Static Tape Allocation

#### The Problem in ML: Reverse-mode AD requires a forward execution "tape" to store intermediate values needed during backpropagation. Frameworks like PyTorch must dynamically allocate and garbage-collect these memory buffers on the GPU at runtime because the shapes are unknown ahead of time, causing massive latency spikes.

#### The Remora/LLVM Solution: Because Remora’s dependent type system computes and guarantees exact multi-dimensional shapes (Shp) at compile time, the AD pass knows the precise byte-size of every intermediate tensor. The compiler can generate a single, static memory allocation layout for the entire forward/backward cycle. LLVM translates this into stack allocations or fixed, immutable heap pointers, entirely eliminating runtime allocations.

### B. Compile-Time Shape Verification of Gradients

#### The Problem in ML: A tensor dimension mismatch (e.g., an incorrect matrix transpose during a custom gradient calculation) usually escapes detection until the code executes, resulting in a runtime crash hours into a training cluster run.The Remora/LLVM Solution: By running the AD pass before type erasure, the generated Vector-Jacobian Product (VJP) functions inherit strict dependent type signatures. If a forward layer maps a tensor from (Shp B F) to (Shp F O), the type engine mathematically proves at compile time that the generated gradient cotangents match the inverted coordinate spaces, preventing shape errors from ever reaching execution.C. Seamless Interleaving with LLVM VectorizationThe Problem in ML: Separating the differentiation engine from the code-generation backend stops optimizations from traversing the boundary between the forward and backward passes.

#### The Remora/LLVM Solution: By emitting both the forward step and the synthesized backward sweep into a unified LLVM IR stream, LLVM’s advanced optimization pipeline treats them as a singular program. It can execute loop fusion (merging the backward gradient computation directly into the forward consumption loop), dead-code elimination on unused intermediate gradients, and aggressive SIMD/vectorization across array frames.

## 3. Implementation Approach

The compiler pipeline is structured as an ahead-of-time (AOT) translator. The architecture is split into three main phases: Type Checking, the Pre-Erasure AD Pass, and Monomorphization/LLVM IR Generation.

```
       [ Source Code ]
              ↓
    [ 1. Dependent Type Checker ]
              ↓
  === [ 2. Pre-Erasure AD Pass ] ===  <-- Target Transformation Phase
              ↓
    [ 3. Type Erasure & Monomorphization ]
              ↓
    [ 4. LLVM IR Code Generation ]
              ↓
       [ Machine Code / GPU Binary ]
```

## Phase 1: Dependent Type Checking

The compiler reads the raw Remora AST and verifies all dependent math variables utilizing its restricted Dependent ML (DML) index language. Every expression \(e\) is verified and assigned a structural shape type:

\(\Gamma \vdash e:(\text{Arr\ }\tau \text{\ }\iota )\)

Where \(\tau \) is the scalar atom type (e.g., Float) and \(\iota \) is the shape index (e.g., (Shp 3 4)).

## Phase 2: Pre-Erasure AD Pass (Source-to-Source)

The compiler intercepts the fully typed AST before shapes are erased. The AD engine transforms every numeric expression computing an array into a dependent pair (\(\Sigma \)-type) called a Differentiable Term:

\(\text{Dual}(e)=\langle \text{Forward\ Value},\text{Pullback\ Function}\rangle \)

For every core array primitive inside Remora, the compiler maintains an explicit translation rule that synthesizes the corresponding typed pullback (VJP).

### Concrete Transformation Mapping:

#### Scalar Atoms: A scalar node \(x\) transforms into a pair of its current value and an identity mapping pullback function.

#### Matrix Multiplication (matmul): Given an expression multiplying matrix \(A\) (shape \([M \times N]\)) and matrix \(B\) (shape \([N \times P]\)), the AD pass generates:

##### Forward element: The standard matrix multiplication result of shape \([M \times P]\).

##### Pullback element: A generated, dependently typed function Π([M Dim], [N Dim], [P Dim]) that takes an incoming cotangent matrix of shape \([M \times P]\) and automatically applies transposed matrix multiplication rules to output gradients matching the exact shapes of \(A\) and \(B\).

#### Rank-Polymorphic Lifting: When a function is lifted over higher-dimensional frames, the AD pass lifts the pullback function along the exact same frame boundaries. The "iteration space" tracked by Remora's types is cleanly duplicated for the reverse execution sweep.

Once the AD pass completes, it outputs a brand new, valid Remora AST where all backward loops are explicitly realized and strictly typed.

## Phase 3: Type Erasure & Monomorphization

With the gradient equations explicitly generated and validated, the compiler executes its type-erasure pass.

### All compile-time Pi (Π) abstractions and shape parameters are stripped away.

### The compiler resolves dimension-polymorphic functions by monomorphizing them—generating specialized, raw execution functions for the specific array sizes used by the model.

## Phase 4: LLVM IR Generation

The monomorphized, flat AST maps directly to the LLVM backend:

### Memory Management: The calculated shapes are converted into static alloca (stack allocation) instructions or immutable malloc blocks at the entry point of the compiled execution kernel.

### Loop Structures: Remora’s implicit iteration frames are translated into heavily optimized br (branching) structures representing explicit, multi-dimensional loops.

### Vectorization: Vector instructions are explicitly decorated with LLVM loop metadata (such as llvm.loop.vectorize.enable and llvm.mem.parallel_loop_access), forcing the LLVM backend to compile the forward/backward sweeps into AVX-512 vector instructions or target specific GPU hardware pipelines via NVPTX.

## 4. Summary of System Impact

| Feature | Legacy ML Frameworks | Integrated LLVM-Remora Architecture |
| Tape Allocation | Dynamic allocations at runtime | Ahead-of-time static layout definition |
| Shape Validation | Runtime crash upon error | Prevented via type engine at compile time |
| Optimization Boundary | Separate forward/backward code | Unified optimization stream inside LLVM |
| Loop Overhead | High interpreter framework tracing | Flat, loop-fused native hardware instructions |

## Open Questions

### The syntax structure for defining custom gradients for new primitives.

### How the type engine should model dynamic batch dimensions within Presburger arithmetic constraints.

### The specific strategy for compiling conditional control flow through the reverse pass. |
