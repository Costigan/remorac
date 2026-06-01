# Remora Dense Core ABI

This document specifies the external execution ABI for Remora Dense Core kernels. It is normative for the first implementation.

## Scope

The ABI covers dense rank-0 through rank-10 arrays with static ranks and static dimensions. It is view-capable: descriptors include offset, sizes, and strides even when the first allocated arrays are contiguous row-major.

Dynamic rank, rank-11+, ragged arrays, boxed arrays, arrays of functions, and hidden-shape values are out of scope.

## Descriptor Rules

All array arguments are passed to exported Remora kernels as pointers to rank-specialized descriptor structs.

Fields:

- `allocated`: base allocation pointer
- `aligned`: aligned base pointer for indexing; initially equal to `allocated`
- `offset`: element offset from `aligned` to the logical first element
- `sizes`: extent per dimension, outermost first
- `strides`: element stride per dimension, outermost first

All offsets, sizes, and strides are signed 64-bit integers. Strides are measured in elements, not bytes.

Contiguous row-major examples:

- rank 1 shape `[n]`: strides `[1]`
- rank 2 shape `[rows, cols]`: strides `[cols, 1]`
- rank 3 shape `[d0, d1, d2]`: strides `[d1*d2, d2, 1]`
- rank 10 follows the same row-major rule: each stride is the product of all inner dimensions

For views, keep `allocated` and `aligned` pointing at the base allocation and represent the logical first element with `offset`. Do not hide view offsets by changing `aligned` unless an external library forces that representation and an ABI test documents it.

## C Layout

Use these layouts for `ctypes`, CPU `ExecutionEngine` calls, and CUDA kernel launches. The descriptor family is defined for every rank `N` where `0 <= N <= 10`.

```c
typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
} RemoraMemRef0;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size0;
  int64_t stride0;
} RemoraMemRef1;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size0;
  int64_t size1;
  int64_t stride0;
  int64_t stride1;
} RemoraMemRef2;

typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size0;
  int64_t size1;
  int64_t size2;
  int64_t stride0;
  int64_t stride1;
  int64_t stride2;
} RemoraMemRef3;

/* For RemoraMemRefN, append size0..sizeN-1 followed by stride0..strideN-1. */
```

Element type is not encoded in the descriptor layout. Kernel metadata records the element type for each descriptor argument.

## Kernel Entry Convention

Exported Remora kernels receive descriptor pointers followed by scalar arguments, in metadata order:

```c
extern "C" __global__
void remora_kernel(RemoraMemRefN *input0,
                   RemoraMemRefM *output0,
                   ... scalar_args);
```

Kernels write array and scalar results through output descriptors. Scalar results use rank-0 descriptors.

If MLIR's internal lowered function uses a different memref convention, code generation must emit an adapter kernel with the external ABI above. `RemoraExecutor` launches only the adapter/exported ABI, never an undocumented internal MLIR ABI.

## Indexing Formula

For rank `r`, the element address in element units is:

```text
linear = offset + sum(indices[i] * strides[i] for i in 0..r-1)
```

The byte address is:

```text
aligned + linear * sizeof(element_type)
```

Rank-0 values use only `offset`.

## Required Tests

`tests/test_abi.py` must validate:

- ctypes struct sizes and field order for ranks 0 through 10
- descriptor construction from contiguous numpy arrays
- descriptor construction from a sliced/transposed numpy view, including byte-stride to element-stride conversion and nonzero element offsets, even before view operations exist in Remora syntax
- CPU `ExecutionEngine` round-trip for representative ranks, including 0, 1, 2, 3, and at least one rank above 3
- CUDA round-trip for representative ranks, including 0, 1, 2, 3, and at least one rank above 3 when a CUDA device is available

The CUDA tests launch tiny kernels that read from an input descriptor and write through an output descriptor. They are ABI tests, not optimizer or lowering tests.
