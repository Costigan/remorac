"""Backend route registry — explicit, testable GPU lowering routes.

Wraps the :func:`~remora.codegen.generate_mlir_descriptor_abi_ptx` dispatch
cascade as a registry of ``Route`` objects with predicates, priorities, and
capability keys.  The goal is to make backend routing visible and audit-ready
without changing any compilation result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from remora.capabilities import Capability
from remora.hir import HIRFunction
from remora.codegen import KernelMeta
from remora.execution_plan import ExecutionPlan


@dataclass(frozen=True)
class RouteContext:
    kernel_name: str
    toolchain: Any
    functions: dict | None = None


@dataclass(frozen=True)
class RouteResult:
    ptx: str
    metas: list[KernelMeta]
    plan: ExecutionPlan | None


@dataclass(frozen=True)
class RouteDecision:
    route_name: str
    accepted: bool
    reason: str
    capability_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class Route:
    name: str
    priority: int
    predicate: Callable[[HIRFunction, RouteContext], bool]
    capability_keys: tuple[str, ...] = ()
    build: Callable[[HIRFunction, RouteContext], RouteResult] | None = None


def select_route(
    function: HIRFunction, ctx: RouteContext
) -> tuple[Route | None, list[RouteDecision]]:
    decisions: list[RouteDecision] = []
    for route in sorted(_all_routes(), key=lambda r: r.priority):
        try:
            accepted = route.predicate(function, ctx)
        except Exception as exc:
            decisions.append(RouteDecision(route.name, False, str(exc), route.capability_keys))
            continue
        if accepted:
            decisions.append(RouteDecision(route.name, True, "accepted", route.capability_keys))
            return route, decisions
        decisions.append(
            RouteDecision(route.name, False, "predicate returned False", route.capability_keys)
        )
    return None, decisions


def build_route_registry_test_routes() -> list[Route]:
    return sorted(_all_routes(), key=lambda r: r.priority)


_RouteBuilder = Callable[[HIRFunction, RouteContext], RouteResult]

_registry: list[Route] | None = None


def _all_routes() -> list[Route]:
    global _registry
    if _registry is None:
        _registry = _build_routes()
    return _registry


def _r(
    name: str,
    priority: int,
    predicate: Callable[[HIRFunction, RouteContext], bool],
    build: _RouteBuilder,
    capability_keys: tuple[str, ...] = (),
) -> Route:
    return Route(name=name, priority=priority, predicate=predicate, capability_keys=capability_keys, build=build)


def _build_routes() -> list[Route]:
    routes: list[Route] = []
    _idx = [0]

    def nxt() -> int:
        _idx[0] += 1
        return _idx[0]

    from remora.hir import (
        HIRAppend,
        HIRApply,
        HIRDrop,
        HIRFilter,
        HIRFold,
        HIRGrade,
        HIRIm2col,
        HIRIndicesOf,
        HIRLambda,
        HIRMap,
        HIRMatmul,
        HIRRavel,
        HIRReplicate,
        HIRReshape,
        HIRReverse,
        HIRRotate,
        HIRScatterAdd,
        HIRSlice,
        HIRSort,
        HIRSubarray,
        HIRTake,
        HIRTranspose,
    )
    from remora.types import ArrayType, INT, BOOL, FLOAT64 as _FLOAT64, ScalarType
    from remora.gpu_lowering import (
        GPUScaffoldError,
        _cell_fold_dot_kernel,
        _sobel_kernel,
        build_descriptor_abi_bitonic_grade_gpu_module,
        build_descriptor_abi_bitonic_sort_gpu_module,
        build_descriptor_abi_bool_map_gpu_module,
        build_descriptor_abi_cell_fold_dot_gpu_module,
        build_descriptor_abi_f32_compound_fold_gpu_module,
        build_descriptor_abi_f32_map_gpu_module,
        build_descriptor_abi_f32_reduction_gpu_module,
        build_descriptor_abi_f32_scan_gpu_module,
        build_descriptor_abi_filter_gpu_module,
        build_descriptor_abi_general_map_gpu_module,
        build_descriptor_abi_grade_gpu_module,
        build_descriptor_abi_i32_map_gpu_module,
        build_descriptor_abi_im2col_gpu_module,
        build_descriptor_abi_indices_of_gpu_module,
        build_descriptor_abi_matmul_gpu_module,
        build_descriptor_abi_multiblock_bitonic_grade_gpu_module,
        build_descriptor_abi_multiblock_bitonic_sort_gpu_module,
        build_descriptor_abi_multiblock_f32_scan_gpu_module,
        build_descriptor_abi_parallel_filter_gpu_module,
        build_descriptor_abi_parallel_replicate_gpu_module,
        build_descriptor_abi_parallel_scatter_add_gpu_module,
        build_descriptor_abi_replicate_gpu_module,
        build_descriptor_abi_reverse_gpu_module,
        build_descriptor_abi_rotate_gpu_module,
        build_descriptor_abi_scatter_add_gpu_module,
        build_descriptor_abi_sobel_gpu_module,
        build_descriptor_abi_sort_gpu_module,
        build_descriptor_abi_take_gpu_module,
        build_descriptor_abi_drop_gpu_module,
        build_descriptor_abi_slice_gpu_module,
        build_descriptor_abi_subarray_gpu_module,
        build_descriptor_abi_reshape_gpu_module,
        build_descriptor_abi_ravel_gpu_module,
        build_descriptor_abi_append_gpu_module,
        build_descriptor_abi_transpose_gpu_module,
        build_descriptor_abi_tiled_matmul_gpu_module,
    )
    from remora._gpu_map_support import (
        analyze_supported_bool_map_function,
        analyze_supported_f32_map_function,
        analyze_supported_i32_map_function,
        F32MapKernel,
        I32MapKernel,
    )
    from remora.codegen import CodegenUnavailable
    from remora.pipeline import translate_mlir_to_llvmir, translate_llvmir_to_nvptx_text
    from remora.gpu_lowering import extract_gpu_module_body_as_module

    def _emit(gpu_module, metas, plan, toolchain):
        device_module = extract_gpu_module_body_as_module(gpu_module.text)
        llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)
        ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=toolchain)
        return RouteResult(ptx, metas, plan)

    # ── Route 1: im2col ──
    def _pred_im2col(fn, ctx):
        if not isinstance(fn.body, HIRIm2col):
            return False
        param_type = fn.params[0].type
        return isinstance(param_type, ArrayType) and param_type.rank == 2

    def _build_im2col(fn, ctx):
        from remora.types import static_shape, static_dim
        im2col = fn.body
        kh, kw = im2col.kernel_shape
        param_type = fn.params[0].type
        h = static_dim(param_type.shape[0])
        w = static_dim(param_type.shape[1])
        ppa = (h - kh) // im2col.stride + 1
        pc = ppa * ppa
        ps = kh * kw
        gpu_module = build_descriptor_abi_im2col_gpu_module(fn, kernel_name=ctx.kernel_name)
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=0,
            num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["f32"],
            output_shape=(pc, ps), output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("im2col", nxt(), _pred_im2col, _build_im2col, ("im2col",)))

    # ── Route 2-3: scatter-add ──
    def _pred_scatter_add(fn, ctx):
        return isinstance(fn.body, HIRScatterAdd)

    def _build_scatter_add_parallel(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_parallel_scatter_add_gpu_module(fn, kernel_name=ctx.kernel_name)
        sa_result = fn.body.result_type
        sa_shape = static_shape(sa_result) if isinstance(sa_result, ArrayType) else ()
        sa_N = sa_shape[0] if sa_shape else 1
        num_array = sum(1 for p in fn.params if isinstance(p.type, ArrayType))
        num_scalar = sum(1 for p in fn.params if not isinstance(p.type, ArrayType))
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=sa_N,
            num_inputs=num_array + num_scalar, num_outputs=1,
            input_elem_types=["f32"] * (num_array + num_scalar),
            output_elem_types=["f32"],
            output_shape=sa_shape, output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    def _build_scatter_add_serial(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_scatter_add_gpu_module(fn, kernel_name=ctx.kernel_name)
        sa_result = fn.body.result_type
        sa_shape = static_shape(sa_result) if isinstance(sa_result, ArrayType) else ()
        num_array = sum(1 for p in fn.params if isinstance(p.type, ArrayType))
        num_scalar = sum(1 for p in fn.params if not isinstance(p.type, ArrayType))
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=1,
            num_inputs=num_array + num_scalar, num_outputs=1,
            input_elem_types=["f32"] * (num_array + num_scalar),
            output_elem_types=["f32"],
            output_shape=sa_shape, output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("scatter_add_parallel", nxt(), _pred_scatter_add, _build_scatter_add_parallel, ("scatter_add",)))
    routes.append(_r("scatter_add_serial", nxt(), _pred_scatter_add, _build_scatter_add_serial, ("scatter_add",)))

    # ── Route 4-5: matmul ──
    def _pred_matmul(fn, ctx):
        return isinstance(fn.body, HIRMatmul)

    def _build_matmul_tiled(fn, ctx):
        from remora.types import static_shape, static_dim
        TILE = 16
        gpu_module = build_descriptor_abi_tiled_matmul_gpu_module(fn, kernel_name=ctx.kernel_name, tile_size=TILE)
        mm_shape = static_shape(fn.body.result_type)
        mm_M = mm_shape[0] if len(mm_shape) >= 1 else 1
        mm_N = mm_shape[1] if len(mm_shape) >= 2 else 1
        gridRows = (mm_M + TILE - 1) // TILE
        gridCols = (mm_N + TILE - 1) // TILE
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=TILE * TILE,
            num_inputs=2, num_outputs=1,
            input_elem_types=["f32", "f32"], output_elem_types=["f32"],
            output_shape=mm_shape, output_dtype="float32",
            grid_size=gridRows * gridCols,
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    def _build_matmul_naive(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_matmul_gpu_module(fn, kernel_name=ctx.kernel_name)
        mm_shape = static_shape(fn.body.result_type)
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=0,
            num_inputs=2, num_outputs=1,
            input_elem_types=["f32", "f32"], output_elem_types=["f32"],
            output_shape=mm_shape, output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("matmul_tiled", nxt(), _pred_matmul, _build_matmul_tiled, ("matmul",)))
    routes.append(_r("matmul_naive", nxt(), _pred_matmul, _build_matmul_naive, ("matmul",)))

    # ── Routes 6-12: sort/grade ──
    def _pred_sort_grade(fn, ctx):
        return isinstance(fn.body, (HIRSort, HIRGrade))

    def _build_sort_radix(fn, ctx):
        from remora._gpu_radix_sort import build_radix_sort_gpu_module
        rx_text, rx_metas, rx_plan = build_radix_sort_gpu_module(fn, kernel_name=ctx.kernel_name)
        rx_dev = extract_gpu_module_body_as_module(rx_text)
        rx_ir = translate_mlir_to_llvmir(rx_dev, toolchain=ctx.toolchain)
        rx_ptx = translate_llvmir_to_nvptx_text(rx_ir, toolchain=ctx.toolchain)
        return RouteResult(rx_ptx, rx_metas, rx_plan)

    def _build_sort_bitonic(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_bitonic_sort_gpu_module(fn, kernel_name=ctx.kernel_name)
        sg_shape = static_shape(fn.body.result_type)
        sg_N = sg_shape[0] if sg_shape else 1
        sg_NP = 1
        while sg_NP < sg_N:
            sg_NP *= 2
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=sg_NP, num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["f32"],
            output_shape=sg_shape, output_dtype="float32", grid_size=1,
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    def _build_grade_bitonic(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_bitonic_grade_gpu_module(fn, kernel_name=ctx.kernel_name)
        sg_shape = static_shape(fn.body.result_type)
        sg_N = sg_shape[0] if sg_shape else 1
        sg_NP = 1
        while sg_NP < sg_N:
            sg_NP *= 2
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=sg_NP, num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["i32"],
            output_shape=sg_shape, output_dtype="int32", grid_size=1,
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    def _build_sort_multiblock(fn, ctx):
        from remora.types import static_shape
        from remora.execution_plan import BufferSpec, ExecutionPlan, KernelStep
        gpu_module = build_descriptor_abi_multiblock_bitonic_sort_gpu_module(fn, kernel_name=ctx.kernel_name)
        sg_shape = static_shape(fn.body.result_type)
        mb_N = sg_shape[0] if sg_shape else 1
        mb_NP = 1
        while mb_NP < mb_N:
            mb_NP *= 2
        mb_BS = 1024
        mb_nblocks = mb_NP // mb_BS
        mb_local_stages = 10
        mb_num_stages = mb_NP.bit_length() - 1
        mb_local_name = f"{ctx.kernel_name}_local"
        mb_all_kernels = [KernelMeta(
            name=mb_local_name, grid_dims=1, block_size=mb_BS, num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["f32"],
            output_shape=(mb_NP,), output_dtype="float32", grid_size=mb_nblocks,
        )]
        mb_steps_list = [KernelStep(mb_local_name, ["input_0"], "sorted_a")]
        mb_step_idx = 0
        for mb_k in range(mb_local_stages, mb_num_stages):
            for mb_j in range(mb_k, -1, -1):
                sn = f"{ctx.kernel_name}_gstep_{mb_step_idx}"
                mb_all_kernels.append(KernelMeta(
                    name=sn, grid_dims=1, block_size=0, num_inputs=1, num_outputs=1,
                    input_elem_types=["f32"], output_elem_types=["f32"],
                    output_shape=(mb_NP,), output_dtype="float32",
                ))
                if mb_step_idx % 2 == 0:
                    mb_steps_list.append(KernelStep(sn, ["sorted_a"], "sorted_b"))
                else:
                    mb_steps_list.append(KernelStep(sn, ["sorted_b"], "sorted_a"))
                mb_step_idx += 1
        mb_final = "sorted_b" if mb_step_idx % 2 == 1 else "sorted_a"
        mb_plan = ExecutionPlan(
            buffers=[BufferSpec("sorted_a", (mb_NP,), "f32"), BufferSpec("sorted_b", (mb_NP,), "f32")],
            steps=mb_steps_list, final_output=mb_final,
            output_shape=sg_shape, output_dtype="f32",
        )
        return _emit(gpu_module, mb_all_kernels, mb_plan, ctx.toolchain)

    def _build_grade_multiblock(fn, ctx):
        from remora.types import static_shape
        from remora.execution_plan import BufferSpec, ExecutionPlan, KernelStep
        gpu_module = build_descriptor_abi_multiblock_bitonic_grade_gpu_module(fn, kernel_name=ctx.kernel_name)
        sg_shape = static_shape(fn.body.result_type)
        mg_N = sg_shape[0] if sg_shape else 1
        mg_NP = 1
        while mg_NP < mg_N:
            mg_NP *= 2
        mg_BS = 1024
        mg_nblocks = mg_NP // mg_BS
        mg_local_stages = 10
        mg_num_stages = mg_NP.bit_length() - 1
        mg_pad = f"{ctx.kernel_name}_pad"
        mg_local = f"{ctx.kernel_name}_local"
        mg_kernels = [
            KernelMeta(name=mg_pad, grid_dims=1, block_size=0, num_inputs=1, num_outputs=1,
                       input_elem_types=["f32"], output_elem_types=["f32"],
                       output_shape=(mg_NP,), output_dtype="float32"),
            KernelMeta(name=mg_local, grid_dims=1, block_size=mg_BS, num_inputs=1, num_outputs=1,
                       input_elem_types=["f32"], output_elem_types=["i32"],
                       output_shape=(mg_NP,), output_dtype="int32", grid_size=mg_nblocks),
        ]
        mg_steps = [
            KernelStep(mg_pad, ["input_0"], "values_padded"),
            KernelStep(mg_local, ["values_padded"], "indices_a"),
        ]
        mg_step_idx = 0
        for mg_k in range(mg_local_stages, mg_num_stages):
            for mg_j in range(mg_k, -1, -1):
                sn = f"{ctx.kernel_name}_gstep_{mg_step_idx}"
                mg_kernels.append(KernelMeta(
                    name=sn, grid_dims=1, block_size=0, num_inputs=2, num_outputs=1,
                    input_elem_types=["f32", "i32"], output_elem_types=["i32"],
                    output_shape=(mg_NP,), output_dtype="int32",
                ))
                if mg_step_idx % 2 == 0:
                    mg_steps.append(KernelStep(sn, ["values_padded", "indices_a"], "indices_b"))
                else:
                    mg_steps.append(KernelStep(sn, ["values_padded", "indices_b"], "indices_a"))
                mg_step_idx += 1
        mg_final = "indices_b" if mg_step_idx % 2 == 1 else "indices_a"
        mg_plan = ExecutionPlan(
            buffers=[
                BufferSpec("values_padded", (mg_NP,), "f32"),
                BufferSpec("indices_a", (mg_NP,), "i32"),
                BufferSpec("indices_b", (mg_NP,), "i32"),
            ],
            steps=mg_steps, final_output=mg_final,
            output_shape=sg_shape, output_dtype="i32",
        )
        return _emit(gpu_module, mg_kernels, mg_plan, ctx.toolchain)

    def _build_sort_simple(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_sort_gpu_module(fn, kernel_name=ctx.kernel_name)
        sg_shape = static_shape(fn.body.result_type)
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=1, num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["f32"],
            output_shape=sg_shape, output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    def _build_grade_simple(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_grade_gpu_module(fn, kernel_name=ctx.kernel_name)
        sg_shape = static_shape(fn.body.result_type)
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=1, num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["i32"],
            output_shape=sg_shape, output_dtype="int32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("sort_radix", nxt(), lambda fn, ctx: isinstance(fn.body, HIRSort), _build_sort_radix, ("sort",)))
    routes.append(_r("sort_bitonic", nxt(), lambda fn, ctx: isinstance(fn.body, HIRSort), _build_sort_bitonic, ("sort",)))
    routes.append(_r("grade_bitonic", nxt(), lambda fn, ctx: isinstance(fn.body, HIRGrade), _build_grade_bitonic, ("grade",)))
    routes.append(_r("sort_multiblock", nxt(), lambda fn, ctx: isinstance(fn.body, HIRSort), _build_sort_multiblock, ("sort",)))
    routes.append(_r("grade_multiblock", nxt(), lambda fn, ctx: isinstance(fn.body, HIRGrade), _build_grade_multiblock, ("grade",)))
    routes.append(_r("sort_simple", nxt(), lambda fn, ctx: isinstance(fn.body, HIRSort), _build_sort_simple, ("sort",)))
    routes.append(_r("grade_simple", nxt(), lambda fn, ctx: isinstance(fn.body, HIRGrade), _build_grade_simple, ("grade",)))

    # ── Route: indices-of ──
    def _pred_indices_of(fn, ctx):
        return isinstance(fn.body, HIRIndicesOf)

    def _build_indices_of(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_indices_of_gpu_module(fn, kernel_name=ctx.kernel_name)
        io_shape = static_shape(fn.body.result_type)
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=0,
            num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["i32"],
            output_shape=io_shape, output_dtype="int32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("indices_of", nxt(), _pred_indices_of, _build_indices_of, ("indices_of",)))

    # ── Routes: filter ──
    def _pred_filter(fn, ctx):
        return isinstance(fn.body, HIRFilter)

    def _build_filter_parallel(fn, ctx):
        from remora.types import static_shape
        from remora.execution_plan import BufferSpec, ExecutionPlan, KernelStep
        gpu_module = build_descriptor_abi_parallel_filter_gpu_module(fn, kernel_name=ctx.kernel_name)
        f_shape = static_shape(fn.params[0].type)
        N = f_shape[0] if f_shape else 1
        pred_name = f"{ctx.kernel_name}_pred"
        scan_name = f"{ctx.kernel_name}_scan"
        scatter_name = f"{ctx.kernel_name}_scatter"
        kernels = [
            KernelMeta(name=pred_name, grid_dims=1, block_size=0, num_inputs=1, num_outputs=1,
                       input_elem_types=["f32"], output_elem_types=["i32"],
                       output_shape=f_shape, output_dtype="int32"),
            KernelMeta(name=scan_name, grid_dims=1, block_size=N, num_inputs=1, num_outputs=1,
                       input_elem_types=["i32"], output_elem_types=["i32"],
                       output_shape=f_shape, output_dtype="int32"),
            KernelMeta(name=scatter_name, grid_dims=1, block_size=0, num_inputs=3, num_outputs=1,
                       input_elem_types=["f32", "i32", "i32"], output_elem_types=["f32"],
                       output_shape=f_shape, output_dtype="float32"),
        ]
        plan = ExecutionPlan(
            buffers=[BufferSpec("pred", f_shape, "i32"), BufferSpec("scan", f_shape, "i32"),
                     BufferSpec("output", f_shape, "f32")],
            steps=[KernelStep(pred_name, ["input_0"], "pred"),
                   KernelStep(scan_name, ["pred"], "scan"),
                   KernelStep(scatter_name, ["input_0", "pred", "scan"], "output")],
            final_output="output", output_shape=f_shape, output_dtype="f32",
        )
        return _emit(gpu_module, kernels, plan, ctx.toolchain)

    def _build_filter_serial(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_filter_gpu_module(fn, kernel_name=ctx.kernel_name)
        f_shape = static_shape(fn.params[0].type)
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=1, num_inputs=1, num_outputs=1,
            input_elem_types=["f32"], output_elem_types=["f32"],
            output_shape=f_shape, output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("filter_parallel", nxt(), _pred_filter, _build_filter_parallel, ("filter",)))
    routes.append(_r("filter_serial", nxt(), _pred_filter, _build_filter_serial, ("filter",)))

    # ── Routes: replicate ──
    def _pred_replicate(fn, ctx):
        return isinstance(fn.body, HIRReplicate)

    def _build_replicate_parallel(fn, ctx):
        from remora.types import static_dim, static_shape
        from remora.execution_plan import BufferSpec, ExecutionPlan, KernelStep
        gpu_module = build_descriptor_abi_parallel_replicate_gpu_module(fn, kernel_name=ctx.kernel_name)
        r_N = static_dim(fn.params[1].type.shape[0]) if isinstance(fn.params[1].type, ArrayType) else 0
        out_N = r_N * r_N
        scan_name_r = f"{ctx.kernel_name}_scan"
        scatter_name_r = f"{ctx.kernel_name}_scatter"
        kernels = [
            KernelMeta(name=scan_name_r, grid_dims=1, block_size=r_N, num_inputs=1, num_outputs=1,
                       input_elem_types=["i32"], output_elem_types=["i32"],
                       output_shape=(r_N,), output_dtype="int32"),
            KernelMeta(name=scatter_name_r, grid_dims=1, block_size=0, num_inputs=3, num_outputs=1,
                       input_elem_types=["i32", "f32", "i32"], output_elem_types=["f32"],
                       output_shape=(out_N,), output_dtype="float32"),
        ]
        plan = ExecutionPlan(
            buffers=[BufferSpec("scan", (r_N,), "i32"), BufferSpec("output", (out_N,), "f32")],
            steps=[KernelStep(scan_name_r, ["input_0"], "scan"),
                   KernelStep(scatter_name_r, ["input_0", "input_1", "scan"], "output")],
            final_output="output", output_shape=(out_N,), output_dtype="f32",
        )
        return _emit(gpu_module, kernels, plan, ctx.toolchain)

    def _build_replicate_serial(fn, ctx):
        from remora.types import static_dim
        gpu_module = build_descriptor_abi_replicate_gpu_module(fn, kernel_name=ctx.kernel_name)
        r_N = static_dim(fn.params[1].type.shape[0]) if isinstance(fn.params[1].type, ArrayType) else 0
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=1, num_inputs=2, num_outputs=1,
            input_elem_types=["i32", "f32"], output_elem_types=["f32"],
            output_shape=(r_N * r_N,), output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("replicate_parallel", nxt(), _pred_replicate, _build_replicate_parallel, ("replicate",)))
    routes.append(_r("replicate_serial", nxt(), _pred_replicate, _build_replicate_serial, ("replicate",)))

    # ── Route: cell-fold-dot ──
    def _pred_cell_fold_dot(fn, ctx):
        try:
            _, (kh, kw), stride = _cell_fold_dot_kernel(fn)
            param_type = fn.params[0].type
            return isinstance(param_type, ArrayType) and param_type.rank == 2
        except GPUScaffoldError:
            return False

    def _build_cell_fold_dot(fn, ctx):
        from remora.types import static_dim
        _, (kh, kw), stride = _cell_fold_dot_kernel(fn)
        param_type = fn.params[0].type
        h = static_dim(param_type.shape[0])
        w = static_dim(param_type.shape[1])
        ppa = (h - kh) // stride + 1
        pc = ppa * ppa
        gpu_module = build_descriptor_abi_cell_fold_dot_gpu_module(fn, kernel_name=ctx.kernel_name)
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=0,
            num_inputs=2, num_outputs=1,
            input_elem_types=["f32", "f32"], output_elem_types=["f32"],
            output_shape=(pc,), output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("cell_fold_dot", nxt(), _pred_cell_fold_dot, _build_cell_fold_dot, ("fold", "map")))

    # ── Route: view ops ──
    def _pred_view_ops(fn, ctx):
        return isinstance(fn.body, (HIRReverse, HIRRotate, HIRTake, HIRDrop, HIRSlice, HIRSubarray,
                                      HIRReshape, HIRRavel, HIRAppend, HIRTranspose))

    def _build_view_ops(fn, ctx):
        from remora.types import static_shape
        body = fn.body
        if isinstance(body, HIRReverse):
            gpu_module = build_descriptor_abi_reverse_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRRotate):
            gpu_module = build_descriptor_abi_rotate_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRTake):
            gpu_module = build_descriptor_abi_take_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRDrop):
            gpu_module = build_descriptor_abi_drop_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRSlice):
            gpu_module = build_descriptor_abi_slice_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRSubarray):
            gpu_module = build_descriptor_abi_subarray_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRReshape):
            gpu_module = build_descriptor_abi_reshape_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRRavel):
            gpu_module = build_descriptor_abi_ravel_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRAppend):
            gpu_module = build_descriptor_abi_append_gpu_module(fn, kernel_name=ctx.kernel_name)
        elif isinstance(body, HIRTranspose):
            gpu_module = build_descriptor_abi_transpose_gpu_module(fn, kernel_name=ctx.kernel_name)
        else:
            raise GPUScaffoldError("not a supported view op")
        rt = body.result_type
        v_shape = static_shape(rt)
        v_total = 1
        for d in v_shape:
            v_total *= d
        v_num_inputs = 2 if isinstance(body, HIRAppend) else 1
        v_ie = ["f32"] * v_num_inputs
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=max(1, min(v_total, 1024)),
            num_inputs=v_num_inputs, num_outputs=1,
            input_elem_types=v_ie, output_elem_types=["f32"],
            output_shape=v_shape, output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("view_ops", nxt(), _pred_view_ops, _build_view_ops, (
        "reverse", "rotate", "take", "drop", "slice", "subarray",
        "reshape", "ravel", "append", "transpose")))

    # ── Route: sobel ──
    def _pred_sobel(fn, ctx):
        try:
            _, (kh, kw), stride = _sobel_kernel(fn)
            param_type = fn.params[0].type
            return isinstance(param_type, ArrayType) and param_type.rank == 2
        except GPUScaffoldError:
            return False

    def _build_sobel(fn, ctx):
        from remora.types import static_dim
        _, (kh, kw), stride = _sobel_kernel(fn)
        param_type = fn.params[0].type
        h = static_dim(param_type.shape[0])
        w = static_dim(param_type.shape[1])
        ppa = (h - kh) // stride + 1
        pc = ppa * ppa
        gpu_module = build_descriptor_abi_sobel_gpu_module(fn, kernel_name=ctx.kernel_name)
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=0,
            num_inputs=3, num_outputs=1,
            input_elem_types=["f32", "f32", "f32"], output_elem_types=["f32"],
            output_shape=(pc,), output_dtype="float32",
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("sobel", nxt(), _pred_sobel, _build_sobel, ("map", "fold")))

    # ── Route: general-map (compound body, HIRMap+HIRLambda) ──
    def _pred_general_map_compound(fn, ctx):
        from remora.lowering.tensor_ops import _body_needs_tensor_lowering
        if not isinstance(fn.body, HIRMap):
            return False
        if not isinstance(fn.body.func, HIRLambda):
            return False
        return _body_needs_tensor_lowering(fn.body.func.body)

    def _build_general_map_compound(fn, ctx):
        from remora.types import static_shape
        gpu_module = build_descriptor_abi_general_map_gpu_module(
            fn, kernel_name=ctx.kernel_name, functions=ctx.functions,
        )
        body_map = fn.body
        result_type = body_map.result_type
        if not isinstance(result_type, ArrayType):
            raise CodegenUnavailable("general GPU map requires an array result type")
        output_shape = static_shape(result_type)
        num_array_inputs = sum(1 for p in fn.params if isinstance(p.type, ArrayType))
        num_scalar_inputs = sum(1 for p in fn.params if not isinstance(p.type, ArrayType))
        total_inputs = num_array_inputs + num_scalar_inputs
        input_elem_types: list[str] = []
        for param in fn.params:
            if isinstance(param.type, ArrayType):
                elem = param.type.element.name
                if elem == "float":
                    input_elem_types.append("f32")
                elif elem == "float64":
                    input_elem_types.append("f64")
                elif elem == "int":
                    input_elem_types.append("i32")
                elif elem == "bool":
                    input_elem_types.append("i1")
                else:
                    input_elem_types.append("f32")
            else:
                input_elem_types.append("f32")
        _out_elem_name = getattr(result_type.element, "name", "float")
        if _out_elem_name == "int":
            _out_et, _out_dtype = "i32", "int32"
        elif _out_elem_name == "bool":
            _out_et, _out_dtype = "i8", "bool"
        elif _out_elem_name == "float64":
            _out_et, _out_dtype = "f64", "float64"
        else:
            _out_et, _out_dtype = "f32", "float32"
        _total = 1
        for d in output_shape:
            _total *= d
        _kind: list[str] = []
        for param in fn.params:
            _kind.append("array" if isinstance(param.type, ArrayType) else "scalar")
        meta = KernelMeta(
            name=ctx.kernel_name, grid_dims=1, block_size=max(1, min(_total, 1024)),
            num_inputs=total_inputs, num_outputs=1,
            input_elem_types=input_elem_types, output_elem_types=[_out_et],
            output_shape=output_shape, output_dtype=_out_dtype, input_kinds=_kind,
        )
        return _emit(gpu_module, [meta], None, ctx.toolchain)

    routes.append(_r("general_map_compound", nxt(), _pred_general_map_compound, _build_general_map_compound, ("map",)))

    # ── Route: general dispatch (the deeply nested f32→i32→bool→reduction→compound-fold→scan→multiblock→general chain) ──
    def _pred_general_dispatch(fn, ctx):
        return True

    def _build_general_dispatch(fn, ctx):
        from remora.types import static_shape, static_dim
        name = ctx.kernel_name
        toolchain = ctx.toolchain

        def _direct_f32_map_kernel(function):
            try:
                return analyze_supported_f32_map_function(
                    function, on_unsupported=CodegenUnavailable, context="direct PTX",
                )
            except CodegenUnavailable as exc:
                message = str(exc).replace("float", "f32").replace(
                    "one or two input parameters", "one or two input descriptors",
                ).replace("literal float section", "literal f32 section constant")
                raise CodegenUnavailable(message) from exc

        def _direct_i32_map_kernel(function):
            try:
                return analyze_supported_i32_map_function(
                    function, on_unsupported=CodegenUnavailable, context="direct MLIR descriptor PTX",
                )
            except CodegenUnavailable as exc:
                message = str(exc).replace("int", "i32").replace(
                    "one or two input parameters", "one or two input descriptors",
                ).replace("literal i32 section", "literal i32 section constant")
                raise CodegenUnavailable(message) from exc

        try:
            map_kernel = _direct_f32_map_kernel(fn)
            rank = len(map_kernel.shape)
            if rank < 1 or rank > 10:
                raise CodegenUnavailable(
                    "MLIR-derived descriptor-ABI PTX currently supports rank-1 through rank-10 f32 maps only"
                )
            if (
                map_kernel.expression is None
                and map_kernel.num_inputs == 1
                and map_kernel.operation.constant is None
                and map_kernel.scalar_count == 0
            ):
                raise CodegenUnavailable(
                    "MLIR-derived descriptor-ABI PTX currently supports unary literal-section or binary f32 maps only"
                )
            if map_kernel.num_inputs not in (1, 2) and map_kernel.expression is None:
                raise CodegenUnavailable(
                    "MLIR-derived descriptor-ABI PTX currently supports one or two f32 input descriptors only"
                )
            gpu_module = build_descriptor_abi_f32_map_gpu_module(fn, kernel_name=name)
            input_kinds = list(map_kernel.input_kinds) if map_kernel.input_kinds else None
            total_inputs = map_kernel.num_inputs + map_kernel.scalar_count
            input_elem_types = ["f32"] * total_inputs if total_inputs > 1 else ["f32"]
            meta = KernelMeta(
                name=name, grid_dims=1, block_size=0,
                num_inputs=map_kernel.num_inputs + map_kernel.scalar_count,
                num_outputs=1, input_elem_types=input_elem_types,
                output_elem_types=["f32"], input_kinds=input_kinds,
                output_shape=map_kernel.shape, output_dtype="float32",
            )
            return _emit(gpu_module, [meta], None, toolchain)
        except CodegenUnavailable as f32_map_error:
            try:
                map_kernel = _direct_i32_map_kernel(fn)
                rank = len(map_kernel.shape)
                if rank < 1 or rank > 10:
                    raise CodegenUnavailable(
                        "MLIR-derived descriptor-ABI PTX currently supports rank-1 through rank-10 i32 maps only"
                    )
                if map_kernel.num_inputs == 1 and map_kernel.operation.constant is None:
                    raise CodegenUnavailable(
                        "MLIR-derived descriptor-ABI PTX currently supports unary literal-section or binary i32 maps only"
                    )
                if map_kernel.num_inputs not in (1, 2):
                    raise CodegenUnavailable(
                        "MLIR-derived descriptor-ABI PTX currently supports one or two i32 input descriptors only"
                    )
                gpu_module = build_descriptor_abi_i32_map_gpu_module(fn, kernel_name=name)
                meta = KernelMeta(
                    name=name, grid_dims=1, block_size=0,
                    num_inputs=map_kernel.num_inputs, num_outputs=1,
                    input_elem_types=["i32"] * map_kernel.num_inputs,
                    output_elem_types=["i32"], output_shape=map_kernel.shape,
                    output_dtype="int32",
                )
                return _emit(gpu_module, [meta], None, toolchain)
            except CodegenUnavailable as i32_map_error:
                try:
                    map_kernel = analyze_supported_bool_map_function(
                        fn, on_unsupported=CodegenUnavailable,
                        context="MLIR-derived descriptor-ABI PTX",
                    )
                    gpu_module = build_descriptor_abi_bool_map_gpu_module(fn, kernel_name=name)
                    meta = KernelMeta(
                        name=name, grid_dims=1, block_size=0,
                        num_inputs=map_kernel.num_inputs, num_outputs=1,
                        input_elem_types=["i8"] * map_kernel.num_inputs,
                        output_elem_types=["i8"], output_shape=map_kernel.shape,
                        output_dtype="bool",
                    )
                    return _emit(gpu_module, [meta], None, toolchain)
                except CodegenUnavailable as bool_map_error:
                    try:
                        gpu_module = build_descriptor_abi_f32_reduction_gpu_module(fn, kernel_name=name)
                        num_inputs = len(fn.params)
                        _red_ename = fn.params[0].type.element.name if isinstance(fn.params[0].type, ArrayType) else "float"
                        _red_et = ("f32" if _red_ename == "float" else "i32" if _red_ename == "int"
                                    else "f64" if _red_ename == "float64" else "f32")
                        _red_dt = ("float32" if _red_ename == "float" else "int32" if _red_ename == "int"
                                    else "float64" if _red_ename == "float64" else "float32")
                        meta = KernelMeta(
                            name=name, grid_dims=1, block_size=0,
                            num_inputs=num_inputs, num_outputs=1,
                            input_elem_types=[_red_et] * num_inputs,
                            output_elem_types=[_red_et],
                            output_shape=(), output_dtype=_red_dt,
                            is_reduction=True,
                        )
                        return _emit(gpu_module, [meta], None, toolchain)
                    except GPUScaffoldError as reduction_error:
                        try:
                            gpu_module = build_descriptor_abi_f32_compound_fold_gpu_module(
                                fn, kernel_name=name, functions=ctx.functions,
                            )
                            meta = KernelMeta(
                                name=name, grid_dims=1, block_size=1,
                                num_inputs=1, num_outputs=1,
                                input_elem_types=["f32"], output_elem_types=["f32"],
                                output_shape=(), output_dtype="float32",
                                is_reduction=True,
                            )
                            return _emit(gpu_module, [meta], None, toolchain)
                        except GPUScaffoldError as compound_fold_error:
                            try:
                                gpu_module = build_descriptor_abi_f32_scan_gpu_module(
                                    fn, kernel_name=name, functions=ctx.functions,
                                )
                                scan_shape = static_shape(fn.params[0].type)
                                _num_inputs = len(fn.params)
                                _scan_elem_types: list[str] = []
                                _scan_kinds: list[str] = []
                                for p in fn.params:
                                    _scan_kinds.append("array")
                                    if isinstance(p.type, ArrayType):
                                        en = p.type.element.name
                                        _scan_elem_types.append("f32" if en == "float" else "i1" if en == "bool" else "f32")
                                    else:
                                        _scan_elem_types.append("f32")
                                _out_ename = fn.params[0].type.element.name if isinstance(fn.params[0].type, ArrayType) else "float"
                                _out_et = ("f32" if _out_ename == "float" else "i1" if _out_ename == "bool"
                                            else "i32" if _out_ename == "int" else "f64" if _out_ename == "float64"
                                            else "f32")
                                _out_dt = ("float32" if _out_ename == "float" else "bool" if _out_ename == "bool"
                                            else "int32" if _out_ename == "int" else "float64" if _out_ename == "float64"
                                            else "float32")
                                meta = KernelMeta(
                                    name=name, grid_dims=1, block_size=scan_shape[0],
                                    num_inputs=_num_inputs, num_outputs=1,
                                    input_elem_types=_scan_elem_types,
                                    output_elem_types=[_out_et],
                                    output_shape=scan_shape, output_dtype=_out_dt,
                                    input_kinds=_scan_kinds,
                                )
                                return _emit(gpu_module, [meta], None, toolchain)
                            except GPUScaffoldError as scan_error:
                                from remora.execution_plan import BufferSpec, ExecutionPlan, KernelStep
                                try:
                                    mb_module, mb_kernels, mb_buffers, mb_steps, sc_shape = build_descriptor_abi_multiblock_f32_scan_gpu_module(fn, kernel_name=name)
                                    mb_plan = ExecutionPlan(
                                        buffers=mb_buffers, steps=mb_steps,
                                        final_output="scanned",
                                        output_shape=sc_shape, output_dtype="f32",
                                    )
                                    return _emit(mb_module, mb_kernels, mb_plan, toolchain)
                                except GPUScaffoldError:
                                    pass
                                try:
                                    if not isinstance(fn.body, (HIRMap, HIRApply)):
                                        raise CodegenUnavailable(
                                            "general GPU fallback requires a HIRMap with HIRLambda"
                                        )
                                    gpu_module = build_descriptor_abi_general_map_gpu_module(
                                        fn, kernel_name=name, functions=ctx.functions,
                                    )
                                    body_map = fn.body
                                    result_type = body_map.result_type
                                    if not isinstance(result_type, ArrayType):
                                        raise CodegenUnavailable(
                                            "general GPU map fallback requires an array result type"
                                        )
                                    output_shape = static_shape(result_type)
                                    num_array_inputs = sum(1 for p in fn.params if isinstance(p.type, ArrayType))
                                    num_scalar_inputs = sum(1 for p in fn.params if isinstance(p.type, ScalarType))
                                    _input_types: list[str] = []
                                    _input_kinds: list[str] = []
                                    for p in fn.params:
                                        if isinstance(p.type, ArrayType):
                                            _input_kinds.append("array")
                                            if p.type.element == INT:
                                                _input_types.append("i32")
                                            elif p.type.element == BOOL:
                                                _input_types.append("i1")
                                            elif p.type.element == _FLOAT64:
                                                _input_types.append("f64")
                                            else:
                                                _input_types.append("f32")
                                        elif isinstance(p.type, ScalarType):
                                            _input_kinds.append("scalar")
                                            if p.type == INT:
                                                _input_types.append("i32")
                                            elif p.type == BOOL:
                                                _input_types.append("i1")
                                            elif p.type == _FLOAT64:
                                                _input_types.append("f64")
                                            else:
                                                _input_types.append("f32")
                                    _out_elem = result_type.element
                                    _out_et = ("i32" if _out_elem == INT else "i1" if _out_elem == BOOL
                                                else "f64" if _out_elem == _FLOAT64 else "f32")
                                    _out_dt = ("int32" if _out_elem == INT else "bool" if _out_elem == BOOL
                                                else "float64" if _out_elem == _FLOAT64 else "float32")
                                    meta = KernelMeta(
                                        name=name, grid_dims=1, block_size=0,
                                        num_inputs=num_array_inputs + num_scalar_inputs,
                                        num_outputs=1,
                                        input_elem_types=_input_types,
                                        output_elem_types=[_out_et],
                                        input_kinds=_input_kinds,
                                        output_shape=output_shape, output_dtype=_out_dt,
                                    )
                                    return _emit(gpu_module, [meta], None, toolchain)
                                except Exception as general_map_error:
                                    for _rec_err in (general_map_error, scan_error, compound_fold_error, reduction_error):
                                        _rec_msg = str(_rec_err)
                                        if "GPU recursion supports" in _rec_msg:
                                            raise CodegenUnavailable(_rec_msg) from _rec_err
                                    raise CodegenUnavailable(
                                        "MLIR-derived descriptor-ABI PTX could not lower function to any GPU kernel: "
                                        f"f32_map={f32_map_error}; i32_map={i32_map_error}; bool_map={bool_map_error}; "
                                        f"reduction={reduction_error}; compound_fold={compound_fold_error}; "
                                        f"scan={scan_error}; general_map={general_map_error}"
                                    ) from general_map_error

    routes.append(_r("general_dispatch", nxt(), _pred_general_dispatch, _build_general_dispatch, (
        "map", "fold", "reduce", "scan")))

    return routes
