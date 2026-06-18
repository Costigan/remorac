"""Tests for the general GPU lowering path (Phase 3 of GPU_GENERAL_LOWERING_PLAN.md).

Tests are split into:
- Direct GPU lowering tests: create HIR directly, feed to GPU lowering, verify PTX
- Program compilation tests: compile Remora source via the full pipeline
- Numeric parity tests: compare CPU vs GPU output (GPU available only)
"""

import numpy as np
import pytest

from remora.codegen import CodegenUnavailable, KernelMeta, generate_mlir_descriptor_abi_ptx
from remora.compiler import (
    compile_function_source,
    compile_function_source_to_mlir_gpu_ptx,
)
from remora.executor import RemoraExecutor, RemoraExecutorError
from remora.gpu_lowering import (
    GPUScaffoldError,
    build_descriptor_abi_general_map_gpu_module,
    extract_gpu_module_body_as_module,
)
from remora.hir import (
    HIRArrayLit,
    HIRCast,
    HIRFold,
    HIRFoldRight,
    HIRFunction,
    HIRIf,
    HIRIndex,
    HIRIota,
    HIRLambda,
    HIRLit,
    HIRMap,
    HIRMatmul,
    HIRParam,
    HIRPrimCallable,
    HIRPrimOp,
    HIRReduce,
    HIRVar,
    HIRAppend,
    HIRCast,
    HIRDrop,
    HIRRavel,
    HIRReshape,
    HIRReverse,
    HIRRotate,
    HIRScatterAdd,
    HIRSort,
    HIRGrade,
    HIRSubarray,
    HIRTake,
    HIRTranspose,
    HIRWithShape,
)
from remora.pipeline import detect_toolchain, translate_llvmir_to_nvptx_text, translate_mlir_to_llvmir
from remora.runtime import CUDARuntime, evaluate_source, RuntimeUnavailable
from remora.types import BOOL, FLOAT, INT, ArrayType, ScalarType, StaticDim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_gpu_runtime():
    """Return a CUDARuntime if available, otherwise return None."""
    try:
        return CUDARuntime()
    except RuntimeUnavailable:
        pass
    import os
    if os.environ.get("REMORA_TEST_GPU") == "1":
        raise RuntimeError("CUDA runtime unavailable but REMORA_TEST_GPU=1 set")
    return None


def _build_and_compile_to_ptx(hfunc, kernel_name):
    """Build a general GPU module and translate to PTX."""
    gpu_module = build_descriptor_abi_general_map_gpu_module(
        hfunc, kernel_name=kernel_name,
    )
    device_module = extract_gpu_module_body_as_module(gpu_module.text)
    tc = detect_toolchain()
    llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=tc)
    ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=tc)
    return ptx, gpu_module


def _compile_source_to_hir(src, name, param_types, **kwargs):
    """Compile Remora source to HIR function."""
    art = compile_function_source(
        src, name, param_types,
        include_prelude=kwargs.get("include_prelude", False),
        syntax=kwargs.get("syntax", "ml"),
    )
    return art.hir_function


# ---------------------------------------------------------------------------
# GPU Expression Compiler Tests (direct HIR, no CPU pipeline)
# ---------------------------------------------------------------------------


class TestGPUExprCompiler:
    """Direct HIR → PTX compilation tests for structurally different patterns."""

    def test_map_with_fold_over_iota(self):
        """map (lambda i -> fold + 0 (map (lambda j -> index pos j) (iota 4))) (iota 3)"""
        N_input = 4
        N_output = 3

        inner_body = HIRIndex(
            array=HIRVar('pos', ArrayType(FLOAT, (StaticDim(N_input),))),
            indices=[HIRVar('j', INT)],
            result_type=FLOAT,
        )
        inner_map = HIRMap(
            frame_shape=(StaticDim(N_input),), cell_shape=(),
            func=HIRLambda(params=[HIRParam('j', INT)], body=inner_body, result_type=None),
            arrays=[HIRIota(StaticDim(N_input), ArrayType(INT, (StaticDim(N_input),)))],
            result_type=ArrayType(FLOAT, (StaticDim(N_input),)),
        )
        inner_fold = HIRFold(
            reduction_dim=StaticDim(N_input),
            func=HIRPrimCallable('+', (FLOAT, FLOAT), FLOAT),
            init=HIRLit(0.0, FLOAT),
            array=inner_map,
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(N_output),))
        outer_map = HIRMap(
            frame_shape=(StaticDim(N_output),), cell_shape=(),
            func=HIRLambda(params=[HIRParam('i', INT)], body=inner_fold, result_type=None),
            arrays=[HIRIota(StaticDim(N_output), ArrayType(INT, (StaticDim(N_output),)))],
            result_type=result_type,
        )
        hfunc = HIRFunction(
            name='test_fold',
            params=[HIRParam('pos', ArrayType(FLOAT, (StaticDim(N_input),)))],
            body=outer_map, return_type=result_type,
        )
        ptx, _scaffold = _build_and_compile_to_ptx(hfunc, "test_fold_kernel")
        assert ".visible .entry test_fold_kernel" in ptx
        assert "f32" in ptx or "float" in ptx.lower()

    def test_map_with_fold_over_iota_scalar(self):
        """map (lambda i -> fold + 0 (iota 4)) (iota 3) — fold over bare iota"""
        N_output = 3
        N_fold = 4

        inner_fold = HIRFold(
            reduction_dim=StaticDim(N_fold),
            func=HIRPrimCallable('+', (FLOAT, FLOAT), FLOAT),
            init=HIRLit(0.0, FLOAT),
            array=HIRIota(StaticDim(N_fold), ArrayType(INT, (StaticDim(N_fold),))),
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(N_output),))
        outer_map = HIRMap(
            frame_shape=(StaticDim(N_output),), cell_shape=(),
            func=HIRLambda(params=[HIRParam('i', INT)], body=inner_fold, result_type=None),
            arrays=[HIRIota(StaticDim(N_output), ArrayType(INT, (StaticDim(N_output),)))],
            result_type=result_type,
        )
        hfunc = HIRFunction(
            name='test_fold_scalar',
            params=[HIRParam('dummy', FLOAT)],
            body=outer_map, return_type=result_type,
        )
        ptx, _scaffold = _build_and_compile_to_ptx(hfunc, "test_fold_scalar")
        assert ".visible .entry test_fold_scalar" in ptx

    def test_map_with_index_on_input(self):
        """map (lambda i -> index pos i) (iota 4) — index directly"""
        N = 4
        body = HIRIndex(
            array=HIRVar('pos', ArrayType(FLOAT, (StaticDim(N),))),
            indices=[HIRVar('i', INT)],
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(N),))
        hfunc = HIRFunction(
            name='test_index',
            params=[HIRParam('pos', ArrayType(FLOAT, (StaticDim(N),)))],
            body=HIRMap(
                frame_shape=(StaticDim(N),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('i', INT)], body=body, result_type=None),
                arrays=[HIRIota(StaticDim(N), ArrayType(INT, (StaticDim(N),)))],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _scaffold = _build_and_compile_to_ptx(hfunc, "test_index")
        assert ".visible .entry test_index" in ptx

    def test_map_with_select_condition(self):
        """map (lambda v -> select (> v 0.0) 1.0 0.0) arr — branchless if"""
        N = 4
        body = HIRIf(
            condition=HIRPrimOp(">f", [HIRVar('v', FLOAT), HIRLit(0.0, FLOAT)], BOOL),
            then_branch=HIRLit(1.0, FLOAT),
            else_branch=HIRLit(0.0, FLOAT),
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(N),))
        hfunc = HIRFunction(
            name='test_select',
            params=[HIRParam('arr', ArrayType(FLOAT, (StaticDim(N),)))],
            body=HIRMap(
                frame_shape=(StaticDim(N),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('v', FLOAT)], body=body, result_type=None),
                arrays=[HIRVar('arr', result_type)],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _scaffold = _build_and_compile_to_ptx(hfunc, "test_select")
        assert ".visible .entry test_select" in ptx

    def test_map_with_binary_arithmetic(self):
        """map (lambda x y -> (+ (* x 2.0) y)) a b — chained binary ops"""
        N = 4
        mul_op = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        add_op = HIRPrimOp("+f", [mul_op, HIRVar('y', FLOAT)], FLOAT)
        result_type = ArrayType(FLOAT, (StaticDim(N),))
        hfunc = HIRFunction(
            name='test_chain',
            params=[
                HIRParam('a', result_type),
                HIRParam('b', result_type),
            ],
            body=HIRMap(
                frame_shape=(StaticDim(N),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT), HIRParam('y', FLOAT)], body=add_op, result_type=None),
                arrays=[HIRVar('a', result_type), HIRVar('b', result_type)],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _scaffold = _build_and_compile_to_ptx(hfunc, "test_chain")
        assert ".visible .entry test_chain" in ptx

    def test_rank2_map_with_index(self):
        """Rank-2: map (lambda i j -> index pos i j) (iota 3 2)"""
        rows, cols = 3, 2
        body = HIRIndex(
            array=HIRVar('pos', ArrayType(FLOAT, (StaticDim(rows), StaticDim(cols)))),
            indices=[HIRVar('i', INT), HIRVar('j', INT)],
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(rows), StaticDim(cols)))
        hfunc = HIRFunction(
            name='test_r2_idx',
            params=[HIRParam('pos', result_type)],
            body=HIRMap(
                frame_shape=(StaticDim(rows), StaticDim(cols)), cell_shape=(),
                func=HIRLambda(params=[HIRParam('i', INT), HIRParam('j', INT)], body=body, result_type=None),
                arrays=[HIRIota(StaticDim(rows*cols), ArrayType(INT, (StaticDim(rows*cols),)))],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _scaffold = _build_and_compile_to_ptx(hfunc, "test_r2_idx")
        assert ".visible .entry test_r2_idx" in ptx

    def test_rank2_array_fold_compiles(self):
        """Rank-2 array-valued fold now compiles (C.5)."""
        init_row = HIRArrayLit([HIRLit(0.0, FLOAT), HIRLit(0.0, FLOAT)],
                               ArrayType(FLOAT, (StaticDim(2),)))
        init_arr = HIRArrayLit([init_row, init_row, init_row],
                               ArrayType(FLOAT, (StaticDim(3), StaticDim(2))))
        fold_arr = HIRFold(
            reduction_dim=StaticDim(3),
            func=HIRPrimCallable("+", (FLOAT, FLOAT), FLOAT),
            init=init_arr,
            array=HIRIota(StaticDim(3), ArrayType(INT, (StaticDim(3),))),
            result_type=ArrayType(FLOAT, (StaticDim(3), StaticDim(2))),
        )
        hfunc = HIRFunction(
            name="r2fold", params=[HIRParam("x", FLOAT)],
            body=HIRMap(
                frame_shape=(StaticDim(3),), cell_shape=(),
                func=HIRLambda(params=[HIRParam("i", INT)], body=fold_arr, result_type=None),
                arrays=[HIRIota(StaticDim(3), ArrayType(INT, (StaticDim(3),)))],
                result_type=ArrayType(FLOAT, (StaticDim(3),)),
            ),
            return_type=ArrayType(FLOAT, (StaticDim(3),)),
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_r2fold")
        assert ".visible .entry test_r2fold" in ptx

    def test_array_valued_fold_compiles(self):
        """Rank-1 array-valued fold compiles (was previously out of scope)."""
        fold_arr = HIRFold(
            reduction_dim=StaticDim(3),
            func=HIRPrimCallable("+", (FLOAT, FLOAT), FLOAT),
            init=HIRArrayLit([HIRLit(0.0, FLOAT)] * 3, ArrayType(FLOAT, (StaticDim(3),))),
            array=HIRIota(StaticDim(3), ArrayType(INT, (StaticDim(3),))),
            result_type=ArrayType(FLOAT, (StaticDim(3),)),
        )
        hfunc = HIRFunction(
            name="arr_fold", params=[HIRParam("x", FLOAT)],
            body=HIRMap(
                frame_shape=(StaticDim(3),), cell_shape=(),
                func=HIRLambda(params=[HIRParam("i", INT)], body=fold_arr, result_type=None),
                arrays=[HIRIota(StaticDim(3), ArrayType(INT, (StaticDim(3),)))],
                result_type=ArrayType(FLOAT, (StaticDim(3),)),
            ),
            return_type=ArrayType(FLOAT, (StaticDim(3),)),
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_arr_fold")
        assert ".visible .entry test_arr_fold" in ptx


# ---------------------------------------------------------------------------
# Phase A: Descriptor-level view ops (GPU_CPU_PARITY_PLAN.md)
# ---------------------------------------------------------------------------


class TestPhaseAViewOps:
    """Direct HIR → PTX tests for descriptor-level view ops."""

    def test_take_compiles(self):
        """A.1: map (* 2.0) (take 3 arr) where arr has shape [5]."""
        arr_type = ArrayType(FLOAT, (StaticDim(5),))
        take_expr = HIRTake(3, HIRVar('arr', arr_type), ArrayType(FLOAT, (StaticDim(3),)))
        result_type = ArrayType(FLOAT, (StaticDim(3),))
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_take',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(3),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[take_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_take")
        assert ".visible .entry test_take" in ptx

    def test_drop_compiles(self):
        """A.2: map (* 2.0) (drop 2 arr) where arr has shape [5]."""
        arr_type = ArrayType(FLOAT, (StaticDim(5),))
        drop_expr = HIRDrop(2, HIRVar('arr', arr_type), ArrayType(FLOAT, (StaticDim(3),)))
        result_type = ArrayType(FLOAT, (StaticDim(3),))
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_drop',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(3),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[drop_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_drop")
        assert ".visible .entry test_drop" in ptx

    def test_subarray_compiles(self):
        """A.3: map (* 2.0) (subarray arr (1,1) (2,2)) where arr has shape [4,4]."""
        arr_type = ArrayType(FLOAT, (StaticDim(4), StaticDim(4)))
        sub_type = ArrayType(FLOAT, (StaticDim(2), StaticDim(2)))
        sub_expr = HIRSubarray(
            HIRVar('arr', arr_type),
            offsets=(StaticDim(1), StaticDim(1)),
            sizes=(StaticDim(2), StaticDim(2)),
            result_type=sub_type,
        )
        result_type = sub_type
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_sub',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(2), StaticDim(2)), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[sub_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_sub")
        assert ".visible .entry test_sub" in ptx

    def test_reverse_compiles(self):
        """A.5: map (* 2.0) (reverse arr) where arr has shape [4]."""
        arr_type = ArrayType(FLOAT, (StaticDim(4),))
        rev_expr = HIRReverse(HIRVar('arr', arr_type), arr_type)
        result_type = arr_type
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_rev',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(4),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[rev_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_rev")
        assert ".visible .entry test_rev" in ptx

    def test_rotate_compiles(self):
        """A.6: map (* 2.0) (rotate 2 arr) where arr has shape [4]."""
        arr_type = ArrayType(FLOAT, (StaticDim(4),))
        rot_expr = HIRRotate(HIRVar('arr', arr_type), StaticDim(2), arr_type)
        result_type = arr_type
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_rot',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(4),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[rot_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_rot")
        assert ".visible .entry test_rot" in ptx

    def test_transpose_compiles(self):
        """A.7: map (* 2.0) (transpose arr) where arr has shape [3,2]."""
        arr_type = ArrayType(FLOAT, (StaticDim(3), StaticDim(2)))
        trans_type = ArrayType(FLOAT, (StaticDim(2), StaticDim(3)))
        trans_expr = HIRTranspose(HIRVar('arr', arr_type), trans_type)
        result_type = trans_type
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_trans',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(2), StaticDim(3)), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[trans_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_trans")
        assert ".visible .entry test_trans" in ptx

    def test_array_lit_in_fold_body_compiles(self):
        """A.8: map (\\_ -> fold (+) 0.0 [1.0, 2.0, 3.0]) (iota 3)."""
        arr_lit = HIRArrayLit(
            [HIRLit(1.0, FLOAT), HIRLit(2.0, FLOAT), HIRLit(3.0, FLOAT)],
            ArrayType(FLOAT, (StaticDim(3),)),
        )
        inner_fold = HIRFold(
            reduction_dim=StaticDim(3),
            func=HIRPrimCallable('+', (FLOAT, FLOAT), FLOAT),
            init=HIRLit(0.0, FLOAT),
            array=arr_lit,
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(3),))
        hfunc = HIRFunction(
            name='test_arrlit',
            params=[HIRParam('dummy', FLOAT)],
            body=HIRMap(
                frame_shape=(StaticDim(3),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('i', INT)], body=inner_fold, result_type=None),
                arrays=[HIRIota(StaticDim(3), ArrayType(INT, (StaticDim(3),)))],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_arrlit")
        assert ".visible .entry test_arrlit" in ptx

    def test_chained_take_drop_compiles(self):
        """Chained: map (* 2.0) (take 2 (drop 1 arr)) where arr has shape [5]."""
        arr_type = ArrayType(FLOAT, (StaticDim(5),))
        drop_type = ArrayType(FLOAT, (StaticDim(4),))
        take_type = ArrayType(FLOAT, (StaticDim(2),))
        drop_expr = HIRDrop(1, HIRVar('arr', arr_type), drop_type)
        take_expr = HIRTake(2, drop_expr, take_type)
        result_type = take_type
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_chain_td',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(2),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[take_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_chain_td")
        assert ".visible .entry test_chain_td" in ptx


# ---------------------------------------------------------------------------
# Phase B: Descriptor reinterpretation ops (GPU_CPU_PARITY_PLAN.md)
# ---------------------------------------------------------------------------


class TestPhaseBReinterpOps:
    """Direct HIR → PTX tests for descriptor reinterpretation ops."""

    def test_reshape_compiles(self):
        """B.1: map (* 2.0) (reshape [2,3] arr) where arr has shape [6]."""
        arr_type = ArrayType(FLOAT, (StaticDim(6),))
        reshaped_type = ArrayType(FLOAT, (StaticDim(2), StaticDim(3)))
        reshape_expr = HIRReshape(HIRVar('arr', arr_type), reshaped_type)
        result_type = reshaped_type
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_reshape',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(2), StaticDim(3)), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[reshape_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_reshape")
        assert ".visible .entry test_reshape" in ptx

    def test_ravel_compiles(self):
        """B.2: map (* 2.0) (ravel arr) where arr has shape [2,3]."""
        arr_type = ArrayType(FLOAT, (StaticDim(2), StaticDim(3)))
        raveled_type = ArrayType(FLOAT, (StaticDim(6),))
        ravel_expr = HIRRavel(HIRVar('arr', arr_type), raveled_type)
        result_type = raveled_type
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_ravel',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(6),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[ravel_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_ravel")
        assert ".visible .entry test_ravel" in ptx

    def test_append_compiles(self):
        """B.3: map (* 2.0) (append left right) where left=[3], right=[2]."""
        left_type = ArrayType(FLOAT, (StaticDim(3),))
        right_type = ArrayType(FLOAT, (StaticDim(2),))
        result_type = ArrayType(FLOAT, (StaticDim(5),))
        append_expr = HIRAppend(
            HIRVar('left', left_type),
            HIRVar('right', right_type),
            result_type,
        )
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_append',
            params=[
                HIRParam('left', left_type),
                HIRParam('right', right_type),
            ],
            body=HIRMap(
                frame_shape=(StaticDim(5),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[append_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_append")
        assert ".visible .entry test_append" in ptx

    def test_withshape_compiles(self):
        """B.4: map (* 2.0) (with-shape [3,4] src) where src has shape [4]."""
        src_type = ArrayType(FLOAT, (StaticDim(4),))
        target_type = ArrayType(FLOAT, (StaticDim(3), StaticDim(4)))
        ws_expr = HIRWithShape(HIRVar('src', src_type), target_type)
        result_type = target_type
        body = HIRPrimOp("*f", [HIRVar('x', FLOAT), HIRLit(2.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_ws',
            params=[HIRParam('src', src_type)],
            body=HIRMap(
                frame_shape=(StaticDim(3), StaticDim(4)), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[ws_expr],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_ws")
        assert ".visible .entry test_ws" in ptx

    def test_scatter_add_compiles(self):
        """B.5: scatter_add(target, idx, val) with literal index."""
        arr_type = ArrayType(FLOAT, (StaticDim(6),))
        sa_body = HIRScatterAdd(
            target=HIRVar('target', arr_type),
            index=HIRLit(2, INT),
            update=HIRLit(1.0, FLOAT),
            result_type=arr_type,
        )
        hfunc = HIRFunction(
            name='test_sa',
            params=[HIRParam('target', arr_type)],
            body=sa_body,
            return_type=arr_type,
        )
        from remora.gpu_lowering import build_descriptor_abi_scatter_add_gpu_module, extract_gpu_module_body_as_module
        from remora.pipeline import detect_toolchain, translate_llvmir_to_nvptx_text, translate_mlir_to_llvmir
        gpu_module = build_descriptor_abi_scatter_add_gpu_module(hfunc, kernel_name="test_sa")
        device_module = extract_gpu_module_body_as_module(gpu_module.text)
        tc = detect_toolchain()
        llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=tc)
        ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=tc)
        assert ".visible .entry test_sa" in ptx


# ---------------------------------------------------------------------------
# Phase C: Hardening (GPU_CPU_PARITY_PLAN.md)
# ---------------------------------------------------------------------------


class TestPhaseCHardening:
    """Tests for type-aware arithmetic, comparisons, and array-typed if."""

    def test_i32_arithmetic_compiles(self):
        """C.1: i32 constant arithmetic cast to f32 inside a map body."""
        N = 4
        i32_mul = HIRPrimOp("*i", [HIRLit(2, INT), HIRLit(3, INT)], INT)
        i32_add = HIRPrimOp("+i", [i32_mul, HIRLit(1, INT)], INT)
        body = HIRCast(i32_add, from_type=INT, to_type=FLOAT, result_type=FLOAT)
        result_type = ArrayType(FLOAT, (StaticDim(N),))
        hfunc = HIRFunction(
            name='test_i32arith',
            params=[HIRParam('arr', result_type)],
            body=HIRMap(
                frame_shape=(StaticDim(N),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[HIRVar('arr', result_type)],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_i32arith")
        assert ".visible .entry test_i32arith" in ptx

    def test_i32_comparison_compiles(self):
        """C.2: i32 comparison producing i1 for select."""
        N = 4
        body = HIRIf(
            HIRPrimOp("<i", [HIRLit(2, INT), HIRLit(3, INT)], BOOL),
            HIRLit(1.0, FLOAT),
            HIRLit(0.0, FLOAT),
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(N),))
        hfunc = HIRFunction(
            name='test_i32cmp',
            params=[HIRParam('arr', result_type)],
            body=HIRMap(
                frame_shape=(StaticDim(N),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[HIRVar('arr', result_type)],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_i32cmp")
        assert ".visible .entry test_i32cmp" in ptx

    def test_array_typed_if_compiles(self):
        """C.4: HIRIf where branches produce arrays → per-component select."""
        N = 3
        arr_type = ArrayType(FLOAT, (StaticDim(N),))
        body = HIRIf(
            HIRPrimOp(">f", [HIRVar('x', FLOAT), HIRLit(0.0, FLOAT)], BOOL),
            HIRArrayLit(
                [HIRLit(1.0, FLOAT), HIRLit(2.0, FLOAT)],
                ArrayType(FLOAT, (StaticDim(2),)),
            ),
            HIRArrayLit(
                [HIRLit(3.0, FLOAT), HIRLit(4.0, FLOAT)],
                ArrayType(FLOAT, (StaticDim(2),)),
            ),
            result_type=ArrayType(FLOAT, (StaticDim(2),)),
        )
        result_type = ArrayType(FLOAT, (StaticDim(N), StaticDim(2)))
        hfunc = HIRFunction(
            name='test_arrif',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(N),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[HIRVar('arr', arr_type)],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_arrif")
        assert ".visible .entry test_arrif" in ptx

    def test_stride_support_subarray(self):
        """C.6: subarray view uses correct strided access (via Phase A test)."""
        arr_type = ArrayType(FLOAT, (StaticDim(6), StaticDim(6)))
        sub_type = ArrayType(FLOAT, (StaticDim(3), StaticDim(3)))
        sub_expr = HIRSubarray(
            HIRVar('arr', arr_type),
            offsets=(StaticDim(2), StaticDim(1)),
            sizes=(StaticDim(3), StaticDim(3)),
            result_type=sub_type,
        )
        body = HIRPrimOp("+f", [HIRVar('x', FLOAT), HIRLit(1.0, FLOAT)], FLOAT)
        hfunc = HIRFunction(
            name='test_stride',
            params=[HIRParam('arr', arr_type)],
            body=HIRMap(
                frame_shape=(StaticDim(3), StaticDim(3)), cell_shape=(),
                func=HIRLambda(params=[HIRParam('x', FLOAT)], body=body, result_type=None),
                arrays=[sub_expr],
                result_type=sub_type,
            ),
            return_type=sub_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_stride")
        assert ".visible .entry test_stride" in ptx

    def test_fold_right_compiles(self):
        """HIRFoldRight: reverse fold over iota inside a map body."""
        N = 4
        inner_fold = HIRFoldRight(
            reduction_dim=StaticDim(N),
            func=HIRPrimCallable('+', (FLOAT, FLOAT), FLOAT),
            init=HIRLit(0.0, FLOAT),
            array=HIRIota(StaticDim(N), ArrayType(INT, (StaticDim(N),))),
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(3),))
        hfunc = HIRFunction(
            name='test_foldr',
            params=[HIRParam('dummy', FLOAT)],
            body=HIRMap(
                frame_shape=(StaticDim(3),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('i', INT)], body=inner_fold, result_type=None),
                arrays=[HIRIota(StaticDim(3), ArrayType(INT, (StaticDim(3),)))],
                result_type=result_type,
            ),
            return_type=result_type,
        )
        ptx, _ = _build_and_compile_to_ptx(hfunc, "test_foldr")
        assert ".visible .entry test_foldr" in ptx

    def test_parallel_scan_compiles(self):
        """Parallel Hillis-Steele scan compiles to PTX."""
        from remora.hir import HIRScan
        from remora.gpu_lowering import build_descriptor_abi_f32_scan_gpu_module, extract_gpu_module_body_as_module
        arr_type = ArrayType(FLOAT, (StaticDim(8),))
        hfunc = HIRFunction(
            name='test_scan',
            params=[HIRParam('xs', arr_type)],
            body=HIRScan(
                reduction_dim=StaticDim(8),
                func=HIRPrimCallable('+', (FLOAT, FLOAT), FLOAT),
                init=HIRLit(0.0, FLOAT),
                array=HIRVar('xs', arr_type),
                exclusive=False,
                right=False,
                result_type=arr_type,
            ),
            return_type=arr_type,
        )
        gpu_module = build_descriptor_abi_f32_scan_gpu_module(hfunc, kernel_name="test_scan")
        device_module = extract_gpu_module_body_as_module(gpu_module.text)
        tc = detect_toolchain()
        llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=tc)
        ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=tc)
        assert ".visible .entry test_scan" in ptx

    def test_matmul_compiles(self):
        """HIRMatmul: GPU matmul kernel compiles to PTX."""
        from remora.gpu_lowering import build_descriptor_abi_matmul_gpu_module, extract_gpu_module_body_as_module
        left_type = ArrayType(FLOAT, (StaticDim(3), StaticDim(4)))
        right_type = ArrayType(FLOAT, (StaticDim(4), StaticDim(2)))
        result_type = ArrayType(FLOAT, (StaticDim(3), StaticDim(2)))
        hfunc = HIRFunction(
            name='test_mm',
            params=[HIRParam('a', left_type), HIRParam('b', right_type)],
            body=HIRMatmul(HIRVar('a', left_type), HIRVar('b', right_type), result_type),
            return_type=result_type,
        )
        gpu_module = build_descriptor_abi_matmul_gpu_module(hfunc, kernel_name="test_mm")
        device_module = extract_gpu_module_body_as_module(gpu_module.text)
        tc = detect_toolchain()
        llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=tc)
        ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=tc)
        assert ".visible .entry test_mm" in ptx

    def test_sort_compiles(self):
        """HIRSort: GPU sort kernel compiles to PTX."""
        from remora.gpu_lowering import build_descriptor_abi_sort_gpu_module, extract_gpu_module_body_as_module
        arr_type = ArrayType(FLOAT, (StaticDim(6),))
        hfunc = HIRFunction(
            name='test_sort',
            params=[HIRParam('xs', arr_type)],
            body=HIRSort(HIRVar('xs', arr_type), arr_type),
            return_type=arr_type,
        )
        gpu_module = build_descriptor_abi_sort_gpu_module(hfunc, kernel_name="test_sort")
        device_module = extract_gpu_module_body_as_module(gpu_module.text)
        tc = detect_toolchain()
        llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=tc)
        ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=tc)
        assert ".visible .entry test_sort" in ptx

    def test_grade_compiles(self):
        """HIRGrade: GPU grade kernel compiles to PTX."""
        from remora.gpu_lowering import build_descriptor_abi_grade_gpu_module, extract_gpu_module_body_as_module
        arr_type = ArrayType(FLOAT, (StaticDim(6),))
        grade_type = ArrayType(INT, (StaticDim(6),))
        hfunc = HIRFunction(
            name='test_grade',
            params=[HIRParam('xs', arr_type)],
            body=HIRGrade(HIRVar('xs', arr_type), grade_type),
            return_type=grade_type,
        )
        gpu_module = build_descriptor_abi_grade_gpu_module(hfunc, kernel_name="test_grade")
        device_module = extract_gpu_module_body_as_module(gpu_module.text)
        tc = detect_toolchain()
        llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=tc)
        ptx = translate_llvmir_to_nvptx_text(llvm_ir, toolchain=tc)
        assert ".visible .entry test_grade" in ptx


# ---------------------------------------------------------------------------
# Program Compilation Tests (full pipeline: source → PTX)
# ---------------------------------------------------------------------------


class TestGPUProgramCompilation:
    """Verify programs compile end-to-end through the full Remora pipeline."""

    def test_simple_unary_map_compiles(self):
        """Simple elementwise map: map (* 2.0) xs"""
        src = "def scale xs = map (* 2.0) xs"
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, "scale", (ArrayType(FLOAT, (StaticDim(4),)),),
        )
        assert ".visible .entry" in ptx
        assert len(kernels) == 1
        assert kernels[0].num_inputs == 1

    def test_simple_binary_map_compiles(self):
        """Simple binary map: map (+) xs ys"""
        src = "def add xs ys = map (+) xs ys"
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, "add",
            (ArrayType(FLOAT, (StaticDim(4),)), ArrayType(FLOAT, (StaticDim(4),))),
        )
        assert ".visible .entry" in ptx
        assert len(kernels) == 1
        assert kernels[0].num_inputs == 2

    def test_i32_map_compiles(self):
        """Integer map on i32."""
        src = "def inc xs = map (+ 1) xs"
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, "inc", (ArrayType(INT, (StaticDim(4),)),),
        )
        assert ".visible .entry" in ptx
        assert len(kernels) == 1

    def test_map_with_condition_lisp_compiles(self):
        """Map with conditional: Lisp syntax."""
        src = (
            "(define/pi () (thresh [x (Array Float 4)] (Array Float 4))"
            " (map (lambda (v) (select (> v 0.0) 1.0 0.0)) x))"
        )
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, "thresh", (ArrayType(FLOAT, (StaticDim(4),)),),
            syntax="lisp", include_prelude=False,
        )
        assert ".visible .entry" in ptx
        assert len(kernels) == 1

    def test_fold_reduction_compiles(self):
        """Fold (reduction) kernel."""
        src = "def sum xs = fold (+) 0.0 xs"
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, "sum", (ArrayType(FLOAT, (StaticDim(4),)),),
            include_prelude=False,
        )
        assert ".visible .entry" in ptx
        assert len(kernels) == 1
        assert kernels[0].is_reduction

    def test_rank2_map_compiles(self):
        """Rank-2 map: 2D array."""
        src = "def scale xs = map (* 2.0) xs"
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, "scale", (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
        )
        assert ".visible .entry" in ptx
        assert kernels[0].output_shape == (2, 3)


# ---------------------------------------------------------------------------
# Dispatch Chain Tests
# ---------------------------------------------------------------------------


class TestDispatchChain:
    """Verify the general path is correctly integrated into the dispatch chain."""

    def test_general_path_catches_compound_map(self):
        """Compound-body maps go through the general path."""
        from remora.lowering.tensor_ops import _body_needs_tensor_lowering

        N = 4
        inner_fold = HIRFold(
            reduction_dim=StaticDim(N),
            func=HIRPrimCallable('+', (FLOAT, FLOAT), FLOAT),
            init=HIRLit(0.0, FLOAT),
            array=HIRIota(StaticDim(N), ArrayType(INT, (StaticDim(N),))),
            result_type=FLOAT,
        )
        result_type = ArrayType(FLOAT, (StaticDim(N),))
        hfunc = HIRFunction(
            name='test_dispatch',
            params=[HIRParam('dummy', FLOAT)],
            body=HIRMap(
                frame_shape=(StaticDim(N),), cell_shape=(),
                func=HIRLambda(params=[HIRParam('i', INT)], body=inner_fold, result_type=None),
                arrays=[HIRIota(StaticDim(N), ArrayType(INT, (StaticDim(N),)))],
                result_type=result_type,
            ),
            return_type=result_type,
        )

        from remora.codegen import generate_mlir_descriptor_abi_ptx
        ptx, kernels = generate_mlir_descriptor_abi_ptx(hfunc, kernel_name="test_dispatch")
        assert len(kernels) == 1
        assert kernels[0].name == "test_dispatch"
        assert "f32" in ptx or "float" in ptx.lower()


# ---------------------------------------------------------------------------
# Numeric Parity Tests (GPU available only)
# ---------------------------------------------------------------------------


class TestGPUNumericParity:
    """Compare GPU general-path output with CPU compiled output."""

    @pytest.fixture(autouse=True)
    def _setup_gpu(self):
        rt = _try_gpu_runtime()
        if rt is None:
            pytest.skip("CUDA runtime not available")
        self._rt = rt
        yield
        self._rt.close()

    def _run_parity(self, src, name, param_types, inputs, include_prelude=False, syntax="ml"):
        """Compile on CPU and GPU, execute both, assert close match."""
        # CPU: use interpreter for reference
        if syntax == "ml":
            call_expr = f"({name} {' '.join(self._format_input(v) for v in inputs)})"
        else:
            call_expr = f"({name} {' '.join(self._format_input(v) for v in inputs)})"
        full_src = f"{src}\n{call_expr}"
        cpu_result = evaluate_source(
            full_src, include_prelude=include_prelude, syntax=syntax,
        )
        cpu_arr = np.asarray(cpu_result, dtype=np.float32)

        # GPU
        ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
            src, name, param_types,
            include_prelude=include_prelude, syntax=syntax,
        )
        executor = RemoraExecutor(ptx, kernels, runtime=self._rt)
        try:
            gpu_result = executor.execute_main(inputs)
        finally:
            if hasattr(executor, 'close'):
                executor.close()

        gpu_arr = np.asarray(gpu_result, dtype=np.float32)
        np.testing.assert_allclose(
            gpu_arr, cpu_arr, rtol=1e-4, atol=1e-5,
            err_msg=f"GPU vs CPU mismatch for '{name}'",
        )

    @staticmethod
    def _format_input(v):
        if isinstance(v, np.ndarray):
            if v.dtype == np.float32:
                rows = []
                for row in v:
                    rows.append("[" + " ".join(f"{x:.6f}" for x in np.atleast_1d(row)) + "]")
                return "[" + " ".join(rows) + "]"
            elif v.dtype == np.int32:
                rows = []
                for row in v:
                    rows.append("[" + " ".join(str(int(x)) for x in np.atleast_1d(row)) + "]")
                return "[" + " ".join(rows) + "]"
            else:
                return str(v.tolist()).replace(",", "")
        elif isinstance(v, (np.floating, float)):
            return f"{float(v):.6f}"
        elif isinstance(v, (np.integer, int)):
            return str(int(v))
        return str(v)

    def test_unary_map_parity(self):
        """scale by 2.0"""
        src = "def scale xs = map (* 2.0) xs"
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self._run_parity(src, "scale", (ArrayType(FLOAT, (StaticDim(4),)),), [x])

    def test_binary_map_parity(self):
        """element-wise add"""
        src = "def add xs ys = map (+) xs ys"
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        self._run_parity(
            src, "add",
            (ArrayType(FLOAT, (StaticDim(4),)), ArrayType(FLOAT, (StaticDim(4),))),
            [x, y],
        )

    def test_i32_map_parity(self):
        """i32 add 1"""
        src = "def inc xs = map (+ 1) xs"
        x = np.array([1, 2, 3, 4], dtype=np.int32)
        self._run_parity(src, "inc", (ArrayType(INT, (StaticDim(4),)),), [x])

    def test_rank2_map_parity(self):
        """rank-2 scale"""
        src = "def scale xs = map (* 2.0) xs"
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        self._run_parity(
            src, "scale",
            (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
            [x],
        )

    def test_reduction_parity(self):
        """fold sum"""
        src = "def sum xs = fold (+) 0.0 xs"
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self._run_parity(
            src, "sum",
            (ArrayType(FLOAT, (StaticDim(4),)),),
            [x],
            include_prelude=False,
        )

    def test_dot_reduction_parity(self):
        """dot product reduction"""
        src = "def dot xs ys = fold (+) 0.0 (map (*) xs ys)"
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        self._run_parity(
            src, "dot",
            (ArrayType(FLOAT, (StaticDim(4),)), ArrayType(FLOAT, (StaticDim(4),))),
            [x, y],
            include_prelude=False,
        )
