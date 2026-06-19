"""Tests for multi-kernel execution plans."""

from __future__ import annotations

import pytest

from remora.execution_plan import (
    BufferSpec,
    ExecutionPlan,
    KernelStep,
    LoopPlan,
)


class TestBufferSpec:

    def test_construction(self):
        b = BufferSpec("pred", (10,), "i32")
        assert b.name == "pred"
        assert b.shape == (10,)
        assert b.dtype == "i32"

    def test_frozen(self):
        b = BufferSpec("x", (5,), "f32")
        with pytest.raises(AttributeError):
            b.name = "y"


class TestKernelStep:

    def test_construction(self):
        s = KernelStep("my_kernel", ["input_0"], "output")
        assert s.kernel_name == "my_kernel"
        assert s.input_refs == ["input_0"]
        assert s.output_ref == "output"
        assert s.is_reduction is False

    def test_reduction(self):
        s = KernelStep("reduce", ["input_0"], "result", is_reduction=True)
        assert s.is_reduction is True


class TestLoopPlan:

    def test_construction(self):
        body = [
            KernelStep("grad", ["params"], "grad_buf"),
            KernelStep("update", ["params", "grad_buf"], "params_new"),
        ]
        loop = LoopPlan(count=200, body=body, swap_pairs=[("params", "params_new")])
        assert loop.count == 200
        assert len(loop.body) == 2
        assert loop.swap_pairs == [("params", "params_new")]

    def test_default_swap_pairs(self):
        loop = LoopPlan(count=10, body=[KernelStep("k", ["a"], "b")])
        assert loop.swap_pairs == []


class TestExecutionPlan:

    def test_simple_pipeline(self):
        plan = ExecutionPlan(
            buffers=[
                BufferSpec("pred", (10,), "i32"),
                BufferSpec("scan", (10,), "i32"),
                BufferSpec("output", (10,), "f32"),
            ],
            steps=[
                KernelStep("eval_pred", ["input_0"], "pred"),
                KernelStep("prefix_sum", ["pred"], "scan"),
                KernelStep("scatter", ["input_0", "pred", "scan"], "output"),
            ],
            final_output="output",
            output_shape=(10,),
            output_dtype="f32",
        )
        assert plan.kernel_names() == {"eval_pred", "prefix_sum", "scatter"}
        assert "input_0" in plan.buffer_names()
        assert "pred" in plan.buffer_names()
        assert "output" in plan.buffer_names()

    def test_loop_plan(self):
        plan = ExecutionPlan(
            buffers=[
                BufferSpec("params", (3,), "f32"),
                BufferSpec("grad", (3,), "f32"),
                BufferSpec("params_new", (3,), "f32"),
            ],
            steps=[
                KernelStep("init", ["input_0"], "params"),
                LoopPlan(
                    count=200,
                    body=[
                        KernelStep("grad_fn", ["params"], "grad"),
                        KernelStep("update", ["params", "grad"], "params_new"),
                    ],
                    swap_pairs=[("params", "params_new")],
                ),
            ],
            final_output="params",
            output_shape=(3,),
            output_dtype="f32",
        )
        assert plan.kernel_names() == {"init", "grad_fn", "update"}
        plan.validate()

    def test_kernel_names_from_loop(self):
        plan = ExecutionPlan(
            buffers=[BufferSpec("a", (5,), "f32"), BufferSpec("b", (5,), "f32")],
            steps=[
                LoopPlan(
                    count=10,
                    body=[KernelStep("step", ["a"], "b")],
                    swap_pairs=[("a", "b")],
                ),
            ],
            final_output="a",
            output_shape=(5,),
            output_dtype="f32",
        )
        assert plan.kernel_names() == {"step"}

    def test_validate_ok(self):
        plan = ExecutionPlan(
            buffers=[BufferSpec("out", (5,), "f32")],
            steps=[KernelStep("k", ["input_0"], "out")],
            final_output="out",
            output_shape=(5,),
            output_dtype="f32",
        )
        plan.validate()

    def test_validate_missing_buffer(self):
        plan = ExecutionPlan(
            buffers=[],
            steps=[KernelStep("k", ["input_0"], "missing")],
            final_output="missing",
            output_shape=(5,),
            output_dtype="f32",
        )
        with pytest.raises(ValueError, match="undeclared buffers"):
            plan.validate()

    def test_validate_bad_final_output(self):
        plan = ExecutionPlan(
            buffers=[BufferSpec("out", (5,), "f32")],
            steps=[KernelStep("k", ["input_0"], "out")],
            final_output="nonexistent",
            output_shape=(5,),
            output_dtype="f32",
        )
        with pytest.raises(ValueError, match="final_output"):
            plan.validate()

    def test_validate_bad_swap_pair(self):
        plan = ExecutionPlan(
            buffers=[BufferSpec("a", (5,), "f32")],
            steps=[
                LoopPlan(
                    count=10,
                    body=[KernelStep("k", ["a"], "a")],
                    swap_pairs=[("a", "nonexistent")],
                ),
            ],
            final_output="a",
            output_shape=(5,),
            output_dtype="f32",
        )
        with pytest.raises(ValueError, match="nonexistent"):
            plan.validate()

    def test_validate_input_refs_ok(self):
        plan = ExecutionPlan(
            buffers=[BufferSpec("out", (5,), "f32")],
            steps=[KernelStep("k", ["input_0", "input_1"], "out")],
            final_output="out",
            output_shape=(5,),
            output_dtype="f32",
        )
        plan.validate()

    def test_final_output_can_be_input(self):
        plan = ExecutionPlan(
            buffers=[],
            steps=[],
            final_output="input_0",
            output_shape=(5,),
            output_dtype="f32",
        )
        plan.validate()
