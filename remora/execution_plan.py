"""Multi-kernel GPU execution plans.

An ``ExecutionPlan`` describes a sequence of GPU kernel launches,
temporary buffer allocations, and optional host-side loops with
buffer swapping.  The ``RemoraExecutor.execute_plan`` method
interprets these plans.

Use cases:
- Two-kernel parallel filter/replicate (prefix-sum + scatter)
- Host-orchestrated optimization loops (CPU loop, GPU kernels per step)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BufferSpec:
    """A temporary device buffer to allocate before plan execution.

    Attributes
    ----------
    name : str
        Unique name for referencing this buffer in plan steps.
    shape : tuple[int, ...]
        Static shape of the buffer.
    dtype : str
        Element type: ``"f32"``, ``"i32"``, or ``"i1"``.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class KernelStep:
    """Launch a single GPU kernel.

    Attributes
    ----------
    kernel_name : str
        Name of the kernel (must exist in the loaded PTX module).
    input_refs : list[str]
        Buffer names to pass as kernel inputs.  ``"input_0"``,
        ``"input_1"``, ... refer to the function's input arrays.
    output_ref : str
        Buffer name to write the kernel's output to.
    is_reduction : bool
        If True, the output buffer is zero-initialized before launch
        and the grid is sized based on the first input rather than
        the output.
    """

    kernel_name: str
    input_refs: list[str]
    output_ref: str
    is_reduction: bool = False


@dataclass(frozen=True)
class LoopPlan:
    """Execute a sequence of kernel steps in a host-side loop.

    After each iteration of the body, the buffers named in
    ``swap_pairs`` have their device pointers exchanged.  This
    implements the double-buffer pattern needed for iterative
    algorithms (e.g. gradient descent: each step reads ``params``
    and writes ``params_new``, then the two are swapped).

    Attributes
    ----------
    count : int
        Number of iterations.
    body : list[KernelStep]
        Kernel steps to execute each iteration.
    swap_pairs : list[tuple[str, str]]
        Pairs of buffer names whose device pointers are swapped
        after each iteration.
    """

    count: int
    body: list[KernelStep]
    swap_pairs: list[tuple[str, str]] = field(default_factory=list)


PlanStep = KernelStep | LoopPlan


@dataclass(frozen=True)
class ExecutionPlan:
    """A multi-kernel GPU execution plan.

    Attributes
    ----------
    buffers : list[BufferSpec]
        Temporary device buffers to allocate before execution.
    steps : list[PlanStep]
        Ordered sequence of kernel launches and/or host loops.
    final_output : str
        Name of the buffer holding the final result.
    output_shape : tuple[int, ...]
        Shape of the final output array.
    output_dtype : str
        Element type of the final output (``"f32"``, ``"i32"``).
    """

    buffers: list[BufferSpec]
    steps: list[PlanStep]
    final_output: str
    output_shape: tuple[int, ...]
    output_dtype: str

    def kernel_names(self) -> set[str]:
        """Return the set of all kernel names referenced by this plan."""
        names: set[str] = set()
        for step in self.steps:
            if isinstance(step, KernelStep):
                names.add(step.kernel_name)
            elif isinstance(step, LoopPlan):
                for body_step in step.body:
                    names.add(body_step.kernel_name)
        return names

    def buffer_names(self) -> set[str]:
        """Return the set of all buffer names referenced in steps."""
        names: set[str] = set()
        for step in self.steps:
            _collect_buffer_refs(step, names)
        return names

    def validate(self) -> None:
        """Check internal consistency.  Raises ``ValueError`` on problems."""
        declared = {b.name for b in self.buffers}
        referenced = self.buffer_names()
        input_refs = {n for n in referenced if n.startswith("input_")}
        non_input = referenced - input_refs
        missing = non_input - declared
        if missing:
            raise ValueError(
                f"Plan references undeclared buffers: {sorted(missing)}"
            )
        final_is_declared = self.final_output in declared
        final_is_input = self.final_output.startswith("input_")
        if not final_is_declared and not final_is_input:
            raise ValueError(
                f"final_output {self.final_output!r} is not a declared buffer or input"
            )
        for step in self.steps:
            if isinstance(step, LoopPlan):
                for a, b in step.swap_pairs:
                    a_ok = a in declared or a.startswith("input_")
                    b_ok = b in declared or b.startswith("input_")
                    if not a_ok:
                        raise ValueError(f"swap_pair name {a!r} is not declared")
                    if not b_ok:
                        raise ValueError(f"swap_pair name {b!r} is not declared")


def _collect_buffer_refs(step: PlanStep, names: set[str]) -> None:
    if isinstance(step, KernelStep):
        names.update(step.input_refs)
        names.add(step.output_ref)
    elif isinstance(step, LoopPlan):
        for body_step in step.body:
            _collect_buffer_refs(body_step, names)
        for a, b in step.swap_pairs:
            names.add(a)
            names.add(b)
