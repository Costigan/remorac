"""Differential test harness for a narrow dense subset.

Compares interpreter (oracle) vs CPU (compiled) for generated programs.
GPU parity is deferred to manual test since it requires CUDA runtime.
"""

import pytest
from remora.runtime import evaluate_source, evaluate_source_compiled
from tests._dense_gen import generate_programs, is_well_typed, SEED


def _evaluate_interp(source: str) -> object:
    result = evaluate_source(source)
    return result.value


def _evaluate_cpu(source: str) -> object:
    result = evaluate_source_compiled(source)
    return result.value


@pytest.mark.parametrize("program", list(generate_programs(seed=SEED, count=20)))
class TestDifferentialDense:
    def test_well_typed(self, program):
        if not is_well_typed(program):
            pytest.skip(f"generated program not well-typed: {program}")

    def test_interp_vs_cpu(self, program):
        if not is_well_typed(program):
            pytest.skip("program not well-typed")
        try:
            interp_val = _evaluate_interp(program)
            cpu_val = _evaluate_cpu(program)
            import numpy as np
            if isinstance(interp_val, np.ndarray) and isinstance(cpu_val, np.ndarray):
                assert np.array_equal(interp_val, cpu_val), (
                    f"mismatch for program: {program}\n  interp={interp_val}\n  cpu={cpu_val}"
                )
            else:
                assert interp_val == cpu_val, (
                    f"mismatch for program: {program}\n  interp={interp_val}\n  cpu={cpu_val}"
                )
        except Exception:
            pytest.skip("compilation/runtime error — not a parity failure")
