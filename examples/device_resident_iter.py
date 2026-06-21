#!/usr/bin/env python
"""Device-resident iteration with Remora's DeviceArray API.

Iterative GPU workflows (fixed-point solvers, simulations, the inner
loop of a Mandelbrot-style recurrence) launch the same kernel many
times.  Copying the array to and from host memory every step dominates
the wall-clock time.  ``RemoraExecutor.execute_to_device`` keeps the
data resident on the GPU between launches, so only one upload and one
download are needed for the whole loop.

This mirrors the inner z = z*z*a + c recurrence used by the Mandelbrot
example, but without the host-side escape masking that forces a
round-trip every step.

Run::

    uv run python examples/device_resident_iter.py
"""

import time

import numpy as np

from remora.compiler import compile_function_source_to_mlir_gpu_ptx
from remora.executor import DeviceArray, RemoraExecutor
from remora.runtime import CUDARuntime, RuntimeUnavailable
from remora.types import FLOAT, ArrayType, StaticDim

N = 1_000_000
STEPS = 100


def main() -> None:
    try:
        runtime = CUDARuntime()
    except RuntimeUnavailable as exc:
        print(f"GPU unavailable, skipping: {exc}")
        return

    # A contracting recurrence so the values stay bounded over many steps.
    ptx, kernels, _ = compile_function_source_to_mlir_gpu_ptx(
        "def step z = map (\\x -> x * x * 0.4 + 0.1) z",
        "step",
        (ArrayType(FLOAT, (StaticDim(N),)),),
        include_prelude=False,
        kernel_name="step",
    )
    xs = np.random.default_rng(0).standard_normal(N).astype(np.float32) * 0.1

    with RemoraExecutor(ptx, kernels, runtime=runtime) as exe:
        # Host reference.
        ref = xs.copy()
        for _ in range(STEPS):
            ref = ref * ref * 0.4 + 0.1

        # Device-resident: one upload, STEPS launches, one download.
        cur = DeviceArray.from_numpy(exe, xs)
        warm = exe.execute_to_device("step", [cur])
        warm.free()
        t0 = time.perf_counter()
        cur = DeviceArray.from_numpy(exe, xs)
        for _ in range(STEPS):
            nxt = exe.execute_to_device("step", [cur])
            cur.free()
            cur = nxt
        device_result = cur.to_numpy()
        cur.free()
        device_ms = (time.perf_counter() - t0) * 1e3

        # Per-call transfer: upload + download every step.
        t0 = time.perf_counter()
        y = xs.copy()
        for _ in range(STEPS):
            y = np.asarray(exe.execute("step", [y]))
        transfer_ms = (time.perf_counter() - t0) * 1e3

    runtime.close()

    ok = np.allclose(device_result, ref, rtol=1e-3, atol=1e-4)
    print(f"N={N}, steps={STEPS}")
    print(f"  device-resident : {device_ms:8.2f} ms")
    print(f"  per-call copy   : {transfer_ms:8.2f} ms")
    print(f"  speedup         : {transfer_ms / device_ms:8.2f}x")
    print(f"  result correct  : {ok}")


if __name__ == "__main__":
    main()
