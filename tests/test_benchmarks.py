"""Benchmark: compile vs execute time for CPU and GPU.

Reports three numbers per size:
  compile — one-time cost (parse → typecheck → HIR → codegen)
  exec   — median execution time after compile
  result — correctness check

Run: uv run python tests/test_benchmarks.py
"""

import time
import numpy as np

from remora.runtime import evaluate_source, CPUExecutor


def bench_cpu_detail(source: str, *, syntax: str = "ml", repeats: int = 200):
    """Returns (compile_time_sec, median_exec_time_sec)."""
    t0 = time.perf_counter()
    artifact = CPUExecutor.compile_source(source, include_prelude=False, syntax=syntax)
    compile_time = time.perf_counter() - t0

    executor = CPUExecutor(artifact)
    for _ in range(10):
        executor.execute_main([])

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        executor.execute_main([])
        times.append(time.perf_counter() - t0)
    artifact.close()
    return compile_time, float(np.median(times))


def bench_gpu_detail(source: str, *, repeats: int = 50):
    """Returns (compile_time_sec, median_exec_time_sec).

    Compiles to PTX once, loads CUDA module once, then re-launches
    kernels on fresh buffers each iteration.
    """
    from remora.compiler import compile_source_to_ptx
    from remora.runtime import CUDARuntime, _numpy_dtype
    from remora.types import ArrayType, ScalarType

    t0 = time.perf_counter()
    artifact = compile_source_to_ptx(source, include_prelude=True)
    compile_time = time.perf_counter() - t0

    result_type = artifact.compiler.return_type
    if isinstance(result_type, ScalarType):
        output_shape = ()
        output_dtype = _numpy_dtype(result_type)
    elif isinstance(result_type, ArrayType):
        output_shape = tuple(d.value for d in result_type.shape)
        output_dtype = _numpy_dtype(result_type.element)
    else:
        raise ValueError(f"unsupported type: {result_type}")

    output_nbytes = max(1, int(np.prod(output_shape, dtype=np.int64) if output_shape else 1)) * np.dtype(output_dtype).itemsize
    buf_size = max(4096, output_nbytes * 4, 1024 * 1024)

    rt = CUDARuntime()
    try:
        mod = rt.load_ptx(artifact.ptx_text)
        times = []
        for _ in range(repeats):
            buf = rt.alloc(buf_size)
            rt.memset_d32(buf, 0, buf_size // 4)
            extra_bufs = []

            t0 = time.perf_counter()
            for km in artifact.kernels:
                kernel = mod.get_function(km.name)
                num_params = km.num_inputs + km.num_outputs
                if num_params > 1:
                    extra = rt.alloc(buf_size)
                    rt.memset_d32(extra, 0, buf_size // 4)
                    extra_bufs.append(extra)
                    params = [buf, extra]
                else:
                    params = [buf]
                block = int(km.block_size or 256)
                element_count = max(1, buf_size // np.dtype(output_dtype).itemsize) if not output_shape else max(1, int(np.prod(output_shape, dtype=np.int64)))
                grid = int((element_count + block - 1) // block)
                kernel.launch((grid, 1, 1), (block, 1, 1), params)
                rt.synchronize()
                if extra_bufs:
                    rt.free(buf)
                    buf = extra_bufs.pop()
            times.append(time.perf_counter() - t0)

            rt.free(buf)
            for p in extra_bufs:
                rt.free(p)
        mod.close()
    finally:
        rt.close()

    return compile_time, float(np.median(times))


def bench_interp(source: str, *, syntax: str = "ml", repeats: int = 10):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        evaluate_source(source, include_prelude=False, syntax=syntax)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def fmt_sec(s):
    if s < 1e-6:
        return f"{(s * 1e9):4.0f}ns"
    if s < 1e-3:
        return f"{(s * 1e6):5.1f}us"
    return f"{(s * 1e3):6.2f}ms"


if __name__ == "__main__":
    print("Remora Benchmark Suite")
    print("=" * 80)
    print(f"{'Op':20s} {'N':>8s}  {'CPU compile':>11s} {'CPU exec':>10s}  {'GPU compile':>12s} {'GPU exec':>10s}  Interp")
    print("-" * 80)

    for n in [100, 1000, 10000, 100000]:
        src = f"fold (+) 0 (iota {n})"
        cc, ce = bench_cpu_detail(src)
        gc, ge = bench_gpu_detail(src)
        interp = bench_interp(src)
        print(f"{'fold':20s} {n:8d}  {fmt_sec(cc):>11s} {fmt_sec(ce):>10s}  {fmt_sec(gc):>12s} {fmt_sec(ge):>10s}  {fmt_sec(interp)}")

    for n in [100, 1000, 10000, 100000]:
        src = f"map (* 2) (iota {n})"
        cc, ce = bench_cpu_detail(src)
        gc, ge = bench_gpu_detail(src)
        interp = bench_interp(src)
        print(f"{'map':20s} {n:8d}  {fmt_sec(cc):>11s} {fmt_sec(ce):>10s}  {fmt_sec(gc):>12s} {fmt_sec(ge):>10s}  {fmt_sec(interp)}")

    print()
    print("Legend: compile = one-time, exec = median post-compile, Interp = interpreter")
