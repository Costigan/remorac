import importlib.util

import pytest


def test_python_package_imports():
    import remora
    import remora.abi
    import remora.defunc
    import remora.errors
    import remora.hir
    import remora.lowering
    import remora.runtime
    import remora.typechecker
    import remora.types

    assert remora.abi.RemoraMemRef0.__name__ == "RemoraMemRef0"


def test_numpy_imports():
    import numpy as np

    assert np.asarray([1, 2, 3]).shape == (3,)


def test_lark_imports():
    import lark

    assert lark.Lark is not None


def test_cuda_python_imports_if_installed():
    if importlib.util.find_spec("cuda") is None:
        pytest.skip("cuda-python is not installed")

    try:
        from cuda import cuda
    except ImportError:
        from cuda.bindings import driver as cuda

    assert cuda is not None


def test_mlir_or_iree_imports_if_available():
    if importlib.util.find_spec("mlir") is not None:
        from mlir.ir import Context

        assert Context is not None
        return

    if importlib.util.find_spec("iree") is not None:
        import iree.compiler

        assert iree.compiler is not None
        return

    pytest.skip("MLIR or IREE compiler bindings are not installed")
