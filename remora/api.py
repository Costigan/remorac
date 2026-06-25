"""Public Python API for Remora Dense Core.

Provides ``RemoraFunction`` — a compiled Remora function callable from
Python with NumPy arrays — and ``compile_function`` for compiling
Remora source into callable wrappers.
"""

from __future__ import annotations

import numpy as np

from remora.runtime import CPUFunctionExecutor, EvaluationResult
from remora.types import ArrayType, FuncType, RemoraType, ScalarType


class RemoraRankMismatchError(TypeError):
    """Input array rank or shape does not match the Remora type signature."""


class RemoraFunction:
    """A compiled Remora function callable from Python.

    Wraps a compiled native function.  Accepts NumPy arrays, performs
    JIT rank/shape checking, and returns NumPy arrays or scalars.

    Example::

        fn = compile_function(
            "(define/pi () (scale [xs (Array Float 4)] (Array Float 4)) (map (* 2.0) xs))",
            "scale",
            syntax="lisp",
        )
        result = fn(np.array([1.0, 2.0, 3.0, 4.0]))
        # result = array([2.0, 4.0, 6.0, 8.0])
    """

    def __init__(self, executor: CPUFunctionExecutor, name: str,
                 param_types: tuple[RemoraType, ...], return_type: RemoraType) -> None:
        self._executor = executor
        self._name = name
        self._param_types = param_types
        self._return_type = return_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def param_types(self) -> tuple[RemoraType, ...]:
        return self._param_types

    @property
    def return_type(self) -> RemoraType:
        return self._return_type

    def __repr__(self) -> str:
        params = ", ".join(str(t) for t in self._param_types)
        return f"RemoraFunction({self._name}({params}) -> {self._return_type})"

    def __call__(self, *args: np.ndarray) -> np.ndarray | np.floating | np.integer:
        if len(args) != len(self._param_types):
            raise TypeError(
                f"{self._name}() expects {len(self._param_types)} argument(s), "
                f"got {len(args)}"
            )

        arrays: list[np.ndarray] = []
        for i, (arg, expected_type) in enumerate(zip(args, self._param_types)):
            arr = np.asarray(arg)
            _check_argument(arr, expected_type, self._name, i)
            if isinstance(expected_type, ArrayType):
                if expected_type.element.name == "float" and arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                elif expected_type.element.name == "float64" and arr.dtype != np.float64:
                    arr = arr.astype(np.float64)
                elif expected_type.element.name == "int" and arr.dtype != np.int32:
                    arr = arr.astype(np.int32)
            arrays.append(arr)

        result = self._executor.execute(*arrays)
        return result.value


def _check_argument(arr: np.ndarray, expected: RemoraType, func_name: str, index: int) -> None:
    if isinstance(expected, ArrayType):
        expected_rank = expected.rank
        if arr.ndim != expected_rank:
            raise RemoraRankMismatchError(
                f"{func_name}() argument {index}: expected rank-{expected_rank} "
                f"array, got rank-{arr.ndim} (shape {arr.shape})"
            )
        for axis, dim in enumerate(expected.shape):
            expected_size = int(dim.value)
            if arr.shape[axis] != expected_size:
                raise RemoraRankMismatchError(
                    f"{func_name}() argument {index}: dimension {axis} expected "
                    f"size {expected_size}, got {arr.shape[axis]} (shape {arr.shape})"
                )
    elif isinstance(expected, ScalarType):
        if arr.ndim != 0:
            raise RemoraRankMismatchError(
                f"{func_name}() argument {index}: expected scalar, "
                f"got rank-{arr.ndim} array (shape {arr.shape})"
            )


def compile_function(
    source: str,
    function_name: str,
    *,
    param_types: tuple[RemoraType, ...] | None = None,
    include_prelude: bool = True,
    syntax: str = "lisp",
) -> RemoraFunction:
    """Compile a Remora function from source and return a callable wrapper.

    If ``param_types`` is not provided, the function's type annotations
    in the source (from ``define/pi``) are used to determine the
    parameter types.

    Parameters
    ----------
    source : str
        Remora source code containing the function definition.
    function_name : str
        Name of the function to compile.
    param_types : tuple[RemoraType, ...], optional
        Explicit parameter types.  If omitted, inferred from the source.
    include_prelude : bool
        Whether to include the standard prelude.
    syntax : str
        Source syntax: ``"lisp"`` (default) or ``"ml"``.

    Returns
    -------
    RemoraFunction
        A callable that accepts NumPy arrays and returns NumPy arrays.
    """
    if param_types is None:
        param_types = _infer_param_types(source, function_name, syntax, include_prelude)

    artifact = CPUFunctionExecutor.compile_source(
        source, function_name, param_types,
        include_prelude=include_prelude, syntax=syntax,
    )
    executor = CPUFunctionExecutor(artifact)
    return RemoraFunction(executor, function_name, param_types, artifact.return_type)


def compile_all(
    source: str,
    *,
    include_prelude: bool = True,
    syntax: str = "lisp",
) -> dict[str, RemoraFunction]:
    """Compile all function definitions in the source.

    Returns a dict mapping function names to ``RemoraFunction`` objects.
    """
    import warnings

    names_and_types = _extract_function_signatures(source, syntax, include_prelude)
    result: dict[str, RemoraFunction] = {}
    for name, ptypes in names_and_types.items():
        try:
            result[name] = compile_function(
                source, name, param_types=ptypes,
                include_prelude=include_prelude, syntax=syntax,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to compile function {name!r}: {exc}",
                stacklevel=2,
            )
    return result


def define(
    source: str,
    *,
    syntax: str = "lisp",
    include_prelude: bool = True,
) -> RemoraFunction | dict[str, RemoraFunction]:
    """Compile Remora source and return callable wrapper(s).

    If the source contains a single function definition, returns a
    ``RemoraFunction``.  If multiple definitions, returns a dict mapping
    names to ``RemoraFunction`` objects.

    Parameters
    ----------
    source : str
        Remora source code containing one or more function definitions.
    syntax : str
        Source syntax: ``"lisp"`` (default) or ``"ml"``.
    include_prelude : bool
        Whether to include the standard prelude.

    Returns
    -------
    RemoraFunction or dict[str, RemoraFunction]

    Examples
    --------
    >>> fn = remora.define(
    ...     "(define/pi () (scale [xs (Array Float 4)] (Array Float 4))"
    ...     "  (map (* 2.0) xs))",
    ... )
    >>> fn(np.array([1.0, 2.0, 3.0, 4.0]))
    array([2., 4., 6., 8.])
    """
    fns = compile_all(source, syntax=syntax, include_prelude=include_prelude)
    if not fns:
        raise ValueError("No function definitions found in source")
    if len(fns) == 1:
        return next(iter(fns.values()))
    return fns


def _infer_param_types(
    source: str, function_name: str, syntax: str, include_prelude: bool,
) -> tuple[RemoraType, ...]:
    """Extract parameter types from the function's type annotations."""
    sigs = _extract_function_signatures(source, syntax, include_prelude)
    if function_name not in sigs:
        raise ValueError(
            f"function {function_name!r} not found in source; "
            f"available: {', '.join(sigs.keys()) or '(none)'}"
        )
    return sigs[function_name]


def _extract_function_signatures(
    source: str, syntax: str, include_prelude: bool,
) -> dict[str, tuple[RemoraType, ...]]:
    """Parse source and extract typed function signatures."""
    from remora.compiler import _parse_source
    from remora.prelude import with_prelude
    from remora.typechecker import TypeChecker, TypeEnv
    from remora.ast_nodes import FuncDef

    program_source = with_prelude(source) if include_prelude and syntax == "ml" else source
    program = _parse_source(program_source, syntax)
    checker = TypeChecker()
    env = TypeEnv()
    signatures: dict[str, tuple[RemoraType, ...]] = {}

    for definition in program.definitions:
        typed_def, env = checker.check_definition(definition, env)
        if isinstance(definition, FuncDef):
            func_type = typed_def.type
            if isinstance(func_type, FuncType):
                signatures[definition.name] = func_type.params

    return signatures
