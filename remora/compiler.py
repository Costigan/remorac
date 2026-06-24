"""Public compiler facade for Remora Dense Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import numpy as np
    from remora.ad_source import GradientSourceArtifact

from remora.codegen import (
    CodegenUnavailable,
    KernelMeta,
    generate_mlir_descriptor_abi_ptx,
    generate_ptx,
    generate_rank1_f32_unary_mlir_descriptor_abi_ptx,
)
from remora.defunc import defunctionalize
from remora.elaborated import CoreProgram
from remora.elaborate import elaborate_program
from remora.erase import erase_to_hir
from remora.execution_plan import ExecutionPlan
from remora.hir import HIRFunction, HIRParam, HIRProgram, lower_expr, lower_to_hir
from remora.lowering import MLIRLowering
from remora.lowering.types import RemoraLoweringError
from remora.parser import parse_program
from remora.lisp_reader import parse_lisp as parse_lisp_program
from remora.pipeline import run_validation_pipeline, verify_module_text
from remora.pipeline import PipelineUnavailable
from remora.prelude import with_prelude
from remora.typechecker import TypeChecker, TypeEnv, TypedApp, TypedGrad, TypedProgram
from remora.index import ShapeExpr
from remora.types import DimExpr, FuncType, INT, PairType, RemoraType, RemoraTypeError
from remora.ast_nodes import AppExpr, FuncDef, IndexAppExpr, Program, VarExpr


def _parse_source(source: str, syntax: str = "ml") -> Program:
    if syntax == "lisp":
        return parse_lisp_program(source)
    return parse_program(source)
from remora.gpu_lowering import (
    GPUModuleScaffold,
    GPUScaffoldError,
    build_descriptor_abi_f32_reduction_gpu_module,
    build_gpu_scaffold_for_function,
)


@dataclass(frozen=True)
class CompilerArtifact:
    source: str
    typed: TypedProgram
    core: CoreProgram
    hir: HIRProgram
    mlir_module: object
    mlir_text: str

    @property
    def return_type(self) -> RemoraType | None:
        rt = self.typed.type
        if rt is not None and isinstance(rt, RemoraType):
            from remora.types import TypeVar as _TV
            if isinstance(rt, _TV):
                # TypeVar from HOF — use the HIR return type
                if self.hir.return_type is not None and not isinstance(
                    self.hir.return_type, _TV
                ):
                    return self.hir.return_type
        return rt


@dataclass(frozen=True)
class FunctionCompilerArtifact:
    source: str
    function_name: str
    function_type: FuncType
    hir_function: HIRFunction
    mlir_module: object
    mlir_text: str
    specialization_name: str | None = None
    index_args: tuple[DimExpr | ShapeExpr, ...] = ()

    @property
    def return_type(self) -> RemoraType:
        return self.function_type.result


@dataclass(frozen=True)
class PreparedFunctionArtifact:
    source: str
    function_name: str
    function_type: FuncType
    hir_function: HIRFunction
    specialization_name: str | None = None
    index_args: tuple[DimExpr | ShapeExpr, ...] = ()


@dataclass(frozen=True)
class PTXArtifact:
    compiler: CompilerArtifact
    ptx_text: str
    kernels: list[KernelMeta]


@dataclass(frozen=True)
class SupportedGPUFunctionArtifact:
    compiler: FunctionCompilerArtifact
    scaffold: GPUModuleScaffold
    ptx_text: str
    kernels: list[KernelMeta]
    plan: ExecutionPlan | None = None


@dataclass(frozen=True)
class GradientCompilerArtifact:
    gradient_source: GradientSourceArtifact
    compiler: FunctionCompilerArtifact


@dataclass(frozen=True)
class GradientGPUArtifact:
    gradient_source: GradientSourceArtifact
    gpu: SupportedGPUFunctionArtifact


def compile_source(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
    export_output_descriptor: bool = False,
    syntax: str = "ml",
) -> CompilerArtifact:
    _maybe_include_prelude = include_prelude and syntax == "ml"
    program_source = with_prelude(source) if _maybe_include_prelude else source
    ast = _parse_source(program_source, syntax)
    typed = TypeChecker().check_program(ast)
    rewritten = _rewrite_applied_source_gradient(
        source,
        ast,
        typed,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    if rewritten is not None:
        typed = TypeChecker().check_program(rewritten)
    core = elaborate_program(typed)
    hir = defunctionalize(erase_to_hir(core))
    hir = _monomorphize_hof_calls(hir, core)
    mlir_module = MLIRLowering().lower_program(
        hir,
        export_output_descriptor=export_output_descriptor,
    ).module
    if verify:
        run_validation_pipeline(mlir_module)
        verify_module_text(str(mlir_module))
    return CompilerArtifact(
        source=source,
        typed=typed,
        core=core,
        hir=hir,
        mlir_module=mlir_module,
        mlir_text=str(mlir_module),
    )


def compile_source_to_mlir(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
    syntax: str = "ml",
) -> str:
    return compile_source(
        source,
        verify=verify,
        include_prelude=include_prelude,
        export_output_descriptor=False,
        syntax=syntax,
    ).mlir_text


def compile_source_to_ptx(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
    syntax: str = "ml",
) -> PTXArtifact:
    artifact = compile_source(
        source,
        verify=verify,
        include_prelude=include_prelude,
        export_output_descriptor=False,
        syntax=syntax,
    )
    ptx_text, kernels = generate_ptx(artifact.mlir_module)
    return PTXArtifact(artifact, ptx_text, kernels)


def compile_function_source_to_rank1_mlir_gpu_ptx(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    include_prelude: bool = True,
    kernel_name: str | None = None,
) -> tuple[str, list[KernelMeta], FunctionCompilerArtifact]:
    """Compile one supported rank-1 unary/binary function to MLIR-derived GPU PTX."""
    artifact = compile_function_source(
        source,
        function_name,
        param_types,
        verify=False,
        include_prelude=include_prelude,
    )
    ptx, kernels, _plan = generate_rank1_f32_unary_mlir_descriptor_abi_ptx(
        artifact.hir_function,
        kernel_name=kernel_name,
    )
    return ptx, kernels, artifact


def compile_function_source_to_mlir_gpu_ptx(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    include_prelude: bool = True,
    kernel_name: str | None = None,
    syntax: str = "ml",
) -> tuple[str, list[KernelMeta], FunctionCompilerArtifact]:
    """Compile one supported function to MLIR-derived GPU PTX."""
    artifact = compile_function_source(
        source,
        function_name,
        param_types,
        verify=False,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    ptx, kernels, _plan = generate_mlir_descriptor_abi_ptx(
        artifact.hir_function,
        kernel_name=kernel_name,
    )
    return ptx, kernels, artifact


def compile_function_source_to_supported_gpu_artifacts(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    include_prelude: bool = True,
    kernel_name: str | None = None,
    syntax: str = "ml",
) -> SupportedGPUFunctionArtifact:
    """Build the current inspection and execution GPU artifacts for one function.

    The returned scaffold is the `gpu.module` artifact for the rank-1 through
    rank-3 float unary/binary map slice. The returned PTX prefers the
    MLIR-derived descriptor-ABI path; when standalone NVPTX tools are not
    available it falls back to the older direct PTX slice so callers can still
    inspect/launch the supported ABI shape in constrained environments.
    """
    artifact = compile_function_source(
        source,
        function_name,
        param_types,
        verify=False,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    kernel = kernel_name or f"remora_{function_name}"
    ptx_text, kernels, plan = generate_mlir_descriptor_abi_ptx(
        artifact.hir_function,
        kernel_name=kernel,
    )
    try:
        scaffold = build_gpu_scaffold_for_function(
            artifact.hir_function,
            kernel_name=kernel,
        )
    except GPUScaffoldError:
        scaffold = build_descriptor_abi_f32_reduction_gpu_module(
            artifact.hir_function,
            kernel_name=kernel,
        )

    return SupportedGPUFunctionArtifact(
        compiler=artifact,
        scaffold=scaffold,
        ptx_text=ptx_text,
        kernels=kernels,
        plan=plan,
    )


def compile_function_source(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    verify: bool = True,
    include_prelude: bool = True,
    export_name: str = "remora_call",
    syntax: str = "ml",
) -> FunctionCompilerArtifact:
    """Compile one top-level function with explicit static parameter types."""
    prepared = prepare_function_source(
        source,
        function_name,
        param_types,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    return compile_prepared_function(
        prepared,
        verify=verify,
        export_name=export_name,
    )


def _monomorphize_hof_calls(program: HIRProgram, core_program) -> HIRProgram:
    """Monomorphize higher-order calls by cloning and substituting.

    Finds all HIRFunction with FuncType params, clones them for each
    concrete call site, substitutes the function argument, and replaces
    HIRCall nodes to call-though-variable with direct HIRCall to the
    resolved function.
    """
    from remora.hir import (
        HIRCall as _HIRCall,
        HIRFunction as _HIRFunction,
        HIRParam as _HIRParam,
        HIRVar as _HIRVar,
        lower_expr as _lower_expr,
    )
    from remora.types import FuncType as _FuncType
    from remora.ast_nodes import FuncDef as _FuncDef

    functions = {f.name: f for f in program.functions}
    hof_functions = {
        name: f for name, f in functions.items()
        if any(isinstance(p.type, _FuncType) for p in f.params)
    }
    if not hof_functions:
        return program

    # Build a name→FuncDef lookup from the core program definitions
    func_defs: dict[str, _FuncDef] = {}
    for d in core_program.definitions:
        if hasattr(d, 'source'):
            src = d.source
            if isinstance(src, _FuncDef):
                func_defs[src.name] = src
        elif isinstance(d, _FuncDef):
            func_defs[d.name] = d

    def _ensure_function(name: str) -> str | None:
        """If *name* is a FuncDef not in HIR functions, lower it and add it."""
        if name in functions:
            return name
        fd = func_defs.get(name)
        if fd is None:
            return None
        # Create a typed lambda by specializing with concrete param types
        from remora.typechecker import TypeChecker as _TC, TypeEnv as _TE
        tc = _TC()
        tc._functions[name] = fd
        try:
            param_types = tuple(
                INT for _ in fd.params
            )
            typed_lam = tc.specialize_top_level_function(fd, param_types, _TE())
            hir_body = _lower_expr(typed_lam.body)
            hir_fn = _HIRFunction(
                name,
                [_HIRParam(pn, pt) for (pn, pt), _ in zip(typed_lam.params, param_types)],
                hir_body,
                typed_lam.type.result,
            )
            functions[name] = hir_fn
            new_functions.append(hir_fn)
            return name
        except Exception:
            return None
    new_functions: list[_HIRFunction] = list(program.functions)

    counter = 0

    def _replace_calls(expr, subs, new_calls):
        """Walk expr and replace HIRCall nodes matching known HOFs."""
        nonlocal counter
        if isinstance(expr, _HIRCall):
            hof = hof_functions.get(expr.func_name)
            if hof is not None:
                new_params = [
                    p for p in hof.params
                    if not isinstance(p.type, _FuncType)
                ]
                concrete: dict[str, _HIRVar] = {}
                for i, (arg, param) in enumerate(zip(expr.args, hof.params)):
                    if isinstance(param.type, _FuncType):
                        if isinstance(arg, _HIRVar):
                            resolved_name = _ensure_function(arg.name)
                            if resolved_name is not None:
                                concrete[param.name] = _HIRVar(
                                    resolved_name,
                                    functions[resolved_name].return_type
                                    if resolved_name in functions
                                    else arg.type,
                                )
                if concrete:
                    counter += 1
                    actual_name = f"__mono_{expr.func_name}_{counter}"
                    new_body = _substitute_hir(hof.body, concrete, functions)
                    # Resolve result type from the concrete function's return
                    resolved_return = hof.return_type
                    for resolved_name in {
                        v.name for v in concrete.values()
                        if isinstance(v, _HIRVar) and v.name in functions
                    }:
                        resolved_return = functions[resolved_name].return_type
                    new_functions.append(
                        _HIRFunction(actual_name, new_params, new_body, resolved_return)
                    )
                    new_args = []
                    for i, (a, param) in enumerate(zip(expr.args, hof.params)):
                        if not isinstance(param.type, _FuncType):
                            new_args.append(_replace_calls(a, subs, new_calls))
                    return _HIRCall(actual_name, new_args, resolved_return)
            return _HIRCall(
                expr.func_name,
                [_replace_calls(a, subs, new_calls) for a in expr.args],
                expr.result_type,
            )
        if isinstance(expr, _HIRVar) and expr.name in subs:
            return subs[expr.name]
        for attr in ("body", "then_branch", "else_branch", "value", "condition",
                     "left", "right", "array", "init"):
            child = getattr(expr, attr, None)
            if child is not None and hasattr(child, "__class__"):
                new_child = _replace_calls(child, subs, new_calls)
                if new_child is not child:
                    kwargs = {}
                    for fld in expr.__dataclass_fields__:
                        kwargs[fld] = getattr(expr, fld)
                    kwargs[attr] = new_child
                    return type(expr)(**kwargs)
        for list_attr in ("args", "arrays", "elements"):
            children = getattr(expr, list_attr, None)
            if isinstance(children, list):
                changed = False
                new_children = []
                for child in children:
                    new_child = _replace_calls(child, subs, new_calls)
                    new_children.append(new_child)
                    if new_child is not child:
                        changed = True
                if changed:
                    kwargs = {}
                    for fld in expr.__dataclass_fields__:
                        kwargs[fld] = getattr(expr, fld)
                    kwargs[list_attr] = new_children
                    return type(expr)(**kwargs)
        return expr

    new_main = _replace_calls(program.main, {}, {}) if program.main else None

    # Remove HOF functions that have been monomorphized — they have
    # FuncType params and the main expression no longer calls them.
    final_functions = [
        f for f in new_functions
        if not any(isinstance(p.type, _FuncType) for p in f.params)
    ]

    # Resolve the program's return type from the monomorphized main
    # expression — it now has concrete types instead of TypeVars.
    resolved_return = new_main.result_type if isinstance(new_main, _HIRCall) else program.return_type

    return HIRProgram(final_functions, new_main, resolved_return)


def _try_monomorphize(
    hir_function: HIRFunction,
    func_type_params: list[tuple[int, "HIRParam"]],
    program: Program,
    function_name: str,
    checker: TypeChecker,
    env: TypeEnv,
) -> HIRFunction:
    """Inline concrete lambda arguments for FuncType parameters.

    When the program body calls the function with concrete lambdas,
    substitute them into the HIR body and remove the FuncType params.
    This enables GPU compilation for higher-order functions.
    """
    from remora.hir import (
        HIRApply as _HIRApply, HIRLambda as _HIRLambda, HIRLet as _HIRLet,
        HIRMap as _HIRMap, HIRPrimCallable as _HIRPrimCallable, HIRVar as _HIRVar,
    )

    body = program.body
    if body is None:
        return hir_function

    all_args: list = []
    cur = body
    while isinstance(cur, AppExpr):
        all_args = list(cur.args) + all_args
        cur = cur.func
    if not (isinstance(cur, VarExpr) and cur.name == function_name):
        return hir_function

    if len(all_args) != len(hir_function.params):
        return hir_function

    substitutions: dict[str, "HIRExpr"] = {}
    from remora.hir import (
        HIRLambda as _HLam, HIRPrimCallable as _HPC,
    )
    from remora.typechecker import TypedLambda as _TLam
    from remora.hir import lower_callable as _lower_callable

    for idx, param in func_type_params:
        if idx >= len(all_args):
            return hir_function
        ast_arg = all_args[idx]
        if not isinstance(param.type, FuncType):
            return hir_function
        try:
            typed_arg = checker.check_callable(ast_arg, param.type, env)
            hir_arg = _lower_callable(typed_arg)
            if not isinstance(hir_arg, (_HLam, _HPC)):
                return hir_function
            substitutions[param.name] = hir_arg
        except Exception:
            return hir_function

    if not substitutions:
        return hir_function

    new_body = _substitute_hir(hir_function.body, substitutions)
    new_params = [p for i, p in enumerate(hir_function.params)
                  if i not in {idx for idx, _ in func_type_params}]
    return HIRFunction(hir_function.name, new_params, new_body, hir_function.return_type)


def _substitute_hir(expr, subs: dict, functions: dict | None = None):
    """Replace HIRVar references in subs with their concrete expressions."""
    from remora.hir import (
        HIRApply as _HIRApply, HIRLambda as _HIRLambda, HIRLet as _HIRLet,
        HIRMap as _HIRMap, HIRPrimCallable as _HIRPrimCallable, HIRVar as _HIRVar,
        HIRCall as _HIRCall, HIRIf as _HIRIf, HIRFold as _HIRFold,
        HIRReduce as _HIRReduce,
    )
    from remora.types import TypeVar as _TV

    if isinstance(expr, _HIRVar) and expr.name in subs:
        resolved = subs[expr.name]
        if isinstance(resolved, _HIRVar):
            return resolved
        return resolved
    if isinstance(expr, _HIRCall):
        new_func_name = expr.func_name
        new_result_type = expr.result_type
        if expr.func_name in subs:
            resolved = subs[expr.func_name]
            if isinstance(resolved, _HIRVar):
                new_func_name = resolved.name
                # Resolve TypeVar result type from the concrete function
                if functions and new_func_name in functions:
                    fn = functions[new_func_name]
                    if not isinstance(fn.return_type, _TV):
                        new_result_type = fn.return_type
        new_args = [_substitute_hir(a, subs, functions) for a in expr.args]
        return _HIRCall(new_func_name, new_args, new_result_type)
    if isinstance(expr, _HIRIf):
        cond = _substitute_hir(expr.condition, subs, functions)
        then_b = _substitute_hir(expr.then_branch, subs, functions)
        else_b = _substitute_hir(expr.else_branch, subs, functions)
        return _HIRIf(cond, then_b, else_b, expr.result_type)
    if isinstance(expr, (_HIRFold, _HIRReduce)):
        array = _substitute_hir(expr.array, subs, functions)
        init = _substitute_hir(expr.init, subs, functions)
        func = expr.func
        if isinstance(func, _HIRVar) and func.name in subs:
            resolved = subs[func.name]
            if isinstance(resolved, (_HIRLambda, _HIRPrimCallable)):
                func = resolved
        return type(expr)(func, init, array, expr.result_type)
    if isinstance(expr, _HIRMap):
        func = expr.func
        if isinstance(func, _HIRVar) and func.name in subs:
            resolved = subs[func.name]
            if isinstance(resolved, (_HIRLambda, _HIRPrimCallable)):
                func = resolved
        arrays = [_substitute_hir(a, subs, functions) for a in expr.arrays]
        return _HIRMap(expr.frame_shape, expr.cell_shape, func, arrays, expr.result_type)
    if isinstance(expr, _HIRApply):
        func = expr.func
        if isinstance(func, _HIRVar) and func.name in subs:
            resolved = subs[func.name]
            if isinstance(resolved, (_HIRLambda, _HIRPrimCallable)):
                func = resolved
        arrays = [_substitute_hir(a, subs, functions) for a in expr.arrays]
        return _HIRApply(expr.frame_shape, expr.cell_shape, func, arrays, expr.result_type)
    if isinstance(expr, _HIRLet):
        value = _substitute_hir(expr.value, subs, functions)
        body = _substitute_hir(expr.body, subs, functions)
        return _HIRLet(expr.name, expr.value_type, value, body, expr.result_type)
    return expr


def prepare_function_source(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    include_prelude: bool = True,
    syntax: str = "ml",
) -> PreparedFunctionArtifact:
    """Parse, specialize, and lower one function to HIR without emitting MLIR."""
    _maybe_include_prelude = include_prelude and syntax == "ml"
    program_source = with_prelude(source) if _maybe_include_prelude else source
    program = _parse_source(program_source, syntax)
    checker = TypeChecker()
    env = TypeEnv()
    function_def: FuncDef | None = None
    for definition in program.definitions:
        typed_definition, env = checker.check_definition(definition, env)
        if isinstance(definition, FuncDef) and definition.name == function_name:
            function_def = definition

    if function_def is None:
        raise ValueError(f"function {function_name!r} is not defined")

    typed_function = checker.specialize_top_level_function(
        function_def,
        param_types,
        env,
    )
    function_type = typed_function.type
    # Ensure the specialized body has no free index variables
    from remora.dependent_types import free_type_index_vars
    free_vars = free_type_index_vars(function_type)
    if free_vars:
        names = ", ".join(sorted(free_vars))
        raise RemoraTypeError(
            f"compiled function {function_name!r} has unspecialized "
            f"index variables: {names}"
        )
    internal_name = typed_function.specialization_name or function_name
    hir_function = HIRFunction(
        internal_name,
        [HIRParam(name, param_type) for name, param_type in typed_function.params],
        lower_expr(typed_function.body),
        function_type.result,
    )

    func_params_with_func_type = [
        (i, p) for i, p in enumerate(hir_function.params)
        if isinstance(p.type, FuncType)
    ]
    if func_params_with_func_type and program.body is not None:
        hir_function = _try_monomorphize(
            hir_function, func_params_with_func_type,
            program, function_name, checker, env,
        )

    return PreparedFunctionArtifact(
        source=source,
        function_name=function_name,
        function_type=function_type,
        hir_function=hir_function,
        specialization_name=typed_function.specialization_name,
        index_args=typed_function.index_args,
    )


def compile_prepared_function(
    prepared: PreparedFunctionArtifact,
    *,
    verify: bool = True,
    export_name: str = "remora_call",
) -> FunctionCompilerArtifact:
    """Emit descriptor MLIR for a function prepared by `prepare_function_source`."""
    try:
        lowered = MLIRLowering().lower_function_descriptor_export(
            prepared.hir_function,
            export_name=export_name,
        )
        if verify:
            run_validation_pipeline(lowered.module)
            verify_module_text(str(lowered.module))
        mlir_module = lowered.module
        mlir_text = str(lowered.module)
    except RemoraLoweringError:
        mlir_module = None
        mlir_text = ""
    return FunctionCompilerArtifact(
        source=prepared.source,
        function_name=prepared.function_name,
        function_type=prepared.function_type,
        hir_function=prepared.hir_function,
        mlir_module=mlir_module,
        mlir_text=mlir_text,
        specialization_name=prepared.specialization_name,
        index_args=prepared.index_args,
    )


def compile_gradient_function_source(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    example_input: np.ndarray | None = None,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    verify: bool = True,
) -> GradientCompilerArtifact:
    """Generate and compile a specialized unary gradient for the CPU path."""
    from remora.ad_source import generate_gradient_function_source

    gradient = generate_gradient_function_source(
        source,
        function_name,
        param_types,
        example_input,
        gradient_name=gradient_name,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    compiler = compile_function_source(
        gradient.source,
        gradient.function_name,
        gradient.param_types,
        verify=verify,
        include_prelude=False,
        syntax="lisp",
    )
    return GradientCompilerArtifact(gradient, compiler)


@dataclass(frozen=True)
class MultiGradientCompilerArtifact:
    """Per-input compiled gradient artifacts for an n-ary function."""
    source: str
    function_name: str
    gradients: list[GradientCompilerArtifact]


def compile_gradient_functions_source(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    example_input: np.ndarray | None = None,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    verify: bool = True,
    differentiate_inputs: Iterable[int] | None = None,
) -> MultiGradientCompilerArtifact:
    """Generate and compile one gradient function per active input.

    For a function f: (A, B) → Float, returns two compiled gradients by
    default: df/dA and df/dB. Pass *differentiate_inputs* to compile only a
    selected ordered subset.
    """
    from remora.ad_source import generate_gradient_function_source

    if len(param_types) < 2:
        raise ValueError("multi-gradient compilation requires at least 2 param types")

    base_name = gradient_name or f"grad_{function_name.replace('-', '_')}"
    gradients: list[GradientCompilerArtifact] = []

    input_indices = (
        tuple(range(len(param_types)))
        if differentiate_inputs is None
        else tuple(differentiate_inputs)
    )
    if any(i < 0 or i >= len(param_types) for i in input_indices):
        raise ValueError("differentiate_inputs contains an out-of-range input index")

    for i in input_indices:
        grad_name = f"{base_name}_{i}"
        gradient = generate_gradient_function_source(
            source,
            function_name,
            param_types,
            example_input,
            gradient_name=grad_name,
            differentiate_input=i,
            include_prelude=include_prelude,
            syntax=syntax,
        )
        compiler = compile_function_source(
            gradient.source,
            gradient.function_name,
            gradient.param_types,
            verify=verify,
            include_prelude=False,
            syntax="lisp",
        )
        gradients.append(GradientCompilerArtifact(gradient, compiler))

    return MultiGradientCompilerArtifact(
        source=source,
        function_name=function_name,
        gradients=gradients,
    )


def compile_value_and_grad_function(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    example_input: np.ndarray | None = None,
    *,
    gradient_name: str | None = None,
    differentiate_inputs: Iterable[int] | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    verify: bool = True,
) -> GradientCompilerArtifact:
    """Generate and compile a single function returning all requested gradients.

    Generates one function whose return type is a nested ``(Pair ...)``
    chain containing every requested gradient.  The forward computation is
    traced once and shared across all backward paths.

    Returns a single ``GradientCompilerArtifact`` (the compiler holds the
    HIR function with the Pair return type).
    """
    from remora.ad_source import generate_value_and_grad_function_source

    if len(param_types) < 2:
        raise ValueError("value-and-grad requires at least 2 parameter types")

    input_indices = (
        tuple(range(len(param_types)))
        if differentiate_inputs is None
        else tuple(differentiate_inputs)
    )
    if any(i < 0 or i >= len(param_types) for i in input_indices):
        raise ValueError("differentiate_inputs contains an out-of-range input index")

    gradient = generate_value_and_grad_function_source(
        source,
        function_name,
        param_types,
        example_input,
        gradient_name=gradient_name,
        differentiate_inputs=input_indices,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    compiler = compile_function_source(
        gradient.source,
        gradient.function_name,
        gradient.param_types,
        verify=verify,
        include_prelude=False,
        syntax="lisp",
    )
    return GradientCompilerArtifact(gradient, compiler)


def compile_gradient_function_source_to_supported_gpu_artifacts(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    example_input: np.ndarray | None = None,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    kernel_name: str | None = None,
) -> GradientGPUArtifact:
    """Generate and compile a specialized unary gradient for the GPU path."""
    from remora.ad_source import generate_gradient_function_source

    gradient = generate_gradient_function_source(
        source,
        function_name,
        param_types,
        example_input,
        gradient_name=gradient_name,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    kernel = kernel_name or f"remora_{gradient.function_name.replace('-', '_')}"
    gpu = compile_function_source_to_supported_gpu_artifacts(
        gradient.source,
        gradient.function_name,
        gradient.param_types,
        include_prelude=False,
        kernel_name=kernel,
        syntax="lisp",
    )
    return GradientGPUArtifact(gradient, gpu)


def compile_source_gradient_function(
    source: str,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    verify: bool = True,
) -> GradientCompilerArtifact:
    """Compile the concrete source-level `(grad f)` request in a program body."""
    function_name, param_types = _source_gradient_request(
        source, include_prelude=include_prelude, syntax=syntax
    )
    return compile_gradient_function_source(
        source,
        function_name,
        param_types,
        gradient_name=gradient_name,
        include_prelude=include_prelude,
        syntax=syntax,
        verify=verify,
    )


def compile_source_gradient_function_to_supported_gpu_artifacts(
    source: str,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    kernel_name: str | None = None,
) -> GradientGPUArtifact:
    """Compile a concrete source-level `(grad f)` request for the GPU path."""
    function_name, param_types = _source_gradient_request(
        source, include_prelude=include_prelude, syntax=syntax
    )
    return compile_gradient_function_source_to_supported_gpu_artifacts(
        source,
        function_name,
        param_types,
        gradient_name=gradient_name,
        include_prelude=include_prelude,
        syntax=syntax,
        kernel_name=kernel_name,
    )


def _source_gradient_request(
    source: str,
    *,
    include_prelude: bool,
    syntax: str,
) -> tuple[str, tuple[RemoraType, ...]]:
    program_source = with_prelude(source) if include_prelude and syntax == "ml" else source
    typed = TypeChecker().check_program(_parse_source(program_source, syntax))
    body = typed.body
    typed_grad = body.func if isinstance(body, TypedApp) else body
    if not isinstance(typed_grad, TypedGrad):
        raise ValueError("program body must be `(grad f)` or an application of it")
    if not isinstance(typed_grad.type, FuncType):
        raise ValueError("source-level gradient must be specialized to concrete parameter types")

    target = typed_grad.expr.func
    if isinstance(target, VarExpr):
        function_name = target.name
    elif isinstance(target, IndexAppExpr) and isinstance(target.func, VarExpr):
        function_name = target.func.name
    else:
        raise ValueError("source-level gradient must target a named function")
    return function_name, typed_grad.type.params


def _collect_typed_grads(expr, _seen=None) -> list:
    """Recursively collect all TypedGrad nodes from a typed AST."""
    if expr is None:
        return []
    if _seen is None:
        _seen = set()
    eid = id(expr)
    if eid in _seen:
        return []
    _seen.add(eid)
    if isinstance(expr, TypedGrad):
        return [expr]
    results: list = []
    for attr in ("func", "body", "condition", "then_branch", "else_branch",
                 "value", "array", "init", "left", "right", "source",
                 "predicate", "image", "columns", "pair", "box_value",
                 "counts", "target", "index", "update"):
        child = getattr(expr, attr, None)
        if child is not None and hasattr(child, '__class__') and child.__class__.__module__.startswith('remora'):
            results.extend(_collect_typed_grads(child, _seen))
    for attr in ("args", "arrays", "elements", "definitions"):
        children = getattr(expr, attr, None)
        if isinstance(children, (list, tuple)):
            for child in children:
                if hasattr(child, '__class__') and child.__class__.__module__.startswith('remora'):
                    results.extend(_collect_typed_grads(child, _seen))
                    child_val = getattr(child, 'value', None)
                    if child_val is not None and hasattr(child_val, '__class__') and child_val.__class__.__module__.startswith('remora'):
                        results.extend(_collect_typed_grads(child_val, _seen))
    return results


def _replace_grad_in_ast(expr, mapping: dict[str, str]):
    """Replace GradExpr nodes in an untyped AST with VarExpr references."""
    from remora.ast_nodes import (
        AppExpr as _App, ArrayLit as _Arr, FoldExpr as _Fold, FoldRightExpr as _FoldR,
        GradExpr as _Grad, IfExpr as _If, IndexAppExpr as _IApp, IotaExpr as _Iota,
        LambdaExpr as _Lam, LetExpr as _Let, MapExpr as _Map, ReduceExpr as _Red,
        ScanExpr as _Scan, SelectExpr as _Sel, VarExpr as _Var,
    )
    if isinstance(expr, _Grad):
        inner = expr.func
        if isinstance(inner, _Var) and inner.name in mapping:
            return _Var(mapping[inner.name], expr.loc)
        if isinstance(inner, _App) and isinstance(inner.func, _Var) and inner.func.name in mapping:
            return _Var(mapping[inner.func.name], expr.loc)
        if isinstance(inner, _IApp) and isinstance(inner.func, _Var) and inner.func.name in mapping:
            return _Var(mapping[inner.func.name], expr.loc)
        return expr
    if isinstance(expr, _App):
        return _App(
            _replace_grad_in_ast(expr.func, mapping),
            [_replace_grad_in_ast(a, mapping) for a in expr.args],
            expr.loc,
        )
    if isinstance(expr, _Lam):
        return _Lam(expr.params, _replace_grad_in_ast(expr.body, mapping), expr.loc)
    if isinstance(expr, _Let):
        return _Let(expr.name, _replace_grad_in_ast(expr.value, mapping),
                    _replace_grad_in_ast(expr.body, mapping), expr.loc)
    if isinstance(expr, _If):
        return _If(_replace_grad_in_ast(expr.condition, mapping),
                   _replace_grad_in_ast(expr.then_branch, mapping),
                   _replace_grad_in_ast(expr.else_branch, mapping), expr.loc)
    if isinstance(expr, _Sel):
        return _Sel(_replace_grad_in_ast(expr.condition, mapping),
                    _replace_grad_in_ast(expr.then_branch, mapping),
                    _replace_grad_in_ast(expr.else_branch, mapping), expr.loc)
    if isinstance(expr, _Fold):
        return _Fold(_replace_grad_in_ast(expr.func, mapping),
                     _replace_grad_in_ast(expr.init, mapping),
                     _replace_grad_in_ast(expr.array, mapping), expr.loc)
    if isinstance(expr, _Map):
        return _Map(_replace_grad_in_ast(expr.func, mapping),
                    [_replace_grad_in_ast(a, mapping) for a in expr.arrays], expr.loc)
    return expr


def _rewrite_applied_source_gradient(
    source: str,
    program: Program,
    typed: TypedProgram,
    *,
    include_prelude: bool,
    syntax: str,
) -> Program | None:
    """Replace all `grad` references with generated gradient function calls.

    Walks the entire typed AST to find TypedGrad nodes (at any depth),
    generates gradient function source for each, adds the definitions,
    and replaces GradExpr nodes in the untyped AST with VarExpr references.
    """
    grads = _collect_typed_grads(typed.body)
    if not grads:
        return None

    if isinstance(typed.body, TypedGrad):
        raise ValueError(
            "bare `(grad f)` is a function value; use "
            "compile_source_gradient_function or apply it to an argument"
        )

    existing_names = {
        d.name for d in program.definitions if isinstance(d, FuncDef)
    }
    from remora.ad_source import generate_gradient_function_source

    new_definitions: list = []
    grad_name_map: dict[str, str] = {}

    for tg in grads:
        func_name = _typed_gradient_target_name(tg)
        if func_name in grad_name_map:
            continue
        if not isinstance(tg.type, FuncType):
            continue
        gen_name = _unique_gradient_name(func_name, existing_names | set(grad_name_map.values()))
        grad_name_map[func_name] = gen_name
        try:
            gradient = generate_gradient_function_source(
                source, func_name, tg.type.params,
                gradient_name=gen_name,
                include_prelude=include_prelude,
                syntax=syntax,
            )
            gen_prog = parse_lisp_program(gradient.source) if syntax == "lisp" else parse_program(gradient.source)
            for d in gen_prog.definitions:
                if isinstance(d, FuncDef):
                    new_definitions.append(d)
                    existing_names.add(d.name)
        except Exception:
            continue

    if not new_definitions:
        return None

    new_body = _replace_grad_in_ast(program.body, grad_name_map) if program.body else None
    return Program(
        [*program.definitions, *new_definitions],
        new_body,
        program.loc,
    )


def _typed_gradient_target_name(typed_grad: TypedGrad) -> str:
    target = typed_grad.expr.func
    if isinstance(target, VarExpr):
        return target.name
    if isinstance(target, IndexAppExpr) and isinstance(target.func, VarExpr):
        return target.func.name
    raise ValueError("source-level gradient must target a named function")


def _unique_gradient_name(function_name: str, existing_names: set[str]) -> str:
    base = f"__remora_grad_{function_name.replace('-', '_')}"
    candidate = base
    suffix = 2
    while candidate in existing_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate
