"""Erase structural typed core programs to the existing backend HIR."""

from __future__ import annotations

from remora.core_verify import verify_core_program
from remora.elaborated import CoreProgram
from remora.hir import (
    HIRFunction,
    HIRLet,
    HIRLoweringError,
    HIRParam,
    HIRProgram,
    _current_func_names,
    _gather_func_def_lambdas,
    body_result_type,
    lower_expr,
)


def erase_to_hir(program: CoreProgram) -> HIRProgram:
    """Erase dependent evidence and lower concrete core expressions to HIR."""
    verify_core_program(program)
    if program.body is None:
        if program.definitions:
            from remora.hir import HIRLoweringError

            raise HIRLoweringError(
                "definition-only programs cannot be lowered to HIR without a body"
            )
        return HIRProgram([], None, None)

    if program.body.typed is None:
        raise HIRLoweringError(
            "cannot erase core expression with no original typed node; "
            "re-run elaboration or use type-only lowering"
        )

    func_lambdas = _gather_func_def_lambdas(program.body.typed)
    import remora.hir as hir_module
    hir_module._current_func_names = set(func_lambdas.keys())
    try:
        main = lower_expr(program.body.typed)
        for definition in reversed(program.definitions):
            if definition.value is None:
                continue
            if definition.value.typed is None:
                raise HIRLoweringError(
                    "cannot erase untransformed core definition "
                    f"{definition.source.name}"
                )
            main = HIRLet(
                definition.source.name,
                definition.type,
                lower_expr(definition.value.typed),
                main,
                body_result_type(main),
            )

        hir_functions: list[HIRFunction] = []
        for name, typed_lam in func_lambdas.items():
            hir_fn_body = lower_expr(typed_lam.body)
            hir_fn = HIRFunction(
                name,
                [HIRParam(p_name, p_type) for p_name, p_type in typed_lam.params],
                hir_fn_body,
                typed_lam.type.result,
            )
            hir_functions.append(hir_fn)

        return HIRProgram(hir_functions, main, program.type)
    finally:
        hir_module._current_func_names = None
