"""Erase structural typed core programs to the existing backend HIR."""

from __future__ import annotations

from remora.core_verify import verify_core_program
from remora.elaborated import CoreProgram
from remora.hir import HIRLet, HIRProgram, body_result_type, lower_expr


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

    main = lower_expr(program.body.typed)
    for definition in reversed(program.definitions):
        if definition.value is None:
            continue
        main = HIRLet(
            definition.source.name,
            definition.type,
            lower_expr(definition.value.typed),
            main,
            body_result_type(main),
        )
    return HIRProgram([], main, program.type)
