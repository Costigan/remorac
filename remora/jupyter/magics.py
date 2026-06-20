"""IPython magic for Remora."""

from __future__ import annotations

import numpy as np
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from remora.api import compile_all, compile_function
from remora.runtime import evaluate_source, evaluate_source_compiled


@magics_class
class RemoraMagics(Magics):
    """IPython magic extension for the Remora array language.

    Usage::

        %%remora
        (define/pi () (scale [xs (Array Float 4)] (Array Float 4))
          (map (* 2.0) xs))

    This compiles ``scale`` and registers it in the notebook namespace.
    Then from a Python cell::

        scale(np.array([1.0, 2.0, 3.0, 4.0]))  # → array([2., 4., 6., 8.])

    If the cell contains a body expression (not just definitions), it is
    evaluated and the result is returned::

        %%remora
        (fold + 0.0 [1.0 2.0 3.0 4.0])
        # → 10.0
    """

    _repl_session = None

    @magic_arguments()
    @argument(
        "--target",
        default="cpu",
        choices=["cpu", "interp", "gpu"],
        help="Execution target (cpu, interp, or gpu)",
    )
    @argument(
        "--syntax",
        default="ml",
        choices=["lisp", "ml"],
        help="Source syntax (lisp or ml)",
    )
    @argument(
        "--out",
        help="Python variable to bind the result to",
    )
    @argument(
        "--types",
        action="store_true",
        help="Print inferred types for all definitions",
    )
    @cell_magic
    def remora(self, line: str, cell: str) -> object:
        """Compile Remora definitions and/or evaluate an expression."""
        args = parse_argstring(self.remora, line)
        source = cell.strip()
        syntax = args.syntax

        compiled_names: list[str] = []

        if args.target == "cpu":
            fns = compile_all(source, syntax=syntax)
            for name, fn in fns.items():
                self.shell.user_ns[name] = fn
                compiled_names.append(name)
                if args.types:
                    print(f"  {name} : {fn.param_types} → {fn.return_type}")

        from remora.compiler import _parse_source
        program = _parse_source(source, syntax)
        has_body = program.body is not None

        if compiled_names and not has_body:
            if not args.types:
                print(f"Compiled: {', '.join(compiled_names)}")
            return None

        if args.target == "interp":
            result = evaluate_source(source, syntax=syntax).value
        elif args.target == "cpu":
            result = evaluate_source_compiled(source, syntax=syntax).value
        elif args.target == "gpu":
            from remora.executor import execute_program_on_gpu
            result = execute_program_on_gpu(source, syntax=syntax)
        else:
            return None

        if args.out:
            self.shell.user_ns[args.out] = result

        return result

    @magic_arguments()
    @argument(
        "--target",
        default="cpu",
        choices=["cpu", "interp"],
        help="Execution target (cpu or interp)",
    )
    @argument(
        "--syntax",
        default="ml",
        choices=["lisp", "ml"],
        help="Source syntax (lisp or ml)",
    )
    @argument(
        "--reset",
        action="store_true",
        help="Reset the REPL session state",
    )
    @argument("expr", nargs="*", help="Remora expression to evaluate")
    @line_magic
    def remora_eval(self, line: str) -> object:
        """Evaluate a Remora expression using a persistent REPL session.

        Maintains state across invocations so definitions accumulate::

            %remora_eval def double x = x + x
            %remora_eval double 21
        """
        from remora.repl import ReplSession

        args = parse_argstring(self.remora_eval, line)

        if self._repl_session is None or args.reset:
            self._repl_session = ReplSession(target=args.target)
            self._repl_session.state.syntax = args.syntax
            if args.reset and not args.expr:
                print("REPL session reset.")
                return None

        expr_text = " ".join(args.expr).strip()
        if not expr_text:
            return None

        output = self._repl_session.eval_input(expr_text)
        if output is not None:
            print(output)
        return None


def load_ipython_extension(ipython: object) -> None:
    """Register the %%remora magic extension with IPython."""
    ipython.register_magics(RemoraMagics)
