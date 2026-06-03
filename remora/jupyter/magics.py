"""IPython magic for Remora."""

from __future__ import annotations

import numpy as np
from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from remora.compiler import compile_source_to_ptx
from remora.executor import RemoraExecutor
from remora.runtime import evaluate_source, evaluate_source_compiled


@magics_class
class RemoraMagics(Magics):
    """IPython magic extension for the Remora array language."""

    @magic_arguments()
    @argument(
        "--target",
        default="cpu",
        choices=["cpu", "interp", "gpu-nvidia"],
        help="Execution target (cpu, interp, gpu-nvidia)",
    )
    @argument(
        "--out",
        help="Python variable to bind the result to",
    )
    @cell_magic
    def remora(self, line: str, cell: str) -> object:
        """Execute Remora code in a cell."""
        args = parse_argstring(self.remora, line)
        source = cell

        if args.target == "interp":
            result = evaluate_source(source).value
        elif args.target == "cpu":
            result = evaluate_source_compiled(source).value
        elif args.target == "gpu-nvidia":
            artifact = compile_source_to_ptx(source)
            with RemoraExecutor(artifact.ptx_text, artifact.kernels) as executor:
                result = executor.execute_main()
        else:
            # This should be caught by choices in argument
            raise ValueError(f"Unknown target: {args.target}")

        if args.out:
            self.shell.user_ns[args.out] = result

        return result


def load_ipython_extension(ipython: object) -> None:
    """Register the %%remora magic extension with IPython."""
    ipython.register_magics(RemoraMagics)
