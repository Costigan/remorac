"""CPU-first Remora REPL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lark import LarkError

from remora.ast_nodes import FuncDef, ValDef
from remora.compiler import compile_source_to_mlir
from remora.display import format_result
from remora.errors import RemoraError
from remora.parser import parse_program, parse_repl_input
from remora.lisp_reader import parse_lisp as parse_lisp_program
from remora.prelude import prelude_definition_sources
from remora.runtime import EvaluationResult, evaluate_source, evaluate_source_compiled
from remora.typechecker import TypeChecker

REPL_TARGETS = ("cpu", "interp", "gpu-nvidia")
REPL_SYNTAXES = ("ml", "lisp")


@dataclass
class ReplState:
    target: str = "cpu"
    debug: bool = False
    syntax: str = "ml"
    definition_sources: list[str] = field(default_factory=prelude_definition_sources)


def make_initial_state(target: str = "cpu") -> ReplState:
    if target not in REPL_TARGETS:
        raise ReplError("available REPL targets: cpu, interp, gpu-nvidia")
    return ReplState(target=target)


class ReplError(RemoraError):
    """Raised for REPL command errors."""


class ReplSession:
    def __init__(self, target: str = "cpu", *, history: bool = False):
        self.state = make_initial_state(target)
        if history:
            self._setup_readline()

    def eval_input(self, text: str) -> str | None:
        text = text.strip()
        if not text:
            return None
        if text.startswith(":"):
            return self._handle_command(text)

        try:
            item = self._parse_repl_input(text)
        except (LarkError, RemoraError) as exc:
            return _format_parse_error(text, exc)

        try:
            if isinstance(item, (FuncDef, ValDef)):
                return self._process_definition(text, item)
            return self._process_expression(text)
        except RemoraError as exc:
            return f"Error: {exc}"

    def _parse_repl_input(self, text: str):
        """Parse REPL input with the current syntax."""
        if self.state.syntax == "lisp":
            return self._parse_repl_input_lisp(text)
        return parse_repl_input(text)

    def _parse_repl_input_lisp(self, text: str):
        try:
            program = parse_lisp_program(text, "<repl>")
            if program.definitions:
                return program.definitions[0]
            if program.body is not None:
                return program.body
            raise LarkError("empty input")
        except Exception:
            raise LarkError("parse error")

    def _process_definition(self, source: str, definition: FuncDef | ValDef) -> str:
        candidate_definitions = [*self.state.definition_sources, source]
        program_source = _program_source(candidate_definitions, "0")
        typed = TypeChecker().check_program(
            self._parse_program(program_source, "<repl>")
        )
        self.state.definition_sources.append(source)
        if isinstance(definition, FuncDef):
            return f"Defined: {definition.name} : <function>"
        return f"Defined: {definition.name} : {typed.definitions[-1].type}"

    def _process_expression(self, source: str) -> str:
        program_source = _program_source(self.state.definition_sources, source)
        result = self._evaluate_program_source(program_source)
        return format_result(result.value, result.type)

    def _parse_program(self, source: str, filename: str = "<repl>"):
        if self.state.syntax == "lisp":
            return parse_lisp_program(source, filename)
        return parse_program(source, filename)

    def _evaluate_program_source(self, program_source: str) -> EvaluationResult:
        if self.state.target == "cpu":
            return evaluate_source_compiled(
                program_source, include_prelude=False, syntax=self.state.syntax
            )
        if self.state.target == "interp":
            return evaluate_source(
                program_source, include_prelude=False, syntax=self.state.syntax
            )
        if self.state.target == "gpu-nvidia":
            from remora.compiler import compile_source_to_ptx
            from remora.codegen import CodegenUnavailable
            try:
                artifact = compile_source_to_ptx(
                    program_source, include_prelude=False, syntax=self.state.syntax
                )
                kernels = artifact.kernels
                if not kernels:
                    raise CodegenUnavailable(
                        "No GPU kernels generated. Try a program with tensor operations like map or fold."
                    )
                info = f"GPU: {len(kernels)} kernel(s) generated"
                return EvaluationResult(info, None)
            except CodegenUnavailable as exc:
                raise ReplError(str(exc)) from exc
        raise ReplError(f"unknown REPL target: {self.state.target}")

    def _handle_command(self, command: str) -> str:
        parts = command.split(None, 1)
        name = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        try:
            if name in (":quit", ":q"):
                raise SystemExit(0)
            if name == ":help":
                return HELP_TEXT.strip()
            if name == ":debug":
                self.state.debug = not self.state.debug
                return f"Debug mode: {'on' if self.state.debug else 'off'}"
            if name == ":target":
                return self._target_command(arg)
            if name == ":syntax":
                return self._syntax_command(arg)
            if name == ":type":
                return self._type_command(arg)
            if name == ":mlir":
                return self._mlir_command(arg)
            if name == ":prelude":
                return self._prelude_command()
            if name == ":defs":
                return self._defs_command()
            if name == ":load":
                return self._load_file(arg)
            if name == ":reset":
                self.state.definition_sources = prelude_definition_sources()
                return "State reset."
            return f"Unknown command: {name}. Type :help for help."
        except RemoraError as exc:
            return f"Error: {exc}"
        except LarkError as exc:
            return _format_parse_error(arg, exc)

    def _target_command(self, arg: str) -> str:
        if not arg:
            return f"Current target: {self.state.target}"
        if arg not in REPL_TARGETS:
            return "Error: available REPL targets: cpu, interp"
        self.state.target = arg
        return f"Target: {arg}"

    def _syntax_command(self, arg: str) -> str:
        if not arg:
            return f"Current syntax: {self.state.syntax}"
        if arg not in REPL_SYNTAXES:
            return "Error: available syntaxes: ml, lisp"
        self.state.syntax = arg
        return f"Syntax: {arg}"

    def _type_command(self, arg: str) -> str:
        if not arg:
            return "Usage: :type <expr>"
        program = self._parse_program(
            _program_source(self.state.definition_sources, arg), "<repl>"
        )
        typed = TypeChecker().check_program(program)
        return f"{arg} : {typed.type}"

    def _mlir_command(self, arg: str) -> str:
        if not arg:
            return "Usage: :mlir <expr>"
        return compile_source_to_mlir(
            _program_source(self.state.definition_sources, arg),
            include_prelude=False,
            syntax=self.state.syntax,
        )

    def _prelude_command(self) -> str:
        return "\n".join(prelude_definition_sources())

    def _defs_command(self) -> str:
        prelude_count = len(prelude_definition_sources())
        definitions = self.state.definition_sources[prelude_count:]
        return "\n".join(definitions) if definitions else "No user definitions."

    def _load_file(self, arg: str) -> str:
        if not arg:
            return "Usage: :load <file>"

        path = Path(arg)
        source = path.read_text(encoding="utf-8")
        # Auto-detect syntax from file extension
        detected_syntax = self.state.syntax
        if path.suffix == ".lisp":
            detected_syntax = "lisp"
        elif path.suffix == ".remora":
            detected_syntax = "ml"

        program = self._parse_program(source, str(path)) if detected_syntax == "ml" else parse_lisp_program(source, str(path))
        messages: list[str] = []
        for definition_source in _top_level_definition_lines(source):
            item = self._parse_repl_input(definition_source)
            if not isinstance(item, (FuncDef, ValDef)):
                continue
            message = self._process_definition(definition_source, item)
            messages.append(message)
        if program.body is not None:
            result = self._evaluate_program_source(
                _program_source(self.state.definition_sources, _body_source(source))
            )
            messages.append(format_result(result.value, result.type))
        return "\n".join(messages) if messages else "Loaded."

    def _collect_full_input(self, first_line: str) -> str:
        buffer = first_line
        while not self._is_complete(buffer):
            buffer = buffer + "\n" + input("...... ")
        return buffer

    def _is_complete(self, text: str) -> bool:
        return _balanced(text, "(", ")") and _balanced(text, "[", "]")

    def run(self) -> None:
        print(f"Remora REPL [target: {self.state.target}, syntax: {self.state.syntax}]")
        print("Type :help for commands, :quit to exit.")
        while True:
            try:
                line = input("remora> ")
                text = self._collect_full_input(line)
                result = self.eval_input(text)
                if result is not None:
                    print(result)
            except EOFError:
                print()
                return
            except KeyboardInterrupt:
                print()

    def _setup_readline(self) -> None:
        try:
            import atexit
            import os
            import readline
        except ImportError:
            return
        history_path = os.path.expanduser("~/.remora_history")
        try:
            readline.read_history_file(history_path)
        except FileNotFoundError:
            pass
        def write_history() -> None:
            try:
                readline.write_history_file(history_path)
            except OSError:
                pass

        atexit.register(write_history)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Remora REPL")
    parser.add_argument("--target", default="cpu", choices=REPL_TARGETS)
    args = parser.parse_args(argv)
    try:
        ReplSession(target=args.target, history=True).run()
        return 0
    except RemoraError as exc:
        print(f"remora: {exc}")
        return 1


def _program_source(definitions: list[str], body: str) -> str:
    if definitions:
        return "\n".join([*definitions, body])
    return body


def _top_level_definition_lines(source: str) -> list[str]:
    return [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("def ")
    ]


def _body_source(source: str) -> str:
    lines = [
        line
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("--") and not line.strip().startswith("def ")
    ]
    return "\n".join(lines)


def _balanced(text: str, open_char: str, close_char: str) -> bool:
    return text.count(open_char) <= text.count(close_char)


def _format_parse_error(text: str, exc: Exception) -> str:
    message = f"Parse error: {exc}"
    hint = _parse_hint(text)
    if hint:
        return f"{message}\nHint: {hint}"
    return message


def _parse_hint(text: str) -> str | None:
    stripped = " ".join(text.strip().split())
    if stripped.startswith(("map \\", "fold \\", "map lambda", "fold lambda")):
        return "parenthesize lambda callables, e.g. map (\\x -> x) ([1, 2, 3])"
    if stripped.startswith("map (") and ") [" in stripped:
        return "parenthesize array literal arguments after callables, e.g. map (\\x -> x) ([1, 2, 3])"
    return None


HELP_TEXT = """
Remora REPL commands:
  :quit, :q      Exit the REPL
  :type <expr>   Show the inferred type of an expression
  :mlir <expr>   Print validated MLIR for an expression
  :prelude       Show built-in prelude definitions
  :defs          Show user definitions in this session
  :load <file>   Load definitions and evaluate the file body
  :reset         Clear accumulated definitions
  :target [cpu|interp|gpu-nvidia]
                 Show or set the current target
  :syntax [ml|lisp]
                 Show or set the current syntax
  :debug         Toggle debug mode
  :help          Show this message
"""
