"""CPU-first Remora REPL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lark import LarkError

from remora.ast_nodes import Definition, FuncDef, Program, ValDef
from remora.compiler import compile_source_to_mlir
from remora.display import format_result
from remora.errors import RemoraError
from remora.parser import parse_program, parse_repl_input
from remora.lisp_reader import parse_lisp as parse_lisp_program
from remora.prelude import prelude_definition_sources_for_syntax
from remora.runtime import EvaluationResult, evaluate_source, evaluate_source_compiled
from remora.typechecker import TypeChecker

REPL_TARGETS = ("cpu", "interp", "cuda")
REPL_SYNTAXES = ("ml", "lisp")


@dataclass
class ReplState:
    target: str = "cpu"
    debug: bool = False
    syntax: str = "ml"
    definition_sources_by_syntax: dict[str, list[str]] = field(
        default_factory=lambda: {
            syntax: prelude_definition_sources_for_syntax(syntax)
            for syntax in REPL_SYNTAXES
        }
    )

    @property
    def definition_sources(self) -> list[str]:
        return self.definition_sources_by_syntax[self.syntax]

    @definition_sources.setter
    def definition_sources(self, sources: list[str]) -> None:
        self.definition_sources_by_syntax[self.syntax] = sources


def make_initial_state(target: str = "cpu") -> ReplState:
    if target not in REPL_TARGETS:
        raise ReplError("available REPL targets: cpu, interp, cuda")
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
        return self._parse_program_with_syntax(source, filename, self.state.syntax)

    def _parse_program_with_syntax(self, source: str, filename: str, syntax: str):
        if syntax == "lisp":
            return parse_lisp_program(source, filename)
        return parse_program(source, filename)

    def load_source(
        self,
        source: str,
        filename: str,
        *,
        syntax: str | None = None,
        evaluate_body: bool = False,
    ) -> str:
        """Load top-level definitions from a source file into the session.

        The load is transactional: parse and type-check the whole source in the
        current session context before appending any definitions. Definition
        source text is sliced from the same raw file text that produced the AST
        locations, so prelude/session context cannot skew line numbers.
        """
        syntax = syntax or self.state.syntax
        program = self._parse_program_with_syntax(source, filename, syntax)
        active_definitions = self.state.definition_sources_by_syntax[syntax]
        candidate_source = _program_source(active_definitions, source)
        typed = TypeChecker().check_program(
            self._parse_program_with_syntax(candidate_source, filename, syntax)
        )

        definition_sources = _definition_sources_from_program(source, program, syntax)
        new_typed_definitions = (
            typed.definitions[-len(program.definitions):]
            if program.definitions else []
        )
        if len(definition_sources) != len(new_typed_definitions):
            raise ReplError(
                f"could not recover top-level definitions from {filename}"
            )

        messages: list[str] = []
        for definition_source, typed_definition in zip(definition_sources, new_typed_definitions):
            active_definitions.append(definition_source)
            definition = typed_definition.definition
            if isinstance(definition, FuncDef):
                messages.append(f"Defined: {definition.name} : <function>")
            else:
                messages.append(f"Defined: {definition.name} : {typed_definition.type}")

        self.state.syntax = syntax
        if evaluate_body and program.body is not None:
            body_source = _body_source_from_program(source, program, syntax)
            result = self._evaluate_program_source(
                _program_source(active_definitions, body_source)
            )
            messages.append(format_result(result.value, result.type))

        return "\n".join(messages) if messages else "Loaded."

    def _evaluate_program_source(self, program_source: str) -> EvaluationResult:
        if self.state.target == "cpu":
            return evaluate_source_compiled(
                program_source, include_prelude=False, syntax=self.state.syntax, strict=False
            )
        if self.state.target == "interp":
            return evaluate_source(
                program_source, include_prelude=False, syntax=self.state.syntax
            )
        if self.state.target == "cuda":
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
                self.state.definition_sources = prelude_definition_sources_for_syntax(
                    self.state.syntax
                )
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
        return "\n".join(prelude_definition_sources_for_syntax(self.state.syntax))

    def _defs_command(self) -> str:
        prelude_count = len(prelude_definition_sources_for_syntax(self.state.syntax))
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

        return self.load_source(
            source,
            str(path),
            syntax=detected_syntax,
            evaluate_body=True,
        )

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


def _definition_sources_from_program(source: str, program: Program, syntax: str) -> list[str]:
    if syntax == "lisp":
        return [
            form for form in _top_level_lisp_forms(source)
            if _is_lisp_definition_form(form)
        ][:len(program.definitions)]

    lines = source.splitlines()
    starts = [_definition_start_line(definition) for definition in program.definitions]
    body_start = getattr(getattr(program.body, "loc", None), "line", 0) if program.body else 0
    boundaries = [*starts, body_start or len(lines) + 1]

    sources: list[str] = []
    for idx, start in enumerate(starts):
        if start <= 0:
            continue
        end = boundaries[idx + 1] - 1
        sources.append(_slice_top_level_source(lines, start, end))
    return sources


def _definition_start_line(definition: Definition) -> int:
    return getattr(getattr(definition, "loc", None), "line", 0)


def _body_source_from_program(source: str, program: Program, syntax: str) -> str:
    if program.body is None:
        return ""
    if syntax == "lisp":
        for form in _top_level_lisp_forms(source):
            if not _is_lisp_definition_form(form):
                return form
        return ""
    start = getattr(getattr(program.body, "loc", None), "line", 0)
    if start <= 0:
        return ""
    return _slice_top_level_source(source.splitlines(), start, len(source.splitlines()))


def _slice_top_level_source(lines: list[str], start_line: int, end_line: int) -> str:
    start = max(start_line - 1, 0)
    end = min(end_line, len(lines))
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end]).strip()


def _top_level_lisp_forms(source: str) -> list[str]:
    forms: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        i = _skip_lisp_ignored(source, i)
        if i >= n:
            break
        start = i
        if source[i] in "([":
            close_for = {"(": ")", "[": "]"}
            stack = [close_for[source[i]]]
            i += 1
            while i < n and stack:
                if source[i] == ";":
                    while i < n and source[i] not in "\r\n":
                        i += 1
                    continue
                if source[i] in close_for:
                    stack.append(close_for[source[i]])
                elif source[i] == stack[-1]:
                    stack.pop()
                i += 1
            forms.append(source[start:i].strip())
            continue

        while i < n and not source[i].isspace():
            if source[i] == ";":
                break
            i += 1
        forms.append(source[start:i].strip())
    return [form for form in forms if form]


def _skip_lisp_ignored(source: str, index: int) -> int:
    n = len(source)
    while index < n:
        if source[index].isspace():
            index += 1
            continue
        if source[index] == ";":
            while index < n and source[index] not in "\r\n":
                index += 1
            continue
        break
    return index


def _is_lisp_definition_form(form: str) -> bool:
    stripped = form.lstrip()
    return (
        stripped.startswith("(define ")
        or stripped.startswith("(define\t")
        or stripped.startswith("(define\n")
        or stripped.startswith("(define\r")
        or stripped.startswith("(define/")
    )


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
  :target [cpu|interp|cuda]
                 Show or set the current target
  :syntax [ml|lisp]
                 Show or set the current syntax
  :debug         Toggle debug mode
  :help          Show this message
"""
