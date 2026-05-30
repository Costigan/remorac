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
from remora.runtime import evaluate_source
from remora.typechecker import TypeChecker


@dataclass
class ReplState:
    target: str = "cpu"
    debug: bool = False
    definition_sources: list[str] = field(default_factory=list)


def make_initial_state(target: str = "cpu") -> ReplState:
    if target != "cpu":
        raise ReplError("only the cpu REPL target is currently available")
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
            item = parse_repl_input(text)
        except (LarkError, RemoraError) as exc:
            return f"Parse error: {exc}"

        try:
            if isinstance(item, (FuncDef, ValDef)):
                return self._process_definition(text, item)
            return self._process_expression(text)
        except RemoraError as exc:
            return f"Error: {exc}"

    def _process_definition(self, source: str, definition: FuncDef | ValDef) -> str:
        if isinstance(definition, FuncDef):
            return "Error: top-level function definitions are deferred"

        candidate_definitions = [*self.state.definition_sources, source]
        program_source = _program_source(candidate_definitions, "0")
        typed = TypeChecker().check_program(parse_program(program_source, "<repl>"))
        self.state.definition_sources.append(source)
        return f"Defined: {definition.name} : {typed.definitions[-1].type}"

    def _process_expression(self, source: str) -> str:
        program_source = _program_source(self.state.definition_sources, source)
        result = evaluate_source(program_source)
        return format_result(result.value, result.type)

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
            if name == ":type":
                return self._type_command(arg)
            if name == ":mlir":
                return self._mlir_command(arg)
            if name == ":load":
                return self._load_file(arg)
            if name == ":reset":
                self.state.definition_sources.clear()
                return "State reset."
            return f"Unknown command: {name}. Type :help for help."
        except RemoraError as exc:
            return f"Error: {exc}"
        except LarkError as exc:
            return f"Parse error: {exc}"

    def _target_command(self, arg: str) -> str:
        if not arg:
            return f"Current target: {self.state.target}"
        if arg != "cpu":
            return "Error: only the cpu REPL target is currently available"
        self.state.target = arg
        return "Target: cpu"

    def _type_command(self, arg: str) -> str:
        if not arg:
            return "Usage: :type <expr>"
        program = parse_program(_program_source(self.state.definition_sources, arg), "<repl>")
        typed = TypeChecker().check_program(program)
        return f"{arg} : {typed.type}"

    def _mlir_command(self, arg: str) -> str:
        if not arg:
            return "Usage: :mlir <expr>"
        return compile_source_to_mlir(_program_source(self.state.definition_sources, arg))

    def _load_file(self, arg: str) -> str:
        if not arg:
            return "Usage: :load <file>"

        path = Path(arg)
        source = path.read_text(encoding="utf-8")
        program = parse_program(source, str(path))
        messages: list[str] = []
        for definition_source in _top_level_definition_lines(source):
            item = parse_repl_input(definition_source, str(path))
            if not isinstance(item, (FuncDef, ValDef)):
                continue
            message = self._process_definition(definition_source, item)
            messages.append(message)
        if program.body is not None:
            result = evaluate_source(_program_source(self.state.definition_sources, _body_source(source)))
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
        print(f"Remora REPL [target: {self.state.target}]")
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

    parser = argparse.ArgumentParser(description="Remora CPU REPL")
    parser.add_argument("--target", default="cpu", choices=("cpu",))
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


HELP_TEXT = """
Remora REPL commands:
  :quit, :q      Exit the REPL
  :type <expr>   Show the inferred type of an expression
  :mlir <expr>   Print validated MLIR for an expression
  :load <file>   Load definitions and evaluate the file body
  :reset         Clear accumulated definitions
  :target [cpu]  Show or set the current target
  :debug         Toggle debug mode
  :help          Show this message
"""
