from enum import Enum, auto
from typing import Any
from parso.python.tokenize import FStringNode, _compile


from re import Pattern

pyxstring: Pattern[str] = _compile(r"(?:[^\{\}<&]|&[#]?[a-zA-Z0-9]+;)+")
pyxstring_end: Pattern[str] = _compile(r"(?:[^<])*</")


class PyxFStringNode(FStringNode):
    parentheses_count: int
    previous_lines: str
    last_string_start_pos: tuple[int, int] | None | Any


class PyxNodeStates(Enum):
    OPEN = auto()
    INNER = auto()
    CLOSE = auto()


class PyxNode(PyxFStringNode):
    def __init__(self, initial_string: str = ""):
        super().__init__('"')
        self.state: PyxNodeStates = PyxNodeStates.OPEN
        self.initial_string = initial_string
        self.is_confirmed: bool = False

    def is_in_str(self):
        return self.state is PyxNodeStates.INNER and self.parentheses_count == 0

    def get_start_pos(self) -> tuple[int, int]:
        if not isinstance(self.last_string_start_pos, tuple):
            raise RuntimeError("No start pos")
        return self.last_string_start_pos

    # @property
    # def expr_level(self):
    #     if self.state is not PyxNodeStates.INNER:
    #         return 0


def pyx_tag_status(pyx_stack: list[PyxNode]) -> bool | None:
    if not pyx_stack:
        return False
    if len(pyx_stack) == 1:
        node = pyx_stack[-1]
        if node.parentheses_count:
            return True
        if node.state is PyxNodeStates.OPEN and not node.is_confirmed:
            return None
    return True


def find_pyx_string(pyx_stack: list[PyxNode], line, lnum, pos) -> tuple[str, int]:
    tos = pyx_stack[-1]

    match = pyxstring.match(line, pos)
    if match is None:
        return tos.previous_lines, pos

    if not tos.previous_lines:
        tos.last_string_start_pos = (lnum, pos)

    string = match.group(0)
    string += tos.initial_string
    tos.initial_string = ""
    new_pos = pos
    new_pos += len(string)
    # even if allow_multiline is False, we still need to check for trailing
    # newlines, because a single-line f-string can contain line continuations
    if string.endswith("\n") or string.endswith("\r"):
        tos.previous_lines += string
        string = ""
    else:
        string = tos.previous_lines + string

    return string, new_pos
