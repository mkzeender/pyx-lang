__all__ = ["tokenize", "parse_to_cst", "compile_to_ast", "compile_to_python"]

import ast
from typing import Literal
from .tokenizer.tokenize import tokenize
from ._parse import parse_to_cst, CstNode
from .compiler.compile import compile_to_ast


def compile_to_python(
    src: str | CstNode, mode: Literal["exec", "eval", "single", "func_type"] = "exec"
) -> str:
    return ast.unparse(compile_to_ast(src, mode))
