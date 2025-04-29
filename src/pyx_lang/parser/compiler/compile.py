from ast import Module
import ast
from collections.abc import Mapping
from typing import Any, Literal, overload
from .._parse import parse_to_cst

from pyx_lang.parser.compiler.ast import CstToAstCompiler, CstNode


class AstModifier(ast.NodeTransformer):
    def __init__(self, replacements: Mapping[tuple[int, int], ast.AST]):
        self.replacements = replacements

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        if (v := self.replacements.get((node.lineno, node.col_offset))) is not None:
            return v
        return super().visit_Tuple(node)


@overload
def compile_to_ast(
    src: str | CstNode, mode: Literal["exec"], filepath: str = "<string>"
) -> ast.Module: ...
@overload
def compile_to_ast(
    src: str | CstNode, mode: Literal["eval"], filepath: str = "<string>"
) -> ast.Expression: ...
@overload
def compile_to_ast(
    src: str | CstNode, mode: Literal["single"], filepath: str = "<string>"
) -> ast.Interactive: ...
@overload
def compile_to_ast(
    src: str | CstNode, mode: Literal["func_type"], filepath: str = "<string>"
) -> ast.FunctionType: ...
def compile_to_ast(
    src: str | CstNode,
    mode: Literal["exec", "eval", "single", "func_type"],
    filepath: str = "<string>",
) -> ast.AST:
    if isinstance(src, str):
        src = parse_to_cst(src)

    compiler = CstToAstCompiler(filename=filepath)
    compiler.visit(src)
    ast_: ast.AST = ast.parse(
        src.get_code(), mode=mode, filename=filepath, type_comments=True
    )

    ast_ = AstModifier(compiler.locs_to_override).visit(ast_)

    return ast_
