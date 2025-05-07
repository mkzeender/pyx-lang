from ast import (
    AST,
    Attribute,
    Call,
    Load,
    Name,
    keyword,
)
import ast
from collections.abc import Iterable
import html
from parso.python.tree import (
    PythonBaseNode as CstBaseNode,
    PythonNode as CstNode,
    Operator as CstOperator,
    Name as CstName,
    PythonErrorLeaf as CstErrorLeaf,
    PythonErrorNode as CstErrorNode,
    FStringString as CstFstringString,
)
from parso.tree import NodeOrLeaf as CstNodeOrLeaf
from typing import Any, Never, TypedDict, Unpack

from pyx_lang.parser.compiler.ast_apply_offset import (
    apply_offset,
    syntaxerror_offset,
)
import pyx_lang.parser.compiler.compile as pyxcomp
from pyx_lang.parser.compiler.positions import (
    OptionalPosDict,
    PosDict,
    PosTuple,
    add,
    position_of,
)

FRAGMENT = "Fragment"


class _Empty(TypedDict):
    pass


def _compile_with_offset(
    code: str,
    filepath: str,
    pos: PosDict,
    root_pos: OptionalPosDict = OptionalPosDict(),
) -> ast.expr:
    try:
        expr = pyxcomp.compile_to_ast(code, mode="eval", filepath=filepath).body
    except SyntaxError as e:
        syntaxerror_offset(e, pos)
        raise

    apply_offset(expr, pos, root_offset=root_pos)

    return expr


def compile_subexpr(node: CstNodeOrLeaf, filepath: str) -> ast.expr:
    code: str = node.get_code(include_prefix=False)  # type: ignore
    return _compile_with_offset(code, filepath, position_of(node))


def compile_fstring_expr(node: CstNode, filepath: str) -> ast.expr:
    code: str = node.get_code(include_prefix=False)
    code = 'f"' + code + '"'

    pos = position_of(node)

    # account for the f" at the start of code
    pos["col_offset"] -= 2
    pos["end_col_offset"] = add(pos["end_col_offset"], -2)

    return _compile_with_offset(code, filepath, pos)


class CstToAstCompiler:
    def __init__(self, code: str | None = None, filename: str = "<string>") -> None:
        self.locs_to_override = dict[tuple[int, int], AST]()
        self.filename = filename
        self.code = code

    def generic_visit[NodeT: CstNodeOrLeaf](self, node: NodeT) -> NodeT:
        if isinstance(node, CstErrorLeaf | CstErrorNode):
            self.generic_error(node)
        if isinstance(node, CstBaseNode | CstNode):
            for i, child in enumerate(node.children):
                node.children[i] = self.visit(child)
        return node

    def generic_error(
        self, node: CstNodeOrLeaf, msg=None, **position: Unpack[OptionalPosDict]
    ) -> Never:
        if isinstance(node, CstErrorNode | CstErrorLeaf):
            child: CstNodeOrLeaf = node
            while children := getattr(child, "children", None):
                for child in children:
                    if isinstance(child, CstErrorNode | CstErrorLeaf):
                        break
                else:
                    break
            if not isinstance(child, CstErrorLeaf | CstErrorNode):
                bad_child = child.get_next_leaf()
            else:
                bad_child = child
            msg = f'"{
                bad_child.get_code(include_prefix=False)
                if bad_child is not None
                else "EOF"
            }" is not understood here.'
            if bad_child is not node and bad_child is not None:
                return self.generic_error(bad_child, msg=msg)

        pos = position_of(node)
        pos.update(position)
        lineno, col_offset, end_lineno, end_col_offset = PosTuple(**pos)

        if self.code is None:
            code_line = None
        else:
            code_line = self.code.splitlines()[lineno - 1]
            if lineno != end_lineno:
                end_col_offset = len(code_line) - 1
                end_lineno = lineno
        if msg is None:
            msg = f"Unexpected {node.type} here."

        if end_col_offset is None:
            end_col_offset = col_offset + 1

        if end_lineno is None:
            end_lineno = lineno

        raise SyntaxError(
            msg,
            (
                self.filename,
                lineno,
                col_offset + 1,
                code_line,
                lineno,
                end_col_offset + 1,
            ),
        )

    def visit[NodeT: CstNodeOrLeaf](self, node: NodeT) -> NodeT:
        return getattr(self, "visit_" + node.type, self.generic_visit)(node)

    def visit_pyxtag(self, node: CstNode) -> CstNodeOrLeaf:
        self.locs_to_override[node.start_pos] = self.create_pyxtag(node)

        prefix: str = node.get_first_leaf().prefix  # type: ignore
        code = node.get_code(include_prefix=False)
        lines = code.splitlines()
        filler: str = ("\n" * (len(lines) - 1)) + (" ") * len(lines[-1])

        return CstNode(
            "atom",
            [
                CstOperator("(", start_pos=node.start_pos, prefix=prefix),
                CstOperator(
                    ")", start_pos=(node.end_pos[0], node.end_pos[1] - 1), prefix=filler
                ),
            ],
        )

    def create_inner(self, node: CstNode, name: str) -> list[ast.expr]:
        inner = list[ast.expr]()
        close_name = FRAGMENT
        close_pos = position_of(node)
        for child in node.children:
            match child:
                case CstName(value=value):
                    close_name = value
                    close_pos = position_of(child)
                case CstOperator():
                    pass
                case CstFstringString(value=value):
                    value = html.unescape(
                        value.replace("\r\n", "\n").replace("\r", "\n")
                    )

                    inner.append(
                        ast.Constant(
                            value=value,
                            **position_of(child),
                        )
                    )
                case CstNode(type="pyxtag"):
                    inner.append(self.create_pyxtag(child))
                case CstNode(type="fstring_expr"):
                    inner.append(compile_fstring_expr(child, self.filename))
                case _:
                    self.generic_error(child)

        if name != close_name:
            self.generic_error(
                node=node, msg=f"{name} tag was not closed.", **close_pos
            )

        return inner

    def create_kwds(self, nodes: Iterable[CstNodeOrLeaf]) -> list[keyword]:
        kwds = list[keyword]()

        for node in nodes:
            match node:
                case CstNode(
                    type="pyxparam", children=[CstName(value=arg) as n, _, expr]
                ):
                    kwds.append(
                        keyword(
                            arg=arg,
                            value=compile_subexpr(expr, self.filename),
                            **position_of(n),
                        )
                    )
                case _:
                    self.generic_error(node, msg=f"Unexpected {node.type} here.")

        return kwds

    def create_pyxtag(self, node: CstNode) -> Call:
        name: str = FRAGMENT
        name_pos = position_of(node)
        name_pos.update(end_lineno=None, end_col_offset=None)
        inner = list[ast.expr]()
        kwds = list[keyword]()
        for child in node.children:
            match child:
                case CstOperator():
                    pass
                case CstNode(type="pyxtagclose"):
                    inner = self.create_inner(child, name)
                case CstNode(
                    type="pyxtagargs",
                    children=[CstName(value=name_) as cstname, *params],
                ):
                    name = name_
                    name_pos = position_of(cstname)
                    kwds = self.create_kwds(params)
                case CstName(value=name_):
                    name = name_
                    name_pos = position_of(child)
                case _:
                    self.generic_error(child, msg=f"Unexpected {child.type} here.")

        expr = Call(
            func=Attribute(
                value=Name(id="_pyx_", ctx=Load(), **position_of(node)),
                attr="create_element",
                ctx=Load(),
                **position_of(node),
            ),
            args=[Name(id=name, ctx=Load(), **name_pos), *inner],  # TODO: actual body!
            keywords=kwds,
            **position_of(node),
        )

        return expr
