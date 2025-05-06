from ast import (
    AST,
    Attribute,
    Call,
    Constant,
    JoinedStr,
    Load,
    Name,
    NodeTransformer,
    keyword,
)
import ast
from collections.abc import Iterable
from parso.python.tree import (
    PythonBaseNode as CstBaseNode,
    PythonNode as CstNode,
    Operator as CstOperator,
    Name as CstName,
    PythonErrorLeaf as CstErrorLeaf,
    PythonErrorNode as CstErrorNode,
)
from parso.tree import NodeOrLeaf as CstNodeOrLeaf
from typing import Any, Never, Protocol, TypeIs


class Located(Protocol):
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None


def has_loc(v) -> TypeIs[Located]:
    return True  # hack
    return hasattr(v, "lineno")


def _pos_dict(node):
    return dict(
        lineno=node.start_pos[0],
        col_offset=node.start_pos[1],
        end_lineno=node.end_pos[0],
        end_col_offset=node.end_pos[1],
    )


class AstOffsetApplier(NodeTransformer):
    def __init__(self, line_offset: int, col_offset: int) -> None:
        super().__init__()
        self.line_offset = line_offset
        self.col_offset = col_offset

    def visit(self, node: AST) -> Any:
        if has_loc(node):
            try:
                node.lineno += self.line_offset
                if node.end_lineno is not None:
                    node.end_lineno += self.line_offset

                node.col_offset += self.col_offset
                if node.end_col_offset is not None:
                    node.end_col_offset += self.col_offset
            except AttributeError:
                pass

        return super().visit(node)


def apply_offset(
    root_node: AST,
    line_offset: int,
    col_offset: int,
    root_col_start: int | None = None,
    root_col_end: int | None = None,
) -> None:
    AstOffsetApplier(line_offset, col_offset).visit(root_node)

    if has_loc(root_node):
        try:
            if root_col_start is not None:
                root_node.col_offset = root_col_start
            if root_col_end is not None:
                root_node.end_col_offset = root_col_end
        except AttributeError:
            pass


def compile_subexpr(node: CstNodeOrLeaf) -> ast.expr:
    code = node.get_code(include_prefix=False)
    assert isinstance(code, str)
    expr_stmt = ast.parse("(" + code + ")", mode="exec").body[0]
    assert isinstance(expr_stmt, ast.Expr)
    expr = expr_stmt.value

    apply_offset(expr, line_offset=node.start_pos[0], col_offset=node.start_pos[1] - 1)

    return expr


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

    def generic_error(self, node: CstNodeOrLeaf, msg=None) -> Never:
        start_line, start_col = node.start_pos
        end_line, end_col = node.end_pos
        if self.code is None:
            code = None
        else:
            code_lines = self.code.splitlines()[start_line - 1 : end_line]
            code = "\n".join(code_lines)
        if msg is None:
            msg = f'"{node.get_last_leaf().get_code()}" is not understood here.'  # type: ignore

        raise SyntaxError(
            msg, (self.filename, start_line, start_col + 1, code, end_line, end_col + 1)
        )

    def visit[NodeT: CstNodeOrLeaf](self, node: NodeT) -> NodeT:
        return getattr(self, "visit_" + node.type, self.generic_visit)(node)

    def visit_pyxtag(self, node: CstNode) -> CstNodeOrLeaf:
        self.locs_to_override[node.start_pos] = self.create_pyxtag(node)

        prefix = node.get_first_leaf().prefix  # type: ignore
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

    def create_inner(self, node: CstNode, name: str | None) -> list[AST]: ...
    def create_kwds(self, nodes: Iterable[CstNodeOrLeaf]) -> list[keyword]:
        kwds = list[keyword]()

        for node in nodes:
            match node:
                case CstNode(
                    type="pyxparam", children=[CstName(value=arg) as n, _, expr]
                ):
                    kwds.append(
                        keyword(arg=arg, value=compile_subexpr(expr), **_pos_dict(n))
                    )
                case _:
                    self.generic_error(node, msg=f"Unexpected {node.type} here.")

        return kwds

    def create_pyxtag(self, node: CstNode) -> AST:
        name: str | None = None
        inner = list[AST]()
        kwds = list[keyword]()
        for child in node.children:
            match child:
                case CstName(value=value):
                    if name not in (None, value):
                        self.generic_error(
                            node=child,
                            msg="Closing tag name must match opening tag name",
                        )
                    name = child.value

                case CstOperator():
                    pass
                case CstNode(type="pyxtagclose"):
                    inner = self.create_inner(child, name)
                case CstNode(
                    type="pyxtagargs", children=[CstName(value=name_), *params]
                ):
                    name = name_
                    kwds = self.create_kwds(params)
                case _:
                    self.generic_error(child, msg=f"Unexpected {child.type} here.")

        if name is None:
            name = "Fragment"

        expr = Call(
            func=Attribute(
                value=Name(id="_pyx_", ctx=Load(), **_pos_dict(node)),
                attr="create_element",
                ctx=Load(),
                **_pos_dict(node),
            ),
            args=[
                Name(id=name, ctx=Load(), **_pos_dict(node)),
                # JoinedStr(values=[Constant(value="hi there")], **_pos_dict(node)),
            ],  # TODO: actual body!
            keywords=kwds,
            **_pos_dict(node),
        )

        return expr
