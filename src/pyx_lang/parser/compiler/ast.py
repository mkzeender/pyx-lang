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
from parso.python.tree import (
    PythonNode as CstNode,
    Operator as CstOperator,
    Name as CstName,
)
from parso.tree import NodeOrLeaf as CstNodeOrLeaf
from typing import Any, Literal, Protocol, TypeIs


class Located(Protocol):
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None


def has_loc(v) -> TypeIs[Located]:
    return hasattr(v, "lineno")


class AstOffsetApplier(NodeTransformer):
    def __init__(self, line_offset: int, col_offset: int) -> None:
        super().__init__()
        self.line_offset = line_offset
        self.col_offset = col_offset

    def visit(self, node: AST) -> Any:
        if has_loc(node):
            node.lineno += self.line_offset
            if node.end_lineno is not None:
                node.end_lineno += self.line_offset

            node.col_offset += self.col_offset
            if node.end_col_offset is not None:
                node.end_col_offset += self.col_offset

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
        if root_col_start is not None:
            root_node.col_offset = root_col_start
        if root_col_end is not None:
            root_node.end_col_offset = root_col_end


def compile_subexpr(node: CstNode) -> ast.expr:
    expr = ast.parse(node.get_code(include_prefix=False), mode="eval").body

    apply_offset(expr, line_offset=node.start_pos[0], col_offset=node.start_pos[1])

    return expr


class CstToAstCompiler:
    def __init__(self) -> None:
        self._locs_to_override = dict[tuple[int, int], AST]()

    def generic_visit[NodeT: CstNodeOrLeaf](self, node: NodeT) -> NodeT:
        if isinstance(node, CstNode):
            for i, child in enumerate(node.children):
                node.children[i] = self.visit(child)
        return node

    def visit[NodeT: CstNodeOrLeaf](self, node: NodeT) -> NodeT:
        return getattr(self, "visit_" + node.type, self.generic_visit)(node)

    def visit_pyxtag(self, node: CstNode) -> CstNodeOrLeaf:
        self._locs_to_override[node.start_pos] = self.create_pyxtag(node)

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

    def create_pyxtag(self, node: CstNode) -> AST:
        name: str | None = None
        kwds = list[keyword]()

        for child in node.children:
            if isinstance(child, CstName):
                if name not in (None, child.value):
                    raise SyntaxError(
                        "Closing tag name must match opening tag name"
                    )  # TODO: better error handling?
                name = child.value
            if child.type == "pyxparam":
                assert isinstance(child, CstNode)
                kwds.append(
                    keyword(
                        arg=child.children[0].value,  # type: ignore
                        value=compile_subexpr(child.children[2]),  # type: ignore
                    )
                )

        assert name is not None

        return Call(
            func=Attribute(
                value=Name(id="_pyx_", ctx=Load()), attr="create_element", ctx=Load()
            ),
            args=[
                Name(id=name, ctx=Load()),
                JoinedStr(values=[Constant(value="hi there")]),
            ],  # TODO: actual body!
            keywords=kwds,
        )


