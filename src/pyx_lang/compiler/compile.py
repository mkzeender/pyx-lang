from ast import AST, NodeTransformer
import ast
from typing import Any, Protocol, Sequence, TypeGuard, TypeIs
from pyx_lang.parser.parse import load_grammar
from parso.tree import Node, NodeOrLeaf, Leaf


def to_python(src: str) -> str:
    gram = load_grammar()

    nodes: Node = gram.parse(src)

    return ''


class CSTtoAst:

    def generic_visit(self, node: NodeOrLeaf) -> Any:
        if isinstance(node, Node):
            ls = [self.visit(child) for child in node.children]
            if len(ls) == 1:
                return ls[0]
        raise SyntaxError(f'Unsupported syntax feature: "{node.type}"')
        
    def visit(self, node: NodeOrLeaf) -> Any:
        visit_method = getattr(self, 'visit_' + node.type, self.generic_visit)
        
        ast_node: AST = visit_method(node)
        
        if has_loc(ast_node):
            ast_node.lineno, ast_node.col_offset = node.start_pos
            ast_node.end_lineno, ast_node.end_col_offset = node.end_pos

        return ast_node


    def visit_STRING(self, node: Leaf) -> AST:
        return ast.Constant(node.value)

    def visit_fstring(self, node: Node) -> AST:
        values = []
        for child in node.children[1:-1]:
            if isinstance(child, Leaf):
                if values and isinstance(values[-1], str):
                    values[-1] += child.value
                else:
                    values.append(child.value)
            else:
                assert isinstance(child, Node)
                for expr in child.children[2:]:
                    
        if len(cs) == 1:
            return cs[0]
        return ast.JoinedStr(
            cs,
        )
