

import ast
from pyx_lang.parser.compiler.compile import compile_to_ast


def test_compiler():
    code = """
v = <hello there=(hi + 2) a=10>wut wut wut</hello>

"""

    v = compile_to_ast(code, mode='exec')

    src = ast.unparse(v)
    compile(v, filename='<string>', mode='exec', flags=0)
    ...
    