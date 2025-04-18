from pyx_lang.parser.parse import load_grammar
from importlib.resources import read_text
from parso.tree import Node
import parso.tree

def test_pyxparse():
    gram = load_grammar()

    code = read_text(__name__, 'pyx_data.pyx')
    v: Node = gram.parse(code)
    assert v.children[0]