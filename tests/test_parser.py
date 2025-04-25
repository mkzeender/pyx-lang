from pyx_lang.parser.parse import load_grammar
from importlib.resources import read_text
from parso.tree import Node
import parso.tree

from ast import parse

code = read_text(__name__, 'pyx_data.pyx')


def test_pyxparse():
    gram = load_grammar()

    v: Node = gram.parse('print("hello world!")', error_recovery=False)
    assert v.children[0]


def test_tag():

    gram = load_grammar()

    v: Node = gram.parse('v = <a href="hi">"do" hooligan(9)</a>', error_recovery=False)

    v.children


def test_code():
    gram = load_grammar()
    v: Node = gram.parse(code, error_recovery=False)
    ...


def test_import():
    from pyx_lang import autoinstall
    from . import pyx_data #type: ignore
