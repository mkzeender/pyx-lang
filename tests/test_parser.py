from pyx_lang.parser._parse import load_grammar
from importlib.resources import read_text
from parso.tree import Node

code = read_text(__name__, 'pyx_data.pyx')


def test_pyxparse():
    gram = load_grammar()

    v: Node = gram.parse('print("hello world!")', error_recovery=False)
    assert v.children[0]


def test_tag():

    gram = load_grammar()

    v: Node = gram.parse('return <a href="hi">do hooligan 9</a>', error_recovery=False)

    v.children


def test_code():
    code = """
def example_func(link):

    return <div>
            <span>
                <a href=link>
                    Hello world!
                </a>
            </span>
        </div>

"""
    gram = load_grammar()
    v: Node = gram.parse(code, error_recovery=True)
    
    ...


def test_import():
    from pyx_lang.importer import autoinstall
    from . import pyx_data #type: ignore

    assert pyx_data.example_func('/woah')
