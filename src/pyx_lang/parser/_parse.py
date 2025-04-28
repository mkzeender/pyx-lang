from parso.grammar import PythonGrammar
from parso.python.parser import Parser
from parso.python.tree import PythonLeaf
from parso.python.diff import DiffParser
from parso.utils import parse_version_string
from importlib.resources import read_text

from parso.python.tree import PythonNode as CstNode
from parso.utils import PythonVersionInfo

from pyx_lang.parser.tokenizer.tokenize import tokenize_lines
from pyx_lang.parser.tokenizer.token_types import PyXTokenTypesNS



def parse(src: str) -> CstNode:
    gram = load_grammar()
    return gram.parse(src)




class PyXStringString(PythonLeaf):
    """
    f-strings contain f-string expressions and normal python strings. These are
    the string parts of f-strings.
    """

    type = "fstring_string"
    __slots__ = ()


class PyXGrammar(PythonGrammar):
    _token_namespace = PyXTokenTypesNS

    def __init__(self, version_info: PythonVersionInfo, bnf_text: str):
        super(PythonGrammar, self).__init__(
            bnf_text,
            tokenizer=self._tokenize_lines,
            parser=PyXParser,
            diff_parser=DiffParser,
        )
        self.version_info = version_info

    def _tokenize_lines(self, lines, **kwargs):
        return tokenize_lines(lines, version_info=self.version_info, **kwargs)


class PyXParser(Parser):
    _leaf_map = Parser._leaf_map | {PyXTokenTypesNS.PYXSTRING_STRING: PyXStringString}


_gram = None


def load_grammar(_py_grammar=False):
    global _gram
    if _gram is None:

        version = parse_version_string()
        gram_text = read_text(
            __name__,
            f"{'' if _py_grammar else 'x'}grammar{version.major}{version.minor}.txt",
        )
        _gram = PyXGrammar(version_info=version, bnf_text=gram_text)

    return _gram
