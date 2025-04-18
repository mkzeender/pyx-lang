from parso.grammar import PythonGrammar
from parso.python.parser import Parser
from parso.python.diff import DiffParser
from parso.utils import parse_version_string
from importlib.resources import read_text

from parso.utils import PythonVersionInfo

class PyXGrammar(PythonGrammar):
    def __init__(self, version_info: PythonVersionInfo, bnf_text: str):
        super(PythonGrammar, self).__init__(bnf_text, tokenizer=self._tokenize_lines, parser=PyXParser, diff_parser=DiffParser)
        self.version_info = version_info

class PyXParser(Parser):
    pass


_gram = None
def load_grammar(_py_grammar=False):
    global _gram
    if _gram is None:
        
        version = parse_version_string()
        gram_text = read_text(__name__, f'{'' if _py_grammar else 'x'}grammar{version.major}{version.minor}.txt')
        _gram = PyXGrammar(version_info=version, bnf_text=gram_text)

    return _gram