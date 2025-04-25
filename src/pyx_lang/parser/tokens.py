from enum import Enum
from types import SimpleNamespace
from parso.python.token import TokenType, PythonTokenTypes


class _PyXTokenTypes(Enum):
    PYXSTRING_STRING = TokenType("PYXSTRING_STRING")


type PyXTokenTypes = _PyXTokenTypes | PythonTokenTypes


class PyXTokenTypesNS:
    STRING = PythonTokenTypes.STRING
    NAME = PythonTokenTypes.NAME
    NUMBER = PythonTokenTypes.NUMBER
    OP = PythonTokenTypes.OP
    NEWLINE = PythonTokenTypes.NEWLINE
    INDENT = PythonTokenTypes.INDENT
    DEDENT = PythonTokenTypes.DEDENT
    ENDMARKER = PythonTokenTypes.ENDMARKER
    ERRORTOKEN = PythonTokenTypes.ERRORTOKEN
    ERROR_DEDENT = PythonTokenTypes.ERROR_DEDENT
    FSTRING_START = PythonTokenTypes.FSTRING_START
    FSTRING_STRING = PythonTokenTypes.FSTRING_STRING
    FSTRING_END = PythonTokenTypes.FSTRING_END

    PYXSTRING_STRING = _PyXTokenTypes.PYXSTRING_STRING
