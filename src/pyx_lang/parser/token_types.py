from parso.python.token import PythonTokenTypes, TokenType
from parso.python.tokenize import PythonToken


from enum import Enum


class PyXTokenTypes(Enum):
    PYXSTRING_STRING = TokenType("PYXSTRING_STRING")


type AllTokenTypes = PyXTokenTypes | PythonTokenTypes


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

    PYXSTRING_STRING = PyXTokenTypes.PYXSTRING_STRING


# class PyXToken(PythonToken):
#     type: PythonTokenTypes
