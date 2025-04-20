from pyx_lang.parser.parse import load_grammar


def to_python(src: str) -> str:
    gram = load_grammar()

    nodes = gram.parse(src)

    return ''
