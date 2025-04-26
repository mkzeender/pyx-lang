

from pyx_lang.parser.tokenize import tokenize


def test_fstring():

    v = tokenize("""f"Hi there! {me:0.1}" """)
    ...