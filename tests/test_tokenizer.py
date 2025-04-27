

from pyx_lang.parser.tokenize import tokenize

def listit(v: str):
    l = list(tokenize(v))
    return l, [x.string for x in l]


def test_fstring():

    v = listit("""f"Hi there! {me:0.1}" """)
    ...


def test_simple_tag():
    v = listit("""<hello>bruh <hoo/>
                      muffin dude!</hello> there""")
    ...


def test_multilines():
    v = listit(
"""
<hello>
    bruh muffin
</hello>

if val1 < val1:
    val2 > val2
cool1
val3 = val3 < val3
val4 > val4
cool2
cool3

"""
    )
    ...