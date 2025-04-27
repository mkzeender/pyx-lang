from typing import Any

from parso.python.tokenize import (
    MAX_UNICODE,
    _compile,
    group,
    _all_string_prefixes,
    maybe,
    TokenCollection,
)

_token_collection_cache: dict[Any, TokenCollection] = {}

Whitespace = r"[ \f\t]*"
Name = "([A-Za-z_0-9\u0080-" + MAX_UNICODE + "]+)"

# 
pyx_break_tokens = {
    '==',
    '<=',
    '!=',
    'in',
    'is',
    'not',
    'and',
    'or',
    'lambda',
    ':',
    ':=',
}


def _create_token_collection(version_info):
    # Note: we use unicode matching for names ("\w") but ascii matching for
    # number literals.
    Whitespace = r"[ \f\t]*"
    whitespace = _compile(Whitespace)
    Comment = r"#[^\r\n]*"

    Hexnumber = r"0[xX](?:_?[0-9a-fA-F])+"
    Binnumber = r"0[bB](?:_?[01])+"
    Octnumber = r"0[oO](?:_?[0-7])+"
    Decnumber = r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)"
    Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
    Exponent = r"[eE][-+]?[0-9](?:_?[0-9])*"
    Pointfloat = group(
        r"[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?", r"\.[0-9](?:_?[0-9])*"
    ) + maybe(Exponent)
    Expfloat = r"[0-9](?:_?[0-9])*" + Exponent
    Floatnumber = group(Pointfloat, Expfloat)
    Imagnumber = group(r"[0-9](?:_?[0-9])*[jJ]", Floatnumber + r"[jJ]")
    Number = group(Imagnumber, Floatnumber, Intnumber)

    # Note that since _all_string_prefixes includes the empty string,
    #  StringPrefix can be the empty string (making it optional).
    possible_prefixes = _all_string_prefixes()
    StringPrefix = group(*possible_prefixes)
    StringPrefixWithF = group(*_all_string_prefixes(include_fstring=True))
    fstring_prefixes = _all_string_prefixes(include_fstring=True, only_fstring=True)
    FStringStart = group(*fstring_prefixes)

    # Tail end of ' string.
    Single = r"(?:\\.|[^'\\])*'"
    # Tail end of " string.
    Double = r'(?:\\.|[^"\\])*"'
    # Tail end of ''' string.
    Single3 = r"(?:\\.|'(?!'')|[^'\\])*'''"
    # Tail end of """ string.
    Double3 = r'(?:\\.|"(?!"")|[^"\\])*"""'
    Triple = group(StringPrefixWithF + "'''", StringPrefixWithF + '"""')

    # Because of leftmost-then-longest match semantics, be sure to put the
    # longest operators first (e.g., if = came before ==, == would get
    # recognized as two instances of =).
    Operator = group(
        r"</", r"/>", r"\*\*=?", r">>=?", r"<<=?", r"//=?", r"->", r"[+\-*/%&@`|^!=<>]=?", r"~"
    )

    Bracket = "[][(){}]"

    special_args = [r"\.\.\.", r"\r\n?", r"\n", r"[;.,@]"]
    if version_info >= (3, 8):
        special_args.insert(0, ":=?")
    else:
        special_args.insert(0, ":")
    Special = group(*special_args)

    Funny = group(Operator, Bracket, Special)

    # First (or only) line of ' or " string.
    ContStr = group(
        StringPrefix
        + r"'[^\r\n'\\]*(?:\\.[^\r\n'\\]*)*"
        + group("'", r"\\(?:\r\n?|\n)"),
        StringPrefix
        + r'"[^\r\n"\\]*(?:\\.[^\r\n"\\]*)*'
        + group('"', r"\\(?:\r\n?|\n)"),
    )
    pseudo_extra_pool = [Comment, Triple]
    all_quotes = '"', "'", '"""', "'''"
    if fstring_prefixes:
        pseudo_extra_pool.append(FStringStart + group(*all_quotes))

    PseudoExtras = group(r"\\(?:\r\n?|\n)|\Z", *pseudo_extra_pool)
    PseudoToken = group(Whitespace, capture=True) + group(
        PseudoExtras, Number, Funny, ContStr, Name, capture=True
    )

    # For a given string prefix plus quotes, endpats maps it to a regex
    #  to match the remainder of that string. _prefix can be empty, for
    #  a normal single or triple quoted string (with no prefix).
    endpats = {}
    for _prefix in possible_prefixes:
        endpats[_prefix + "'"] = _compile(Single)
        endpats[_prefix + '"'] = _compile(Double)
        endpats[_prefix + "'''"] = _compile(Single3)
        endpats[_prefix + '"""'] = _compile(Double3)

    # A set of all of the single and triple quoted string prefixes,
    #  including the opening quotes.
    single_quoted = set()
    triple_quoted = set()
    fstring_pattern_map = {}
    for t in possible_prefixes:
        for quote in '"', "'":
            single_quoted.add(t + quote)

        for quote in '"""', "'''":
            triple_quoted.add(t + quote)

    for t in fstring_prefixes:
        for quote in all_quotes:
            fstring_pattern_map[t + quote] = quote

    ALWAYS_BREAK_TOKENS = (
        ";",
        "import",
        "class",
        "def",
        "try",
        "except",
        "finally",
        "while",
        "with",
        "return",
        "continue",
        "break",
        "del",
        "pass",
        "global",
        "assert",
        "nonlocal",
    )
    pseudo_token_compiled = _compile(PseudoToken)
    return TokenCollection(
        pseudo_token_compiled,
        single_quoted,
        triple_quoted,
        endpats,
        whitespace,
        fstring_pattern_map,
        set(ALWAYS_BREAK_TOKENS),  # type: ignore
    )


def get_token_collection(version_info):
    try:
        return _token_collection_cache[tuple(version_info)]
    except KeyError:
        _token_collection_cache[tuple(version_info)] = result = (
            _create_token_collection(version_info)
        )
        return result
