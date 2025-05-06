from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path
import re
import sys
from textwrap import dedent

from pyx_lang.parser import compile_to_python

parser = ArgumentParser(
    description=dedent("""
        Note: Use "python -m" to run a .pyxx file as a script.
        """)
)

parser.add_argument(
    "-f",
    "--save-files",
    action="store_true",
    help="Save output to .py files",
)


parser.add_argument("FILE", nargs="+")


def main(args: list[str] | None = None) -> None:
    ns = parser.parse_args(args)

    if ns.FILE == ["-"]:
        if ns.save_files:
            print("--save-files does not work with stdin.", file=sys.stderr)
            raise SystemExit(1)
        print(compile_to_python(sys.stdin.read()))

    else:
        for n_file, file in enumerate(_resolve_file_patterns(ns.FILE)):
            pyth = compile_to_python(file.read_text(newline="\n"))

            if ns.save_files:
                file.with_suffix(".py").write_text(pyth)
            elif n_file > 0:
                print(
                    "\n\nError: Only 1 file can be compiled when writing to stdout",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            else:
                print(pyth)


_is_glob = re.compile(r"[\*\?\[]").search


def _resolve_file_patterns(patterns: list[str]) -> Iterator[Path]:
    for pattern in patterns:
        if _is_glob(pattern):
            yield from Path(".").glob(pattern)
            continue

        fp = Path(pattern).resolve()
        if fp.is_dir():
            yield from fp.glob("**/*.pyxx")
            continue
        yield fp
