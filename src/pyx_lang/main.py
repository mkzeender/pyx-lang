
from argparse import ArgumentParser
import runpy
runpy.run_module
root = ArgumentParser('pyx-lang', description='Note: use "python -m script_name" to run a script.')

subcommands = root.add_subparsers()
compile_ = subcommands.add_parser(
    'compile'
)

def main() -> None:
    print("Hello from pyx-lang!")
