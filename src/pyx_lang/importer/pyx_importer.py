from __future__ import annotations

# Limit how much we import. This module is loaded at Python startup!!!
# Don't import things until they are needed.
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import decode_source, spec_from_loader

TYPE_CHECKING = False
if TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Buffer, Sequence
    from ast import Expression, Interactive, Module
    from types import CodeType, ModuleType


def _decode_path(path: str | PathLike | Buffer) -> str:
    if isinstance(path, str):
        return path
    try:
        from pathlib import Path

        return str(Path(path))  # type: ignore
    except:
        return bytes(path).decode()  # type: ignore


class PyXFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        from pathlib import Path

        name = fullname.split(".")[-1]
        if path is None:
            path = sys.path
        for dir in path:
            for file in Path(dir).glob(f"{name}.[pP][yY][xX][xX]"):
                fp = str(file.resolve())
                return spec_from_loader(
                    name=fullname, loader=PyXLoader(fullname=fullname, path=fp)
                )


pyx_importer = PyXFinder()


class PyXLoader(SourceFileLoader):
    def __init__(self, fullname: str, path: str) -> None:
        super().__init__(fullname, path)

    def source_to_code(
        self,
        data: Buffer | str | Module | Expression | Interactive,
        path: Buffer | str | PathLike[str],
    ) -> CodeType:
        path = _decode_path(path)

        if isinstance(data, str):
            pass
        else:
            from collections.abc import Buffer

            if isinstance(data, Buffer):
                data = decode_source(data)
            else:
                # already valid Python AST
                return super().source_to_code(data, path)

        from pyx_lang.parser import compile_to_ast

        ast = compile_to_ast(data, mode="exec", filepath=path)

        return super().source_to_code(ast, path)

    def exec_module(self, module: ModuleType) -> None:
        # inject the namespace for creating nodes, etc
        from pyx_lang.hooks import _pyx_

        module._pyx_ = _pyx_  # type: ignore

        return super().exec_module(module)
