

from ast import Expression, Interactive, Module
from collections.abc import Buffer
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import spec_from_loader, decode_source
import os
from pathlib import Path
import sys
from types import CodeType, ModuleType
from typing import Sequence

AstNodeType = Module | Expression | Interactive

class PyXFinder(MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None) -> ModuleSpec | None:
        name = fullname.split('.')[-1]
        if path is None:
            path = sys.path
        for dir in path:
            for file in Path(dir).glob(f'{name}.[pP][yY][xX]'):
                fp = str(file.resolve())
                return spec_from_loader(
                    name=fullname,
                    loader=PyXLoader(fullname=fullname, path=fp)
                )
                
pyx_importer = PyXFinder()

class PyXLoader(SourceFileLoader):

    def __init__(self, fullname: str, path: str) -> None:

        super().__init__(fullname, path)

    def source_to_code(self, data: Buffer | str | AstNodeType, path: Buffer | str | os.PathLike[str]) -> CodeType:
        print('\n\n\n\n\n\n\ncompiling!\n\n\n\n\n\n\n')

        if isinstance(data, AstNodeType):
            # already valid Python AST
            return super().source_to_code(data, path)

        if not isinstance(data, str):
            data = decode_source(data)

        from pyx_lang.compiler.compile import to_python

        data = to_python(data)
                
        return super().source_to_code(data, path)

    def exec_module(self, module: ModuleType) -> None:

        # inject the namespace for creating nodes, etc
        from pyx_lang.hooks import _pyx_
        module._pyx_ = _pyx_ # type: ignore
        
        return super().exec_module(module)

