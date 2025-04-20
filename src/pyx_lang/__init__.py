
from sys import meta_path

from pyx_lang.importer.pyx_importer import pyx_importer


def install():
    """
    Installs the import hook, allowing you to import pyx files. Typically, this is called automatically when Python starts
    """
    if pyx_importer not in meta_path:
        meta_path.insert(0, pyx_importer)