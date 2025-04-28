from sys import meta_path

from pyx_lang.importer.pyx_importer import pyx_importer

_installed = False

def install():
    """
    Installs the import hook, allowing you to import pyx files.
    Typically, this is called automatically when Python starts.
    """
    global _installed
    
    if not _installed:
        _installed = True
        meta_path.insert(0, pyx_importer)

