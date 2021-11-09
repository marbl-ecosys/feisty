"""
This module operates like a local Sphinx extension,
but the only thing it does is add the book directory
to PYTHONPATH, thereby enabling Sphinx to import
the modules that are in the API documentation included
in the book.

This is accomlished by adding the following to _config.yml:

  local_extensions:
    add_book_to_path: ../

"""
import sys

sys.path.append('./docs/source')


def setup(dummy):
    pass
