import os
import sys

from . import conftest


def test_add_to_path():
    import add_book_to_path

    assert conftest.path_to_here.replace('tests', 'docs') in sys.path
