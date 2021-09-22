.. image:: https://img.shields.io/github/workflow/status/NCAR/feisty/CI?logo=github&style=for-the-badge
    :target: https://github.com/NCAR/feisty/actions
    :alt: GitHub Workflow CI Status

.. image:: https://img.shields.io/github/workflow/status/NCAR/feisty/code-style?label=Code%20Style&style=for-the-badge
    :target: https://github.com/NCAR/feisty/actions
    :alt: GitHub Workflow Code Style Status

.. image:: https://img.shields.io/codecov/c/github/NCAR/feisty.svg?style=for-the-badge
    :target: https://codecov.io/gh/NCAR/feisty

.. If you want the following badges to be visible, please remove this line, and unindent the lines below
    .. image:: https://img.shields.io/readthedocs/feisty/latest.svg?style=for-the-badge
        :target: https://feisty.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

    .. image:: https://img.shields.io/pypi/v/feisty.svg?style=for-the-badge
        :target: https://pypi.org/project/feisty
        :alt: Python Package Index

    .. image:: https://img.shields.io/conda/vn/conda-forge/feisty.svg?style=for-the-badge
        :target: https://anaconda.org/conda-forge/feisty
        :alt: Conda Version


feisty
======

Development
------------

For a development install, do the following in the repository directory:

.. code-block:: bash

    conda env update -f ci/environment.yml
    conda activate sandbox-devel
    python -m pip install -e .

Also, please install `pre-commit` hooks from the root directory of the created project by running::

      pre-commit install

These code style pre-commit hooks (black, isort, flake8, ...) will run every time you are about to commit code.

.. If you want the following badges to be visible, please remove this line, and unindent the lines below
    Re-create notebooks with Pangeo Binder
    --------------------------------------

    Try notebooks hosted in this repo on Pangeo Binder. Note that the session is ephemeral.
    Your home directory will not persist, so remember to download your notebooks if you
    made changes that you need to use at a later time!

    .. image:: https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=GCE+us-central1&color=blue&style=for-the-badge
        :target: https://binder.pangeo.io/v2/gh/NCAR/feisty/master?urlpath=lab
        :alt: Binder
