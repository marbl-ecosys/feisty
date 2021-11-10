================================
Contributing to the FEISTY model
================================

Contributions are highly welcomed and appreciated.

The following sections cover some general guidelines
regarding development in feisty for maintainers and contributors.
Nothing here is set in stone and can't be changed.
Feel free to suggest improvements or changes in the workflow.


.. _pull-requests:

Developing FEISTY
-----------------

The best approach to contributing to the FEISTY code or documentation is to fork the repository, clone your fork locally, apply your edits, push back to your fork, then submit a pull request on GitHub.

Here are some detailed instructions describing these steps.

#. Fork the
   `feisty GitHub repository <https://github.com/marbl-ecosys/feisty>`__.  It's
   fine to use ``feisty`` as your fork repository name because it will live
   under your user.

#. Clone your fork locally using `git <https://git-scm.com/>`_ and create a branch::

    $ git clone git@github.com:YOUR_GITHUB_USERNAME/feisty.git
    $ cd feisty

#. Consider creating your own branch off "main"::

    $ git checkout -b your-bugfix-feature-branch-name main


#. Install `pre-commit <https://pre-commit.com>`_ and its hook on the feisty repo::

     $ pip install --user pre-commit
     $ pre-commit install

   Afterwards ``pre-commit`` will run whenever you commit.

   https://pre-commit.com/ is a framework for managing and maintaining multi-language pre-commit hooks
   to ensure code-style and code formatting is consistent.

#. Install dependencies into a new conda environment::

    $ conda env update -f ci/environment.yml


#. Run all the tests

   Now running tests is as simple as issuing this command::

    $ conda activate sandbox-devel
    $ pytest --junitxml=test-reports/junit.xml --cov=./ --verbose


   This command will run tests via the "pytest" tool against the latest Python version.

#. You can now edit your local working copy and run the tests again as necessary. Please follow PEP-8 for naming.

   When committing, ``pre-commit`` will re-format the files if necessary.

#. Commit and push once your tests pass and you are happy with your change(s)::

    $ git commit -a -m "<commit message>"
    $ git push -u


#. Finally, submit a pull request through the GitHub website using this data::

    head-fork: YOUR_GITHUB_USERNAME/feisty
    compare: your-branch-name

    base-fork: NCAR/feisty
    base: master


.. _documentation:

Write documentation
-------------------

Documentation is critical to enabling a robust, usable code base. Documentation files are is in the ``docs/`` directory and are built with the `Jupyter Book <https://jupyterbook.org/intro.html>`_ utility. Documentation files can be Jupyter Notebooks, `MyST Markdown <https://myst-parser.readthedocs.io/en/latest/>`_, or `reStructuredText <https://docutils.sourceforge.io/rst.html>`_. The layout of the documentation is expressed in ``docs/_toc.yml``.

Documentation is built automatically using GitHub Actions.

You can edit documentation files directly in the GitHub web interface,
without using a local copy.  This can be convenient for small fixes.

.. note::
    Build the documentation locally with the following command:

    .. code:: bash

        $ conda env update -f ci/environment.yml
        $  jupyter-book build docs/ --all

    The built documentation should be available in the ``docs/_build/`` and can be viewed using a local browser at::

      file:///path/to/feisty/docs/_build/html/index.html

.. _submitfeedback:

Feature requests and feedback
-----------------------------

Feature requests, bug reports and questions can be submitted via the GitHub `issue tracker <https://github.com/marbl-ecosys/feisty/issues>`_.
