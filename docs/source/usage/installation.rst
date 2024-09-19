Installation
============

Since we are still in the final development process, the package should be installed by cloning the `repository <https://github.com/Center-for-Health-Data-Science/multiDGD/tree/main>`_ and installing it from source (from the project directory). If you are new to cloning repositories, check out `this post <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_.

We recommend working in an environment like conda (if not already used with scvi-tools):

.. code-block:: bash

    conda create -n multiDGD-env python=3.9 # same as for scvi-tools installation
    conda activate multiDGD-env

After cloning the repository, navigate to the project directory and install the package by running the following command:

.. code-block:: bash
    
    pip install .

.. note::
    The model is compatible with scverse and will soon be installable via pip.

.. warning::
    Please note that the package requires Python version 3.8 or higher.