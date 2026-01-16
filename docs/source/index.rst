.. figure:: _static/logo_sbss.png
  :height: 140px
  :align: center

SBSS Documentation
==================

The Scalable BSS Toolkit (SBSS) provides end-to-end components for blind source
separation research, including neural full-rank spatial covariance analysis
(neural FCA/FastFCA), dataset recipes, and reusable PyTorch Lightning modules.

.. code-block:: bash

   pip install git+https://github.com/b-sigpro/sbss

Key Features
------------

:octicon:`zap;1em` Reproducible Recipes
    End-to-end recipes document every stage (data prep, training, inference, evaluation) for reproducible studies.

:octicon:`cpu;1em` HPC Ready
    Recipes and Makefiles tuned for ABCI, TSUBAME, and other clusters, yet still runnable on a single workstation.

:octicon:`server;1em` Highly Modular
    Lightning tasks, common datasets, models, and utilities built to swap components and run ablations with minimal friction.


Acknowledgments
---------------

- Part of this work was developed under a commissioned project of the New Energy and
  Industrial Technology Development Organization (NEDO).
- Part of this software was developed by using ABCI 3.0 provided by AIST and AIST
  Solutions.
- Part of this software was developed by using the TSUBAME4.0 supercomputer at Institute
  of Science Tokyo.



.. toctree::
    :hidden:
    :maxdepth: 2

    user_guide/index
    recipes/index
    api_reference/index.rst
