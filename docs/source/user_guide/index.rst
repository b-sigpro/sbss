User Guide
==========

Installation
------------
Pip install (inference only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For inference only, install the toolkit from PyPI:

.. code-block:: bash

   pip install git+https://github.com/b-sigpro/sbss

Pixi install (training)
~~~~~~~~~~~~~~~~~~~~~~~
If you want to train models or develop SBSS, follow these steps:

1. (If needed) Set up Pixi by following the
   `official installation guide <https://pixi.prefix.dev/latest/>`_.

2. Clone the repo:

   .. code-block:: bash

      git clone github.com:b-sigpro/sbss

3. Adjust the MPI-related entries in ``pyproject.toml`` to match your environment:

   .. code-block:: toml

      openmpi = { version = "4.1.*", build = "external_*" }
      hdf5 = { version = "*", build = "mpi_openmpi*" }
      h5py = { version = "*", build = "mpi_openmpi*" }

   -  If you do not have MPI installed locally (e.g., on a laptop or a plain AWS instance),
      remove the ``build = "external_*"`` part so Pixi can install OpenMPI for you.
            
      .. code-block:: toml

         openmpi = { version = "4.1.*" }
         hdf5 = { version = "*", build = "mpi_openmpi*" }
         h5py = { version = "*", build = "mpi_openmpi*" }

   -  If you are on a cluster, check the available MPI version via ``module available``
      or ``mpirun --version``, and set the version of ``mpi`` (or another MPI provider) accordingly.
      The default MPI is ``openmpi``, but if your environment uses another one (e.g., MPICH),
      replace the package and build selectors as needed.
         
      .. code-block:: toml

         mpich = { version = "4.3.*" }
         hdf5 = { version = "*", build = "mpi_mpich*" }
         h5py = { version = "*", build = "mpi_mpich*" }

   .. note::
      OpenMPI is ABI-compatible within the same major version, so pick the closest version
      even if the exact one is not available.
      For example, on ABCI 3.0, HPC-X 2.20 provides OpenMPI 4.1.7, but keeping ``openmpi = "4.1.*"``
      lets Pixi install 4.1.6, and it works without issues.

4. Install dependencies with Pixi:

   .. code-block:: bash

      pixi install

Recommended reading
-------------------

SBSS is built on top of several ``aiaccel`` components (``aiaccel.job``,
``aiaccel.config``, and ``aiaccel.torch``). For details on submitting jobs,
configuring experiments, and using the Lightning integration layer, refer to the
`aiaccel user guide <https://aiaccel.readthedocs.io/en/latest/user_guide/>`_
before diving into the SBSS-specific sections.
