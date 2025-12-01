Spatialized WSJ0-2mix
=====================

The ``recipes/sp_wsj02mix`` directory provides an end-to-end pipeline for the
Spatialized WSJ0-2mix speech separation benchmark. The recipe automates data
generation, preprocessing, model training, and evaluation on noisy,
reverberant two-speaker mixtures rendered with spatial room impulse responses.

The Makefile is organized into GNU Make stages (``stage0``–``stage5``). Each
stage drops ``.done`` files so you can resume from intermediate results. Run a
stage with ``make stageN`` or execute the complete pipeline via ``make all``.
Important variables such as ``data``, ``duration``, and ``train_path`` can be
overridden on the command line. For example::

    make stage2 duration=96000 train_path=models/nfca/unet

Configurable Make variables
---------------------------

You can inspect the full ``Makefile`` on GitHub:
`recipes/sp_wsj02mix/Makefile <https://github.com/b-sigpro/sbss/blob/main/recipes/sp_wsj02mix/Makefile>`_.

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Configurable
     - Description
   * - ``data``
     - ``derev``
     - ✓
     - Selects which preprocessed data stream (e.g., dereverberated vs. reverberant) to pack into HDF5 files.
   * - ``duration``
     - ``64000``
     - ✓
     - Number of audio samples per excerpt when building the HDF5 dataset.
   * - ``train_path``
     - ``models/nfca/unet``
     - ✓
     - Directory that stores the model training configuration, checkpoints, and job logs.
   * - ``tag``
     - ``nfca``
     - ✓
     - Short identifier used when naming inference runs.
   * - ``inference_name``
     - ``$(tag)_<timestamp>_<rand>``
     - ✓
     - Inference results directory name; automatically generated but can be overridden for resumption.
   * - ``inference_command``
     - ``python -m sbss.nfca.pipelines.separate batch $(train_path) $$src_path $$dst_path``
     - ✓
     - Command executed during Stage 4 for inference; modify to plug in alternative pipelines.
   * - ``inference_path``
     - ``results/$(inference_name)``
     - ✓
     - Output directory for separated signals, logs, and evaluation artifacts.
   * - ``cmd`` / ``job_ops``
     - inherited from ``recipes/globals.mk``
     - ✓
     - Cluster submission command plus optional job arguments; default invokes ``aiaccel-job local``.

Stage-by-stage guide
--------------------

Stage 0: dataset preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``make stage0`` calls ``scripts/0_prepare_dataset.py`` to download/generate the
dry, noisy, and reverberant Spatialized WSJ0-2mix waveforms. Ensure the `hdf5/`
directory exists before launching this step.

Stage 1: preprocessing
~~~~~~~~~~~~~~~~~~~~~~~

``make stage1`` launches two commands per split (``tr``, ``cv``, ``tt``):

- ``scripts/1_add_noise.py`` adds diffuse noise to the clean mixtures
- ``scripts/1_dereverberate.py`` removes the simulated room effect to provide
  auxiliary dereverberated targets

Both commands are submitted through ``$(cmd)`` so they can fan out across job
slots on a cluster.

Stage 2: HDF5 generation
~~~~~~~~~~~~~~~~~~~~~~~~~

``make stage2`` converts the processed audio into chunked HDF5 datasets by
executing ``scripts/2_make_hdf5_unsupervised.py`` with the current ``data`` and
``duration`` arguments. Only the ``tr`` and ``cv`` splits are packaged, as
defined by ``HDF5_SPLITS``.

Stage 3: model training
~~~~~~~~~~~~~~~~~~~~~~~

``make stage3`` triggers ``aiaccel.torch.apps.train`` with the Lightning/Aiaccel
configuration stored under ``$(train_path)/config.yaml``. Before training kicks
off, existing checkpoints and logs are removed after an interactive
confirmation. The step finishes when ``$(train_path)/.train.done`` is created.

Stage 4: inference
~~~~~~~~~~~~~~~~~~~

``make stage4`` separates the ``cv`` and ``tt`` sets by running
``python -m sbss.nfca.pipelines.separate batch`` with the trained checkpoint
specified by ``train_path``. Outputs (and job logs) are stored in
``results/<inference_name>``. Each split produces a stamp file named
``.4_inference.<split>.done`` under the corresponding results directory.

Stage 5: evaluation
~~~~~~~~~~~~~~~~~~~~

``make stage5`` evaluates SDR, STOI, and PESQ via the scripts
``scripts/5_evaluate_sdr.py``, ``scripts/5_evaluate_stoi.py``, and
``scripts/5_evaluate_pesq.py``. After all metric jobs succeed, the ``scores``
target runs ``scripts/summarize_scores.py`` to aggregate the measurements for
the current ``inference_name``.


Configuration file
------------------

The dataset configuration is stored in 
`recipes/sp_wsj02mix/config.yaml <https://github.com/b-sigpro/sbss/blob/main/recipes/sp_wsj02mix/config.yaml>`_,
which inherits ``recipes/globals.yaml``. Key fields include:

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Configurable
     - Description
   * - ``n_mixtures.tr`` / ``n_mixtures.cv`` / ``n_mixtures.tt``
     - ``20000`` / ``5000`` / ``3000``
     - 
     - Numbers of mixtures for the corresponding splits.
   * - ``n_mics``
     - ``4``
     - ✓
     - Microphone channels assumed in STFTs and SCMs.
   * - ``n_fft`` / ``hop_length``
     - ``512`` / ``128``
     - ✓
     - STFT window size and hop length shared by encoders/decoders.
   * - ``snr``
     - ``30``
     - ✓
     - Target SNR (dB) for adding white noise when generating mixtures.
   * - ``wpe.taps`` / ``wpe.delay``
     - ``10`` / ``3``
     - ✓
     - WPE dereverberation hyperparameters used in preprocessing scripts.
   * - ``blacklist``
     - list of WSJ0 filenames
     - ✓
     - Mixtures excluded due to anomalies when rendering Spatialized WSJ0-2mix.
