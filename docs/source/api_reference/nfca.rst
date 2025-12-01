Neural FCA
==========

This package provides neural blind source separation methods, namely neural full-rank spatial covariance analysis (neural FCA) [Bando2021]_ and neural fast FCA (neural FastFCA) [Bando2023]_.


Tasks
-----

.. currentmodule:: sbss.nfca.tasks

.. autosummary::
    :toctree: generated/

    AviTask
    JdAviTask


Encoders
--------

.. currentmodule:: sbss.nfca.encoders

.. autosummary::
    :toctree: generated/

    DilcnvEncoder
    UNetEncoder
    JdUNetEncoder


Decoders
--------

.. currentmodule:: sbss.nfca.decoders

.. autosummary::
    :toctree: generated/

    ResDecoder
    ResLinearDecoder


Lightning Callbacks
-------------------

.. currentmodule:: sbss.nfca.callbacks

.. autosummary::
    :toctree: generated/

    PsdVisualizerCallback
    XtVisualizerCallback

References
----------

.. [Bando2021] Yoshiaki Bando, Kouhei Sekiguchi, Yoshiki Masuyama, Aditya Arie Nugraha, Mathieu Fontaine, and Kazuyoshi Yoshii, "Neural full-rank spatial covariance analysis for blind source separation," *IEEE Signal Processing Letters*, vol. 28, pp. 1670-1674, 2021. `[PDF] <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9506855>`__
.. [Bando2023] Yoshiaki Bando, Yoshiki Masuyama, Aditya Arie Nugraha, and Kazuyoshi Yoshii, "Neural fast full-rank spatial covariance analysis for blind source separation," in *Proc. 31st European Signal Processing Conference (EUSIPCO)*, pp. 51-55, 2023. `[PDF] <https://arxiv.org/pdf/2306.10240>`__
