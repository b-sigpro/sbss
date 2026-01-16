<div align="center"><img src="https://raw.githubusercontent.com/b-sigpro/sbss/master/docs/image/logo_sbss.png" width="600"/></div>
<br>
<div align="center">
    <img src="https://img.shields.io/github/license/b-sigpro/sbss.svg" />
    <img src="https://img.shields.io/badge/Python-3.10-blue" />
    <a href="https://sbss.readthedocs.io/en/latest/index.html">
        <img src="https://app.readthedocs.org/projects/sbss/badge/?version=latest" />
    </a>
</div>

# Scalable BSS Toolkit

SBSS is a research-oriented toolkit for scalable blind source separation, including neural FCA/FastFCA models, dataset recipes, and Lightning integrations.

## Key Features

- **Reproducible recipes** – End-to-end pipelines document every stage (data prep, training, inference, evaluation) for reproducible studies.
- **HPC ready** – Recipes and Makefiles are tuned for ABCI, TSUBAME, and other clusters, yet remain runnable on a single workstation.
- **Highly modular** – Lightning tasks, common datasets, models, and utilities are built to swap components and run ablations with minimal friction.

## Getting started
For only inference, you can install SBSS by using pip (or uv) as:
```bash
pip install git+https://github.com/b-sigpro/sbss
```

For development, we recommend to use Pixi for installing the dependencies:
```bash
git clone github.com:b-sigpro/sbss
pixi install
```
Full instructions are provided in https://sbss.readthedocs.io/en/latest/user_guide/index.html

## Acknowledgement
* Part of this software was developed in a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
* Part of this software was developed by using ABCI 3.0 provided by AIST and AIST Solutions.
* Part of this software was developed by using the TSUBAME4.0 supercomputer at Institute of Science Tokyo.
