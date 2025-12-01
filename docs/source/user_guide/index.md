# User Guide

## Installation
For only inference, you can install SBSS by using pip (or uv) as:
```bash
pip install git+https://github.com/b-sigpro/sbss
```

For development, we recommend to use Pixi for installing the dependencies:
```bash
git clone github.com:b-sigpro/sbss

pixi run -e pre-install build-hdf5
pixi run -e pre-install clean-cache

pixi install
```

## Recommended reading

SBSS is built on top of several `aiaccel` components (``aiaccel.job``,
``aiaccel.config``, and ``aiaccel.torch``). For details on submitting jobs,
configuring experiments, and using the Lightning integration layer, refer to the
[aiaccel user guide](https://aiaccel.readthedocs.io/en/latest/user_guide/)
before diving into the SBSS-specific sections.
