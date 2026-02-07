[![PyPI version](https://badge.fury.io/py/GeneralTmm.svg)](https://badge.fury.io/py/GeneralTmm)
[![Pytest](https://github.com/ardiloot/GeneralTmm/actions/workflows/pytest.yml/badge.svg)](https://github.com/ardiloot/GeneralTmm/actions/workflows/pytest.yml)
[![Build and upload to PyPI](https://github.com/ardiloot/GeneralTmm/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/ardiloot/GeneralTmm/actions/workflows/publish-to-pypi.yml)

# General 4×4 Transfer-Matrix Method (TMM)

A Python library for optical simulations of **isotropic and anisotropic multilayer structures** using the 4×4 transfer-matrix method, based on Hodgkinson, Kassam & Wu (1997), *Journal of Computational Physics*, 133(1), 75–83.

<p align="center">
  <img src="docs/images/spp_fields_2d.png" alt="2D electromagnetic field map of surface plasmons" width="700">
</p>

## Features

- **Isotropic and anisotropic (birefringent) layers** — full 4×4 matrix for uniaxial/biaxial crystals with arbitrary orientation
- **Parameter sweeps** — over wavelength, angle (β), layer thickness, refractive index, and crystal rotation angles
- **1D and 2D electromagnetic field profiles** — E and H field distributions through the structure
- **Field enhancement and optimization** — built-in simplex optimizer to find resonance conditions (e.g. SPP)
- **Wavelength-dependent materials** — interpolated from measured optical data
- **Cross-polarization coefficients** — R₁₂, R₂₁, T₃₂, T₄₁ for polarization coupling in anisotropic media
- **High performance** — C++ core (Eigen) with Cython bindings
- **Cross-platform wheels** — Linux, Windows, macOS; Python 3.10–3.14

## Installation

```bash
pip install GeneralTmm
```

Pre-built wheels are available for most platforms. A C++ compiler is only needed when installing from source.

## Quick Start

Simulate total internal reflection at a glass/air interface:

```python
import numpy as np
from GeneralTmm import Tmm, Material

# Materials: glass prism and air
prism = Material.Static(1.5)
substrate = Material.Static(1.0)

# Set up TMM solver at 532 nm wavelength
tmm = Tmm(wl=532e-9)
tmm.AddIsotropicLayer(float("inf"), prism)      # semi-infinite prism
tmm.AddIsotropicLayer(float("inf"), substrate)   # semi-infinite air

# Sweep over effective mode index beta = n * sin(theta)
betas = np.linspace(0.0, 1.49, 100)
result = tmm.Sweep("beta", betas)

# Reflection coefficients for p- and s-polarization
R_p = result["R11"]  # p → p reflection
R_s = result["R22"]  # s → s reflection
```

<p align="center">
  <img src="docs/images/tir_reflection.png" alt="Total internal reflection" width="650">
</p>

## Examples

### Total Internal Reflection — [ExampleTIR.py](Examples/ExampleTIR.py)

Basic two-layer glass/air interface showing the critical-angle transition for both polarizations. A minimal starting example.

### Surface Plasmon Polaritons — [ExampleSPP.py](Examples/ExampleSPP.py)

Kretschmann configuration (glass | 50 nm Ag | air) with wavelength-dependent silver data from Johnson & Christy (1972). Demonstrates reflection sweeps, enhancement optimization, and 1D/2D field visualization.

<p align="center">
  <img src="docs/images/spp_reflection.png" alt="SPP reflection and enhancement" width="600">
</p>
<p align="center">
  <img src="docs/images/spp_fields_1d.png" alt="1D field profile at SPP resonance" width="600">
</p>

### Wave Plates — [ExampleAnisotropic.py](Examples/ExampleAnisotropic.py)

Half-wave and quarter-wave plates simulated as birefringent slabs (Δn = 0.1) at normal incidence. Sweeps the plate rotation angle ξ to show how a HWP fully converts p- to s-polarization at 45°, while a QWP produces circular polarization. A textbook result verified with the full 4×4 method.

<p align="center">
  <img src="docs/images/anisotropic_wave_plates.png" alt="Half-wave and quarter-wave plate polarization conversion" width="600">
</p>

## References

> Hodgkinson, I. J., Kassam, S., & Wu, Q. H. (1997). Eigenequation and free energy optimization of film thicknesses in multicavity systems. *Journal of Computational Physics*, 133(1), 75–83.

## Development

### Setup

```bash
git clone https://github.com/ardiloot/GeneralTmm.git
cd GeneralTmm

# Install uv if not already installed:
# https://docs.astral.sh/uv/getting-started/installation/

# Create venv, build the C++ extension, and install all dependencies
uv sync
```

### Running tests

```bash
uv run pytest -v
```

### Code formatting and linting

[Pre-commit](https://pre-commit.com/) hooks are configured to enforce formatting (ruff, clang-format) and catch common issues. To install the git hook locally:

```bash
uvx pre-commit install
```

To run all checks manually:

```bash
uvx pre-commit run --all-files
```

### Regenerating README images

```bash
uv run python docs/generate_images.py
```

### CI overview

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| [Pytest](.github/workflows/pytest.yml) | Push to `master` / PRs | Tests on {ubuntu, windows, macos} × Python {3.10 – 3.14} |
| [Pre-commit](.github/workflows/pre-commit.yml) | Push to `master` / PRs | Runs ruff, clang-format, and other checks |
| [Publish to PyPI](.github/workflows/publish-to-pypi.yml) | Release published | Builds wheels + sdist via cibuildwheel, uploads to PyPI |
| [Dependabot](.github/dependabot.yml) | Weekly | Keeps GitHub Actions and pip dependencies up to date |

## Releasing

Versioning is handled automatically by [setuptools-scm](https://github.com/pypa/setuptools-scm) from git tags.

1. **Ensure CI is green** on the `master` branch.
2. **Create a new release** on GitHub:
   - Go to [Releases](https://github.com/ardiloot/GeneralTmm/releases) → **Draft a new release**
   - Create a new tag following [PEP 440](https://peps.python.org/pep-0440/) (e.g. `v1.2.0`)
   - Target the `master` branch (or a specific commit on master)
   - Click **Generate release notes** for auto-generated changelog
   - For pre-releases (e.g. `v1.2.0rc1`), check **Set as a pre-release** — these upload to TestPyPI instead of PyPI
3. **Publish the release** — the workflow builds wheels for Linux (x86_64 + aarch64), Windows (AMD64 + ARM64), and macOS (ARM64) and uploads to [PyPI](https://pypi.org/project/GeneralTmm/).

## License

MIT
