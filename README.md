[![PyPI version](https://badge.fury.io/py/GeneralTmm.svg)](https://badge.fury.io/py/GeneralTmm)
[![Pytest](https://github.com/ardiloot/GeneralTmm/actions/workflows/pytest.yml/badge.svg)](https://github.com/ardiloot/GeneralTmm/actions/workflows/pytest.yml)
[![Build and upload to PyPI](https://github.com/ardiloot/GeneralTmm/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/ardiloot/GeneralTmm/actions/workflows/publish-to-pypi.yml)

# General 4x4 transfer-matrix method (TMM)

A Python library for 4x4 anisotropic transfer-matrix method (TMM) optical simulations,
based on the algorithm from Hodgkinson, Kassam & Wu (1997), Journal of Computational Physics, 133(1) 75-83.

The computational core is written in C++ (using Eigen) and wrapped via Cython for high performance.

## Installation

```bash
pip install GeneralTmm
```

Pre-built wheels are available for most platforms. A C++ compiler is only needed when installing from source.

## Quick Start

```python
import numpy as np
from GeneralTmm import Tmm, Material

# Define materials
prism = Material.Static(1.5)
substrate = Material.Static(1.0)

# Set up TMM
tmm = Tmm(wl=532e-9)
tmm.AddIsotropicLayer(float("inf"), prism)
tmm.AddIsotropicLayer(float("inf"), substrate)

# Sweep over angles (via effective mode index beta)
betas = np.linspace(0.0, 1.49, 100)
result = tmm.Sweep("beta", betas)

# Access reflection/transmission coefficients
R_p = result["R11"]  # p-polarization reflection
R_s = result["R22"]  # s-polarization reflection
```

See the [Examples](Examples/) directory for more detailed usage including surface plasmon polariton (SPP) calculations and field visualizations.

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
