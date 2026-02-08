"""Comprehensive C++ vs Python reference comparison tests.

Compares the C++ TMM extension against the pure-Python reference
implementation (TmmPy) across many parameter combinations:

- TIR sweep: various n1/n2/wl/polarization
- SPP sweep: semi-infinite metal layers
- Anisotropic sweep: extensive psi/xi orientation coverage
- SPP and anisotropic field profiles
"""

import numpy as np
import pytest
from _general_tmm_py import TmmPy

from GeneralTmm import Material, Tmm

# ---------------------------------------------------------------------------
# Silver refractive-index data — Johnson & Christy (1972)
# ---------------------------------------------------------------------------

WLS_AG = np.array(
    [
        4.0e-07, 4.2e-07, 4.4e-07, 4.6e-07, 4.8e-07,
        5.0e-07, 5.2e-07, 5.4e-07, 5.6e-07, 5.8e-07,
        6.0e-07, 6.2e-07, 6.4e-07, 6.6e-07, 6.8e-07,
        7.0e-07, 7.2e-07, 7.4e-07, 7.6e-07, 7.8e-07,
        8.0e-07, 8.2e-07, 8.4e-07, 8.6e-07, 8.8e-07,
        9.0e-07, 9.2e-07, 9.4e-07, 9.6e-07, 9.8e-07,
    ]
)  # fmt: skip

NS_AG = np.array(
    [
        0.050 + 2.104j, 0.046 + 2.348j, 0.040 + 2.553j, 0.044 + 2.751j, 0.050 + 2.948j,
        0.050 + 3.131j, 0.050 + 3.316j, 0.057 + 3.505j, 0.057 + 3.679j, 0.051 + 3.841j,
        0.055 + 4.010j, 0.059 + 4.177j, 0.055 + 4.332j, 0.050 + 4.487j, 0.045 + 4.645j,
        0.041 + 4.803j, 0.037 + 4.960j, 0.033 + 5.116j, 0.031 + 5.272j, 0.034 + 5.421j,
        0.037 + 5.570j, 0.040 + 5.719j, 0.040 + 5.883j, 0.040 + 6.048j, 0.040 + 6.213j,
        0.040 + 6.371j, 0.040 + 6.519j, 0.040 + 6.667j, 0.040 + 6.815j, 0.040 + 6.962j,
    ]
)  # fmt: skip

SWEEP_KEYS = ["R11", "R22", "R12", "R21", "T31", "T32", "T41", "T42"]


def _prepare_tmm_pair(wl, layers):
    """Create matching Tmm (C++) and TmmPy (Python) instances."""
    tmm = Tmm()
    tmm.SetParams(wl=wl)
    tmm_py = TmmPy()
    tmm_py.SetConf(wl=wl)

    for layer in layers:
        if layer[0] == "iso":
            _, d, n = layer
            tmm.AddIsotropicLayer(d, Material.Static(n))
            tmm_py.AddIsotropicLayer(d, n)
        else:
            _, d, n1, n2, n3, psi, xi = layer
            tmm.AddLayer(d, Material.Static(n1), Material.Static(n2), Material.Static(n3), psi, xi)
            tmm_py.AddLayer(d, n1, n2, n3, psi, xi)

    return tmm, tmm_py


def _assert_sweep_match(wl, layers, betas, pol):
    """Assert that C++ and Python sweeps produce identical results."""
    tmm, tmm_py = _prepare_tmm_pair(wl, layers)
    res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
    res_py = tmm_py.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)
    for k in SWEEP_KEYS:
        np.testing.assert_allclose(res[k], res_py[k], rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(res["enh"], res_py["enh"], rtol=1e-7, atol=1e-14)


def _assert_fields_match(wl, beta, layers, xs, pol):
    """Assert that C++ and Python 1D field profiles match."""
    tmm, tmm_py = _prepare_tmm_pair(wl, layers)
    tmm.beta = beta
    E, H = tmm.CalcFields1D(xs, np.array(pol))
    tmm_py.Solve(wl, beta)
    E_py, H_py = tmm_py.CalcFields1D(xs, pol)

    for i in range(3):
        np.testing.assert_allclose(abs(E[:, i]), abs(E_py[:, i]), rtol=1e-7, atol=1e-15)
        np.testing.assert_allclose(abs(H[:, i]), abs(H_py[:, i]), rtol=1e-7, atol=1e-15)


# ---------------------------------------------------------------------------
# TIR sweep
# ---------------------------------------------------------------------------


class TestTIRSweep:
    """TIR sweep across wavelength, refractive index, and polarization."""

    @pytest.mark.parametrize("wl", [400e-9, 800e-9])
    @pytest.mark.parametrize("n1", [1.2, 1.5])
    @pytest.mark.parametrize("n2", [1.0, 2.0])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_tir_sweep(self, wl, n1, n2, pol):
        layers = [("iso", float("inf"), n1), ("iso", float("inf"), n2)]
        betas = np.linspace(0.0, n1 - 1e-3, 30)
        _assert_sweep_match(wl, layers, betas, pol)


# ---------------------------------------------------------------------------
# SPP sweep (semi-infinite metal)
# ---------------------------------------------------------------------------


class TestSPPSweep:
    """SPP sweep with semi-infinite metal layer."""

    @pytest.mark.parametrize("wl", [400e-9, 500e-9, 800e-9])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_spp_sweep(self, wl, pol):
        silver = Material(WLS_AG, NS_AG)
        layers = [("iso", float("inf"), 1.5), ("iso", float("inf"), silver(wl)), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.0, 1.499, 30)
        _assert_sweep_match(wl, layers, betas, pol)


# ---------------------------------------------------------------------------
# Anisotropic sweep
# ---------------------------------------------------------------------------

_PSI_VALUES = np.linspace(0.0, 2 * np.pi, 5).tolist()
_XI_VALUES = np.linspace(0.0, np.pi, 5).tolist()


class TestAnisotropicSweep:
    """Anisotropic crystal sweep across orientations."""

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    @pytest.mark.parametrize("psi", _PSI_VALUES)
    @pytest.mark.parametrize("xi", _XI_VALUES)
    def test_semi_infinite(self, pol, psi, xi):
        """Semi-infinite anisotropic substrate."""
        wl = 500e-9
        layers = [("iso", float("inf"), 1.5), ("aniso", float("inf"), 1.76, 1.8, 1.9, psi, xi)]
        betas = np.linspace(0.0, 1.5 - 1e-3, 30)
        _assert_sweep_match(wl, layers, betas, pol)

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    @pytest.mark.parametrize("psi", _PSI_VALUES)
    @pytest.mark.parametrize("xi", _XI_VALUES)
    def test_finite_layer(self, pol, psi, xi):
        """Finite anisotropic layer sandwiched between isotropic media."""
        wl = 500e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 200e-9, 1.76, 1.8, 1.9, psi, xi),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.5 - 1e-3, 30)
        _assert_sweep_match(wl, layers, betas, pol)


# ---------------------------------------------------------------------------
# Field profiles
# ---------------------------------------------------------------------------


class TestSPPFields:
    """SPP field profile comparison."""

    @pytest.mark.parametrize("wl", [400e-9, 500e-9, 800e-9])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_spp_fields(self, wl, pol):
        silver = Material(WLS_AG, NS_AG)
        layers = [("iso", float("inf"), 1.5), ("iso", float("inf"), silver(wl)), ("iso", float("inf"), 1.0)]
        xs = np.linspace(-1e-6, 1e-6, 1000)
        _assert_fields_match(wl, 0.5, layers, xs, pol)


class TestAnisotropicFields:
    """Anisotropic field profile comparison."""

    @pytest.mark.parametrize("wl", [400e-9, 500e-9, 800e-9])
    @pytest.mark.parametrize("psi", _PSI_VALUES)
    @pytest.mark.parametrize("xi", _XI_VALUES)
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_aniso_fields(self, wl, psi, xi, pol):
        layers = [("iso", float("inf"), 1.5), ("aniso", float("inf"), 1.76, 1.8, 1.9, psi, xi)]
        xs = np.linspace(-1e-6, 1e-6, 100)
        _assert_fields_match(wl, 0.5, layers, xs, pol)
