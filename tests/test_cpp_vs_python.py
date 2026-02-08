"""C++ vs Python reference-implementation comparison tests.

Every test builds matching Tmm (C++) and TmmPy (Python) instances and
asserts their outputs are numerically identical.
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

ALL_KEYS = ["R11", "R22", "R12", "R21", "T31", "T32", "T41", "T42"]

# Orientation angle grids for anisotropic parametrize
_PSI_VALUES = np.linspace(0.0, 2 * np.pi, 5).tolist()
_XI_VALUES = np.linspace(0.0, np.pi, 5).tolist()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


def _assert_sweep_match(wl, layers, betas, pol, *, enh_interface=-1, enh_dist=0.0, keys=ALL_KEYS):
    """Assert that C++ and Python sweeps produce identical results."""
    tmm, tmm_py = _prepare_tmm_pair(wl, layers)
    res = tmm.Sweep("beta", betas, (pol, enh_interface, enh_dist))
    res_py = tmm_py.SolveFor("beta", betas, polarization=pol, enhInterface=enh_interface, enhDist=enh_dist)
    for k in keys:
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


@pytest.fixture
def silver():
    """Silver material with full wavelength data."""
    return Material(WLS_AG, NS_AG)


# ---------------------------------------------------------------------------
# 1. Simple interfaces
# ---------------------------------------------------------------------------


class TestNormalIncidence:
    """Fresnel reflection at normal incidence (beta = 0) — simplest case."""

    @pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.0, 2.0), (1.5, 2.5)])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_fresnel_reflection(self, n1, n2, pol):
        wl = 532e-9
        layers = [("iso", float("inf"), n1), ("iso", float("inf"), n2)]
        betas = np.array([0.0])
        _assert_sweep_match(wl, layers, betas, pol, keys=["R11", "R22", "T31", "T42"])

    @pytest.mark.parametrize("wl", [400e-9, 532e-9, 633e-9, 800e-9])
    def test_wavelength_dependence(self, wl):
        layers = [("iso", float("inf"), 1.5), ("iso", float("inf"), 1.0)]
        betas = np.array([0.0])
        _assert_sweep_match(wl, layers, betas, (1.0, 0.0), keys=["R11", "T31"])


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
# 2. Thin-film stacks (dielectric)
# ---------------------------------------------------------------------------


class TestThinFilmStack:
    """Dielectric thin-film and multilayer stacks."""

    @pytest.mark.parametrize("thickness", [50e-9, 100e-9, 200e-9])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_dielectric_thin_film(self, thickness, pol):
        wl = 532e-9
        layers = [("iso", float("inf"), 1.0), ("iso", thickness, 1.5), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.0, 0.99, 20)
        _assert_sweep_match(wl, layers, betas, pol, keys=["R11", "R22", "T31", "T42"])

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_multilayer_stack(self, pol):
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 100e-9, 2.0),
            ("iso", 150e-9, 1.8),
            ("iso", 80e-9, 1.6),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 30)
        _assert_sweep_match(wl, layers, betas, pol)


# ---------------------------------------------------------------------------
# 3. Metallic (absorbing) films and SPP
# ---------------------------------------------------------------------------


class TestSPPSweep:
    """SPP sweep with semi-infinite metal layer."""

    @pytest.mark.parametrize("wl", [400e-9, 500e-9, 800e-9])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_spp_sweep(self, wl, pol, silver):
        layers = [("iso", float("inf"), 1.5), ("iso", float("inf"), silver(wl)), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.0, 1.499, 30)
        _assert_sweep_match(wl, layers, betas, pol)


class TestMetallicFilms:
    """Metallic thin films at various thicknesses and SPP resonances."""

    @pytest.mark.parametrize("thickness", [20e-9, 50e-9, 100e-9])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_metal_film_sweep(self, silver, thickness, pol):
        wl = 600e-9
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", thickness, n_ag), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.0, 1.4, 30)
        _assert_sweep_match(wl, layers, betas, pol, keys=["R11", "R22", "T31", "T42"])

    @pytest.mark.parametrize("wl", [500e-9, 600e-9, 700e-9, 800e-9])
    def test_spp_resonance(self, silver, wl):
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_ag), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.8, 1.4, 50)
        _assert_sweep_match(wl, layers, betas, (1.0, 0.0), enh_interface=2, keys=["R11"])


# ---------------------------------------------------------------------------
# 4. Anisotropic layers
# ---------------------------------------------------------------------------


class TestAnisotropicSweep:
    """Anisotropic crystal sweep across many orientations."""

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    @pytest.mark.parametrize("psi", _PSI_VALUES)
    @pytest.mark.parametrize("xi", _XI_VALUES)
    def test_semi_infinite(self, pol, psi, xi):
        wl = 500e-9
        layers = [("iso", float("inf"), 1.5), ("aniso", float("inf"), 1.76, 1.8, 1.9, psi, xi)]
        betas = np.linspace(0.0, 1.5 - 1e-3, 30)
        _assert_sweep_match(wl, layers, betas, pol)

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    @pytest.mark.parametrize("psi", _PSI_VALUES)
    @pytest.mark.parametrize("xi", _XI_VALUES)
    def test_finite_layer(self, pol, psi, xi):
        wl = 500e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 200e-9, 1.76, 1.8, 1.9, psi, xi),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.5 - 1e-3, 30)
        _assert_sweep_match(wl, layers, betas, pol)

    @pytest.mark.parametrize("psi", [0.0, np.pi / 4, np.pi / 2, np.pi])
    @pytest.mark.parametrize("xi", [0.0, np.pi / 6, np.pi / 3])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_uniaxial_crystal(self, psi, xi, pol):
        """Uniaxial crystal (nx = ny != nz) at various orientations."""
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 500e-9, 1.55, 1.55, 1.65, psi, xi),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.0, 1.4, 20)
        _assert_sweep_match(wl, layers, betas, pol)

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_biaxial_crystal(self, pol):
        """Biaxial crystal (all three n different)."""
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 300e-9, 1.50, 1.55, 1.60, np.pi / 4, np.pi / 6),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 20)
        _assert_sweep_match(wl, layers, betas, pol, keys=["R11", "R22", "T31", "T42"])


class TestCrossPolarization:
    """Cross-polarization terms (R12, R21, T32, T41) in anisotropic media."""

    @pytest.mark.parametrize("psi", [np.pi / 8, np.pi / 4, 3 * np.pi / 8])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_finite_layer(self, psi, pol):
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 200e-9, 1.76, 1.8, 1.9, psi, np.pi / 6),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 25)
        _assert_sweep_match(wl, layers, betas, pol, keys=["R12", "R21", "T32", "T41"])

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_thick_crystal(self, pol):
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 1000e-9, 1.55, 1.60, 1.65, np.pi / 3, np.pi / 5),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 30)
        _assert_sweep_match(wl, layers, betas, pol)


# ---------------------------------------------------------------------------
# 5. Field profiles
# ---------------------------------------------------------------------------


class TestFieldProfiles:
    """1D electromagnetic field profile comparisons."""

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_isotropic_finite_layer(self, pol):
        wl = 532e-9
        xs = np.linspace(-500e-9, 500e-9, 100)
        layers = [("iso", float("inf"), 1.5), ("iso", 200e-9, 1.8), ("iso", float("inf"), 1.0)]
        _assert_fields_match(wl, 0.3, layers, xs, pol)

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_anisotropic_finite_layer(self, pol):
        wl = 532e-9
        xs = np.linspace(-500e-9, 500e-9, 80)
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 300e-9, 1.55, 1.58, 1.62, np.pi / 6, np.pi / 4),
            ("iso", float("inf"), 1.0),
        ]
        _assert_fields_match(wl, 0.3, layers, xs, pol)

    @pytest.mark.parametrize("beta", [0.0, 0.3, 0.6, 0.9])
    def test_beta_dependence(self, beta):
        wl = 532e-9
        xs = np.linspace(-300e-9, 300e-9, 50)
        layers = [("iso", float("inf"), 1.5), ("iso", 150e-9, 2.0), ("iso", float("inf"), 1.3)]
        _assert_fields_match(wl, beta, layers, xs, (1.0, 0.0))

    @pytest.mark.parametrize("wl", [400e-9, 500e-9, 800e-9])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_spp_fields(self, wl, pol, silver):
        layers = [("iso", float("inf"), 1.5), ("iso", float("inf"), silver(wl)), ("iso", float("inf"), 1.0)]
        xs = np.linspace(-1e-6, 1e-6, 1000)
        _assert_fields_match(wl, 0.5, layers, xs, pol)

    @pytest.mark.parametrize("wl", [400e-9, 500e-9, 800e-9])
    @pytest.mark.parametrize("psi", _PSI_VALUES)
    @pytest.mark.parametrize("xi", _XI_VALUES)
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_anisotropic_sweep(self, wl, psi, xi, pol):
        layers = [("iso", float("inf"), 1.5), ("aniso", float("inf"), 1.76, 1.8, 1.9, psi, xi)]
        xs = np.linspace(-1e-6, 1e-6, 100)
        _assert_fields_match(wl, 0.5, layers, xs, pol)


# ---------------------------------------------------------------------------
# 6. Enhancement
# ---------------------------------------------------------------------------


class TestEnhancement:
    """Field enhancement at interfaces."""

    @pytest.mark.parametrize("enh_interface", [2, -1], ids=["film", "substrate"])
    def test_at_interface(self, silver, enh_interface):
        wl = 600e-9
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_ag), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.9, 1.3, 30)
        _assert_sweep_match(wl, layers, betas, (1.0, 0.0), enh_interface=enh_interface)

    def test_spp_near_resonance(self, silver):
        """Enhancement near SPP resonance at restricted beta range."""
        wl = 600e-9
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_ag), ("iso", float("inf"), 1.0)]
        betas = np.linspace(1.01, 1.3, 30)
        _assert_sweep_match(wl, layers, betas, (1.0, 0.0), enh_interface=2)
