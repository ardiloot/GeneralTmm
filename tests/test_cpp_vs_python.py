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
        np.testing.assert_allclose(res[k], res_py[k], rtol=1e-7, atol=1e-12)
    if enh_interface >= 0:
        np.testing.assert_allclose(res["enh"], res_py["enh"], rtol=1e-7, atol=1e-12)


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


def _assert_general_sweep_match(
    wl, layers, param, values, pol, *, beta=0.0, enh_interface=-1, enh_dist=0.0, keys=ALL_KEYS
):
    """Assert that C++ and Python sweeps match for any sweep parameter."""
    tmm, tmm_py = _prepare_tmm_pair(wl, layers)
    tmm.beta = beta
    tmm_py.SetConf(beta=beta)
    res = tmm.Sweep(param, values, (pol, enh_interface, enh_dist))
    res_py = tmm_py.SolveFor(param, values, polarization=pol, enhInterface=enh_interface, enhDist=enh_dist)
    for k in keys:
        np.testing.assert_allclose(res[k], res_py[k], rtol=1e-7, atol=1e-12)
    if enh_interface >= 0:
        np.testing.assert_allclose(res["enh"], res_py["enh"], rtol=1e-7, atol=1e-12)


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
# 4. SPP Kretschmann with metal film variations (ExampleSPP)
# ---------------------------------------------------------------------------


class TestSPPKretschmann:
    """SPP in Kretschmann config: enhancement, field profiles, and wl sweep.

    From ExampleSPP — glass / Ag thin film / air.
    """

    @pytest.mark.parametrize("d_metal", [30e-9, 50e-9, 70e-9])
    def test_spp_enhancement_vs_thickness(self, silver, d_metal):
        """Enhancement sweep near SPP resonance for different metal thicknesses."""
        wl = 800e-9
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", d_metal, n_ag), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.9, 1.4, 100)
        _assert_sweep_match(wl, layers, betas, (1.0, 0.0), enh_interface=2, enh_dist=0.0)

    @pytest.mark.parametrize("enh_dist", [0.0, 10e-9, 50e-9])
    def test_enhancement_at_distance(self, silver, enh_dist):
        """Enhancement at various distances from the metal/air interface."""
        wl = 600e-9
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_ag), ("iso", float("inf"), 1.0)]
        # Start above beta = 1.0 (air critical angle) to avoid eigenvalue zeros
        betas = np.linspace(1.01, 1.3, 100)
        _assert_sweep_match(wl, layers, betas, (1.0, 0.0), enh_interface=2, enh_dist=enh_dist)

    @pytest.mark.parametrize("wl", [500e-9, 700e-9, 900e-9])
    def test_spp_fields_with_metal_film(self, silver, wl):
        """1D fields through the prism/metal-film/air stack."""
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_ag), ("iso", float("inf"), 1.0)]
        xs = np.linspace(-0.5e-6, 2e-6, 500)
        _assert_fields_match(wl, 1.05, layers, xs, (1.0, 0.0))

    def test_spp_wl_sweep(self):
        """Wavelength sweep at a fixed beta near SPP resonance (static metal)."""
        n_metal = 0.05 + 4.0j
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_metal), ("iso", float("inf"), 1.0)]
        wls = np.linspace(400e-9, 900e-9, 100)
        _assert_general_sweep_match(
            600e-9,
            layers,
            "wl",
            wls,
            (1.0, 0.0),
            beta=1.05,
        )


# ---------------------------------------------------------------------------
# 5. Enhancement
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


# ---------------------------------------------------------------------------
# 6. Dielectric multilayer filters (ExampleFilter)
# ---------------------------------------------------------------------------


class TestBraggMirror:
    """Bragg mirror: (HL)^N H — wavelength sweep against stopband.

    From ExampleFilter — TiO2/SiO2 quarter-wave pairs on BK7 glass.
    """

    WL0 = 550e-9
    N_H, N_L = 2.30, 1.46  # TiO2 / SiO2
    N_SUB = 1.52  # BK7

    @staticmethod
    def _bragg_layers(n_h, n_l, n_sub, wl0, n_pairs):
        d_h = wl0 / (4 * n_h)
        d_l = wl0 / (4 * n_l)
        layers = [("iso", float("inf"), n_sub)]
        for _ in range(n_pairs):
            layers.append(("iso", d_h, n_h))
            layers.append(("iso", d_l, n_l))
        layers.append(("iso", d_h, n_h))
        layers.append(("iso", float("inf"), 1.0))
        return layers

    @pytest.mark.parametrize("n_pairs", [3, 5, 7])
    def test_bragg_wl_sweep(self, n_pairs):
        """Wavelength sweep across the Bragg stopband."""
        layers = self._bragg_layers(self.N_H, self.N_L, self.N_SUB, self.WL0, n_pairs)
        wls = np.linspace(350e-9, 800e-9, 100)
        _assert_general_sweep_match(
            self.WL0,
            layers,
            "wl",
            wls,
            (1.0, 0.0),
            beta=0.0,
            keys=["R11", "R22", "T31", "T42"],
        )

    def test_bragg_beta_sweep(self):
        """Beta sweep at the design wavelength."""
        layers = self._bragg_layers(self.N_H, self.N_L, self.N_SUB, self.WL0, 5)
        betas = np.linspace(0.0, 1.4, 100)
        _assert_sweep_match(self.WL0, layers, betas, (1.0, 0.0))

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_bragg_polarization(self, pol):
        """Both polarizations through a 7-pair Bragg mirror."""
        layers = self._bragg_layers(self.N_H, self.N_L, self.N_SUB, self.WL0, 7)
        wls = np.linspace(400e-9, 700e-9, 100)
        _assert_general_sweep_match(self.WL0, layers, "wl", wls, pol, beta=0.0)

    def test_bragg_fields(self):
        """1D field profile through a Bragg mirror at the design wavelength."""
        layers = self._bragg_layers(self.N_H, self.N_L, self.N_SUB, self.WL0, 5)
        xs = np.linspace(-500e-9, 1500e-9, 300)
        _assert_fields_match(self.WL0, 0.0, layers, xs, (1.0, 0.0))


class TestFabryPerot:
    """Fabry-Perot bandpass: (HL)^N H 2L H (LH)^N — narrow transmission peak.

    From ExampleFilter — two Bragg mirrors sandwich a half-wave SiO2 cavity.
    """

    WL0 = 550e-9
    N_H, N_L = 2.30, 1.46
    N_SUB = 1.52

    @staticmethod
    def _fp_layers(n_h, n_l, n_sub, wl0, n_pairs):
        d_h = wl0 / (4 * n_h)
        d_l = wl0 / (4 * n_l)
        layers = [("iso", float("inf"), n_sub)]
        for _ in range(n_pairs):
            layers.append(("iso", d_h, n_h))
            layers.append(("iso", d_l, n_l))
        layers.append(("iso", d_h, n_h))
        layers.append(("iso", 2 * d_l, n_l))  # half-wave cavity
        layers.append(("iso", d_h, n_h))
        for _ in range(n_pairs):
            layers.append(("iso", d_l, n_l))
            layers.append(("iso", d_h, n_h))
        layers.append(("iso", float("inf"), 1.0))
        return layers

    @pytest.mark.parametrize("n_pairs", [3, 5])
    def test_fp_wl_sweep(self, n_pairs):
        """Wavelength sweep showing the narrow transmission peak."""
        layers = self._fp_layers(self.N_H, self.N_L, self.N_SUB, self.WL0, n_pairs)
        wls = np.linspace(400e-9, 700e-9, 150)
        _assert_general_sweep_match(
            self.WL0,
            layers,
            "wl",
            wls,
            (1.0, 0.0),
            beta=0.0,
            keys=["R11", "R22", "T31", "T42"],
        )

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_fp_polarization(self, pol):
        """Both polarizations through a 5-pair Fabry-Perot cavity."""
        layers = self._fp_layers(self.N_H, self.N_L, self.N_SUB, self.WL0, 5)
        wls = np.linspace(450e-9, 650e-9, 100)
        _assert_general_sweep_match(self.WL0, layers, "wl", wls, pol, beta=0.0)

    def test_fp_fields(self):
        """1D field profile showing cavity mode through the structure."""
        layers = self._fp_layers(self.N_H, self.N_L, self.N_SUB, self.WL0, 3)
        xs = np.linspace(-500e-9, 2000e-9, 400)
        _assert_fields_match(self.WL0, 0.0, layers, xs, (1.0, 0.0))


# ---------------------------------------------------------------------------
# 7. Anisotropic layers
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
# 8. Wave plates — optic-axis rotation sweep (ExampleAnisotropic)
# ---------------------------------------------------------------------------


class TestWavePlate:
    """Birefringent wave plates: xi sweep at normal and off-normal incidence.

    From ExampleAnisotropic — half-wave and quarter-wave plates sweeping
    the crystal rotation angle xi.
    """

    N_O, N_E = 1.50, 1.60
    WL = 632.8e-9

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_half_wave_plate_xi_sweep(self, pol):
        """HWP xi sweep: p→s conversion peaks near xi = 45°."""
        d = self.WL / (2 * (self.N_E - self.N_O))
        layers = [
            ("iso", float("inf"), 1.0),
            ("aniso", d, self.N_O, self.N_E, self.N_O, 0.0, 0.0),
            ("iso", float("inf"), 1.0),
        ]
        xi_values = np.linspace(0.0, np.pi, 100)
        _assert_general_sweep_match(self.WL, layers, "xi_1", xi_values, pol, beta=0.0)

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_quarter_wave_plate_xi_sweep(self, pol):
        """QWP xi sweep at normal incidence."""
        d = self.WL / (4 * (self.N_E - self.N_O))
        layers = [
            ("iso", float("inf"), 1.0),
            ("aniso", d, self.N_O, self.N_E, self.N_O, 0.0, 0.0),
            ("iso", float("inf"), 1.0),
        ]
        xi_values = np.linspace(0.0, np.pi, 100)
        _assert_general_sweep_match(self.WL, layers, "xi_1", xi_values, pol, beta=0.0)

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_hwp_off_normal_incidence(self, pol):
        """HWP with beta != 0 to test combined angular + rotation behavior."""
        d = self.WL / (2 * (self.N_E - self.N_O))
        layers = [
            ("iso", float("inf"), 1.0),
            ("aniso", d, self.N_O, self.N_E, self.N_O, 0.0, 0.0),
            ("iso", float("inf"), 1.0),
        ]
        xi_values = np.linspace(0.0, np.pi, 100)
        _assert_general_sweep_match(self.WL, layers, "xi_1", xi_values, pol, beta=0.3)


# ---------------------------------------------------------------------------
# 9. Cholesteric liquid crystal — many anisotropic layers (ExampleCholesteric)
# ---------------------------------------------------------------------------


class TestCholestericLC:
    """Cholesteric LC: helical stack with many anisotropic layers, wl sweep.

    From ExampleCholesteric — a discretized helix of birefringent layers that
    selectively reflects one circular polarization in a Bragg band.
    """

    N_O, N_E = 1.5, 1.7
    PITCH = 350e-9

    @staticmethod
    def _build_cholesteric_layers(n_o, n_e, pitch, n_layers_per_pitch, n_pitches):
        d_layer = pitch / n_layers_per_pitch
        layers = [("iso", float("inf"), 1.0)]
        for i in range(n_layers_per_pitch * n_pitches):
            xi = 2.0 * np.pi * i / n_layers_per_pitch
            layers.append(("aniso", d_layer, n_o, n_e, n_o, 0.0, xi))
        layers.append(("iso", float("inf"), 1.0))
        return layers

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_bragg_band_wl_sweep(self, pol):
        """Wavelength sweep through the cholesteric Bragg band (5 pitches)."""
        layers = self._build_cholesteric_layers(self.N_O, self.N_E, self.PITCH, 10, 5)
        wls = np.linspace(400e-9, 750e-9, 80)
        _assert_general_sweep_match(550e-9, layers, "wl", wls, pol, beta=0.0)

    def test_cross_polarization_in_bragg_band(self):
        """Cross-polarization terms should be significant in cholesteric."""
        layers = self._build_cholesteric_layers(self.N_O, self.N_E, self.PITCH, 10, 5)
        wls = np.linspace(450e-9, 650e-9, 60)
        _assert_general_sweep_match(
            550e-9,
            layers,
            "wl",
            wls,
            (1.0, 0.0),
            beta=0.0,
            keys=["R11", "R12", "R21", "R22", "T31", "T32", "T41", "T42"],
        )

    def test_many_layers(self):
        """10 pitches with 10 layers/pitch (100 anisotropic layers)."""
        layers = self._build_cholesteric_layers(self.N_O, self.N_E, self.PITCH, 10, 10)
        wls = np.linspace(450e-9, 650e-9, 50)
        _assert_general_sweep_match(550e-9, layers, "wl", wls, (1.0, 0.0), beta=0.0)


# ---------------------------------------------------------------------------
# 10. Dyakonov SPP — metal + anisotropic substrate (ExampleDSPP)
# ---------------------------------------------------------------------------


class TestDSPP:
    """Dyakonov SPP: ZnSe / Ag / KTP Kretschmann — beta and xi sweeps.

    From ExampleDSPP — leaky Dyakonov surface plasmon polaritons at
    a metal-birefringent interface.
    """

    WL = 900e-9
    N_O, N_E = 1.740, 1.830  # KTP
    N_AG = 0.040 + 6.371j
    N_PRISM = 2.5  # ZnSe

    @pytest.mark.parametrize("xi_deg", [0, 45, 69, 90])
    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_dspp_beta_sweep(self, xi_deg, pol):
        """Kretschmann DSPP at various optic-axis angles.

        Beta range avoids eigenvalue degeneracies at beta = n_o, n_e.
        """
        layers = [
            ("iso", float("inf"), self.N_PRISM),
            ("iso", 60e-9, self.N_AG),
            ("aniso", float("inf"), self.N_O, self.N_E, self.N_O, 0.0, np.radians(xi_deg)),
        ]
        # Scan 47.5°–49.5° → betas above n_e = 1.83, avoiding eigenvalue zeros
        betas = self.N_PRISM * np.sin(np.radians(np.linspace(47.5, 49.5, 100)))
        _assert_sweep_match(self.WL, layers, betas, pol)

    def test_dspp_xi_sweep(self):
        """Sweep the optic-axis angle itself at a fixed incidence angle."""
        layers = [
            ("iso", float("inf"), self.N_PRISM),
            ("iso", 60e-9, self.N_AG),
            ("aniso", float("inf"), self.N_O, self.N_E, self.N_O, 0.0, 0.0),
        ]
        xi_values = np.radians(np.linspace(0, 90, 100))
        # beta above n_e avoids eigenvalue zeros for all xi
        beta_fixed = self.N_PRISM * np.sin(np.radians(48))
        # Higher atol: cross-pol terms are ~zero; noise differs between implementations
        _assert_general_sweep_match(
            self.WL,
            layers,
            "xi_2",
            xi_values,
            (1.0, 0.0),
            beta=beta_fixed,
        )

    def test_dspp_enhancement(self):
        """Enhancement at the metal/crystal interface."""
        layers = [
            ("iso", float("inf"), self.N_PRISM),
            ("iso", 60e-9, self.N_AG),
            ("aniso", float("inf"), self.N_O, self.N_E, self.N_O, 0.0, np.radians(69)),
        ]
        betas = self.N_PRISM * np.sin(np.radians(np.linspace(47.5, 49.5, 100)))
        _assert_sweep_match(self.WL, layers, betas, (1.0, 0.0), enh_interface=2)

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_dspp_fields(self, pol):
        """1D field profile at DSPP resonance."""
        layers = [
            ("iso", float("inf"), self.N_PRISM),
            ("iso", 60e-9, self.N_AG),
            ("aniso", float("inf"), self.N_O, self.N_E, self.N_O, 0.0, np.radians(69)),
        ]
        xs = np.linspace(-0.5e-6, 1e-6, 500)
        beta = self.N_PRISM * np.sin(np.radians(46))
        _assert_fields_match(self.WL, beta, layers, xs, pol)


# ---------------------------------------------------------------------------
# 11. Field profiles
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
# 12. Polarization variations
# ---------------------------------------------------------------------------


class TestMixedPolarization:
    """45° linear input (equal p + s) — exercises cross-polarization paths."""

    POL_45 = (1.0, 1.0)

    def test_single_interface(self):
        layers = [("iso", float("inf"), 1.5), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.0, 0.99, 100)
        _assert_sweep_match(532e-9, layers, betas, self.POL_45)

    def test_thin_film(self):
        layers = [("iso", float("inf"), 1.0), ("iso", 100e-9, 1.5), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.0, 0.99, 100)
        _assert_sweep_match(532e-9, layers, betas, self.POL_45)

    def test_anisotropic_film(self):
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 200e-9, 1.55, 1.60, 1.65, np.pi / 4, np.pi / 6),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.0, 1.4, 100)
        _assert_sweep_match(532e-9, layers, betas, self.POL_45)

    def test_spp(self, silver):
        wl = 600e-9
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_ag), ("iso", float("inf"), 1.0)]
        # Start above beta = 1.0 to avoid air critical-angle eigenvalue zeros
        betas = np.linspace(1.01, 1.4, 100)
        _assert_sweep_match(wl, layers, betas, self.POL_45)

    def test_dspp(self):
        layers = [
            ("iso", float("inf"), 2.5),
            ("iso", 60e-9, 0.040 + 6.371j),
            ("aniso", float("inf"), 1.74, 1.83, 1.74, 0.0, np.radians(69)),
        ]
        # Scan above n_e to avoid eigenvalue zeros
        betas = 2.5 * np.sin(np.radians(np.linspace(47.5, 49.5, 100)))
        _assert_sweep_match(900e-9, layers, betas, self.POL_45)

    def test_mixed_fields(self):
        """Fields with mixed polarization through an anisotropic structure."""
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 300e-9, 1.55, 1.60, 1.65, np.pi / 4, np.pi / 6),
            ("iso", float("inf"), 1.0),
        ]
        xs = np.linspace(-500e-9, 500e-9, 200)
        _assert_fields_match(532e-9, 0.5, layers, xs, self.POL_45)


class TestArbitraryPolarization:
    """Arbitrary polarization inputs — exercises non-trivial p/s ratios."""

    @pytest.mark.parametrize("pol", [(0.3, 0.7), (0.7, 0.3), (0.1, 0.9), (0.5, 0.8), (1.0, 0.2)])
    def test_single_interface_sweep(self, pol):
        layers = [("iso", float("inf"), 1.5), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.0, 0.99, 100)
        _assert_sweep_match(532e-9, layers, betas, pol)

    @pytest.mark.parametrize("pol", [(0.3, 0.7), (0.7, 0.3), (0.1, 0.9)])
    def test_thin_film_sweep(self, pol):
        layers = [("iso", float("inf"), 1.0), ("iso", 100e-9, 1.5), ("iso", float("inf"), 1.0)]
        betas = np.linspace(0.0, 0.99, 100)
        _assert_sweep_match(532e-9, layers, betas, pol)

    @pytest.mark.parametrize("pol", [(0.3, 0.7), (0.7, 0.3), (0.5, 0.8)])
    def test_anisotropic_film_sweep(self, pol):
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 200e-9, 1.55, 1.60, 1.65, np.pi / 4, np.pi / 6),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.0, 1.4, 100)
        _assert_sweep_match(532e-9, layers, betas, pol)

    @pytest.mark.parametrize("pol", [(0.3, 0.7), (0.7, 0.3), (0.1, 0.9), (0.5, 0.8)])
    def test_fields_isotropic(self, pol):
        """Field profiles with odd polarizations through isotropic layers."""
        layers = [("iso", float("inf"), 1.5), ("iso", 200e-9, 2.0), ("iso", float("inf"), 1.0)]
        xs = np.linspace(-500e-9, 500e-9, 200)
        _assert_fields_match(532e-9, 0.3, layers, xs, pol)

    @pytest.mark.parametrize("pol", [(0.3, 0.7), (0.7, 0.3), (0.5, 0.8)])
    def test_fields_anisotropic(self, pol):
        """Field profiles with odd polarizations through anisotropic film."""
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 300e-9, 1.55, 1.60, 1.65, np.pi / 4, np.pi / 6),
            ("iso", float("inf"), 1.0),
        ]
        xs = np.linspace(-500e-9, 500e-9, 200)
        _assert_fields_match(532e-9, 0.5, layers, xs, pol)

    @pytest.mark.parametrize("pol", [(0.3, 0.7), (0.7, 0.3)])
    def test_spp_sweep(self, silver, pol):
        """SPP beta sweep with odd polarizations."""
        wl = 600e-9
        n_ag = silver(wl)
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_ag), ("iso", float("inf"), 1.0)]
        betas = np.linspace(1.01, 1.4, 100)
        _assert_sweep_match(wl, layers, betas, pol)

    @pytest.mark.parametrize("pol", [(0.3, 0.7), (0.7, 0.3)])
    def test_dspp_sweep(self, pol):
        """DSPP beta sweep with odd polarizations."""
        layers = [
            ("iso", float("inf"), 2.5),
            ("iso", 60e-9, 0.040 + 6.371j),
            ("aniso", float("inf"), 1.74, 1.83, 1.74, 0.0, np.radians(69)),
        ]
        betas = 2.5 * np.sin(np.radians(np.linspace(47.5, 49.5, 100)))
        _assert_sweep_match(900e-9, layers, betas, pol)


# ---------------------------------------------------------------------------
# 13. Wavelength sweep (various structures)
# ---------------------------------------------------------------------------


class TestWavelengthSweep:
    """Wavelength sweep for various structures — tests a sweep dimension
    not covered by the beta-sweep tests above."""

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_single_dielectric_film(self, pol):
        layers = [("iso", float("inf"), 1.0), ("iso", 200e-9, 1.5), ("iso", float("inf"), 1.0)]
        wls = np.linspace(300e-9, 900e-9, 100)
        _assert_general_sweep_match(500e-9, layers, "wl", wls, pol, beta=0.0)

    @pytest.mark.parametrize("pol", [(1.0, 0.0), (0.0, 1.0)])
    def test_multilayer_dielectric(self, pol):
        """4-layer dielectric stack, off-normal incidence."""
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 100e-9, 2.0),
            ("iso", 150e-9, 1.8),
            ("iso", 80e-9, 1.6),
            ("iso", float("inf"), 1.0),
        ]
        wls = np.linspace(350e-9, 800e-9, 100)
        _assert_general_sweep_match(500e-9, layers, "wl", wls, pol, beta=0.3)

    def test_anisotropic_film_wl_sweep(self):
        """Anisotropic film wavelength sweep at oblique incidence."""
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 500e-9, 1.55, 1.60, 1.65, np.pi / 4, np.pi / 6),
            ("iso", float("inf"), 1.0),
        ]
        wls = np.linspace(400e-9, 800e-9, 100)
        _assert_general_sweep_match(600e-9, layers, "wl", wls, (1.0, 0.0), beta=0.3)

    def test_metal_film_wl_sweep(self):
        """Metal film with static n — wavelength sweep changes interference."""
        n_metal = 0.05 + 4.0j
        layers = [("iso", float("inf"), 1.5), ("iso", 50e-9, n_metal), ("iso", float("inf"), 1.0)]
        wls = np.linspace(400e-9, 900e-9, 100)
        # beta = 0.5 avoids the critical-angle eigenvalue zero at beta = n_air = 1.0
        _assert_general_sweep_match(600e-9, layers, "wl", wls, (1.0, 0.0), beta=0.5)
