"""Tests comparing C++ TMM bindings against reference Python implementation.

These tests ensure that refactoring of C++ code does not break functionality
by comparing results against the pure Python reference implementation (TmmPy).

Test coverage includes:
- Normal incidence Fresnel reflection
- Thin film interference effects
- Metallic (absorbing) layers and SPP resonances
- Anisotropic (birefringent) crystals at various orientations
- 1D electromagnetic field profiles
- Field enhancement calculations
- Total internal reflection (TIR)
- Cross-polarization coupling in anisotropic media
"""

import numpy as np
import pytest

from GeneralTmm import Material, Tmm, TmmPy

# Full silver data from Johnson and Christy (1972)
SILVER_WLS_FULL = np.array([
    4.0e-07, 4.2e-07, 4.4e-07, 4.6e-07, 4.8e-07, 5.0e-07, 5.2e-07, 5.4e-07,
    5.6e-07, 5.8e-07, 6.0e-07, 6.2e-07, 6.4e-07, 6.6e-07, 6.8e-07, 7.0e-07,
    7.2e-07, 7.4e-07, 7.6e-07, 7.8e-07, 8.0e-07, 8.2e-07, 8.4e-07, 8.6e-07,
    8.8e-07, 9.0e-07, 9.2e-07, 9.4e-07, 9.6e-07, 9.8e-07,
])
SILVER_NS_FULL = np.array([
    0.050 + 2.104j, 0.046 + 2.348j, 0.040 + 2.553j, 0.044 + 2.751j,
    0.050 + 2.948j, 0.050 + 3.131j, 0.050 + 3.316j, 0.057 + 3.505j,
    0.057 + 3.679j, 0.051 + 3.841j, 0.055 + 4.010j, 0.059 + 4.177j,
    0.055 + 4.332j, 0.050 + 4.487j, 0.045 + 4.645j, 0.041 + 4.803j,
    0.037 + 4.960j, 0.033 + 5.116j, 0.031 + 5.272j, 0.034 + 5.421j,
    0.037 + 5.570j, 0.040 + 5.719j, 0.040 + 5.883j, 0.040 + 6.048j,
    0.040 + 6.213j, 0.040 + 6.371j, 0.040 + 6.519j, 0.040 + 6.667j,
    0.040 + 6.815j, 0.040 + 6.962j,
])


def prepare_tmm_pair(wl, layers):
    """Create matching Tmm (C++) and TmmPy (Python) instances."""
    tmm = Tmm()
    tmm.SetParams(wl=wl)
    for layer in layers:
        if layer[0] == "iso":
            _, d, n = layer
            mat = Material.Static(n)
            tmm.AddIsotropicLayer(d, mat)
        else:
            _, d, n1, n2, n3, psi, xi = layer
            matx = Material.Static(n1)
            maty = Material.Static(n2)
            matz = Material.Static(n3)
            tmm.AddLayer(d, matx, maty, matz, psi, xi)

    oldTmm = TmmPy()
    oldTmm.SetConf(wl=wl)
    for layer in layers:
        if layer[0] == "iso":
            _, d, n = layer
            oldTmm.AddIsotropicLayer(d, n)
        else:
            _, d, n1, n2, n3, psi, xi = layer
            oldTmm.AddLayer(d, n1, n2, n3, psi, xi)

    return tmm, oldTmm


class TestNormalIncidence:
    """Tests at normal incidence (beta=0)."""

    @pytest.mark.parametrize(
        "n1,n2",
        [
            (1.0, 1.5),
            (1.5, 1.0),
            (1.0, 2.0),
            (1.5, 2.5),
        ],
    )
    def test_fresnel_reflection_normal(self, n1, n2):
        """Compare Fresnel reflection at normal incidence."""
        wl = 532e-9
        layers = [("iso", float("inf"), n1), ("iso", float("inf"), n2)]
        betas = np.array([0.0])

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            for k in ["R11", "R22", "T31", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-10)

    @pytest.mark.parametrize("wl", [400e-9, 532e-9, 633e-9, 800e-9])
    def test_wavelength_sweep_normal(self, wl):
        """Compare results at different wavelengths at normal incidence."""
        layers = [("iso", float("inf"), 1.5), ("iso", float("inf"), 1.0)]
        betas = np.array([0.0])

        tmm, oldTmm = prepare_tmm_pair(wl, layers)
        res = tmm.Sweep("beta", betas, ((1.0, 0.0), -1, 0.0))
        resOld = oldTmm.SolveFor("beta", betas, polarization=(1.0, 0.0), enhInterface=-1, enhDist=0.0)

        np.testing.assert_allclose(res["R11"], resOld["R11"], rtol=1e-10)
        np.testing.assert_allclose(res["T31"], resOld["T31"], rtol=1e-10)


class TestThinFilmStack:
    """Tests for thin film multilayer structures."""

    @pytest.mark.parametrize("thickness", [50e-9, 100e-9, 200e-9])
    def test_dielectric_thin_film(self, thickness):
        """Compare thin dielectric film on substrate."""
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.0),
            ("iso", thickness, 1.5),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.0, 0.99, 20)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            for k in ["R11", "R22", "T31", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7)

    def test_multilayer_stack(self):
        """Compare multilayer dielectric stack."""
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 100e-9, 2.0),
            ("iso", 150e-9, 1.8),
            ("iso", 80e-9, 1.6),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 30)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            for k in ["R11", "R22", "R12", "R21", "T31", "T32", "T41", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7)


class TestMetallicFilms:
    """Tests for metallic (absorbing) films."""

    @pytest.fixture
    def silver(self):
        """Silver material with full wavelength data."""
        return Material(SILVER_WLS_FULL, SILVER_NS_FULL)

    @pytest.mark.parametrize("thickness", [20e-9, 50e-9, 100e-9])
    def test_metal_film_sweep(self, silver, thickness):
        """Compare metal thin film at various thicknesses."""
        wl = 600e-9
        n_ag = silver(wl)
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", thickness, n_ag),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.0, 1.4, 30)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            for k in ["R11", "R22", "T31", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7)

    @pytest.mark.parametrize("wl", [500e-9, 600e-9, 700e-9, 800e-9])
    def test_spp_resonance(self, silver, wl):
        """Compare SPP resonance at different wavelengths."""
        n_ag = silver(wl)
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 50e-9, n_ag),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.8, 1.4, 50)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)
        pol = (1.0, 0.0)

        res = tmm.Sweep("beta", betas, (pol, 2, 0.0))
        resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=2, enhDist=0.0)

        np.testing.assert_allclose(res["R11"], resOld["R11"], rtol=1e-7)
        np.testing.assert_allclose(res["enh"], resOld["enh"], rtol=1e-7)


class TestAnisotropicLayers:
    """Tests for anisotropic (birefringent) materials."""

    @pytest.mark.parametrize("psi", [0.0, np.pi / 4, np.pi / 2, np.pi])
    @pytest.mark.parametrize("xi", [0.0, np.pi / 6, np.pi / 3])
    def test_uniaxial_crystal_orientation(self, psi, xi):
        """Compare uniaxial crystal at various orientations."""
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 500e-9, 1.55, 1.55, 1.65, psi, xi),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.0, 1.4, 20)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            for k in ["R11", "R22", "R12", "R21", "T31", "T32", "T41", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7, atol=1e-14)

    def test_biaxial_crystal(self):
        """Compare biaxial crystal (all n different)."""
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 300e-9, 1.50, 1.55, 1.60, np.pi / 4, np.pi / 6),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 20)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            for k in ["R11", "R22", "T31", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7)


class TestFieldCalculations:
    """Tests for electromagnetic field calculations."""

    def test_field_profile_isotropic(self):
        """Compare 1D field profile in isotropic layers."""
        wl = 532e-9
        beta = 0.3
        xs = np.linspace(-500e-9, 500e-9, 100)
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 200e-9, 1.8),
            ("iso", float("inf"), 1.0),
        ]

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            tmm.beta = beta
            E, H = tmm.CalcFields1D(xs, np.array(pol))
            oldTmm.Solve(wl, beta)
            EOld, HOld = oldTmm.CalcFields1D(xs, pol)

            for i in range(3):
                np.testing.assert_allclose(abs(E[:, i]), abs(EOld[:, i]), rtol=1e-7, atol=1e-15)
                np.testing.assert_allclose(abs(H[:, i]), abs(HOld[:, i]), rtol=1e-7, atol=1e-15)

    def test_field_profile_anisotropic(self):
        """Compare 1D field profile in anisotropic layers."""
        wl = 532e-9
        beta = 0.3
        xs = np.linspace(-500e-9, 500e-9, 80)
        psi, xi = np.pi / 6, np.pi / 4
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 300e-9, 1.55, 1.58, 1.62, psi, xi),
            ("iso", float("inf"), 1.0),
        ]

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            tmm.beta = beta
            E, H = tmm.CalcFields1D(xs, np.array(pol))
            oldTmm.Solve(wl, beta)
            EOld, HOld = oldTmm.CalcFields1D(xs, pol)

            for i in range(3):
                np.testing.assert_allclose(abs(E[:, i]), abs(EOld[:, i]), rtol=1e-7, atol=1e-15)
                np.testing.assert_allclose(abs(H[:, i]), abs(HOld[:, i]), rtol=1e-7, atol=1e-15)

    @pytest.mark.parametrize("beta", [0.0, 0.3, 0.6, 0.9])
    def test_field_profile_beta_dependence(self, beta):
        """Compare field profiles at various angles of incidence."""
        wl = 532e-9
        xs = np.linspace(-300e-9, 300e-9, 50)
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 150e-9, 2.0),
            ("iso", float("inf"), 1.3),
        ]

        tmm, oldTmm = prepare_tmm_pair(wl, layers)
        pol = (1.0, 0.0)

        tmm.beta = beta
        E, H = tmm.CalcFields1D(xs, np.array(pol))
        oldTmm.Solve(wl, beta)
        EOld, HOld = oldTmm.CalcFields1D(xs, pol)

        for i in range(3):
            np.testing.assert_allclose(abs(E[:, i]), abs(EOld[:, i]), rtol=1e-7, atol=1e-15)


class TestEnhancement:
    """Tests for field enhancement calculations."""

    def test_enhancement_at_interface(self):
        """Compare field enhancement at interface."""
        wl = 600e-9
        silver = Material(SILVER_WLS_FULL, SILVER_NS_FULL)
        n_ag = silver(wl)
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 50e-9, n_ag),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.9, 1.3, 30)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)
        pol = (1.0, 0.0)

        res = tmm.Sweep("beta", betas, (pol, 2, 0.0))
        resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=2, enhDist=0.0)

        np.testing.assert_allclose(res["enh"], resOld["enh"], rtol=1e-7)

    def test_enhancement_substrate_interface(self):
        """Compare field enhancement at substrate using enhInterface=-1."""
        wl = 600e-9
        silver = Material(SILVER_WLS_FULL, SILVER_NS_FULL)
        n_ag = silver(wl)
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 50e-9, n_ag),
            ("iso", float("inf"), 1.0),
        ]
        betas = np.linspace(0.9, 1.3, 30)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)
        pol = (1.0, 0.0)

        res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
        resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

        np.testing.assert_allclose(res["enh"], resOld["enh"], rtol=1e-7)


class TestMixedPolarization:
    """Tests with mixed polarization states."""

    @pytest.mark.parametrize(
        "pol",
        [
            (1.0, 0.0),
            (0.0, 1.0),
        ],
    )
    def test_polarization_states(self, pol):
        """Compare results for different polarization states."""
        wl = 532e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", 100e-9, 1.8),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 20)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
        resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

        np.testing.assert_allclose(res["enh"], resOld["enh"], rtol=1e-7)


class TestTotalInternalReflection:
    """Tests for total internal reflection scenarios."""

    @pytest.mark.parametrize("n1,n2", [(1.5, 1.0), (1.8, 1.3), (2.0, 1.5)])
    def test_tir_transition(self, n1, n2):
        """Compare reflection coefficients across TIR transition."""
        wl = 532e-9
        layers = [("iso", float("inf"), n1), ("iso", float("inf"), n2)]
        # Stay below n1 to avoid numerical issues
        betas = np.linspace(0.0, n1 - 1e-3, 30)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            for k in ["R11", "R22", "R12", "R21", "T31", "T32", "T41", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7)

    @pytest.mark.parametrize("wl", [400e-9, 600e-9, 800e-9])
    def test_tir_wavelength_dependence(self, wl):
        """Compare TIR at different wavelengths."""
        n1, n2 = 1.5, 1.0
        layers = [("iso", float("inf"), n1), ("iso", float("inf"), n2)]
        betas = np.linspace(0.0, n1 - 1e-3, 30)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            np.testing.assert_allclose(res["R11"], resOld["R11"], rtol=1e-7)
            np.testing.assert_allclose(res["T31"], resOld["T31"], rtol=1e-7)


class TestCrossPolarization:
    """Tests for cross-polarization coupling in anisotropic media."""

    @pytest.mark.parametrize("psi", [np.pi/8, np.pi/4, 3*np.pi/8])
    def test_cross_polarization_finite_layer(self, psi):
        """Compare cross-polarization terms for finite anisotropic layer."""
        wl = 532e-9
        xi = np.pi / 6
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 200e-9, 1.76, 1.8, 1.9, psi, xi),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 25)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            # Cross-polarization terms are key for anisotropic layers
            for k in ["R12", "R21", "T32", "T41"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7, atol=1e-14)

    def test_cross_polarization_thick_crystal(self):
        """Compare cross-polarization for thick anisotropic crystal."""
        wl = 532e-9
        psi, xi = np.pi / 3, np.pi / 5
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 1000e-9, 1.55, 1.60, 1.65, psi, xi),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.4, 30)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            for k in ["R11", "R22", "R12", "R21", "T31", "T32", "T41", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7, atol=1e-14)


class TestAnisotropicFiniteLayers:
    """Tests for finite-thickness anisotropic layers (matching original test_tmm.py)."""

    @pytest.mark.parametrize("psi", [0.0, np.pi/4, np.pi/2, 3*np.pi/4])
    @pytest.mark.parametrize("xi", [0.0, np.pi/4, np.pi/2, 3*np.pi/4])
    def test_aniso_sweep_finite_layer(self, psi, xi):
        """Compare anisotropic finite layer at various orientations (matching testAnisoSweep2)."""
        wl = 500e-9
        layers = [
            ("iso", float("inf"), 1.5),
            ("aniso", 200e-9, 1.76, 1.8, 1.9, psi, xi),
            ("iso", float("inf"), 1.5),
        ]
        betas = np.linspace(0.0, 1.5 - 1e-3, 30)

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            res = tmm.Sweep("beta", betas, (pol, -1, 0.0))
            resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=-1, enhDist=0.0)

            # Main reflection/transmission terms
            for k in ["R11", "R22", "T31", "T42"]:
                np.testing.assert_allclose(res[k], resOld[k], rtol=1e-7)

            np.testing.assert_allclose(res["enh"], resOld["enh"], rtol=1e-7)


class TestSPPFieldsAndEnhancement:
    """Tests for SPP fields comparing to reference implementation."""

    @pytest.fixture
    def silver(self):
        """Silver material with full wavelength data."""
        return Material(SILVER_WLS_FULL, SILVER_NS_FULL)

    @pytest.mark.parametrize("wl", [400e-9, 500e-9, 800e-9])
    def test_spp_field_profile(self, silver, wl):
        """Compare 1D field profiles for SPP structure."""
        beta = 0.5
        xs = np.linspace(-1e-6, 1e-6, 100)
        n_ag = silver(wl)
        layers = [
            ("iso", float("inf"), 1.5),
            ("iso", float("inf"), n_ag),
            ("iso", float("inf"), 1.0),
        ]

        tmm, oldTmm = prepare_tmm_pair(wl, layers)

        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            tmm.beta = beta
            E, H = tmm.CalcFields1D(xs, np.array(pol))
            oldTmm.Solve(wl, beta)
            EOld, HOld = oldTmm.CalcFields1D(xs, pol)

            for i in range(3):
                np.testing.assert_allclose(abs(E[:, i]), abs(EOld[:, i]), rtol=1e-7, atol=1e-15)
                np.testing.assert_allclose(abs(H[:, i]), abs(HOld[:, i]), rtol=1e-7, atol=1e-15)
