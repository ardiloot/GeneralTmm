"""Tests comparing C++ TMM bindings against reference Python implementation.

These tests ensure that refactoring of C++ code does not break functionality
by comparing results against the pure Python reference implementation (TmmPy).

Note: This file complements test_tmm.py which contains comprehensive tests for:
- TIR sweep (testTIRSweep) - many n1/n2/wl combinations
- SPP sweep (testSppSweep) - semi-infinite metal layers
- Anisotropic sweep (testAnisoSweep, testAnisoSweep2) - extensive psi/xi coverage
- SPP fields (testSppFields) - field profiles in metal structures
- Anisotropic fields (testAnisoFields) - extensive field calculations

This file adds additional scenarios not covered by test_tmm.py:
- Normal incidence Fresnel reflection
- Thin film interference effects with finite layers
- Metallic thin films (finite thickness)
- Uniaxial and biaxial crystals
- Field calculations with isotropic finite layers
- Field enhancement at specific interfaces
- Cross-polarization coupling tests
"""

import numpy as np
import pytest

from GeneralTmm import Material, Tmm, TmmPy

# Full silver data from Johnson and Christy (1972)
SILVER_WLS_FULL = np.array(
    [
        4.0e-07,
        4.2e-07,
        4.4e-07,
        4.6e-07,
        4.8e-07,
        5.0e-07,
        5.2e-07,
        5.4e-07,
        5.6e-07,
        5.8e-07,
        6.0e-07,
        6.2e-07,
        6.4e-07,
        6.6e-07,
        6.8e-07,
        7.0e-07,
        7.2e-07,
        7.4e-07,
        7.6e-07,
        7.8e-07,
        8.0e-07,
        8.2e-07,
        8.4e-07,
        8.6e-07,
        8.8e-07,
        9.0e-07,
        9.2e-07,
        9.4e-07,
        9.6e-07,
        9.8e-07,
    ]
)
SILVER_NS_FULL = np.array(
    [
        0.050 + 2.104j,
        0.046 + 2.348j,
        0.040 + 2.553j,
        0.044 + 2.751j,
        0.050 + 2.948j,
        0.050 + 3.131j,
        0.050 + 3.316j,
        0.057 + 3.505j,
        0.057 + 3.679j,
        0.051 + 3.841j,
        0.055 + 4.010j,
        0.059 + 4.177j,
        0.055 + 4.332j,
        0.050 + 4.487j,
        0.045 + 4.645j,
        0.041 + 4.803j,
        0.037 + 4.960j,
        0.033 + 5.116j,
        0.031 + 5.272j,
        0.034 + 5.421j,
        0.037 + 5.570j,
        0.040 + 5.719j,
        0.040 + 5.883j,
        0.040 + 6.048j,
        0.040 + 6.213j,
        0.040 + 6.371j,
        0.040 + 6.519j,
        0.040 + 6.667j,
        0.040 + 6.815j,
        0.040 + 6.962j,
    ]
)


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


class TestCrossPolarization:
    """Tests for cross-polarization coupling in anisotropic media.

    Note: test_tmm.py covers anisotropic layers extensively but uses loops
    that hide individual test cases. These tests explicitly test cross-polarization
    terms (R12, R21, T32, T41) which are key for validating anisotropic behavior.
    """

    @pytest.mark.parametrize("psi", [np.pi / 8, np.pi / 4, 3 * np.pi / 8])
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


class TestOptimizeEnhancement:
    """Tests for enhancement optimization functionality."""

    def test_enhancement_sweep_spp(self):
        """Compare enhancement sweep for SPP structure at valid beta range."""
        wl = 600e-9
        silver = Material(SILVER_WLS_FULL, SILVER_NS_FULL)
        n_ag = silver(wl)

        # Setup C++ TMM
        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(50e-9, Material(SILVER_WLS_FULL, SILVER_NS_FULL))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        # Setup Python TMM
        oldTmm = TmmPy()
        oldTmm.SetConf(wl=wl)
        oldTmm.AddIsotropicLayer(float("inf"), 1.5)
        oldTmm.AddIsotropicLayer(50e-9, n_ag)
        oldTmm.AddIsotropicLayer(float("inf"), 1.0)

        pol = (1.0, 0.0)
        enhPos = (pol, 2, 0.0)

        # Sweep beta values around SPP resonance
        betas = np.linspace(1.01, 1.3, 30)
        res = tmm.Sweep("beta", betas, enhPos)
        resOld = oldTmm.SolveFor("beta", betas, polarization=pol, enhInterface=2, enhDist=0.0)

        # Enhancement values should match
        np.testing.assert_allclose(res["enh"], resOld["enh"], rtol=1e-7)


class TestWavelengthSweep:
    """Tests for wavelength parameter sweeps."""

    def test_wl_sweep_simple(self):
        """Test sweeping over wavelength parameter."""
        # This tests the "wl" sweep parameter which is different from beta sweep
        wls = np.linspace(500e-9, 700e-9, 10)

        tmm = Tmm()
        tmm.SetParams(beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        res = tmm.Sweep("wl", wls)

        # Check that results have correct length
        assert len(res["R11"]) == len(wls)
        assert len(res["T31"]) == len(wls)

        # R and T should be bounded
        assert np.all(res["R11"] >= 0)
        assert np.all(res["R11"] <= 1)
        assert np.all(res["T31"] >= 0)


class TestMaterialInterpolation:
    """Tests for Material class wavelength interpolation."""

    def test_material_interpolation(self):
        """Test that Material interpolates correctly at intermediate wavelengths."""
        mat = Material(SILVER_WLS_FULL, SILVER_NS_FULL)

        # Test at a known data point
        n_500 = mat(500e-9)
        assert n_500.real == pytest.approx(0.050, rel=0.01)
        assert n_500.imag == pytest.approx(3.131, rel=0.01)

        # Test interpolation between points - just verify finite values
        n_550 = mat(550e-9)
        assert np.isfinite(n_550.real)
        assert np.isfinite(n_550.imag)
        # Verify imaginary part is reasonable (between nearby data points)
        assert n_550.imag > 3.0
        assert n_550.imag < 5.0

    def test_material_static(self):
        """Test Material.Static helper method."""
        mat = Material.Static(1.5)
        assert mat(400e-9) == pytest.approx(1.5)
        assert mat(800e-9) == pytest.approx(1.5)

        mat_complex = Material.Static(1.5 + 0.1j)
        n = mat_complex(532e-9)
        assert n.real == pytest.approx(1.5)
        assert n.imag == pytest.approx(0.1)


# =============================================================================
# Physics-based validation tests (not comparing against Python reference)
# =============================================================================


class TestEnergyConservation:
    """Tests for energy conservation (R + T = 1 for lossless materials)."""

    @pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.5, 2.0)])
    def test_energy_conservation_interface(self, n1, n2):
        """R + T should equal 1 for lossless interface."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        betas = np.linspace(0.0, min(n1, n2) - 0.01, 20)
        res = tmm.Sweep("beta", betas)

        # For p-polarization: R11 + T31 = 1
        np.testing.assert_allclose(res["R11"] + res["T31"], 1.0, rtol=1e-10)
        # For s-polarization: R22 + T42 = 1
        np.testing.assert_allclose(res["R22"] + res["T42"], 1.0, rtol=1e-10)

    def test_energy_conservation_thin_film(self):
        """R + T should equal 1 for lossless thin film."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.5))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        betas = np.linspace(0.0, 0.99, 20)
        res = tmm.Sweep("beta", betas)

        np.testing.assert_allclose(res["R11"] + res["T31"], 1.0, rtol=1e-7)
        np.testing.assert_allclose(res["R22"] + res["T42"], 1.0, rtol=1e-7)

    def test_energy_conservation_multilayer(self):
        """R + T should equal 1 for lossless multilayer."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(80e-9, Material.Static(2.0))
        tmm.AddIsotropicLayer(120e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))

        betas = np.linspace(0.0, 1.4, 30)
        res = tmm.Sweep("beta", betas)

        np.testing.assert_allclose(res["R11"] + res["T31"], 1.0, rtol=1e-7)
        np.testing.assert_allclose(res["R22"] + res["T42"], 1.0, rtol=1e-7)


class TestBrewsterAngle:
    """Tests for Brewster angle physics.
    
    At Brewster's angle, p-polarized light has zero reflection when
    passing from medium n1 to n2. The Brewster angle θ_B satisfies:
    tan(θ_B) = n2/n1
    
    In terms of beta (= n1*sin(θ)): beta_B = n1*sin(arctan(n2/n1))
    """

    def test_brewster_angle_zero_reflection(self):
        """p-polarization reflection should be very small at Brewster angle."""
        wl = 532e-9
        n1, n2 = 1.0, 1.5

        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        # Brewster angle: tan(theta_B) = n2/n1
        theta_brewster = np.arctan(n2 / n1)
        beta_brewster = n1 * np.sin(theta_brewster)

        # Sweep around Brewster angle with fine resolution
        betas = np.linspace(0.0, 0.99, 200)
        res = tmm.Sweep("beta", betas)

        # Find minimum in p-polarization reflection
        min_idx = np.argmin(res["R11"])
        beta_at_min = betas[min_idx]
        r_at_min = res["R11"][min_idx]

        # Minimum should be near Brewster angle
        np.testing.assert_allclose(beta_at_min, beta_brewster, rtol=0.05)
        # p-polarization reflection should be very small at Brewster angle
        assert r_at_min < 1e-5  # Numerical precision limits

    def test_brewster_no_s_pol_minimum(self):
        """s-polarization should NOT have zero reflection at Brewster angle."""
        wl = 532e-9
        n1, n2 = 1.0, 1.5

        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        theta_brewster = np.arctan(n2 / n1)
        beta_brewster = n1 * np.sin(theta_brewster)

        # Set beta near Brewster angle
        tmm.SetParams(beta=beta_brewster)
        R = tmm.GetIntensityMatrix()

        # s-polarization (R22) should have significant reflection
        assert R[1, 1] > 0.01  # Should be around 7-8% for glass


class TestTotalInternalReflectionPhysics:
    """Tests for TIR physics.
    
    Total internal reflection occurs when light travels from higher to lower
    refractive index medium at angle greater than critical angle.
    Critical angle: sin(θ_c) = n2/n1 where n1 > n2
    In terms of beta: beta_critical = n1 * sin(θ_c) = n2
    """

    def test_tir_complete_reflection(self):
        """Beyond critical angle, R should equal 1."""
        wl = 532e-9
        n1, n2 = 1.5, 1.0
        # Critical beta is simply n2 (the lower refractive index)
        beta_critical = n2

        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        # Sweep beyond critical angle (beta > n2)
        betas = np.linspace(beta_critical + 0.01, n1 - 0.01, 20)
        res = tmm.Sweep("beta", betas)

        # Beyond critical angle, reflection should be 1
        np.testing.assert_allclose(res["R11"], 1.0, rtol=1e-10)
        np.testing.assert_allclose(res["R22"], 1.0, rtol=1e-10)

    def test_tir_zero_transmission(self):
        """Beyond critical angle, T should equal 0."""
        wl = 532e-9
        n1, n2 = 1.5, 1.0
        beta_critical = n2

        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        betas = np.linspace(beta_critical + 0.01, n1 - 0.01, 20)
        res = tmm.Sweep("beta", betas)

        np.testing.assert_allclose(res["T31"], 0.0, atol=1e-10)
        np.testing.assert_allclose(res["T42"], 0.0, atol=1e-10)

    def test_tir_transition(self):
        """Test behavior across critical angle transition."""
        wl = 532e-9
        n1, n2 = 1.5, 1.0
        beta_critical = n2

        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        # Sweep below and above critical angle (avoid exact critical point)
        betas_below = np.linspace(0.5, beta_critical - 0.05, 20)
        betas_above = np.linspace(beta_critical + 0.05, n1 - 0.1, 20)

        res_below = tmm.Sweep("beta", betas_below)
        res_above = tmm.Sweep("beta", betas_above)

        # Below critical: R < 1 and T > 0
        assert np.all(res_below["R11"] < 1.0)
        assert np.all(res_below["T31"] > 0.0)

        # Above critical: R = 1 and T = 0
        np.testing.assert_allclose(res_above["R11"], 1.0, rtol=1e-10)
        np.testing.assert_allclose(res_above["T31"], 0.0, atol=1e-10)


class TestFresnelEquations:
    """Tests verifying Fresnel equations at normal incidence.
    
    At normal incidence (θ=0), the Fresnel reflection coefficient is:
    r = (n1 - n2) / (n1 + n2)
    R = |r|² = ((n1 - n2) / (n1 + n2))²
    """

    @pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.0, 2.0), (1.5, 2.5)])
    def test_fresnel_normal_incidence(self, n1, n2):
        """Verify Fresnel reflection at normal incidence matches analytic formula."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        R_matrix = tmm.GetIntensityMatrix()

        # Analytic Fresnel reflection at normal incidence
        R_expected = ((n1 - n2) / (n1 + n2)) ** 2

        np.testing.assert_allclose(R_matrix[0, 0], R_expected, rtol=1e-10)
        np.testing.assert_allclose(R_matrix[1, 1], R_expected, rtol=1e-10)


class TestQuarterWaveCoating:
    """Tests for quarter-wave antireflection coating.
    
    A quarter-wave layer of thickness d = λ/(4n) with n = sqrt(n1*n2)
    gives zero reflection at normal incidence when placed between
    media with refractive indices n1 and n2.
    """

    def test_quarter_wave_antireflection(self):
        """Quarter-wave coating should give zero reflection at design wavelength."""
        wl = 532e-9
        n_air = 1.0
        n_glass = 1.5

        # Optimal coating: n_coating = sqrt(n_air * n_glass)
        n_coating = np.sqrt(n_air * n_glass)
        # Quarter-wave thickness
        d_coating = wl / (4 * n_coating)

        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_air))
        tmm.AddIsotropicLayer(d_coating, Material.Static(n_coating))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_glass))

        R = tmm.GetIntensityMatrix()

        # Reflection should be zero at design wavelength
        np.testing.assert_allclose(R[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(R[1, 1], 0.0, atol=1e-10)

    def test_quarter_wave_off_design(self):
        """Quarter-wave coating should have higher reflection away from design wavelength."""
        wl_design = 532e-9
        n_air = 1.0
        n_glass = 1.5
        n_coating = np.sqrt(n_air * n_glass)
        d_coating = wl_design / (4 * n_coating)

        tmm = Tmm()
        tmm.SetParams(beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_air))
        tmm.AddIsotropicLayer(d_coating, Material.Static(n_coating))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_glass))

        # Check at design wavelength vs off-design
        wls = np.array([wl_design, 400e-9, 700e-9])
        res = tmm.Sweep("wl", wls)

        # At design wavelength, R should be ~0
        assert res["R11"][0] < 1e-10
        # Off design, R should be larger
        assert res["R11"][1] > 0.001
        assert res["R11"][2] > 0.001


class TestNormalIncidenceSymmetry:
    """Tests for symmetry at normal incidence."""

    def test_polarization_symmetry_normal(self):
        """At normal incidence, p and s polarizations should give same result."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.5))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        betas = np.array([0.0])
        res = tmm.Sweep("beta", betas)

        # At normal incidence, R11 should equal R22
        np.testing.assert_allclose(res["R11"], res["R22"], rtol=1e-10)
        # And T31 should equal T42
        np.testing.assert_allclose(res["T31"], res["T42"], rtol=1e-10)


class TestCalcFields2D:
    """Tests for 2D field calculations."""

    def test_calc_fields_2d_shape(self):
        """Test that CalcFields2D returns correct shape."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        xs = np.linspace(-200e-9, 200e-9, 20)
        ys = np.linspace(-100e-9, 100e-9, 15)
        pol = np.array([1.0, 0.0])

        E, H = tmm.CalcFields2D(xs, ys, pol)

        # Should have shape (len(xs), len(ys), 3)
        assert E.shape == (len(xs), len(ys), 3)
        assert H.shape == (len(xs), len(ys), 3)

    def test_calc_fields_2d_consistency(self):
        """2D fields at y=0 should match 1D fields."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        xs = np.linspace(-200e-9, 200e-9, 30)
        ys = np.array([0.0])
        pol = np.array([1.0, 0.0])

        E_2d, H_2d = tmm.CalcFields2D(xs, ys, pol)
        E_1d, H_1d = tmm.CalcFields1D(xs, pol)

        # 2D fields at y=0 should match 1D fields (up to phase)
        np.testing.assert_allclose(np.abs(E_2d[:, 0, :]), np.abs(E_1d), rtol=1e-7)
        np.testing.assert_allclose(np.abs(H_2d[:, 0, :]), np.abs(H_1d), rtol=1e-7)


class TestClearLayers:
    """Tests for layer management."""

    def test_clear_layers(self):
        """Test that ClearLayers resets the structure."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        # Get result with 3 layers
        betas = np.linspace(0.0, 0.5, 10)
        res1 = tmm.Sweep("beta", betas)
        r11_original = res1["R11"].copy()

        # Clear and add different structure
        tmm.ClearLayers()
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))

        res2 = tmm.Sweep("beta", betas)

        # Results should be different
        assert not np.allclose(r11_original, res2["R11"], rtol=0.01)


class TestMatrixAccess:
    """Tests for amplitude and intensity matrix access."""

    def test_get_intensity_matrix(self):
        """Test GetIntensityMatrix returns valid matrix."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        R = tmm.GetIntensityMatrix()

        # Should be 4x4 matrix
        assert R.shape == (4, 4)
        # Diagonal elements should be non-negative
        assert R[0, 0] >= 0
        assert R[1, 1] >= 0

    def test_get_amplitude_matrix(self):
        """Test GetAmplitudeMatrix returns valid matrix."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        r = tmm.GetAmplitudeMatrix()

        # Should be 4x4 complex matrix
        assert r.shape == (4, 4)
        assert np.iscomplexobj(r)


class TestFieldsAtInterface:
    """Tests for field calculations at specific interfaces."""

    def test_fields_at_interface(self):
        """Test CalcFieldsAtInterface returns valid fields."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        pol = (1.0, 0.0)
        enhPos = (pol, 2, 0.0)

        E, H = tmm.CalcFieldsAtInterface(enhPos)

        # Should return 3-component vectors
        assert len(E) == 3
        assert len(H) == 3
        # Fields should be finite
        assert np.all(np.isfinite(E))
        assert np.all(np.isfinite(H))
