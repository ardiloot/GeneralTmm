"""Physics-based validation tests for TMM calculations.

These tests verify that the TMM implementation produces physically correct results
by checking against known analytical solutions and fundamental physics principles.

Test categories:
1. Energy Conservation - R + T = 1 for lossless materials
2. Fresnel Equations - Analytical formulas at normal incidence
3. Brewster's Angle - Zero p-polarization reflection
4. Total Internal Reflection - Complete reflection beyond critical angle
5. Quarter-Wave Coatings - Antireflection at design wavelength
6. Symmetry - p/s polarization equivalence at normal incidence
"""

import numpy as np
import pytest

from GeneralTmm import Material, Tmm


class TestEnergyConservation:
    """Tests for energy conservation (R + T = 1 for lossless materials).

    For lossless (real refractive index) materials, the sum of reflection
    and transmission coefficients must equal 1 due to energy conservation.
    """

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

        np.testing.assert_allclose(res["R11"] + res["T31"], 1.0, rtol=1e-10)
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
        R_expected = ((n1 - n2) / (n1 + n2)) ** 2

        np.testing.assert_allclose(R_matrix[0, 0], R_expected, rtol=1e-10)
        np.testing.assert_allclose(R_matrix[1, 1], R_expected, rtol=1e-10)


class TestBrewsterAngle:
    """Tests for Brewster angle physics.

    At Brewster's angle, p-polarized light has zero reflection when
    passing from medium n1 to n2. The Brewster angle θ_B satisfies:
    tan(θ_B) = n2/n1

    s-polarization does NOT have zero reflection at Brewster's angle.
    """

    def test_brewster_angle_zero_reflection(self):
        """p-polarization reflection should be very small at Brewster angle."""
        wl = 532e-9
        n1, n2 = 1.0, 1.5

        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        theta_brewster = np.arctan(n2 / n1)
        beta_brewster = n1 * np.sin(theta_brewster)

        betas = np.linspace(0.0, 0.99, 200)
        res = tmm.Sweep("beta", betas)

        min_idx = np.argmin(res["R11"])
        beta_at_min = betas[min_idx]
        r_at_min = res["R11"][min_idx]

        np.testing.assert_allclose(beta_at_min, beta_brewster, rtol=0.05)
        assert r_at_min < 1e-5

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

        tmm.SetParams(beta=beta_brewster)
        R = tmm.GetIntensityMatrix()

        assert R[1, 1] > 0.01


class TestTotalInternalReflection:
    """Tests for total internal reflection (TIR) physics.

    TIR occurs when light travels from higher to lower refractive index
    medium at angle greater than critical angle.

    Critical angle: sin(θ_c) = n2/n1 where n1 > n2
    In terms of beta: beta_critical = n1 * sin(θ_c) = n2

    Beyond critical angle: R = 1, T = 0
    """

    def test_tir_complete_reflection(self):
        """Beyond critical angle, R should equal 1."""
        wl = 532e-9
        n1, n2 = 1.5, 1.0
        beta_critical = n2

        tmm = Tmm()
        tmm.SetParams(wl=wl)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        betas = np.linspace(beta_critical + 0.01, n1 - 0.01, 20)
        res = tmm.Sweep("beta", betas)

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

        betas_below = np.linspace(0.5, beta_critical - 0.05, 20)
        betas_above = np.linspace(beta_critical + 0.05, n1 - 0.1, 20)

        res_below = tmm.Sweep("beta", betas_below)
        res_above = tmm.Sweep("beta", betas_above)

        assert np.all(res_below["R11"] < 1.0)
        assert np.all(res_below["T31"] > 0.0)

        np.testing.assert_allclose(res_above["R11"], 1.0, rtol=1e-10)
        np.testing.assert_allclose(res_above["T31"], 0.0, atol=1e-10)


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
        n_coating = np.sqrt(n_air * n_glass)
        d_coating = wl / (4 * n_coating)

        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_air))
        tmm.AddIsotropicLayer(d_coating, Material.Static(n_coating))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_glass))

        R = tmm.GetIntensityMatrix()

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

        wls = np.array([wl_design, 400e-9, 700e-9])
        res = tmm.Sweep("wl", wls)

        assert res["R11"][0] < 1e-10
        assert res["R11"][1] > 0.001
        assert res["R11"][2] > 0.001


class TestPolarizationSymmetry:
    """Tests for polarization symmetry at normal incidence.

    At normal incidence (β=0), there is no distinction between p and s
    polarizations for isotropic media, so R_pp = R_ss and T_pp = T_ss.
    """

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

        np.testing.assert_allclose(res["R11"], res["R22"], rtol=1e-10)
        np.testing.assert_allclose(res["T31"], res["T42"], rtol=1e-10)

    def test_polarization_symmetry_multilayer(self):
        """Multilayer at normal incidence should have p/s symmetry."""
        wl = 532e-9
        tmm = Tmm()
        tmm.SetParams(wl=wl, beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(80e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(120e-9, Material.Static(1.4))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))

        R = tmm.GetIntensityMatrix()

        np.testing.assert_allclose(R[0, 0], R[1, 1], rtol=1e-10)
