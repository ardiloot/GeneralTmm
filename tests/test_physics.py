"""Physics validation tests against analytical solutions."""

import numpy as np
import pytest

from GeneralTmm import Material, Tmm


class TestEnergyConservation:
    """R + T = 1 for lossless (real n) materials."""

    @pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.5, 2.0)])
    def test_single_interface(self, n1, n2):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        betas = np.linspace(0.0, min(n1, n2) - 0.01, 20)
        res = tmm.Sweep("beta", betas)
        np.testing.assert_allclose(res["R11"] + res["T31"], 1.0, rtol=1e-10)
        np.testing.assert_allclose(res["R22"] + res["T42"], 1.0, rtol=1e-10)

    def test_thin_film(self):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.5))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        betas = np.linspace(0.0, 0.99, 20)
        res = tmm.Sweep("beta", betas)
        np.testing.assert_allclose(res["R11"] + res["T31"], 1.0, rtol=1e-7)
        np.testing.assert_allclose(res["R22"] + res["T42"], 1.0, rtol=1e-7)

    def test_multilayer(self):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(80e-9, Material.Static(2.0))
        tmm.AddIsotropicLayer(120e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))

        betas = np.linspace(0.0, 1.4, 30)
        res = tmm.Sweep("beta", betas)
        np.testing.assert_allclose(res["R11"] + res["T31"], 1.0, rtol=1e-7)
        np.testing.assert_allclose(res["R22"] + res["T42"], 1.0, rtol=1e-7)


class TestFresnelEquations:
    """R = ((n₁ − n₂) / (n₁ + n₂))² at normal incidence."""

    @pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.0, 2.0), (1.5, 2.5)])
    def test_normal_incidence(self, n1, n2):
        tmm = Tmm(wl=532e-9, beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        R = tmm.GetIntensityMatrix()
        R_expected = ((n1 - n2) / (n1 + n2)) ** 2
        np.testing.assert_allclose(R[0, 0], R_expected, rtol=1e-10)
        np.testing.assert_allclose(R[1, 1], R_expected, rtol=1e-10)


class TestBrewsterAngle:
    """tan(θ_B) = n₂/n₁ gives R_p ≈ 0 while R_s > 0."""

    def test_p_pol_minimum(self):
        n1, n2 = 1.0, 1.5
        beta_brewster = n1 * np.sin(np.arctan(n2 / n1))

        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        betas = np.linspace(0.0, 0.99, 200)
        res = tmm.Sweep("beta", betas)
        min_idx = np.argmin(res["R11"])

        np.testing.assert_allclose(betas[min_idx], beta_brewster, rtol=0.05)
        assert res["R11"][min_idx] < 1e-5

    def test_s_pol_no_zero(self):
        n1, n2 = 1.0, 1.5
        beta_brewster = n1 * np.sin(np.arctan(n2 / n1))

        tmm = Tmm(wl=532e-9, beta=beta_brewster)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n2))

        assert tmm.GetIntensityMatrix()[1, 1] > 0.01


class TestTotalInternalReflection:
    """β > n₂ → R = 1, T = 0 when n₁ > n₂ (glass → air)."""

    N1, N2 = 1.5, 1.0
    BETA_C = N2  # critical beta = n₂

    @pytest.fixture
    def glass_air(self):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(self.N1))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(self.N2))
        return tmm

    def test_beyond_critical(self, glass_air):
        """R = 1 and T = 0 above the critical angle."""
        betas = np.linspace(self.BETA_C + 0.01, self.N1 - 0.01, 20)
        res = glass_air.Sweep("beta", betas)
        np.testing.assert_allclose(res["R11"], 1.0, rtol=1e-10)
        np.testing.assert_allclose(res["R22"], 1.0, rtol=1e-10)
        np.testing.assert_allclose(res["T31"], 0.0, atol=1e-10)
        np.testing.assert_allclose(res["T42"], 0.0, atol=1e-10)

    def test_transition(self, glass_air):
        """Partial reflection below critical, total above."""
        res_below = glass_air.Sweep("beta", np.linspace(0.5, self.BETA_C - 0.05, 20))
        res_above = glass_air.Sweep("beta", np.linspace(self.BETA_C + 0.05, self.N1 - 0.1, 20))

        assert np.all(res_below["R11"] < 1.0)
        assert np.all(res_below["T31"] > 0.0)
        np.testing.assert_allclose(res_above["R11"], 1.0, rtol=1e-10)
        np.testing.assert_allclose(res_above["T31"], 0.0, atol=1e-10)


class TestQuarterWaveCoating:
    """d = λ/(4n), n = √(n₁ n₂) → R = 0 at the design wavelength."""

    def test_antireflection(self):
        n_air, n_glass = 1.0, 1.5
        n_coat = np.sqrt(n_air * n_glass)
        wl = 532e-9

        tmm = Tmm(wl=wl, beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_air))
        tmm.AddIsotropicLayer(wl / (4 * n_coat), Material.Static(n_coat))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_glass))

        R = tmm.GetIntensityMatrix()
        np.testing.assert_allclose(R[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(R[1, 1], 0.0, atol=1e-10)

    def test_off_design_wavelength(self):
        """Reflection is non-zero away from the design wavelength."""
        wl_design = 532e-9
        n_air, n_glass = 1.0, 1.5
        n_coat = np.sqrt(n_air * n_glass)

        tmm = Tmm(beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_air))
        tmm.AddIsotropicLayer(wl_design / (4 * n_coat), Material.Static(n_coat))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(n_glass))

        res = tmm.Sweep("wl", np.array([wl_design, 400e-9, 700e-9]))
        assert res["R11"][0] < 1e-10
        assert res["R11"][1] > 0.001
        assert res["R11"][2] > 0.001


class TestPolarizationSymmetry:
    """At normal incidence (β = 0), R_pp = R_ss for isotropic media."""

    def test_single_film(self):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.5))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        res = tmm.Sweep("beta", np.array([0.0]))
        np.testing.assert_allclose(res["R11"], res["R22"], rtol=1e-10)
        np.testing.assert_allclose(res["T31"], res["T42"], rtol=1e-10)

    def test_multilayer(self):
        tmm = Tmm(wl=532e-9, beta=0.0)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(80e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(120e-9, Material.Static(1.4))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))

        R = tmm.GetIntensityMatrix()
        np.testing.assert_allclose(R[0, 0], R[1, 1], rtol=1e-10)
