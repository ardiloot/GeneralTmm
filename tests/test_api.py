"""API tests for the TMM library.

Verifies return types, shapes, and basic functionality — not physics
correctness (see test_physics.py) or C++/Python parity (see
test_cpp_vs_python.py).
"""

import numpy as np
import pytest

from GeneralTmm import Material, Tmm

SILVER_WLS = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
SILVER_NS = np.array([0.050 + 2.104j, 0.050 + 3.131j, 0.055 + 4.010j, 0.041 + 4.803j, 0.037 + 5.570j])


@pytest.fixture
def glass_film_air():
    """Glass (1.5) | 100 nm film (1.8) | air — 532 nm, β = 0.3."""
    tmm = Tmm(wl=532e-9, beta=0.3)
    tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
    tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
    tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
    return tmm


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


class TestMaterialClass:
    def test_creation_and_interpolation(self):
        mat = Material(SILVER_WLS, SILVER_NS)
        n = mat(500e-9)
        assert n.real == pytest.approx(0.050, rel=0.01)
        assert n.imag == pytest.approx(3.131, rel=0.01)

        # Intermediate wavelength returns a sensible value
        n_mid = mat(550e-9)
        assert 3.0 < n_mid.imag < 5.0

    def test_interpolation_endpoint(self):
        """Evaluation at the last wavelength should return the exact data point."""
        mat = Material(SILVER_WLS, SILVER_NS)
        n = mat(SILVER_WLS[-1])
        assert n.real == pytest.approx(SILVER_NS[-1].real)
        assert n.imag == pytest.approx(SILVER_NS[-1].imag)

    def test_static_real(self):
        mat = Material.Static(1.5)
        assert mat(400e-9) == pytest.approx(1.5)
        assert mat(800e-9) == pytest.approx(1.5)

    def test_static_complex(self):
        mat = Material.Static(1.5 + 0.1j)
        n = mat(532e-9)
        assert n.real == pytest.approx(1.5)
        assert n.imag == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Layer management
# ---------------------------------------------------------------------------


class TestTmmLayerManagement:
    def test_add_isotropic_layer(self):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.beta = 0.3
        assert tmm.GetIntensityMatrix().shape == (4, 4)

    def test_add_anisotropic_layer(self):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddLayer(
            100e-9,
            Material.Static(1.55),
            Material.Static(1.58),
            Material.Static(1.62),
            np.pi / 4,
            np.pi / 6,
        )
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.beta = 0.3
        assert tmm.GetIntensityMatrix().shape == (4, 4)

    def test_clear_layers(self):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        betas = np.linspace(0.0, 0.5, 10)
        r11_before = tmm.Sweep("beta", betas)["R11"].copy()

        tmm.ClearLayers()
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))

        r11_after = tmm.Sweep("beta", betas)["R11"]
        assert not np.allclose(r11_before, r11_after, rtol=0.01)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class TestTmmParameters:
    def test_set_wl(self):
        tmm = Tmm(wl=532e-9)
        assert tmm.wl == pytest.approx(532e-9)
        tmm.SetParams(wl=600e-9)
        assert tmm.wl == pytest.approx(600e-9)

    def test_set_beta(self):
        tmm = Tmm(wl=532e-9, beta=0.5)
        assert tmm.beta == pytest.approx(0.5)
        tmm.beta = 0.3
        assert tmm.beta == pytest.approx(0.3)

    def test_properties(self):
        tmm = Tmm()
        tmm.wl = 632e-9
        tmm.beta = 0.4
        assert tmm.wl == pytest.approx(632e-9)
        assert tmm.beta == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


class TestSweepFunctions:
    def test_beta_sweep(self):
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        betas = np.linspace(0.0, 0.95, 20)
        res = tmm.Sweep("beta", betas)
        for key in ("R11", "R22", "T31", "T42"):
            assert len(res[key]) == len(betas)

    def test_wl_sweep(self, glass_film_air):
        wls = np.linspace(500e-9, 700e-9, 10)
        res = glass_film_air.Sweep("wl", wls)
        assert len(res["R11"]) == len(wls)

    def test_sweep_with_enhancement(self, glass_film_air):
        betas = np.linspace(0.0, 0.95, 20)
        res = glass_film_air.Sweep("beta", betas, ((1.0, 0.0), 2, 0.0))
        assert len(res["enh"]) == len(betas)
        assert np.all(res["enh"] >= 0)

    def test_sweep_invalid_alpha_layer(self):
        """alphaLayer outside the valid range should raise."""
        tmm = Tmm(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        with pytest.raises((RuntimeError, ValueError)):
            tmm.Sweep("beta", np.linspace(0.0, 0.95, 5), alphaLayer=5)


# ---------------------------------------------------------------------------
# Field calculations
# ---------------------------------------------------------------------------


class TestFieldCalculations:
    def test_fields_1d(self, glass_film_air):
        xs = np.linspace(-200e-9, 200e-9, 50)
        E, H = glass_film_air.CalcFields1D(xs, np.array([1.0, 0.0]))
        assert E.shape == (len(xs), 3) and H.shape == (len(xs), 3)
        assert np.all(np.isfinite(E)) and np.all(np.isfinite(H))

    def test_fields_2d(self, glass_film_air):
        xs = np.linspace(-200e-9, 200e-9, 20)
        ys = np.linspace(-100e-9, 100e-9, 15)
        E, H = glass_film_air.CalcFields2D(xs, ys, np.array([1.0, 0.0]))
        assert E.shape == (len(xs), len(ys), 3)
        assert H.shape == (len(xs), len(ys), 3)

    def test_fields_2d_matches_1d_at_y0(self, glass_film_air):
        """2D fields at y = 0 should match 1D fields."""
        xs = np.linspace(-200e-9, 200e-9, 30)
        pol = np.array([1.0, 0.0])
        E_2d, H_2d = glass_film_air.CalcFields2D(xs, np.array([0.0]), pol)
        E_1d, H_1d = glass_film_air.CalcFields1D(xs, pol)
        np.testing.assert_allclose(np.abs(E_2d[:, 0, :]), np.abs(E_1d), rtol=1e-7)
        np.testing.assert_allclose(np.abs(H_2d[:, 0, :]), np.abs(H_1d), rtol=1e-7)

    def test_fields_at_interface(self, glass_film_air):
        E, H = glass_film_air.CalcFieldsAtInterface(((1.0, 0.0), 2, 0.0))
        assert len(E) == 3 and len(H) == 3
        assert np.all(np.isfinite(E)) and np.all(np.isfinite(H))


# ---------------------------------------------------------------------------
# Matrix access
# ---------------------------------------------------------------------------


class TestMatrixAccess:
    def test_intensity_matrix(self, glass_film_air):
        R = glass_film_air.GetIntensityMatrix()
        assert R.shape == (4, 4)
        assert R[0, 0] >= 0 and R[1, 1] >= 0

    def test_amplitude_matrix(self, glass_film_air):
        r = glass_film_air.GetAmplitudeMatrix()
        assert r.shape == (4, 4)
        assert np.iscomplexobj(r)
