"""API functionality tests for TMM library.

These tests verify that the TMM API functions work correctly,
including proper return types, shapes, and basic functionality.

This does NOT test physics correctness (see test_physics.py) or
comparison against reference implementation (see test_cpp_vs_python.py).

Test categories:
1. Material Class - Creation, interpolation, static helper
2. Tmm Class - Layer management, parameter setting
3. Sweep Functions - beta and wavelength sweeps
4. Field Calculations - 1D and 2D field computations
5. Matrix Access - Amplitude and intensity matrices
"""

import numpy as np
import pytest

from GeneralTmm import Material, Tmm

# Sample silver data for tests
SILVER_WLS = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
SILVER_NS = np.array(
    [
        0.050 + 2.104j,
        0.050 + 3.131j,
        0.055 + 4.010j,
        0.041 + 4.803j,
        0.037 + 5.570j,
    ]
)


class TestMaterialClass:
    """Tests for Material class functionality."""

    def test_material_creation(self):
        """Test basic Material creation with wavelength and n data."""
        mat = Material(SILVER_WLS, SILVER_NS)
        n = mat(500e-9)
        assert np.isfinite(n.real)
        assert np.isfinite(n.imag)

    def test_material_static(self):
        """Test Material.Static helper for constant refractive index."""
        mat = Material.Static(1.5)
        assert mat(400e-9) == pytest.approx(1.5)
        assert mat(800e-9) == pytest.approx(1.5)

    def test_material_static_complex(self):
        """Test Material.Static with complex refractive index."""
        mat = Material.Static(1.5 + 0.1j)
        n = mat(532e-9)
        assert n.real == pytest.approx(1.5)
        assert n.imag == pytest.approx(0.1)

    def test_material_interpolation(self):
        """Test that Material interpolates at intermediate wavelengths."""
        mat = Material(SILVER_WLS, SILVER_NS)
        n_500 = mat(500e-9)
        assert n_500.real == pytest.approx(0.050, rel=0.01)
        assert n_500.imag == pytest.approx(3.131, rel=0.01)

        n_550 = mat(550e-9)
        assert np.isfinite(n_550.real)
        assert np.isfinite(n_550.imag)
        assert n_550.imag > 3.0
        assert n_550.imag < 5.0


class TestTmmLayerManagement:
    """Tests for Tmm layer management."""

    def test_add_isotropic_layer(self):
        """Test adding isotropic layers."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        tmm.SetParams(beta=0.3)
        R = tmm.GetIntensityMatrix()
        assert R.shape == (4, 4)

    def test_add_anisotropic_layer(self):
        """Test adding anisotropic layers."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9)
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

        tmm.SetParams(beta=0.3)
        R = tmm.GetIntensityMatrix()
        assert R.shape == (4, 4)

    def test_clear_layers(self):
        """Test that ClearLayers resets the structure."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        betas = np.linspace(0.0, 0.5, 10)
        res1 = tmm.Sweep("beta", betas)
        r11_original = res1["R11"].copy()

        tmm.ClearLayers()
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))

        res2 = tmm.Sweep("beta", betas)

        assert not np.allclose(r11_original, res2["R11"], rtol=0.01)


class TestTmmParameters:
    """Tests for Tmm parameter setting."""

    def test_set_params_wl(self):
        """Test setting wavelength parameter."""
        tmm = Tmm(wl=532e-9)
        assert tmm.wl == pytest.approx(532e-9)

        tmm.SetParams(wl=600e-9)
        assert tmm.wl == pytest.approx(600e-9)

    def test_set_params_beta(self):
        """Test setting beta parameter."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.5)
        assert tmm.beta == pytest.approx(0.5)

        tmm.beta = 0.3
        assert tmm.beta == pytest.approx(0.3)

    def test_set_params_via_property(self):
        """Test setting parameters via properties."""
        tmm = Tmm()
        tmm.wl = 632e-9
        tmm.beta = 0.4

        assert tmm.wl == pytest.approx(632e-9)
        assert tmm.beta == pytest.approx(0.4)


class TestSweepFunctions:
    """Tests for sweep functionality."""

    def test_beta_sweep(self):
        """Test sweeping over beta parameter."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        betas = np.linspace(0.0, 0.95, 20)
        res = tmm.Sweep("beta", betas)

        assert len(res["R11"]) == len(betas)
        assert len(res["R22"]) == len(betas)
        assert len(res["T31"]) == len(betas)
        assert len(res["T42"]) == len(betas)

    def test_wl_sweep(self):
        """Test sweeping over wavelength parameter."""
        tmm = Tmm()
        tmm.SetParams(beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        wls = np.linspace(500e-9, 700e-9, 10)
        res = tmm.Sweep("wl", wls)

        assert len(res["R11"]) == len(wls)
        assert len(res["T31"]) == len(wls)

    def test_sweep_with_enhancement(self):
        """Test sweep with enhancement position."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        betas = np.linspace(0.0, 0.95, 20)
        pol = (1.0, 0.0)
        enhPos = (pol, 2, 0.0)

        res = tmm.Sweep("beta", betas, enhPos)

        assert len(res["enh"]) == len(betas)
        assert np.all(res["enh"] >= 0)


class TestFieldCalculations:
    """Tests for field calculation functions."""

    def test_calc_fields_1d_shape(self):
        """Test CalcFields1D returns correct shape."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        xs = np.linspace(-200e-9, 200e-9, 50)
        pol = np.array([1.0, 0.0])

        E, H = tmm.CalcFields1D(xs, pol)

        assert E.shape == (len(xs), 3)
        assert H.shape == (len(xs), 3)

    def test_calc_fields_1d_finite(self):
        """Test CalcFields1D returns finite values."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        xs = np.linspace(-200e-9, 200e-9, 50)
        pol = np.array([1.0, 0.0])

        E, H = tmm.CalcFields1D(xs, pol)

        assert np.all(np.isfinite(E))
        assert np.all(np.isfinite(H))

    def test_calc_fields_2d_shape(self):
        """Test CalcFields2D returns correct shape."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        xs = np.linspace(-200e-9, 200e-9, 20)
        ys = np.linspace(-100e-9, 100e-9, 15)
        pol = np.array([1.0, 0.0])

        E, H = tmm.CalcFields2D(xs, ys, pol)

        assert E.shape == (len(xs), len(ys), 3)
        assert H.shape == (len(xs), len(ys), 3)

    def test_calc_fields_2d_consistency(self):
        """2D fields at y=0 should match 1D fields."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        xs = np.linspace(-200e-9, 200e-9, 30)
        ys = np.array([0.0])
        pol = np.array([1.0, 0.0])

        E_2d, H_2d = tmm.CalcFields2D(xs, ys, pol)
        E_1d, H_1d = tmm.CalcFields1D(xs, pol)

        np.testing.assert_allclose(np.abs(E_2d[:, 0, :]), np.abs(E_1d), rtol=1e-7)
        np.testing.assert_allclose(np.abs(H_2d[:, 0, :]), np.abs(H_1d), rtol=1e-7)

    def test_calc_fields_at_interface(self):
        """Test CalcFieldsAtInterface returns valid fields."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        pol = (1.0, 0.0)
        enhPos = (pol, 2, 0.0)

        E, H = tmm.CalcFieldsAtInterface(enhPos)

        assert len(E) == 3
        assert len(H) == 3
        assert np.all(np.isfinite(E))
        assert np.all(np.isfinite(H))


class TestMatrixAccess:
    """Tests for amplitude and intensity matrix access."""

    def test_get_intensity_matrix_shape(self):
        """Test GetIntensityMatrix returns 4x4 matrix."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        R = tmm.GetIntensityMatrix()

        assert R.shape == (4, 4)

    def test_get_intensity_matrix_values(self):
        """Test GetIntensityMatrix returns valid values."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        R = tmm.GetIntensityMatrix()

        assert R[0, 0] >= 0
        assert R[1, 1] >= 0

    def test_get_amplitude_matrix_shape(self):
        """Test GetAmplitudeMatrix returns 4x4 complex matrix."""
        tmm = Tmm()
        tmm.SetParams(wl=532e-9, beta=0.3)
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.5))
        tmm.AddIsotropicLayer(100e-9, Material.Static(1.8))
        tmm.AddIsotropicLayer(float("inf"), Material.Static(1.0))

        r = tmm.GetAmplitudeMatrix()

        assert r.shape == (4, 4)
        assert np.iscomplexobj(r)
