"""Tests for Material class functionality."""

import numpy as np
import pytest

from GeneralTmm import Material


class TestMaterialStatic:
    """Tests for static material creation."""

    def test_static_real_refractive_index(self):
        """Static material with real refractive index."""
        mat = Material.Static(1.5)
        assert mat(532e-9) == pytest.approx(1.5)

    def test_static_complex_refractive_index(self):
        """Static material with complex refractive index."""
        n_complex = 1.5 + 0.1j
        mat = Material.Static(n_complex)
        result = mat(532e-9)
        assert result.real == pytest.approx(1.5)
        assert result.imag == pytest.approx(0.1)

    def test_static_wavelength_independent(self):
        """Static material should return same n at different wavelengths."""
        mat = Material.Static(2.0)
        assert mat(400e-9) == pytest.approx(2.0)
        assert mat(600e-9) == pytest.approx(2.0)
        assert mat(800e-9) == pytest.approx(2.0)


class TestMaterialInterpolated:
    """Tests for interpolated material from tabulated data."""

    @pytest.fixture
    def silver_material(self):
        """Silver material data from Johnson and Christy."""
        wls = np.array([400e-9, 500e-9, 600e-9, 700e-9, 800e-9])
        ns = np.array(
            [
                0.050 + 2.104j,
                0.050 + 3.131j,
                0.055 + 4.010j,
                0.041 + 4.803j,
                0.037 + 5.570j,
            ]
        )
        return Material(wls, ns)

    def test_interpolation_at_data_points(self, silver_material):
        """Material should return exact values at tabulated wavelengths."""
        result = silver_material(500e-9)
        assert result.real == pytest.approx(0.050, rel=1e-3)
        assert result.imag == pytest.approx(3.131, rel=1e-3)

    def test_interpolation_between_points(self, silver_material):
        """Material should interpolate between data points."""
        result = silver_material(550e-9)
        # Should be between values at 500nm and 600nm
        assert 0.050 <= result.real <= 0.055
        assert 3.131 <= result.imag <= 4.010

    def test_extrapolation_warning(self, silver_material):
        """Material should handle wavelengths outside data range."""
        # Test that it doesn't crash for wavelengths outside range
        result = silver_material(350e-9)
        assert np.isfinite(result.real)
        assert np.isfinite(result.imag)


class TestMaterialEdgeCases:
    """Tests for edge cases in Material class."""

    def test_unity_refractive_index(self):
        """Material with n=1 (vacuum)."""
        mat = Material.Static(1.0)
        assert mat(532e-9) == pytest.approx(1.0)

    def test_high_refractive_index(self):
        """Material with high refractive index."""
        mat = Material.Static(4.0)
        assert mat(532e-9) == pytest.approx(4.0)

    def test_lossy_material(self):
        """Material with significant absorption."""
        mat = Material.Static(0.1 + 5.0j)
        result = mat(532e-9)
        assert result.real == pytest.approx(0.1)
        assert result.imag == pytest.approx(5.0)
