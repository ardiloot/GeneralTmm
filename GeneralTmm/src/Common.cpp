#include "Common.h"

namespace TmmModel {

	template<typename T> T Interpolate(double x, const ArrayXd & xs, const Eigen::Array<T, Eigen::Dynamic, 1> & ys) {
		// xs must be sorted

		// Check range
		if (x < xs(0) || x >= xs(xs.size() - 1)) {
			throw std::runtime_error("Interpolation out of range");
		}

		if (xs(0) >= xs(xs.size() - 1)) {
			throw std::runtime_error("Interpolation: xs must be sorted");
		}

		// Binary search (last element, that is less or equal than x)
		int b = 0, e = xs.size() - 1;
		while (b < e) {
			int a = (b + e) / 2;
			if (xs(b) >= xs(e)) {
				throw std::runtime_error("Interpolation: xs must be sorted");
			}

			if (xs(a) > x) {
				// [b..a[
				e = a - 1;

				if (xs(e) <= x) {
					b = e;
				}
			}
			else {
				// [a..e]
				b = a;
				if (xs(a + 1) > x) {
					e = a;
				}
			}
		}
		// Linear interpolation in range x[b]..x[b+1]
		double dx = xs(b + 1) - xs(b);
		T dy = ys(b + 1) - ys(b);
		T res = ys(b) + dy / dx * (x - xs(b));
		return res;
	}

	template dcomplex Interpolate<dcomplex>(double, const ArrayXd &, const ArrayXcd &);
	template double Interpolate<double>(double, const ArrayXd &, const ArrayXd &);

	double sqr(double a) {
		return a * a;
	}
	dcomplex sqr(dcomplex a) {
		return a * a;
	}

	Matrix3cd RotationSx(double phi) {
		Matrix3cd res;
		res << 1.0, 0.0, 0.0,
			0.0, cos(phi), -sin(phi),
			0.0, sin(phi), cos(phi);
		return res;
	}

	Matrix3cd RotationSz(double phi) {
		Matrix3cd res;
		res << cos(phi), -sin(phi), 0.0,
			sin(phi), cos(phi), 0.0,
			0.0, 0.0, 1.0;
		return res;
	}

	Param::Param(ParamType pType_) {
		pType = pType_;
		layerId = -1;
	}

	Param::Param(ParamType pType_, int layerId_) {
		pType = pType_;
		layerId = layerId_;
	}

	ParamType Param::GetParamType() {
		return pType;
	}

	int Param::GetLayerID() {
		return layerId;
	}
	PositionSettings::PositionSettings(RowVector2d polarization_, int interfaceId_, double distFromInterface_) {
		polarization = polarization_;
		interfaceId = interfaceId_;
		distFromInterface = distFromInterface_;
		enabled = true;
	}
	PositionSettings::PositionSettings(double polCoef1, double polCoef2, int interfaceId_, double distFromInterface_) {
		polarization << polCoef1, polCoef2;
		interfaceId = interfaceId_;
		distFromInterface = distFromInterface_;
		enabled = true;
	}
	PositionSettings::PositionSettings() {
		polarization.setZero();
		interfaceId = -1;
		distFromInterface = 0.0;
		enabled = false;
	}
	RowVector2d PositionSettings::GetPolarization() {
		return polarization;
	}
	int PositionSettings::GetInterfaceId() {
		return interfaceId;
	}
	double PositionSettings::GetDistFromInterface() {
		return distFromInterface;
	}
	bool PositionSettings::IsEnabled() {
		return enabled;
	}
}