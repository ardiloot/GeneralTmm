#include "Common.h"

namespace tmm {

template <typename T> T Interpolate(double x, const ArrayXd& xs, const Eigen::Array<T, Eigen::Dynamic, 1>& ys) {
    // xs must be sorted

    // Check range
    if (x < xs(0) || x >= xs(xs.size() - 1)) {
        throw std::runtime_error("Interpolation out of range");
    }

    if (xs(0) >= xs(xs.size() - 1)) {
        throw std::runtime_error("Interpolation: xs must be sorted");
    }

    // Binary search (last element, that is less or equal than x)
    Eigen::Index b = 0, e = xs.size() - 1;
    while (b < e) {
        Eigen::Index a = (b + e) / 2;
        if (xs(b) >= xs(e)) {
            throw std::runtime_error("Interpolation: xs must be sorted");
        }

        if (xs(a) > x) {
            // [b..a[
            e = a - 1;

            if (xs(e) <= x) {
                b = e;
            }
        } else {
            // [a..e]
            b = a;
            if (xs(a + 1) > x) {
                e = a;
            }
        }
    }
    // Linear interpolation in range x[b]..x[b+1]
    const double dx = xs(b + 1) - xs(b);
    const T dy = ys(b + 1) - ys(b);
    return ys(b) + dy / dx * (x - xs(b));
}

template dcomplex Interpolate<dcomplex>(double, const ArrayXd&, const ArrayXcd&);
template double Interpolate<double>(double, const ArrayXd&, const ArrayXd&);

Matrix3cd RotationSx(double phi) noexcept {
    Matrix3cd res;
    res << 1.0, 0.0, 0.0, 0.0, cos(phi), -sin(phi), 0.0, sin(phi), cos(phi);
    return res;
}

Matrix3cd RotationSz(double phi) noexcept {
    Matrix3cd res;
    res << cos(phi), -sin(phi), 0.0, sin(phi), cos(phi), 0.0, 0.0, 0.0, 1.0;
    return res;
}

Param::Param(ParamType pType) : pType_(pType), layerId_(-1) {}

Param::Param(ParamType pType, int layerId) : pType_(pType), layerId_(layerId) {}

ParamType Param::GetParamType() const noexcept {
    return pType_;
}

int Param::GetLayerID() const noexcept {
    return layerId_;
}
PositionSettings::PositionSettings(const RowVector2d& polarization, int interfaceId, double distFromInterface)
    : enabled_(true), polarization_(polarization), interfaceId_(interfaceId), distFromInterface_(distFromInterface) {}

PositionSettings::PositionSettings(double polCoef1, double polCoef2, int interfaceId, double distFromInterface)
    : enabled_(true), polarization_{polCoef1, polCoef2}, interfaceId_(interfaceId),
      distFromInterface_(distFromInterface) {}

PositionSettings::PositionSettings() = default;
const RowVector2d& PositionSettings::GetPolarization() const noexcept {
    return polarization_;
}
int PositionSettings::GetInterfaceId() const noexcept {
    return interfaceId_;
}
double PositionSettings::GetDistFromInterface() const noexcept {
    return distFromInterface_;
}
bool PositionSettings::IsEnabled() const noexcept {
    return enabled_;
}
} // namespace tmm
