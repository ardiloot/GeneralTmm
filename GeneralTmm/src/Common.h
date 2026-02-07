#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <complex>
#include <string>
#include <map>
#include <limits>
#include <stdexcept>
#include <Eigen/Dense>

namespace tmm {
//---------------------------------------------------------------
// Namespaces
//---------------------------------------------------------------

using Eigen::Array2cd;
using Eigen::ArrayXcd;
using Eigen::ArrayXd;
using Eigen::Matrix2cd;
using Eigen::Matrix3cd;
using Eigen::Matrix4cd;
using Eigen::Matrix4d;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::RowVector2d;
using Eigen::Vector2d;
using Eigen::Vector4cd;
using Eigen::Vector4d;
using Eigen::Vector4i;
using Eigen::VectorXd;
using Eigen::VectorXi;

using std::abs;
using std::norm;

//---------------------------------------------------------------
// Type aliases
//---------------------------------------------------------------

using dcomplex = std::complex<double>;
using ComplexVectorMap = std::map<std::string, ArrayXcd>;
using DoubleVectorMap = std::map<std::string, ArrayXd>;

//---------------------------------------------------------------
// Constants
//---------------------------------------------------------------

inline constexpr double PI = 3.14159265358979323846;
inline constexpr double Z0 = 119.9169832 * PI;
inline constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr dcomplex I{0.0, 1.0};

//---------------------------------------------------------------
// Functions
//---------------------------------------------------------------

[[nodiscard]] inline constexpr double sqr(double a) noexcept {
    return a * a;
}
[[nodiscard]] inline dcomplex sqr(dcomplex a) noexcept {
    return a * a;
}
template <typename T>
[[nodiscard]] T Interpolate(double x, const ArrayXd& xs, const Eigen::Array<T, Eigen::Dynamic, 1>& ys);
[[nodiscard]] Matrix3cd RotationSx(double phi) noexcept;
[[nodiscard]] Matrix3cd RotationSz(double phi) noexcept;

//---------------------------------------------------------------------
// Enums
//---------------------------------------------------------------------

enum class ParamType {
    WL,
    BETA,
    ENH_OPT_REL,
    ENH_OPT_MAX_ITERS,
    ENH_INITIAL_STEP,
    LAYER_D,
    LAYER_N,
    LAYER_NX,
    LAYER_NY,
    LAYER_NZ,
    LAYER_PSI,
    LAYER_XI,
    LAYER_MAT_NX,
    LAYER_MAT_NY,
    LAYER_MAT_NZ,
    NOT_DEFINED
};

enum class WaveDirection { WD_FORWARD, WD_BACKWARD, WD_BOTH };

//---------------------------------------------------------------------
// Param
//---------------------------------------------------------------------

class Param {
public:
    Param() : Param(ParamType::NOT_DEFINED) {}
    explicit Param(ParamType pType);
    Param(ParamType pType, int layerId);
    [[nodiscard]] ParamType GetParamType() const noexcept;
    [[nodiscard]] int GetLayerID() const noexcept;

    bool operator==(const Param& p) const noexcept { return (pType_ == p.pType_ && layerId_ == p.layerId_); }

private:
    ParamType pType_;
    int layerId_;
};

//---------------------------------------------------------------------
// PositionSettings
//---------------------------------------------------------------------

struct PositionSettings {
    PositionSettings();
    PositionSettings(const RowVector2d& polarization, int interfaceId, double distFromInterface);
    PositionSettings(double polCoef1, double polCoef2, int interfaceId, double distFromInterface);
    [[nodiscard]] const RowVector2d& GetPolarization() const noexcept;
    [[nodiscard]] int GetInterfaceId() const noexcept;
    [[nodiscard]] double GetDistFromInterface() const noexcept;
    [[nodiscard]] bool IsEnabled() const noexcept;

private:
    bool enabled_ = false;
    RowVector2d polarization_ = RowVector2d::Zero();
    int interfaceId_ = -1;
    double distFromInterface_ = 0.0;
};


//---------------------------------------------------------------------
// EMFields
//---------------------------------------------------------------------

struct EMFields {
    ArrayXcd E;
    ArrayXcd H;

    EMFields() : E(3), H(3) {}
};

//---------------------------------------------------------------------
// EMFieldsList
//---------------------------------------------------------------------

struct EMFieldsList {
    MatrixXcd E;
    MatrixXcd H;

    explicit EMFieldsList(int size = 0) : E(size, 3), H(size, 3) {}
};

//---------------------------------------------------------------------
// LayerIndices
//---------------------------------------------------------------------

struct LayerIndices {
    VectorXi indices;
    VectorXd ds;
};

//---------------------------------------------------------------------
// SweepRes
//---------------------------------------------------------------------

struct SweepRes {
    ComplexVectorMap mapComplex;
    DoubleVectorMap mapDouble;
};


} // namespace tmm
