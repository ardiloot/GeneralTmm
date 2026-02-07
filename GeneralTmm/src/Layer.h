#pragma once
#include "Common.h"
#include "Material.h"
#include <Eigen/Eigenvalues>

namespace TmmModel {

//---------------------------------------------------------------------
// Layer
//---------------------------------------------------------------------

class Layer {
    friend class Tmm;

public:
    Layer(double d, Material* n);
    Layer(double d, Material* nx, Material* ny, Material* nz, double psi, double xi);
    void SetParam(Param param, int value);
    void SetParam(Param param, double value);
    void SetParam(Param param, dcomplex value);
    [[nodiscard]] int GetParamInt(Param param) const;
    [[nodiscard]] double GetParamDouble(Param param) const;
    [[nodiscard]] dcomplex GetParamComplex(Param param) const;
    [[nodiscard]] double GetD() const noexcept;
    [[nodiscard]] dcomplex GetNx(double wl) const;
    [[nodiscard]] dcomplex GetNy(double wl) const;
    [[nodiscard]] dcomplex GetNz(double wl) const;
    void SolveLayer(double wl, double beta);
    [[nodiscard]] EMFields GetFields(double wl, double beta, double x, const Vector4cd& coefs,
                                     WaveDirection waveDirection) const;

private:
    bool solved_ = false;
    bool isotropicLayer_ = false;
    bool epsilonRefractiveIndexChanged_ = true;
    double wlEpsilonCalc_ = 0.0;

    double d_ = 0.0;
    Material* nx_ = nullptr;
    Material* ny_ = nullptr;
    Material* nz_ = nullptr;
    double psi_ = 0.0;
    double xi_ = 0.0;

    Eigen::ComplexEigenSolver<Matrix4cd> ces_;
    Matrix3cd epsTensor_;
    Vector4cd alpha_;
    Vector4d poyntingX_;
    Matrix4cd F_;
    Matrix4cd invF_;
    Matrix4cd phaseMatrix_;
    Matrix4cd M_;

    void SolveEpsilonMatrix(double wl);
    void SolveEigenFunction(double beta);
};
} // namespace TmmModel