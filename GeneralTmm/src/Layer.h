#pragma once
#include "Common.h"
#include "Material.h"
#include <Eigen/Eigenvalues>

namespace tmm {

//---------------------------------------------------------------------
// Layer
//---------------------------------------------------------------------

class Layer {
    // Tmm accesses internal Eigen matrices (M_, F_, invF_, alpha_, poyntingX_,
    // fieldCoefs_) directly for performance in the matrix chain multiplication.
    friend class Tmm;

public:
    Layer(double d, Material* n);
    Layer(double d, Material* nx, Material* ny, Material* nz, double psi, double xi);

    // Rule of five: ensure noexcept moves for efficient std::vector<Layer> reallocation.
    Layer(const Layer&) = default;
    Layer& operator=(const Layer&) = default;
    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;
    ~Layer() = default;

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
    [[nodiscard]] EMFields GetFields(double wl, double beta, double x, const Eigen::Ref<const Vector4cd>& coefs,
                                     WaveDirection waveDirection) const;

private:
    bool solved_ = false;
    bool isotropicLayer_ = false;
    bool epsilonRefractiveIndexChanged_ = true;
    double wlEpsilonCalc_ = 0.0;

    double d_ = 0.0;
    // Non-owning pointers. Lifetime managed by caller (Cython materialsCache).
    Material* nx_ = nullptr;
    Material* ny_ = nullptr;
    Material* nz_ = nullptr;
    double psi_ = 0.0;
    double xi_ = 0.0;

    Eigen::ComplexEigenSolver<Matrix4cd> ces_{4};
    Matrix3cd epsTensor_ = Matrix3cd::Zero();
    Vector4cd alpha_ = Vector4cd::Zero();
    Vector4d poyntingX_ = Vector4d::Zero();
    Matrix4cd F_ = Matrix4cd::Zero();
    Matrix4cd invF_ = Matrix4cd::Zero();
    Matrix4cd phaseMatrix_ = Matrix4cd::Identity();
    Matrix4cd M_ = Matrix4cd::Zero();

    void SolveEpsilonMatrix(double wl);
    void SolveEigenFunction(double beta);
};
} // namespace tmm
