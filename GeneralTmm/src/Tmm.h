#pragma once
#include "Common.h"
#include "Layer.h"
#include "Material.h"
#include <array>

namespace tmm {
//---------------------------------------------------------------------
// Tmm
//---------------------------------------------------------------------

class Tmm {
public:
    Tmm();
    void SetParam(Param param, int value);
    void SetParam(Param param, double value);
    void SetParam(Param param, dcomplex value);
    [[nodiscard]] int GetParamInt(Param param) const;
    [[nodiscard]] double GetParamDouble(Param param) const;
    [[nodiscard]] dcomplex GetParamComplex(Param param) const;
    void AddIsotropicLayer(double d, Material* mat);
    void AddLayer(double d, Material* matx, Material* maty, Material* matz, double psi, double xi);
    void ClearLayers() noexcept;
    [[nodiscard]] const Matrix4d& GetIntensityMatrix();
    [[nodiscard]] const Matrix4cd& GetAmplitudeMatrix();
    [[nodiscard]] SweepRes Sweep(Param sweepParam, const Eigen::Map<Eigen::ArrayXd>& sweepValues,
                                 const PositionSettings& enhpos, int alphasLayer);
    [[nodiscard]] SweepRes Sweep(Param sweepParam, const Eigen::Map<Eigen::ArrayXd>& sweepValues);
    [[nodiscard]] EMFieldsList CalcFields1D(const Eigen::Map<Eigen::ArrayXd>& xs,
                                            const Eigen::Map<Eigen::Array2d>& polarization,
                                            WaveDirection waveDirection);
    [[nodiscard]] EMFields CalcFieldsAtInterface(const PositionSettings& pos, WaveDirection waveDirection);
    [[nodiscard]] double OptimizeEnhancement(const std::vector<Param>& optParams, const ArrayXd& optInitial,
                                             const PositionSettings& pos);

private:
    double wl_;
    double beta_;
    double enhOptMaxRelError_;
    double enhOptInitialStep_;
    long enhOptMaxIters_;
    std::vector<Layer> layers_;

    static constexpr int kMatrixSize = 4;
    std::array<std::array<std::string, kMatrixSize>, kMatrixSize> names_R_;
    std::array<std::array<std::string, kMatrixSize>, kMatrixSize> names_r_;

    Matrix4cd A_;
    bool needToSolve_;
    bool needToCalcFieldCoefs_;
    RowVector2d lastFieldCoefsPol_;
    Matrix4d R_;
    Matrix4cd r_;

    double normCoef_;
    MatrixXcd fieldCoefs_;

    void Solve();
    void CalcFieldCoefs(const RowVector2d& polarization);
    LayerIndices CalcLayerIndices(const Eigen::Map<Eigen::ArrayXd>& xs);
};

//---------------------------------------------------------------------
// EnhFitStruct
//---------------------------------------------------------------------

class EnhFitStruct {
    friend Tmm;

public:
    using DataType = double;
    using ParameterType = VectorXd;

    // Non-owning pointer. Lifetime managed by caller.
    EnhFitStruct(Tmm* tmm, const std::vector<Param>& optParams, const PositionSettings& enhpos);

    DataType operator()(const ParameterType& params) const {
        SetParams(params);
        EMFields r = tmm_->CalcFieldsAtInterface(enhPos_, WaveDirection::WD_BOTH);
        return -r.E.matrix().norm();
    }

private:
    Tmm* tmm_;
    std::vector<Param> optParams_;
    PositionSettings enhPos_;
    void SetParams(const ParameterType& params) const;
};
} // namespace tmm
