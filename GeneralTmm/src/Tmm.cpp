#include "Tmm.h"
#include "criteria.h"
#include "simplex.h"

namespace tmm {

//---------------------------------------------------------------------
// Tmm
//---------------------------------------------------------------------

Tmm::Tmm()
    : wl_(500e-9), beta_(0.0), enhOptMaxRelError_(1e-10), enhOptInitialStep_(0.1), enhOptMaxIters_(100),
      needToSolve_(true), needToCalcFieldCoefs_(true), normCoef_(0.0) {
    for (int i = 0; i < kMatrixSize; i++) {
        for (int j = 0; j < kMatrixSize; j++) {
            auto numbers = std::to_string(i + 1) + std::to_string(j + 1);
            if ((i < 2 && j < 2) || (i >= 2 && j >= 2)) {
                names_R_[i][j] = "R" + numbers;
                names_r_[i][j] = "r" + numbers;
            } else {
                names_R_[i][j] = "T" + numbers;
                names_r_[i][j] = "t" + numbers;
            }
        }
    }
}

void Tmm::SetParam(Param param, int value) {
    needToSolve_ = true;
    if (param.GetLayerID() < 0) {
        switch (param.GetParamType()) {
        case ParamType::ENH_OPT_MAX_ITERS:
            enhOptMaxIters_ = static_cast<long>(value);
            break;
        default:
            throw std::invalid_argument("Invalid param int");
        }
    } else {
        layers_[param.GetLayerID()].SetParam(param, value);
    }
}

void Tmm::SetParam(Param param, double value) {
    needToSolve_ = true;
    if (param.GetLayerID() < 0) {
        switch (param.GetParamType()) {
        case ParamType::WL:
            wl_ = value;
            break;
        case ParamType::BETA:
            beta_ = value;
            break;
        case ParamType::ENH_OPT_REL:
            enhOptMaxRelError_ = value;
            break;
        case ParamType::ENH_INITIAL_STEP:
            enhOptInitialStep_ = value;
            break;
        default:
            throw std::invalid_argument("Invalid param double");
        }
    } else {
        layers_[param.GetLayerID()].SetParam(param, value);
    }
}

void Tmm::SetParam(Param param, dcomplex value) {
    needToSolve_ = true;
    if (param.GetLayerID() < 0) {
        throw std::invalid_argument("Invalid param complex");
    } else {
        layers_[param.GetLayerID()].SetParam(param, value);
    }
}

int Tmm::GetParamInt(Param param) const {
    if (param.GetLayerID() < 0) {
        switch (param.GetParamType()) {
        case ParamType::ENH_OPT_MAX_ITERS:
            return static_cast<int>(enhOptMaxIters_);
        default:
            throw std::invalid_argument("Get invalid param int");
        }
    } else {
        return layers_[param.GetLayerID()].GetParamInt(param);
    }
}

double Tmm::GetParamDouble(Param param) const {
    if (param.GetLayerID() < 0) {
        switch (param.GetParamType()) {
        case ParamType::WL:
            return wl_;
        case ParamType::BETA:
            return beta_;
        case ParamType::ENH_OPT_REL:
            return enhOptMaxRelError_;
        case ParamType::ENH_INITIAL_STEP:
            return enhOptInitialStep_;
        default:
            throw std::invalid_argument("Get invalid param double");
        }
    } else {
        return layers_[param.GetLayerID()].GetParamDouble(param);
    }
}

dcomplex Tmm::GetParamComplex(Param param) const {
    if (param.GetLayerID() < 0) {
        throw std::invalid_argument("Get invalid param complex");
    } else {
        return layers_[param.GetLayerID()].GetParamComplex(param);
    }
}

void Tmm::AddIsotropicLayer(double d, Material* mat) {
    needToSolve_ = true;
    layers_.emplace_back(d, mat);
}

void Tmm::AddLayer(double d, Material* matx, Material* maty, Material* matz, double psi, double xi) {
    needToSolve_ = true;
    layers_.emplace_back(d, matx, maty, matz, psi, xi);
}

void Tmm::ClearLayers() noexcept {
    layers_.clear();
    needToSolve_ = true;
}

const Eigen::Matrix4d& Tmm::GetIntensityMatrix() {
    Solve();
    return R_;
}

const Eigen::Matrix4cd& Tmm::GetAmplitudeMatrix() {
    Solve();
    return r_;
}

void Tmm::Solve() {
    if (!needToSolve_) {
        return;
    }
    if (layers_.size() < 2) {
        throw std::invalid_argument("At least two layers are required to solve.");
    }
    needToSolve_ = false;
    needToCalcFieldCoefs_ = true;

    for (auto& layer : layers_) {
        layer.SolveLayer(wl_, beta_);
    }

    // System matrix
    A_ = layers_[0].invF_;
    for (size_t i = 1; i + 1 < layers_.size(); i++) {
        Layer& layer = layers_[i];
        A_ = A_ * layer.M_;
    }
    A_ = A_ * layers_.back().F_;

    // r - matrix
    Eigen::Matrix4cd invr1;
    dcomplex t = A_(0, 0) * A_(2, 2) - A_(0, 2) * A_(2, 0);
    invr1 << -(A_(1, 0) * A_(2, 2) - A_(1, 2) * A_(2, 0)) / t, 1, -(A_(0, 0) * A_(1, 2) - A_(0, 2) * A_(1, 0)) / t, 0,
        (A_(2, 0) * A_(3, 2) - A_(2, 2) * A_(3, 0)) / t, 0, -(A_(0, 0) * A_(3, 2) - A_(0, 2) * A_(3, 0)) / t, 1,
        -A_(2, 2) / t, 0, A_(0, 2) / t, 0, A_(2, 0) / t, 0, -A_(0, 0) / t, 0;

    Eigen::Matrix4cd r2;
    r2 << -1.0, 0.0, A_(0, 1), A_(0, 3), 0.0, 0.0, A_(1, 1), A_(1, 3), 0.0, -1.0, A_(2, 1), A_(2, 3), 0.0, 0.0,
        A_(3, 1), A_(3, 3);

    r_ = invr1 * r2;

    Eigen::Vector4d& poyntingXF = layers_.front().poyntingX_;
    Eigen::Vector4d& poyntingXL = layers_.back().poyntingX_;

    Eigen::Vector4d pBackward, pForward;
    pBackward << poyntingXF(1), poyntingXF(3), poyntingXL(1), poyntingXL(3);
    pForward << poyntingXF(0), poyntingXF(2), poyntingXL(0), poyntingXL(2);

    R_(0, 0) = norm(r_(0, 0)) * abs(pBackward(0) / pForward(0));
    R_(0, 1) = norm(r_(0, 1)) * abs(pBackward(0) / pForward(1));
    R_(0, 2) = NaN;
    R_(0, 3) = NaN;
    R_(1, 0) = norm(r_(1, 0)) * abs(pBackward(1) / pForward(0));
    R_(1, 1) = norm(r_(1, 1)) * abs(pBackward(1) / pForward(1));
    R_(1, 2) = NaN;
    R_(1, 3) = NaN;
    R_(2, 0) = norm(r_(2, 0)) * abs(pForward(2) / pForward(0));
    R_(2, 1) = norm(r_(2, 1)) * abs(pForward(2) / pForward(1));
    R_(2, 2) = NaN;
    R_(2, 3) = NaN;
    R_(3, 0) = norm(r_(3, 0)) * abs(pForward(3) / pForward(0));
    R_(3, 1) = norm(r_(3, 1)) * abs(pForward(3) / pForward(1));
    R_(3, 2) = NaN;
    R_(3, 3) = NaN;
}

SweepRes Tmm::Sweep(Param sweepParam, const Eigen::Map<Eigen::ArrayXd>& sweepValues, const PositionSettings& enhpos,
                    int alphasLayer) {
    SweepRes res;
    auto& resComplex = res.mapComplex;
    auto& resDouble = res.mapDouble;
    DoubleVectorMap::iterator data_R[kMatrixSize][kMatrixSize];
    ComplexVectorMap::iterator data_r[kMatrixSize][kMatrixSize];

    ComplexVectorMap::iterator alphas0;
    ComplexVectorMap::iterator alphas1;
    ComplexVectorMap::iterator alphas2;
    ComplexVectorMap::iterator alphas3;
    bool alphasEnabled = (alphasLayer >= 0);

    if (alphasEnabled) {
        if (alphasLayer >= static_cast<int>(layers_.size())) {
            throw std::invalid_argument("alphaLayer is out of range.");
        }
        alphas0 = resComplex.try_emplace("alphas0", sweepValues.size()).first;
        alphas1 = resComplex.try_emplace("alphas1", sweepValues.size()).first;
        alphas2 = resComplex.try_emplace("alphas2", sweepValues.size()).first;
        alphas3 = resComplex.try_emplace("alphas3", sweepValues.size()).first;
    }

    DoubleVectorMap::iterator enhs;
    DoubleVectorMap::iterator enhExs;
    DoubleVectorMap::iterator enhEys;
    DoubleVectorMap::iterator enhEzs;

    if (enhpos.IsEnabled()) {
        enhs = resDouble.try_emplace("enh", sweepValues.size()).first;
        enhExs = resDouble.try_emplace("enhEx", sweepValues.size()).first;
        enhEys = resDouble.try_emplace("enhEy", sweepValues.size()).first;
        enhEzs = resDouble.try_emplace("enhEz", sweepValues.size()).first;
    }

    for (int i = 0; i < kMatrixSize; i++) {
        for (int j = 0; j < 2; j++) {
            data_R[i][j] = resDouble.try_emplace(names_R_[i][j], sweepValues.size()).first;
            data_r[i][j] = resComplex.try_emplace(names_r_[i][j], sweepValues.size()).first;
        }
    }

    EMFields fields;
    for (Eigen::Index i = 0; i < sweepValues.size(); i++) {
        SetParam(sweepParam, sweepValues[i]);
        Solve();

        for (int j = 0; j < kMatrixSize; j++) {
            for (int k = 0; k < 2; k++) {
                data_R[j][k]->second(i) = R_(j, k);
                data_r[j][k]->second(i) = r_(j, k);
            }
        }

        if (alphasEnabled) {
            alphas0->second(i) = layers_[alphasLayer].alpha_(0);
            alphas1->second(i) = layers_[alphasLayer].alpha_(1);
            alphas2->second(i) = layers_[alphasLayer].alpha_(2);
            alphas3->second(i) = layers_[alphasLayer].alpha_(3);
        }

        if (enhpos.IsEnabled()) {
            fields = CalcFieldsAtInterface(enhpos, WaveDirection::WD_BOTH);
            enhs->second(i) = fields.E.matrix().norm();
            enhExs->second(i) = abs(fields.E(0));
            enhEys->second(i) = abs(fields.E(1));
            enhEzs->second(i) = abs(fields.E(2));
        }
    }

    return res;
}

SweepRes Tmm::Sweep(Param sweepParam, const Eigen::Map<Eigen::ArrayXd>& sweepValues) {
    PositionSettings enhpos;
    return Sweep(sweepParam, sweepValues, enhpos, -1);
}

EMFieldsList Tmm::CalcFields1D(const Eigen::Map<Eigen::ArrayXd>& xs, const Eigen::Map<Eigen::Array2d>& polarization,
                               WaveDirection waveDirection) {
    Solve();
    RowVector2d pol(polarization(0), polarization(1));
    CalcFieldCoefs(pol);

    EMFieldsList res(xs.size());
    LayerIndices layerP = CalcLayerIndices(xs);
    for (Eigen::Index i = 0; i < xs.size(); i++) {
        int layerId = layerP.indices(i);
        EMFields f = layers_[layerId].GetFields(wl_, beta_, layerP.ds(i), fieldCoefs_.row(layerId), waveDirection);
        res.E.row(i) = f.E / normCoef_;
        res.H.row(i) = f.H / normCoef_;
    }
    return res;
}

EMFields Tmm::CalcFieldsAtInterface(const PositionSettings& pos, WaveDirection waveDirection) {
    if (!pos.IsEnabled()) {
        throw std::invalid_argument("Position settings must be enabled.");
    }

    int layerId;
    if (pos.GetInterfaceId() < 0) {
        layerId = static_cast<int>(layers_.size()) + pos.GetInterfaceId();
    } else {
        layerId = pos.GetInterfaceId();
    }
    if (layerId < 0 || layerId >= static_cast<int>(layers_.size())) {
        throw std::invalid_argument("Interface layer index is out of range.");
    }

    Solve();
    CalcFieldCoefs(pos.GetPolarization());
    EMFields res =
        layers_[layerId].GetFields(wl_, beta_, pos.GetDistFromInterface(), fieldCoefs_.row(layerId), waveDirection);
    res.E /= normCoef_;
    res.H /= normCoef_;
    return res;
}

double Tmm::OptimizeEnhancement(const std::vector<Param>& optParams, const Eigen::ArrayXd& optInitial,
                                const PositionSettings& pos) {
    EnhFitStruct fitFunc(this, optParams, pos);
    auto criterion =
        Optimization::Local::make_and_criteria(Optimization::Local::IterationCriterion(enhOptMaxIters_),
                                               Optimization::Local::RelativeValueCriterion<double>(enhOptMaxRelError_));
    auto optimizer = Optimization::Local::build_simplex(fitFunc, criterion);

    optimizer.set_start_point(optInitial);
    optimizer.set_delta(enhOptInitialStep_); // TODO, set deltas
    optimizer.optimize(fitFunc);


    fitFunc.SetParams(optimizer.get_best_parameters());
    double res = -optimizer.get_best_value();
    long nIterations = optimizer.get_number_of_iterations();

    if (nIterations >= enhOptMaxIters_) {
        throw std::runtime_error("Maximum number of iterations reached: " + std::to_string(nIterations) + "/" +
                                 std::to_string(enhOptMaxIters_));
    }

    return res;
}

void Tmm::CalcFieldCoefs(const RowVector2d& polarization) {
    if (!needToCalcFieldCoefs_ && polarization == lastFieldCoefsPol_) {
        return;
    }
    needToCalcFieldCoefs_ = false;
    lastFieldCoefsPol_ = polarization;

    // Calc normalization factor
    Eigen::Vector4cd incCoefs;
    incCoefs << polarization(0), 0.0, polarization(1), 0.0;
    Eigen::Vector3cd Einc = layers_.front().GetFields(wl_, beta_, 0.0, incCoefs, WaveDirection::WD_BOTH).E;

    dcomplex n1 = sqrt(sqr(beta_) + sqr(layers_.front().alpha_(0)));
    dcomplex n2 = sqrt(sqr(beta_) + sqr(layers_.front().alpha_(2)));
    double nEff = (polarization(0) * real(n1) + polarization(1) * real(n2)) /
                  (polarization(0) + polarization(1)); // Maybe not fully correct
    normCoef_ = Einc.norm() * sqrt(nEff);

    // Calc output coefs
    Eigen::Vector4cd inputFields;
    inputFields << polarization(0), polarization(1), 0.0, 0.0;
    Eigen::Vector4cd outputFields = r_ * inputFields;

    // Calc coefs in all layers
    Eigen::Vector4cd coefsSubstrate;
    coefsSubstrate << outputFields(2), 0.0, outputFields(3), 0.0;

    Eigen::Matrix4cd mat = layers_.back().F_;
    fieldCoefs_.resize(layers_.size(), 4);
    for (Eigen::Index i = static_cast<Eigen::Index>(layers_.size()) - 1; i >= 0; i--) {
        mat = layers_[i].M_ * mat;
        fieldCoefs_.row(i) = layers_[i].invF_ * mat * coefsSubstrate;
    }
    fieldCoefs_(layers_.size() - 1, 1) = fieldCoefs_(layers_.size() - 1, 3) = 0.0;
}

LayerIndices Tmm::CalcLayerIndices(const Eigen::Map<Eigen::ArrayXd>& xs) {
    LayerIndices res;
    res.indices.resize(xs.size());
    res.ds.resize(xs.size());

    int curLayer = 0;
    double curDist = 0.0;
    double prevDist = 0.0;
    int numLayers = static_cast<int>(layers_.size());

    for (Eigen::Index i = 0; i < xs.size(); i++) {
        while (xs[i] >= curDist) {
            curLayer++;
            prevDist = curDist;
            if (curLayer >= numLayers - 1) {
                curDist = std::numeric_limits<double>::infinity();
                curLayer = numLayers - 1;
            }
            curDist += layers_[curLayer].GetD();
        }
        res.indices(i) = curLayer;
        res.ds(i) = xs(i) - prevDist;
    }
    return res;
}
EnhFitStruct::EnhFitStruct(Tmm* tmm, const std::vector<Param>& optParams, const PositionSettings& enhpos)
    : tmm_(tmm), optParams_(optParams), enhPos_(enhpos) {}

void EnhFitStruct::SetParams(const ParameterType& params) const {
    for (Eigen::Index i = 0; i < params.size(); i++) {
        tmm_->SetParam(optParams_[i], params[i]);
    }
}
} // namespace tmm
