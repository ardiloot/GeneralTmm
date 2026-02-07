#include "Layer.h"
#include <sstream>

namespace tmm {

//---------------------------------------------------------------------
// Layer
//---------------------------------------------------------------------

Layer::Layer(double d, Material* n) : isotropicLayer_(true), d_(d), nx_(n), ny_(n), nz_(n), psi_(0.0), xi_(0.0) {}

Layer::Layer(double d, Material* nx, Material* ny, Material* nz, double psi, double xi)
    : isotropicLayer_(false), d_(d), nx_(nx), ny_(ny), nz_(nz), psi_(psi), xi_(xi) {}

void Layer::SetParam(Param param, [[maybe_unused]] int value) {
    switch (param.GetParamType()) {
    default:
        throw std::invalid_argument("Invalid layer param int");
    }
}

void Layer::SetParam(Param param, double value) {
    switch (param.GetParamType()) {
    case ParamType::LAYER_D:
        d_ = value;
        break;
    case ParamType::LAYER_PSI:
        psi_ = value;
        epsilonRefractiveIndexChanged_ = true;
        break;
    case ParamType::LAYER_XI:
        xi_ = value;
        epsilonRefractiveIndexChanged_ = true;
        break;
    default:
        throw std::invalid_argument("Invalid layer param double");
    }
}

void Layer::SetParam(Param param, dcomplex value) {
    switch (param.GetParamType()) {
    case ParamType::LAYER_N:
        nx_->SetStatic(value);
        ny_->SetStatic(value);
        nz_->SetStatic(value);
        epsilonRefractiveIndexChanged_ = true;
        isotropicLayer_ = true;
        break;
    case ParamType::LAYER_NX:
        nx_->SetStatic(value);
        epsilonRefractiveIndexChanged_ = true;
        isotropicLayer_ = false;
        break;
    case ParamType::LAYER_NY:
        ny_->SetStatic(value);
        epsilonRefractiveIndexChanged_ = true;
        isotropicLayer_ = false;
        break;
    case ParamType::LAYER_NZ:
        nz_->SetStatic(value);
        epsilonRefractiveIndexChanged_ = true;
        isotropicLayer_ = false;
        break;
    default:
        throw std::invalid_argument("Invalid layer param complex");
    }
}

int Layer::GetParamInt(Param param) const {
    switch (param.GetParamType()) {
    default:
        throw std::invalid_argument("Get invalid layer param int");
    }
}

double Layer::GetParamDouble(Param param) const {
    switch (param.GetParamType()) {
    case ParamType::LAYER_D:
        return d_;
    case ParamType::LAYER_PSI:
        return psi_;
    case ParamType::LAYER_XI:
        return xi_;
    default:
        throw std::invalid_argument("Get invalid layer param double");
    }
}

dcomplex Layer::GetParamComplex(Param param) const {
    switch (param.GetParamType()) {
    case ParamType::LAYER_N:
        if (!isotropicLayer_) {
            throw std::runtime_error("To get LAYER_N, the layer must be isotropic");
        }

        if (!nx_->IsStatic()) {
            throw std::runtime_error("To get LAYER_N, the material must be static");
        }
        return nx_->n(0.0);
    case ParamType::LAYER_NX:
        if (!nx_->IsStatic()) {
            throw std::runtime_error("To get LAYER_NX, the material must be static");
        }
        return nx_->n(0.0);
    case ParamType::LAYER_NY:
        if (!ny_->IsStatic()) {
            throw std::runtime_error("To get LAYER_NY, the material must be static");
        }
        return ny_->n(0.0);
    case ParamType::LAYER_NZ:
        if (!nz_->IsStatic()) {
            throw std::runtime_error("To get LAYER_NZ, the material must be static");
        }
        return nz_->n(0.0);
    default:
        throw std::invalid_argument("Invalid layer param complex");
    }
}


double Layer::GetD() const noexcept {
    return d_;
}

dcomplex Layer::GetNx(double wl) const {
    return nx_->n(wl);
}

dcomplex Layer::GetNy(double wl) const {
    return ny_->n(wl);
}

dcomplex Layer::GetNz(double wl) const {
    return nz_->n(wl);
}

void Layer::SolveLayer(double wl, double beta) {
    SolveEpsilonMatrix(wl);
    SolveEigenFunction(beta);

    // Phase matrix
    phaseMatrix_.setIdentity();
    if (!std::isinf(d_)) {
        dcomplex expParam = 2.0 * PI / wl * d_ * (-I);
        for (int i = 0; i < 4; i++) {
            phaseMatrix_(i, i) = exp(expParam * alpha_(i));
        }
    }

    // InvF

    if (isotropicLayer_) {
        invF_ << 0.5 / F_(0, 0), 0.5, 0, 0, 0.5 / F_(0, 1), 0.5, 0, 0, 0, 0, 0.5 / F_(2, 2), 0.5, 0, 0, 0.5 / F_(2, 3),
            0.5;
    } else {
        invF_ = F_.inverse();
    }
    M_ = F_ * phaseMatrix_ * invF_;
    solved_ = true;
}

EMFields Layer::GetFields(double wl, double beta, double x, const Eigen::Ref<const Vector4cd>& coefs,
                          WaveDirection waveDirection) const {
    EMFields res;
    res.E.setZero();
    res.H.setZero();
    double k0 = 2.0 * PI / wl;

    for (int mode = 0; mode < 4; mode++) {
        if (waveDirection == WaveDirection::WD_BACKWARD && (mode == 0 || mode == 2)) {
            continue;
        } else if (waveDirection == WaveDirection::WD_FORWARD && (mode == 1 || mode == 3)) {
            continue;
        }

        dcomplex a = alpha_[mode];
        dcomplex epsXX = epsTensor_(0, 0);
        dcomplex epsXY = epsTensor_(0, 1);
        dcomplex epsXZ = epsTensor_(0, 2);

        dcomplex mEy = coefs[mode] * F_(0, mode);
        dcomplex mHz = coefs[mode] * F_(1, mode);
        dcomplex mEz = coefs[mode] * F_(2, mode);
        dcomplex mHy = coefs[mode] * F_(3, mode);
        dcomplex mEx = -(epsXY * mEy + epsXZ * mEz + beta * Z0 * mHz) / epsXX;
        dcomplex mHx = (beta / Z0) * mEz;
        dcomplex phase = exp(I * k0 * a * x);

        res.E(0) += mEx * phase;
        res.E(1) += mEy * phase;
        res.E(2) += mEz * phase;
        res.H(0) += mHx * phase;
        res.H(1) += mHy * phase;
        res.H(2) += mHz * phase;
    }

    return res;
}


void Layer::SolveEpsilonMatrix(double wl) {
    if (wl == wlEpsilonCalc_ && !epsilonRefractiveIndexChanged_) {
        return;
    }
    Matrix3cd epsTensorCrystal = Matrix3cd::Zero();
    dcomplex nxTmp = GetNx(wl);
    dcomplex nyTmp = GetNy(wl);
    dcomplex nzTmp = GetNz(wl);
    epsTensorCrystal(0, 0) = sqr(nxTmp);
    epsTensorCrystal(1, 1) = sqr(nyTmp);
    epsTensorCrystal(2, 2) = sqr(nzTmp);

    if (epsTensorCrystal(0, 0) == epsTensorCrystal(1, 1) && epsTensorCrystal(1, 1) == epsTensorCrystal(2, 2)) {
        isotropicLayer_ = true;
        epsTensor_ = epsTensorCrystal;
    } else {
        isotropicLayer_ = false;
        epsTensor_ = RotationSx(xi_) * RotationSz(psi_) * epsTensorCrystal * RotationSz(-psi_) * RotationSx(-xi_);
    }

    wlEpsilonCalc_ = wl;
    epsilonRefractiveIndexChanged_ = false;
}


void Layer::SolveEigenFunction(double beta) {
    Eigen::ComplexEigenSolver<Eigen::Matrix4cd>::EigenvalueType eigenvalues;
    Eigen::ComplexEigenSolver<Eigen::Matrix4cd>::EigenvectorType eigenvectors;
    if (isotropicLayer_) {
        dcomplex eps = epsTensor_(0, 0);
        dcomplex a = sqrt(eps - sqr(beta));
        dcomplex p1 = Z0 * a / eps, p2 = Z0 / a;
        eigenvalues << a, -a, a, -a;
        eigenvectors << p1, -p1, 0, 0, 1, 1, 0, 0, 0, 0, -p2, p2, 0, 0, 1, 1;
    } else {
        dcomplex epsXX = epsTensor_(0, 0);
        dcomplex epsYY = epsTensor_(1, 1);
        dcomplex epsZZ = epsTensor_(2, 2);
        dcomplex epsXY = epsTensor_(0, 1);
        dcomplex epsXZ = epsTensor_(0, 2);
        dcomplex epsYZ = epsTensor_(1, 2);


        Matrix4cd mBeta = Matrix4cd::Zero();
        mBeta(0, 0) = -beta * epsXY / epsXX;
        mBeta(0, 1) = Z0 - (Z0 * sqr(beta)) / epsXX;
        mBeta(0, 2) = -beta * epsXZ / epsXX;
        // mBeta(0, 3) = 0.0;
        mBeta(1, 0) = epsYY / Z0 - (sqr(epsXY)) / (Z0 * epsXX);
        mBeta(1, 1) = (-beta * epsXY) / epsXX;
        mBeta(1, 2) = epsYZ / Z0 - (epsXY * epsXZ) / (Z0 * epsXX);
        // mBeta(1, 3) = 0.0;
        // mBeta(2, 0) = 0.0;
        // mBeta(2, 1) = 0.0;
        // mBeta(2, 2) = 0.0;
        mBeta(2, 3) = -Z0;
        mBeta(3, 0) = (-epsYZ / Z0) + (epsXY * epsXZ) / (Z0 * epsXX);
        mBeta(3, 1) = beta * epsXZ / epsXX;
        mBeta(3, 2) = (sqr(beta)) / Z0 + (sqr(epsXZ)) / (Z0 * epsXX) - epsZZ / Z0;
        // mBeta(3, 3) = 0.0;

        // Calc eigenvalues
        ces_.compute(mBeta, true);
        eigenvalues = ces_.eigenvalues();
        eigenvectors = ces_.eigenvectors();
    }

    // Sort eigenvalues
    Vector4d poyntingXTmp;
    int countF = 0, countB = 0;
    std::array<int, 4> forward{}, backward{};

    for (int i = 0; i < 4; i++) {
        bool movingForward = false;
        poyntingXTmp(i) =
            0.5 * real(eigenvectors(0, i) * conj(eigenvectors(1, i)) - eigenvectors(2, i) * conj(eigenvectors(3, i)));

        if (abs(poyntingXTmp(i)) > 1e-10) {
            movingForward = poyntingXTmp(i) > 0.0;
        } else {
            movingForward = imag(eigenvalues(i)) > 0.0;
        }

        if (movingForward) {
            forward[countF++] = i;
        } else {
            backward[countB++] = i;
        }
    }

    if (countF != 2) {
        std::ostringstream oss;
        oss << "Wrong number of forward waves (" << countF << "). "
            << "eigenvalues: " << eigenvalues.transpose();
        throw std::runtime_error(oss.str());
    }

    if (abs(real(eigenvalues(forward[0])) - real(eigenvalues(forward[1]))) < 1e-10) {
        double normUp0 = eigenvectors.block(0, forward[0], 2, 1).norm();
        double normUp1 = eigenvectors.block(0, forward[1], 2, 1).norm();
        if (normUp1 > normUp0) {
            std::swap(forward[0], forward[1]);
        }
    } else if (real(eigenvalues(forward[0])) < real(eigenvalues(forward[1]))) {
        std::swap(forward[0], forward[1]);
    }

    if (abs(real(eigenvalues(backward[0])) - real(eigenvalues(backward[1]))) < 1e-10) {
        double normUp0 = eigenvectors.block(0, backward[0], 2, 1).norm();
        double normUp1 = eigenvectors.block(0, backward[1], 2, 1).norm();
        if (normUp1 > normUp0) {
            std::swap(backward[0], backward[1]);
        }
    } else if (real(eigenvalues(backward[0])) > real(eigenvalues(backward[1]))) {
        std::swap(backward[0], backward[1]);
    }

    // Ordering
    Vector4i order;
    order << forward[0], backward[0], forward[1], backward[1];

    // Save result
    for (int i = 0; i < 4; i++) {
        int index = order[i];
        alpha_(i) = eigenvalues(index);
        poyntingX_(i) = poyntingXTmp(index);
        F_.col(i) = eigenvectors.col(index);
    }
}
} // namespace tmm
