#include "tmm.h"

namespace TmmModel{

	//---------------------------------------------------------------------
	// Layer
	//---------------------------------------------------------------------

	Layer::Layer(double d_, Material n_){
		Init();
		d = d_;
		nx = n_;
		ny = n_;
		nz = n_;
		psi = 0.0;
		xi = 0.0;
		isotropicLayer = true;
	}

	Layer::Layer(double d_, Material nx_, Material ny_, Material nz_, double psi_, double xi_){
		Init();
		d = d_;
		nx = nx_;
		ny = ny_;
		nz = nz_;
		psi = psi_;
		xi = xi_;
		isotropicLayer = false;
	}

	void Layer::SetParam(Param param, int value){
		switch (param.GetParamType())
		{
		default:
			throw invalid_argument("Invalid layer param int");
			break;
		}
	}

	void Layer::SetParam(Param param, double value){
		switch (param.GetParamType())
		{
		case LAYER_D:
			d = value;
			break;
		case LAYER_PSI:
			psi = value;
			epsilonRefractiveIndexChanged = true;
			break;
		case LAYER_XI:
			xi = value;
			epsilonRefractiveIndexChanged = true;
			break;
		default:
			throw invalid_argument("Invalid layer param double");
			break;
		}
	}

	void Layer::SetParam(Param param, dcomplex value){
		switch (param.GetParamType())
		{
		case LAYER_N:
			nx = Material(value);
			ny = Material(value);
			nz = Material(value);
			epsilonRefractiveIndexChanged = true;
			isotropicLayer = true;
			break;
		case LAYER_NX:
			nx = Material(value);
			epsilonRefractiveIndexChanged = true;
			isotropicLayer = false;
			break;
		case LAYER_NY:
			ny = Material(value);
			epsilonRefractiveIndexChanged = true;
			isotropicLayer = false;
			break;
		case LAYER_NZ:
			nz = Material(value);
			epsilonRefractiveIndexChanged = true;
			isotropicLayer = false;
			break;
		default:
			throw invalid_argument("Invalid layer param complex");
			break;
		}
	}

	int Layer::GetParamInt(Param param){
		switch (param.GetParamType())
		{
		default:
			throw invalid_argument("Get invalid layer param int");
			break;
		}
	}

	double Layer::GetParamDouble(Param param){
		switch (param.GetParamType())
		{
		case LAYER_D:
			return d;
			break;
		case LAYER_PSI:
			return psi;
			break;
		case LAYER_XI:
			return xi;
			break;
		default:
			throw invalid_argument("Get invalid layer param double");
			break;
		}
	}

	dcomplex Layer::GetParamComplex(Param param){
		switch (param.GetParamType())
		{
		case LAYER_N:
			if (!isotropicLayer){
				throw runtime_error("To get LAYER_N, the layer must be isotropic");
			}

			if (!nx.IsStatic()){
				throw runtime_error("To get LAYER_N, the material must be static");
			}
			return nx.n(0.0);
			break;
		case LAYER_NX:
			if (!nx.IsStatic()){
				throw runtime_error("To get LAYER_NX, the material must be static");
			}
			return nx.n(0.0);
			break;
		case LAYER_NY:
			if (!ny.IsStatic()){
				throw runtime_error("To get LAYER_NY, the material must be static");
			}
			return ny.n(0.0);
			break;
		case LAYER_NZ:
			if (!nz.IsStatic()){
				throw runtime_error("To get LAYER_NZ, the material must be static");
			}
			return nz.n(0.0);
			break;
		default:
			throw invalid_argument("Invalid layer param complex");
			break;
		}
	}


	double Layer::GetD(){
		return d;
	}

	dcomplex Layer::GetNx(double wl){
		return nx.n(wl);
	}

	dcomplex Layer::GetNy(double wl){
		return ny.n(wl);
	}

	dcomplex Layer::GetNz(double wl){
		return nz.n(wl);
	}

	void Layer::SolveLayer(double wl, double beta){
		SolveEpsilonMatrix(wl);
		SolveEigenFunction(beta);

		// Phase matrix
		phaseMatrix.setIdentity();
		if (d != INFINITY){
			dcomplex expParam = 2.0 * M_PI / wl *  d * dcomplex(0.0, -1.0);
			for (int i = 0; i < 4; i++){
				phaseMatrix(i, i) = exp(expParam * alpha(i));
			}
		}

		//InvF

		if (isotropicLayer) {
			invF << 0.5 / F(0, 0), 0.5, 0, 0,
				0.5 / F(0, 1), 0.5, 0, 0,
				0, 0, 0.5 / F(2, 2), 0.5,
				0, 0, 0.5 / F(2, 3), 0.5;
		}
		else {
			invF = F.inverse();
		}
		M = F * phaseMatrix * invF;
		solved = true;
	}

	EMFields Layer::GetFields(double wl, double beta, double x, Eigen::Vector4cd coefs){
		EMFields res;
		res.E.setZero();
		res.H.setZero();
		double z0 = 119.9169832 * M_PI;
		double k0 = 2.0 * M_PI / wl;

		for (int mode = 0; mode < 4; mode++){
			dcomplex a = alpha[mode];
			dcomplex epsXX = epsTensor(0, 0);
			dcomplex epsXY = epsTensor(0, 1);
			dcomplex epsXZ = epsTensor(0, 2);

			dcomplex mEy = coefs[mode] * F(0, mode);
			dcomplex mHz = coefs[mode] * F(1, mode);
			dcomplex mEz = coefs[mode] * F(2, mode);
			dcomplex mHy = coefs[mode] * F(3, mode);
			dcomplex mEx = -(epsXY * mEy + epsXZ * mEz + beta * z0 * mHz) / epsXX;
			dcomplex mHx = (beta / z0) * mEz;
			dcomplex phase = exp(dcomplex(0.0, 1.0) * k0 * a * x);

			res.E(0) += mEx * phase;
			res.E(1) += mEy * phase;
			res.E(2) += mEz * phase;
			res.H(0) += mHx * phase;
			res.H(1) += mHy * phase;
			res.H(2) += mHz * phase;
		}

		return res;
	}


	void Layer::Init(){
		solved = false;
		epsilonRefractiveIndexChanged = true;
		wlEpsilonCalc = 0.0;
	}

	void Layer::SolveEpsilonMatrix(double wl){
		if (wl == wlEpsilonCalc && !epsilonRefractiveIndexChanged){
			return;
		}
		//cout << "SolveEpsilonMatrix " << wl << " " << wlEpsilonCalc << " " << epsilonRefractiveIndexChanged << endl;

		Eigen::Matrix3cd epsTensorCrystal = Eigen::Matrix3cd::Zero();
		dcomplex nxTmp = GetNx(wl);
		dcomplex nyTmp = GetNy(wl);
		dcomplex nzTmp = GetNz(wl);
		epsTensorCrystal(0, 0) = sqr(nxTmp);
		epsTensorCrystal(1, 1) = sqr(nyTmp);
		epsTensorCrystal(2, 2) = sqr(nzTmp);

		if (epsTensorCrystal(0, 0) == epsTensorCrystal(1, 1) && epsTensorCrystal(1, 1) == epsTensorCrystal(2, 2)){
			isotropicLayer = true;
			epsTensor = epsTensorCrystal;
		}
		else {
			isotropicLayer = false;
			epsTensor = RotationSx(xi) * RotationSz(psi) * epsTensorCrystal * RotationSz(-psi) * RotationSx(-xi);
		}

		wlEpsilonCalc = wl;
		epsilonRefractiveIndexChanged = false;
	}


	void Layer::SolveEigenFunction(double beta){
		double z0 = 119.9169832 * M_PI;

		Eigen::ComplexEigenSolver<Eigen::Matrix4cd>::EigenvalueType eigenvalues;
		Eigen::ComplexEigenSolver<Eigen::Matrix4cd>::EigenvectorType eigenvectors;
		if (isotropicLayer)
		{
			dcomplex eps = epsTensor(0, 0);
			dcomplex a = sqrt(eps - sqr(beta));
			dcomplex p1 = z0 * a / eps, p2 = z0 / a;
			eigenvalues << a, -a, a, -a;
			eigenvectors << p1, -p1, 0, 0,
				1, 1, 0, 0,
				0, 0, -p2, p2,
				0, 0, 1, 1;
		}
		else
		{
			dcomplex epsXX = epsTensor(0, 0);
			dcomplex epsYY = epsTensor(1, 1);
			dcomplex epsZZ = epsTensor(2, 2);
			dcomplex epsXY = epsTensor(0, 1);
			dcomplex epsXZ = epsTensor(0, 2);
			dcomplex epsYZ = epsTensor(1, 2);


			Eigen::Matrix4cd mBeta = Eigen::Matrix4cd::Zero();
			mBeta(0, 0) = -beta * epsXY / epsXX;
			mBeta(0, 1) = z0 - (z0 * sqr(beta)) / epsXX;
			mBeta(0, 2) = -beta * epsXZ / epsXX;
			//mBeta(0, 3) = 0.0;
			mBeta(1, 0) = epsYY / z0 - (sqr(epsXY)) / (z0 * epsXX);
			mBeta(1, 1) = (-beta * epsXY) / epsXX;
			mBeta(1, 2) = epsYZ / z0 - (epsXY * epsXZ) / (z0 * epsXX);
			//mBeta(1, 3) = 0.0;
			//mBeta(2, 0) = 0.0;
			//mBeta(2, 1) = 0.0;
			//mBeta(2, 2) = 0.0;
			mBeta(2, 3) = -z0;
			mBeta(3, 0) = (-epsYZ / z0) + (epsXY * epsXZ) / (z0 * epsXX);
			mBeta(3, 1) = beta * epsXZ / epsXX;
			mBeta(3, 2) = (sqr(beta)) / z0 + (sqr(epsXZ)) / (z0 * epsXX) - epsZZ / z0;
			//mBeta(3, 3) = 0.0;

			// Calc eigenvalues
			ces.compute(mBeta, true, false);
			eigenvalues = ces.eigenvalues();
			eigenvectors = ces.eigenvectors();
		}

		// Sort eigenvalues
		Eigen::Vector4d poyntingXTmp;
		int countF = 0, countB = 0;
		int forward[4], backward[4];

		for (int i = 0; i < 4; i++){
			bool movingForward = false;
			poyntingXTmp(i) = 0.5 * real(eigenvectors(0, i) * conj(eigenvectors(1, i))
				- eigenvectors(2, i) * conj(eigenvectors(3, i)));

			if (abs(poyntingXTmp(i)) > 1e-10){
				movingForward = poyntingXTmp(i) > 0.0;
			}
			else {
				movingForward = imag(eigenvalues(i)) > 0.0;
			}

			if (movingForward){
				forward[countF++] = i;
			}
			else {
				backward[countB++] = i;
			}
		}

		if (countF != 2){
			cerr << "eigenvalues" << endl;
			cerr << eigenvalues << endl;
			cerr << "eigenvectors" << endl;
			cerr << eigenvectors << endl;
			cerr << "Wrong number of forward moving waves: " << endl;
			throw runtime_error("wrong number of forward waves");
		}

		if (abs(real(eigenvalues(forward[0])) - real(eigenvalues(forward[1]))) < 1e-10){
			double normUp0 = eigenvectors.block(0, forward[0], 2, 1).norm();
			double normUp1 = eigenvectors.block(0, forward[1], 2, 1).norm();
			if (normUp1 > normUp0){
				swap(forward[0], forward[1]);
			}
		}
		else if (real(eigenvalues(forward[0])) < real(eigenvalues(forward[1]))){
			swap(forward[0], forward[1]);
		}

		if (abs(real(eigenvalues(backward[0])) - real(eigenvalues(backward[1]))) < 1e-10){
			double normUp0 = eigenvectors.block(0, backward[0], 2, 1).norm();
			double normUp1 = eigenvectors.block(0, backward[1], 2, 1).norm();
			if (normUp1 > normUp0){
				swap(backward[0], backward[1]);
			}
		}
		else if (real(eigenvalues(backward[0])) > real(eigenvalues(backward[1]))){
			swap(backward[0], backward[1]);
		}

		// Ordering
		Eigen::Vector4i order;
		order << forward[0], backward[0], forward[1], backward[1];

		// Save result
		for (int i = 0; i < 4; i++){
			int index = order[i];
			alpha(i) = eigenvalues(index);
			poyntingX(i) = poyntingXTmp(index);
			for (int j = 0; j < 4; j++){
				F(j, i) = eigenvectors(j, index);
			}
		}

		//cout << "alpha" << alpha << endl;
		//cout << "F" << endl << F << endl << endl;
		//cout << "poyntingX" << endl << poyntingX << endl << endl;
	}
} // Namespace