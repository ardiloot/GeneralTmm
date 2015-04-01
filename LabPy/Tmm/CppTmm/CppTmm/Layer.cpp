#include "tmm.h"

//---------------------------------------------------------------------
// Layer
//---------------------------------------------------------------------

Layer::Layer(double d_, dcomplex n_){
	Init();
	d = d_;
	nx = Material(n_);
	ny = Material(n_);
	nz = Material(n_);
	psi = 0.0;
	xi = 0.0;
}

Layer::Layer(double d_, dcomplex nx_, dcomplex ny_, dcomplex nz_, double psi_, double xi_){
	Init();
	d = d_;
	nx = Material(nx_);
	ny = Material(ny_);
	nz = Material(nz_);
	psi = psi_;
	xi = xi_;
}

void Layer::SetParam(Param param, double value){
	switch (param.GetParamType())
	{
	case LAYER_D:
		d = value;
		break;
	case LAYER_PSI:
		psi = value;
		break;
	case LAYER_XI:
		xi = value;
		break;
	default:
		throw invalid_argument("Invalid param");
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
		break;
	case LAYER_NX:
		nx = Material(value);
		epsilonRefractiveIndexChanged = true;
		break;
	case LAYER_NY:
		ny = Material(value);
		epsilonRefractiveIndexChanged = true;
		break;
	case LAYER_NZ:
		nz = Material(value);
		epsilonRefractiveIndexChanged = true;
		break;
	default:
		throw invalid_argument("Invalid param");
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

void Layer::SolveLayer(double wl, double beta, bool calcInvF, bool calcPhaseAndTransfer){
	SolveEpsilonMatrix(wl);
	SolveEigenFunction(beta);

	//phaseMatrix = Matrix4cd::Identity();
	//invF.fill(0);

	if (calcPhaseAndTransfer){
		// Phase matrix
		if (d != INFINITY){
			for (int i = 0; i < 4; i++){
				dcomplex phi = 2.0 * M_PI / wl * alpha(i) * d;
				phaseMatrix(i, i) = exp(dcomplex(0.0, -1.0) * phi);
			}
		}
	}

	if (calcInvF && calcPhaseAndTransfer){
		invF = F.inverse();
		M = F * phaseMatrix * invF;
	}
	else if (calcInvF && !calcPhaseAndTransfer){
		invF = F.inverse();
	}


	solved = true;
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

	Matrix3cd epsTensorCrystal = Matrix3cd::Zero();
	epsTensorCrystal(0, 0) = sqr(GetNx(wl));
	epsTensorCrystal(1, 1) = sqr(GetNy(wl));
	epsTensorCrystal(2, 2) = sqr(GetNz(wl));
	epsTensor = RotationSx(xi) * RotationSz(psi) * epsTensorCrystal * RotationSz(-psi) * RotationSx(-xi);

	wlEpsilonCalc = wl;
	epsilonRefractiveIndexChanged = false;
}

void Layer::SolveEigenFunction(double beta){
	double z0 = 119.9169832 * M_PI;
	dcomplex epsXX = epsTensor(0, 0);
	dcomplex epsYY = epsTensor(1, 1);
	dcomplex epsZZ = epsTensor(2, 2);
	dcomplex epsXY = epsTensor(0, 1);
	dcomplex epsXZ = epsTensor(0, 2);
	dcomplex epsYZ = epsTensor(1, 2);


	Matrix4cd mBeta = Matrix4cd::Zero();
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
	ComplexEigenSolver<Matrix4cd>::EigenvalueType eigenvalues = ces.eigenvalues();
	ComplexEigenSolver<Matrix4cd>::EigenvectorType eigenvectors = ces.eigenvectors();

	// Sort eigenvalues

	Vector4d poyntingXTmp;
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
	Vector4i order;
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
