#include "tmm.h"

//---------------------------------------------------------------------
// Tmm
//---------------------------------------------------------------------

Tmm::Tmm(){
	solved = false;
	needToSolve = true;
	needToCalcFieldCoefs = true;
	wl = 500e-9;
	beta = 0.0;

	names_R = vector<vector<string> >(4, vector<string>(4));
	names_r = vector<vector<string> >(4, vector<string>(4));
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			ostringstream ss;
			ss << i + 1 << j + 1;
			string numbers = ss.str();
			if ((i < 2 && j < 2) || (i >= 2 && j >= 2)){
				names_R[i][j] = "R" + numbers;
				names_r[i][j] = "r" + numbers;
			}
			else {
				names_R[i][j] = "T" + numbers;
				names_r[i][j] = "t" + numbers;
			}
		}
	}
}

void Tmm::SetParam(Param param, double value){
	needToSolve = true;
	if (param.GetLayerID() < 0){
		switch (param.GetParamType())
		{
		case WL:
			wl = value;
			break;
		case BETA:
			beta = value;
			break;
		default:
			throw invalid_argument("Invalid param");
			break;
		}
	}
	else {
		layers[param.GetLayerID()].SetParam(param, value);
	}
}

void Tmm::SetParam(Param param, dcomplex value){
	needToSolve = true;
	if (param.GetLayerID() < 0){
		throw invalid_argument("Invalid param");
	}
	else {
		layers[param.GetLayerID()].SetParam(param, value);
	}
}

void Tmm::AddIsotropicLayer(double d, dcomplex n){
	needToSolve = true;
	layers.push_back(Layer(d, n));
}

void Tmm::AddLayer(double d, dcomplex nx, dcomplex ny, dcomplex nz, double psi, double xi){
	needToSolve = true;
	layers.push_back(Layer(d, nx, ny, nz, psi, xi));
}

Eigen::Matrix4d Tmm::GetIntensityMatrix(){
	if (!solved){
		throw runtime_error("TMM has to be solved before calling this function.");
	}
	return R;
}

Eigen::Matrix4cd Tmm::GetAmplitudeMatrix(){
	if (!solved){
		throw runtime_error("TMM has to be solved before calling this function.");
	}
	return r;
}

void Tmm::Solve(){
	if (!needToSolve){
		return;
	}
	needToSolve = false;
	needToCalcFieldCoefs = true;

	for (int i = 0; i < len(layers); i++){
		layers[i].SolveLayer(wl, beta);
	}

	// System matrix
	A = layers[0].invF;
	for (int i = 1; i < len(layers) - 1; i++){
		Layer &layer = layers[i];
		A = A * layer.M;
	}
	A = A * layers[len(layers) - 1].F;

	//r - matrix
	Eigen::Matrix4cd invr1;
	dcomplex t = A(0, 0)*A(2, 2) - A(0, 2) * A(2, 0);
	invr1 << -(A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) / t, 1, -(A(0, 0) * A(1, 2) - A(0, 2) * A(1, 0)) / t, 0,
		(A(2, 0) * A(3, 2) - A(2, 2) * A(3, 0)) / t, 0, -(A(0, 0) * A(3, 2) - A(0, 2) * A(3, 0)) / t, 1,
		-A(2, 2) / t, 0, A(0, 2) / t, 0,
		A(2, 0) / t, 0, -A(0, 0) / t, 0;

	Eigen::Matrix4cd r2;
	r2 << -1.0, 0.0, A(0, 1), A(0, 3),
		0.0, 0.0, A(1, 1), A(1, 3),
		0.0, -1.0, A(2, 1), A(2, 3),
		0.0, 0.0, A(3, 1), A(3, 3);

	r = invr1 * r2;

	Eigen::Vector4d &poyningXF = layers[0].poyntingX;
	Eigen::Vector4d &poyningXL = layers[len(layers) - 1].poyntingX;

	Eigen::Vector4d pBackward, pForward;
	pBackward << poyningXF(1), poyningXF(3), poyningXL(1), poyningXL(3);
	pForward << poyningXF(0), poyningXF(2), poyningXL(0), poyningXL(2);

	R(0, 0) = norm(r(0, 0)) * abs(pBackward(0) / pForward(0));
	R(0, 1) = norm(r(0, 1)) * abs(pBackward(0) / pForward(1));
	R(0, 2) = NAN;
	R(0, 3) = NAN;
	R(1, 0) = norm(r(1, 0)) * abs(pBackward(1) / pForward(0));
	R(1, 1) = norm(r(1, 1)) * abs(pBackward(1) / pForward(1));
	R(1, 2) = NAN;
	R(1, 3) = NAN;
	R(2, 0) = norm(r(2, 0)) * abs(pForward(2) / pForward(0));
	R(2, 1) = norm(r(2, 1)) * abs(pForward(2) / pForward(1));
	R(2, 2) = NAN;
	R(2, 3) = NAN;
	R(3, 0) = norm(r(3, 0)) * abs(pForward(3) / pForward(0));
	R(3, 1) = norm(r(3, 1)) * abs(pForward(3) / pForward(1));
	R(3, 2) = NAN;
	R(3, 3) = NAN;
	solved = true;
}

ComplexVectorMap Tmm::Sweep(Param sweepParam, Eigen::VectorXd sweepValues){

	ComplexVectorMap res;
	ComplexVectorMap::iterator data_R[4][4];
	ComplexVectorMap::iterator data_r[4][4];
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 2; j++){
			data_R[i][j] = res.insert(make_pair(names_R[i][j], Eigen::RowVectorXcd(len(sweepValues)))).first;
			data_r[i][j] = res.insert(make_pair(names_r[i][j], Eigen::RowVectorXcd(len(sweepValues)))).first;
		
		}
	}

	for (int i = 0; i < len(sweepValues); i++){
		SetParam(sweepParam, sweepValues[i]);
		Solve();

		for (int j = 0; j < 4; j++){
			for (int k = 0; k < 2; k++){
				data_R[j][k]->second(i) = R(j, k);
				data_r[j][k]->second(i) = r(j, k);
			}
		}
	}

	return res;
}


EMFieldsList Tmm::CalcFields1D(Eigen::VectorXd xs, Eigen::VectorXd polarization){
	Solve();

	EMFieldsList res(len(xs));
	CalcFieldCoefs(polarization);
	LayerIndices layerP = CalcLayerIndices(xs);
	for (int i = 0; i < len(xs); i++){
		int layerId = layerP.indices(i);
		EMFields f = layers[layerId].GetFields(wl, beta, layerP.ds(i), fieldCoefs.row(layerId));
		res.E.row(i) = f.E / normCoef;
		res.H.row(i) = f.H / normCoef;
	}
	return res;
}


void Tmm::CalcFieldCoefs(Eigen::Vector2d polarization){	
	if (!needToCalcFieldCoefs && polarization == lastFieldCoefsPol){
		return;
	}
	needToCalcFieldCoefs = false;
	lastFieldCoefsPol = polarization;

	//Calc normalization factor
	Eigen::Vector4cd incCoefs;
	incCoefs << polarization(0), 0.0, polarization(1), 0.0;
	Eigen::Vector3cd Einc = layers[0].GetFields(wl, beta, 0.0, incCoefs).E;
	
	dcomplex n1 = sqrt(sqr(beta) + sqr(layers[0].alpha(0)));
	dcomplex n2 = sqrt(sqr(beta) + sqr(layers[0].alpha(2)));
	double nEff = (polarization(0) * real(n1) + polarization(1) * real(n2)) / (polarization(0) + polarization(1)); // Maybe not fully correct
	normCoef = Einc.norm() * sqrt(nEff);
		
	//Calc output coefs
	Eigen::Vector4cd inputFields;
	inputFields << polarization(0), polarization(1), 0.0, 0.0;
	Eigen::Vector4cd outputFields = r * inputFields;

	//Calc coefs in all layers
	Eigen::Vector4cd coefsSubstrate;
	coefsSubstrate << outputFields(2), 0.0, outputFields(3), 0.0;

	Eigen::Matrix4cd mat = layers[len(layers) - 1].F;
	fieldCoefs.resize(len(layers), 4);
	for (int i = len(layers) - 1; i >= 0; i--){
		mat = layers[i].M * mat;
		fieldCoefs.row(i) = layers[i].invF * mat * coefsSubstrate;
	}
	fieldCoefs(len(layers) - 1, 1) = fieldCoefs(len(layers) - 1, 3) = 0.0;
}

LayerIndices Tmm::CalcLayerIndices(Eigen::VectorXd &xs){
	LayerIndices res;
	res.indices.resize(len(xs));
	res.ds.resize(len(xs));

	int curLayer = 0;
	double curDist = 0.0;
	double prevDist = 0.0;

	for (int i = 0; i < len(xs); i++){
		while (xs[i] >= curDist){
			curLayer++;
			prevDist = curDist;
			if (curLayer >= len(layers) - 1){
				curDist = INFINITY;
				curLayer = len(layers) - 1;
			}
			curDist += layers[curLayer].GetD();
		}
		res.indices(i) = curLayer;
		res.ds(i) = xs(i) - prevDist;
	}
	return res;
}
