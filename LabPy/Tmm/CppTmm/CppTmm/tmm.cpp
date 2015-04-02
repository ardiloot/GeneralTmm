#include "tmm.h"

//---------------------------------------------------------------------
// Tmm
//---------------------------------------------------------------------

Tmm::Tmm(){
	solved = false;
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
	//cout << "SetParamDouble " << int(param.GetParamType()) << " " << value << endl;
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
	//cout << "SetParamComplex " << int(param.GetParamType()) << " " << value << endl;
	if (param.GetLayerID() < 0){
		throw invalid_argument("Invalid param");
	}
	else {
		layers[param.GetLayerID()].SetParam(param, value);
	}
}

void Tmm::AddIsotropicLayer(double d, dcomplex n){
	layers.push_back(Layer(d, n));
}

void Tmm::AddLayer(double d, dcomplex nx, dcomplex ny, dcomplex nz, double psi, double xi){
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

	for (int i = 0; i < len(layers); i++){
		//bool first = (i == 0);
		bool last = (i == len(layers) - 1);
		layers[i].SolveLayer(wl, beta, !last);
	}

	// System matrix
	A = layers[0].invF;
	for (int i = 1; i < len(layers) - 1; i++){
		Layer &layer = layers[i];
		A = A * layer.F * layer.phaseMatrix * layer.invF;
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

	Eigen::VectorXcd data_R[4][4];
	Eigen::VectorXcd data_r[4][4];
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 2; j++){
			data_R[i][j] = Eigen::VectorXcd(len(sweepValues));
			data_r[i][j] = Eigen::VectorXcd(len(sweepValues));
		}
	}


	for (int i = 0; i < len(sweepValues); i++){
		SetParam(sweepParam, sweepValues[i]);
		Solve();

		for (int j = 0; j < 4; j++){
			for (int k = 0; k < 2; k++){
				data_R[j][k](i) = R(j, k);
				data_r[j][k](i) = r(j, k);
			}
		}
	}

	ComplexVectorMap res;
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 2; j++){
			res.insert(make_pair(names_R[i][j], data_R[i][j]));
			res.insert(make_pair(names_r[i][j], data_r[i][j]));
		}
	}
	return res;
}
