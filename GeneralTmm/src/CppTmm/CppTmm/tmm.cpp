#include "tmm.h"


//---------------------------------------------------------------------
// Tmm
//---------------------------------------------------------------------
namespace TmmModel {
	Tmm::Tmm(){
		solved = false;
		needToSolve = true;
		needToCalcFieldCoefs = true;
		wl = 500e-9;
		beta = 0.0;
		enhOptMaxIters = 100;
		enhOptMaxRelError = 1e-10;
		enhOptInitialStep = 0.1;

		names_R = std::vector<std::vector<std::string> >(4, std::vector<std::string>(4));
		names_r = std::vector<std::vector<std::string> >(4, std::vector<std::string>(4));
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				std::ostringstream ss;
				ss << i + 1 << j + 1;
				std::string numbers = ss.str();
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

	void Tmm::SetParam(Param param, int value){
		needToSolve = true;
		if (param.GetLayerID() < 0){
			switch (param.GetParamType())
			{
			case ENH_OPT_MAX_ITERS:
				enhOptMaxIters = value;
				break;
			default:
				throw std::invalid_argument("Invalid param int");
				break;
			}
		}
		else {
			layers[param.GetLayerID()].SetParam(param, value);
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
			case ENH_OPT_REL:
				enhOptMaxRelError = value;
				break;
			case ENH_INITIAL_STEP:
				enhOptInitialStep = value;
				break;
			default:
				throw std::invalid_argument("Invalid param double");
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
			throw std::invalid_argument("Invalid param complex");
		}
		else {
			layers[param.GetLayerID()].SetParam(param, value);
		}
	}

	int Tmm::GetParamInt(Param param){
		if (param.GetLayerID() < 0){
			switch (param.GetParamType())
			{
			case ENH_OPT_MAX_ITERS:
				return enhOptMaxIters;
				break;
			default:
				throw std::invalid_argument("Get invalid param int");
				break;
			}
		}
		else {
			return layers[param.GetLayerID()].GetParamInt(param);
		}
	}

	double Tmm::GetParamDouble(Param param){
		if (param.GetLayerID() < 0){
			switch (param.GetParamType())
			{
			case WL:
				return wl;
				break;
			case BETA:
				return beta;
				break;
			case ENH_OPT_REL:
				return enhOptMaxRelError;
				break;
			case ENH_INITIAL_STEP:
				return enhOptInitialStep;
				break;
			default:
				throw std::invalid_argument("Get invalid param double");
				break;
			}
		}
		else {
			return layers[param.GetLayerID()].GetParamDouble(param);
		}
	}

	dcomplex Tmm::GetParamComplex(Param param){
		if (param.GetLayerID() < 0){
			throw std::invalid_argument("Get invalid param complex");
		}
		else {
			return layers[param.GetLayerID()].GetParamComplex(param);
		}
	}

	void Tmm::AddIsotropicLayer(double d, Material *mat){
		needToSolve = true;
		layers.push_back(Layer(d, mat));
	}

	void Tmm::AddLayer(double d, Material *matx, Material *maty, Material *matz, double psi, double xi){
		needToSolve = true;
		layers.push_back(Layer(d, matx, maty, matz, psi, xi));
	}

	/*
	void Tmm::AddLayer(double d, boost::python::object &matX, boost::python::object &matY, boost::python::object &matZ, double psi, double xi){
		needToSolve = true;
		layers.push_back(Layer(d, Material(matX), Material(matY), Material(matZ), psi, xi));
	}
	*/

	void Tmm::ClearLayers(){
		layers.clear();
	}

	Eigen::Matrix4d Tmm::GetIntensityMatrix(){
		Solve();
		return R;
	}

	Eigen::Matrix4cd Tmm::GetAmplitudeMatrix(){
		Solve();
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

	SweepRes Tmm::Sweep(Param sweepParam, const Eigen::Map<Eigen::ArrayXd> &sweepValues, PositionSettings enhpos, int alphasLayer){
		SweepRes res;
		ComplexVectorMap &resComplex = res.mapComplex;
		DoubleVectorMap &resDouble = res.mapDouble;
		DoubleVectorMap::iterator data_R[4][4];
		ComplexVectorMap::iterator data_r[4][4];
	
		
		ComplexVectorMap::iterator alphas0;
		ComplexVectorMap::iterator alphas1;
		ComplexVectorMap::iterator alphas2;
		ComplexVectorMap::iterator alphas3;
		bool alphasEnabled = bool(alphasLayer >= 0);
		
		if (alphasEnabled){
			alphas0 = resComplex.insert(std::make_pair("alphas0", Eigen::RowVectorXcd(len(sweepValues)))).first;
			alphas1 = resComplex.insert(std::make_pair("alphas1", Eigen::RowVectorXcd(len(sweepValues)))).first;
			alphas2 = resComplex.insert(std::make_pair("alphas2", Eigen::RowVectorXcd(len(sweepValues)))).first;
			alphas3 = resComplex.insert(std::make_pair("alphas3", Eigen::RowVectorXcd(len(sweepValues)))).first;
		}

		DoubleVectorMap::iterator enhs;
		DoubleVectorMap::iterator enhExs;
		DoubleVectorMap::iterator enhEys;
		DoubleVectorMap::iterator enhEzs;

		if (enhpos.IsEnabled()){
			enhs = resDouble.insert(std::make_pair("enh", Eigen::RowVectorXd(len(sweepValues)))).first;
			enhExs = resDouble.insert(std::make_pair("enhEx", Eigen::RowVectorXd(len(sweepValues)))).first;
			enhEys = resDouble.insert(std::make_pair("enhEy", Eigen::RowVectorXd(len(sweepValues)))).first;
			enhEzs = resDouble.insert(std::make_pair("enhEz", Eigen::RowVectorXd(len(sweepValues)))).first;
		}

		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 2; j++){
				data_R[i][j] = resDouble.insert(std::make_pair(names_R[i][j], Eigen::RowVectorXd(len(sweepValues)))).first;
				data_r[i][j] = resComplex.insert(std::make_pair(names_r[i][j], Eigen::RowVectorXcd(len(sweepValues)))).first;
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

			if (alphasEnabled){
				alphas0->second(i) = layers[alphasLayer].alpha(0);
				alphas1->second(i) = layers[alphasLayer].alpha(1);
				alphas2->second(i) = layers[alphasLayer].alpha(2);
				alphas3->second(i) = layers[alphasLayer].alpha(3);
			}

			if (enhpos.IsEnabled()){
				EMFields fields = CalcFieldsAtInterface(enhpos, WD_BOTH);
				enhs->second(i) = fields.E.matrix().norm();
				enhExs->second(i) = abs(fields.E(0));
				enhEys->second(i) = abs(fields.E(1));
				enhEzs->second(i) = abs(fields.E(2));
			}
		}

		return res;
	}

	SweepRes Tmm::Sweep(Param sweepParam, const Eigen::Map<Eigen::ArrayXd> &sweepValues){
		PositionSettings enhpos;
		return Sweep(sweepParam, sweepValues, enhpos, -1);
	}

	EMFieldsList Tmm::CalcFields1D(const Eigen::Map<Eigen::ArrayXd> &xs, const Eigen::Map<Eigen::Array2d> &polarization, WaveDirection waveDirection){
		Solve();
		CalcFieldCoefs(polarization);

		EMFieldsList res(len(xs));
		LayerIndices layerP = CalcLayerIndices(xs);
		for (int i = 0; i < len(xs); i++){
			int layerId = layerP.indices(i);
			EMFields f = layers[layerId].GetFields(wl, beta, layerP.ds(i), fieldCoefs.row(layerId), waveDirection);
			res.E.row(i) = f.E / normCoef;
			res.H.row(i) = f.H / normCoef;
		}
		return res;
	}

	EMFields Tmm::CalcFieldsAtInterface(PositionSettings pos, WaveDirection waveDirection){
		if (!pos.IsEnabled()){
			throw std::invalid_argument("Position settings must be enabled.");
		}
	
		int layerId;
		if (pos.GetInterfaceId() < 0){
			layerId = len(layers) + pos.GetInterfaceId();
		}
		else{
			layerId = pos.GetInterfaceId();
		}

		Solve();
		CalcFieldCoefs(pos.GetPolarization());
		EMFields res = layers[layerId].GetFields(wl, beta, pos.GetDistFromInterface(), fieldCoefs.row(layerId), waveDirection);
		res.E /= normCoef;
		res.H /= normCoef;
		return res;
	}

	double Tmm::OptimizeEnhancement(std::vector<Param> optParams, Eigen::ArrayXd optInitial, PositionSettings pos){
		EnhFitStuct fitFunc(this, optParams, pos);
		auto criterion = Optimization::Local::make_and_criteria(Optimization::Local::IterationCriterion(enhOptMaxIters), Optimization::Local::RelativeValueCriterion<double>(enhOptMaxRelError));
		auto optimizer = Optimization::Local::build_simplex(fitFunc, criterion);

		optimizer.set_start_point(optInitial);
		optimizer.set_delta(enhOptInitialStep); // TODO, set deltas
		optimizer.optimize(fitFunc);

		
		fitFunc.SetParams(optimizer.get_best_parameters());
		double res = -optimizer.get_best_value();
		int nIterations = optimizer.get_number_of_iterations();

		if (nIterations >= enhOptMaxIters){
			cerr << "Maximum number of iterations reached: " << nIterations << "/" << enhOptMaxIters << endl;
		}

		return res;
	}

	/*
	double Tmm::OptimizeEnhancementPython(boost::python::list optParams, Eigen::VectorXd optInitial, PositionSettings pos){
		std::vector<Param> optParamsVector;
		ssize_t length = PyObject_Length(optParams.ptr());
		for (int i = 0; i < length; ++i)
		{
			optParamsVector.push_back(boost::python::extract<Param>(optParams[i]));
		}
		double res = Tmm::OptimizeEnhancement(optParamsVector, optInitial, pos);
		return res;
	}
	*/

	void Tmm::CalcFieldCoefs(Eigen::Vector2d polarization){	
		if (!needToCalcFieldCoefs && polarization == lastFieldCoefsPol){
			return;
		}
		needToCalcFieldCoefs = false;
		lastFieldCoefsPol = polarization;

		//Calc normalization factor
		Eigen::Vector4cd incCoefs;
		incCoefs << polarization(0), 0.0, polarization(1), 0.0;
		Eigen::Vector3cd Einc = layers[0].GetFields(wl, beta, 0.0, incCoefs, WD_BOTH).E;
	
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

	LayerIndices Tmm::CalcLayerIndices(const Eigen::Map<Eigen::ArrayXd> &xs){
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
} // Namespace