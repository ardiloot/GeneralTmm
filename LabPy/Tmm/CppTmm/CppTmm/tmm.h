#pragma once
#include <iostream>
#include <complex>
#include <exception>
#include <vector>
#include <cmath>
#include <string>
#include <ctime>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include "simplex.h"
#include "criteria.h"
#include <boost/python.hpp>

#define sqr(a) ((a) * (a))
#define len(a) int((a).size())

namespace TmmModel
{
	using namespace std;
	using namespace Eigen;

	typedef complex<double> dcomplex;
	typedef map<string, RowVectorXcd> ComplexVectorMap;
	typedef map<string, RowVectorXd> DoubleVectorMap;

	//---------------------------------------------------------------------
	// Param Type
	//---------------------------------------------------------------------

	enum ParamType {
		WL,
		BETA,
		ENH_OPT_REL,
		ENH_OPT_MAX_ITERS,
		ENH_INITIAL_STEP,
		LAYER_D,
		LAYER_N,
		LAYER_NX,
		LAYER_NY,
		LAYER_NZ,
		LAYER_PSI,
		LAYER_XI,
		LAYER_MAT_NX,
		LAYER_MAT_NY,
		LAYER_MAT_NZ
	};

	//---------------------------------------------------------------------
	// Fuctions
	//---------------------------------------------------------------------

	Matrix3cd RotationSx(double phi);
	Matrix3cd RotationSz(double phi);

	//---------------------------------------------------------------------
	// Param
	//---------------------------------------------------------------------

	class Param {
	public:

		Param(ParamType pType_);
		Param(ParamType pType_, int layerId_);
		ParamType GetParamType();
		int GetLayerID();

		bool operator == (const Param &p) const
		{
			return (pType == p.pType && layerId == p.layerId);
		}

	private:
		ParamType pType;
		int layerId;
	};

	//---------------------------------------------------------------------
	// Material
	//---------------------------------------------------------------------

	class Material {
	public:

		Material();
		Material(dcomplex staticN_);
		Material(boost::python::object &materialClass);
		dcomplex n(double wl);
		bool IsStatic();

	private:
		bool isStatic;
		dcomplex staticN;
		boost::python::object materialClass;
	};

	//---------------------------------------------------------------------
	// EMFields
	//---------------------------------------------------------------------

	struct EMFields {
	public:
		RowVectorXcd E;
		RowVectorXcd H;

		EMFields() : E(3), H(3) {

		}

		RowVectorXcd GetE(){
			return E;
		}

		RowVectorXcd GetH(){
			return H;
		}
	};

	//---------------------------------------------------------------------
	// EMFields
	//---------------------------------------------------------------------

	struct EMFieldsList {
	public:
		MatrixXcd E;
		MatrixXcd H;

		EMFieldsList(int size) : E(size, 3), H(size, 3) {

		}

		MatrixXcd GetE(){
			return E;
		}

		MatrixXcd GetH(){
			return H;
		}
	};

	//---------------------------------------------------------------------
	// LayerIndices
	//---------------------------------------------------------------------

	struct LayerIndices {
	public:
		VectorXi indices;
		VectorXd ds;
	};

	//---------------------------------------------------------------------
	// SweepRes
	//---------------------------------------------------------------------

	struct SweepRes {
	public:
		ComplexVectorMap mapComplex;
		DoubleVectorMap mapDouble;

		ComplexVectorMap GetComplexMap(){
			return mapComplex;
		}

		DoubleVectorMap GetDoubleMap(){
			return mapDouble;
		}
	};

	//---------------------------------------------------------------------
	// PositionSettings
	//---------------------------------------------------------------------

	struct PositionSettings {
		//	friend class Tmm;
	public:
		PositionSettings(RowVector2d polarization_, int interfaceId_, double distFromInterface_){
			polarization = polarization_;
			interfaceId = interfaceId_;
			distFromInterface = distFromInterface_;
			enabled = true;
		}

		PositionSettings(){
			polarization.setZero();
			interfaceId = -1;
			distFromInterface = 0.0;
			enabled = false;
		}

		RowVector2d GetPolarization(){
			return polarization;
		}

		int GetInterfaceId(){
			return interfaceId;
		}

		double GetDistFromInterface(){
			return distFromInterface;
		}

		bool IsEnabled(){
			return enabled;
		}

	private:
		bool enabled;
		RowVector2d polarization;
		int interfaceId;
		double distFromInterface;
	};

	//---------------------------------------------------------------------
	// Layer
	//---------------------------------------------------------------------

	class Layer{
		friend class Tmm;

	public:

		Layer(double d_, Material n_);
		Layer(double d_, Material nx_, Material ny_, Material nz_, double psi_, double xi_);
		void SetParam(Param param, int value);
		void SetParam(Param param, double value);
		void SetParam(Param param, dcomplex value);
		int GetParamInt(Param param);
		double GetParamDouble(Param param);
		dcomplex GetParamComplex(Param param);
		double GetD();
		dcomplex GetNx(double wl);
		dcomplex GetNy(double wl);
		dcomplex GetNz(double wl);
		void SolveLayer(double wl, double beta);
		EMFields GetFields(double wl, double beta, double x, Vector4cd coefs);

	private:
		bool solved;
		bool isotropicLayer;
		bool epsilonRefractiveIndexChanged;
		double wlEpsilonCalc;

		double d;
		Material nx, ny, nz;
		double psi;
		double xi;

		ComplexEigenSolver<Matrix4cd> ces;
		Matrix3cd epsTensor;
		Vector4cd alpha;
		Vector4d poyntingX;
		Matrix4cd F;
		Matrix4cd invF;
		Matrix4cd phaseMatrix;
		Matrix4cd M;

		void Init();
		void SolveEpsilonMatrix(double wl);
		void SolveEigenFunction(double beta);

	};


	//---------------------------------------------------------------------
	// Tmm
	//---------------------------------------------------------------------

	class Tmm {
	public:

		Tmm();
		void SetParam(Param param, int value);
		void SetParam(Param param, double value);
		void SetParam(Param param, dcomplex value);
		int GetParamInt(Param param);
		double GetParamDouble(Param param); 
		dcomplex GetParamComplex(Param param);
		void AddIsotropicLayer(double d, dcomplex n);
		void AddIsotropicLayer(double d, boost::python::object &materialClass);
		void AddLayer(double d, dcomplex nx, dcomplex ny, dcomplex nz, double psi, double xi);
		void AddLayer(double d, boost::python::object &matX, boost::python::object &matY, boost::python::object &matZ, double psi, double xi);
		void ClearLayers();
		Matrix4d GetIntensityMatrix();
		Matrix4cd GetAmplitudeMatrix();
		SweepRes Sweep(Param sweepParam, VectorXd sweepValues, PositionSettings enhpos, int alphasLayer);
		SweepRes Sweep(Param sweepParam, VectorXd sweepValues);
		EMFieldsList CalcFields1D(VectorXd xs, VectorXd polarization);
		EMFields CalcFieldsAtInterface(PositionSettings pos);
		double OptimizeEnhancement(vector<Param> optParams, VectorXd optInitial, PositionSettings pos);
		double OptimizeEnhancementPython(boost::python::list optParams, VectorXd optInitial, PositionSettings pos);


	private:
		double wl;
		double beta;
		double enhOptMaxRelError;
		double enhOptInitialStep;
		int enhOptMaxIters;
		vector<Layer> layers;
		vector<vector<string> > names_R;
		vector<vector<string> > names_r;
		Matrix4cd A;
		bool solved;
		bool needToSolve;
		bool needToCalcFieldCoefs;
		Vector2d lastFieldCoefsPol;
		Matrix4d R;
		Matrix4cd r;

		double normCoef;
		MatrixXcd fieldCoefs;

		void Solve();
		void CalcFieldCoefs(Vector2d polarization);
		LayerIndices CalcLayerIndices(VectorXd &xs);
	};

	//---------------------------------------------------------------------
	// EnhFitStuct
	//---------------------------------------------------------------------

	class EnhFitStuct{
		friend Tmm;
	public:
		typedef double DataType;
		typedef VectorXd ParameterType;

		EnhFitStuct(Tmm *tmm_, vector<Param> optParams_, PositionSettings enhpos_){
			tmm = tmm_;
			optParams = optParams_;
			enhPos = enhpos_;
		}

		DataType operator()(const ParameterType &params) const
		{
			//cout << "fit function call res " << params << endl;
			SetParams(params);
			EMFields r = tmm->CalcFieldsAtInterface(enhPos);
			double res = -r.E.norm();
			return res;
		}
	private:
		Tmm *tmm;
		vector<Param> optParams;
		PositionSettings enhPos;

		void SetParams(const ParameterType &params) const
		{
			for (int i = 0; i < len(params); i++){
				tmm->SetParam(optParams[i], params[i]);
			}
		}

	};
} // Namespace