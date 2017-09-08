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
#include "Common.h"
#include "Material.h"
//#include <boost/python.hpp>

#define sqr(a) ((a) * (a))
#define len(a) int((a).size())

namespace TmmModel
{
	typedef std::map<std::string, ArrayXcd> ComplexVectorMap;
	typedef std::map<std::string, ArrayXd> DoubleVectorMap;
	

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
		LAYER_MAT_NZ,
		NOT_DEFINED
	};

	//---------------------------------------------------------------------
	// WaveDirection
	//---------------------------------------------------------------------

	enum WaveDirection {
		WD_FORWARD,
		WD_BACKWARD,
		WD_BOTH
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
		Param() : Param(NOT_DEFINED) {}
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
	// EMFields
	//---------------------------------------------------------------------

	struct EMFields {
	public:
		ArrayXcd E;
		ArrayXcd H;

		EMFields() : E(3), H(3) {

		}

		ArrayXcd GetE(){
			return E;
		}

		ArrayXcd GetH(){
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

		EMFieldsList(int size = 0) : E(size, 3), H(size, 3) {

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

		PositionSettings(double polCoef1, double polCoef2, int interfaceId_, double distFromInterface_) {
			polarization << polCoef1, polCoef2;
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

		Layer(double d_, Material *n_);
		Layer(double d_, Material *nx_, Material *ny_, Material *nz_, double psi_, double xi_);
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
		EMFields GetFields(double wl, double beta, double x, Vector4cd coefs, WaveDirection waveDirection);

	private:
		bool solved;
		bool isotropicLayer;
		bool epsilonRefractiveIndexChanged;
		double wlEpsilonCalc;

		double d;
		Material *nx, *ny, *nz;
		double psi;
		double xi;

		Eigen::ComplexEigenSolver<Matrix4cd> ces;
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
		void AddIsotropicLayer(double d, Material *mat);
		void AddLayer(double d, Material *matx, Material *maty, Material *matz, double psi, double xi);
		void ClearLayers();
		Matrix4d GetIntensityMatrix();
		Matrix4cd GetAmplitudeMatrix();
		SweepRes Sweep(Param sweepParam, const Eigen::Map<Eigen::ArrayXd> &sweepValues, PositionSettings enhpos, int alphasLayer);
		SweepRes Sweep(Param sweepParam, const Eigen::Map<Eigen::ArrayXd> &sweepValues);
		EMFieldsList CalcFields1D(const Eigen::Map<Eigen::ArrayXd> &xs, const Eigen::Map<Eigen::Array2d> &polarization, WaveDirection waveDirection);
		EMFields CalcFieldsAtInterface(PositionSettings pos, WaveDirection waveDirection);
		double OptimizeEnhancement(std::vector<Param> optParams, ArrayXd optInitial, PositionSettings pos);
		//double OptimizeEnhancementPython(boost::python::list optParams, VectorXd optInitial, PositionSettings pos);


	private:
		double wl;
		double beta;
		double enhOptMaxRelError;
		double enhOptInitialStep;
		int enhOptMaxIters;
		std::vector<Layer> layers;
		std::vector<std::vector<std::string> > names_R;
		std::vector<std::vector<std::string> > names_r;
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
		LayerIndices CalcLayerIndices(const Eigen::Map<Eigen::ArrayXd> &xs);
	};

	//---------------------------------------------------------------------
	// EnhFitStuct
	//---------------------------------------------------------------------

	class EnhFitStuct{
		friend Tmm;
	public:
		typedef double DataType;
		typedef VectorXd ParameterType;

		EnhFitStuct(Tmm *tmm_, std::vector<Param> optParams_, PositionSettings enhpos_){
			tmm = tmm_;
			optParams = optParams_;
			enhPos = enhpos_;
		}

		DataType operator()(const ParameterType &params) const
		{
			//cout << "fit function call res " << params << endl;
			SetParams(params);
			EMFields r = tmm->CalcFieldsAtInterface(enhPos, WD_BOTH);
			double res = -r.E.matrix().norm();
			return res;
		}
	private:
		Tmm *tmm;
		std::vector<Param> optParams;
		PositionSettings enhPos;

		void SetParams(const ParameterType &params) const
		{
			for (int i = 0; i < len(params); i++){
				tmm->SetParam(optParams[i], params[i]);
			}
		}

	};
} // Namespace