#pragma once
#define PYTHON_WRAP
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

#ifdef PYTHON_WRAP
#include <boost/python.hpp>
#endif

using namespace std;
//using namespace Eigen;

#define sqr(a) ((a) * (a))
#define len(a) int((a).size())

typedef complex<double> dcomplex;
typedef map<string, Eigen::RowVectorXcd> ComplexVectorMap;
typedef map<string, Eigen::RowVectorXd> DoubleVectorMap;

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
	LAYER_MAT_N,
	LAYER_MAT_NX,
	LAYER_MAT_NY,
	LAYER_MAT_NZ,
};

//---------------------------------------------------------------------
// Fuctions
//---------------------------------------------------------------------

Eigen::Matrix3cd RotationSx(double phi);
Eigen::Matrix3cd RotationSz(double phi);

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
	dcomplex n(double wl);

private:
	bool isStatic;
	dcomplex staticN;
};

//---------------------------------------------------------------------
// EMFields
//---------------------------------------------------------------------

struct EMFields {
public:
	Eigen::RowVectorXcd E;
	Eigen::RowVectorXcd H;

	EMFields() : E(3), H(3) {

	}

	Eigen::RowVectorXcd GetE(){
		return E;
	}

	Eigen::RowVectorXcd GetH(){
		return H;
	}
};

//---------------------------------------------------------------------
// EMFields
//---------------------------------------------------------------------

struct EMFieldsList {
public:
	Eigen::MatrixXcd E;
	Eigen::MatrixXcd H;

	EMFieldsList(int size) : E(size, 3), H(size, 3) {

	}

	Eigen::MatrixXcd GetE(){
		return E;
	}

	Eigen::MatrixXcd GetH(){
		return H;
	}
};

//---------------------------------------------------------------------
// LayerIndices
//---------------------------------------------------------------------

struct LayerIndices {
public:
	Eigen::VectorXi indices;
	Eigen::VectorXd ds;
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
	PositionSettings(Eigen::RowVector2d polarization_, int interfaceId_, double distFromInterface_){
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

	Eigen::RowVector2d GetPolarization(){
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
	Eigen::RowVector2d polarization;
	int interfaceId;
	double distFromInterface;
};

//---------------------------------------------------------------------
// Layer
//---------------------------------------------------------------------

class Layer{
	friend class Tmm;

public:

	Layer(double d_, dcomplex n_);
	Layer(double d_, dcomplex nx_, dcomplex ny_, dcomplex nz_, double psi_, double xi_);
	void SetParam(Param param, int value);
	void SetParam(Param param, double value);
	void SetParam(Param param, dcomplex value);
	double GetD();
	dcomplex GetNx(double wl);
	dcomplex GetNy(double wl);
	dcomplex GetNz(double wl);
	void SolveLayer(double wl, double beta);
	EMFields GetFields(double wl, double beta, double x, Eigen::Vector4cd coefs);

private:
	bool solved;
	bool isotropicLayer;
	bool epsilonRefractiveIndexChanged;
	double wlEpsilonCalc;

	double d;
	Material nx, ny, nz;
	double psi;
	double xi;

	Eigen::ComplexEigenSolver<Eigen::Matrix4cd> ces;
	Eigen::Matrix3cd epsTensor;
	Eigen::Vector4cd alpha;
	Eigen::Vector4d poyntingX;
	Eigen::Matrix4cd F;
	Eigen::Matrix4cd invF;
	Eigen::Matrix4cd phaseMatrix;
	Eigen::Matrix4cd M;

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
	void AddIsotropicLayer(double d, dcomplex n);
	void AddLayer(double d, dcomplex nx, dcomplex ny, dcomplex nz, double psi, double xi);
	Eigen::Matrix4d GetIntensityMatrix();
	Eigen::Matrix4cd GetAmplitudeMatrix();
	SweepRes Sweep(Param sweepParam, Eigen::VectorXd sweepValues, PositionSettings enhpos);
	SweepRes Sweep(Param sweepParam, Eigen::VectorXd sweepValues);
	EMFieldsList CalcFields1D(Eigen::VectorXd xs, Eigen::VectorXd polarization);
	EMFields CalcFieldsAtInterface(PositionSettings pos);
	double OptimizeEnhancement(vector<Param> optParams, Eigen::VectorXd optInitial, PositionSettings pos);

#ifdef PYTHON_WRAP
	double OptimizeEnhancementPython(boost::python::list optParams, Eigen::VectorXd optInitial, PositionSettings pos);
#endif

private:
	double wl;
	double beta;
	double enhOptMaxRelError;
	double enhOptInitialStep;
	int enhOptMaxIters;
	vector<Layer> layers;
	vector<vector<string> > names_R;
	vector<vector<string> > names_r;
	Eigen::Matrix4cd A;
	bool solved;
	bool needToSolve;
	bool needToCalcFieldCoefs;
	Eigen::Vector2d lastFieldCoefsPol;
	Eigen::Matrix4d R;
	Eigen::Matrix4cd r;

	double normCoef;
	Eigen::MatrixXcd fieldCoefs;

	void Solve();
	void CalcFieldCoefs(Eigen::Vector2d polarization);
	LayerIndices CalcLayerIndices(Eigen::VectorXd &xs);
};

//---------------------------------------------------------------------
// EnhFitStuct
//---------------------------------------------------------------------

class EnhFitStuct{
	friend Tmm;
public:
	typedef double DataType;
	typedef Eigen::VectorXd ParameterType;

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
