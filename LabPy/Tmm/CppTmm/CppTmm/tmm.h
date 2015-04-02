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


using namespace std;
//using namespace Eigen;


#define sqr(a) ((a) * (a))
#define len(a) int((a).size())

typedef complex<double> dcomplex;
typedef map<string, Eigen::RowVectorXcd> ComplexVectorMap;

//---------------------------------------------------------------------
// Param Type
//---------------------------------------------------------------------

enum ParamType {
	WL,
	BETA,
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
// Layer
//---------------------------------------------------------------------

class Layer{
	friend class Tmm;

public:

	Layer(double d_, dcomplex n_);
	Layer(double d_, dcomplex nx_, dcomplex ny_, dcomplex nz_, double psi_, double xi_);
	void SetParam(Param param, double value);
	void SetParam(Param param, dcomplex value);
	double GetD();
	dcomplex GetNx(double wl);
	dcomplex GetNy(double wl);
	dcomplex GetNz(double wl);
	void SolveLayer(double wl, double beta, bool calcInvF);

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
	void SetParam(Param param, double value);
	void SetParam(Param param, dcomplex value);
	void AddIsotropicLayer(double d, dcomplex n);
	void AddLayer(double d, dcomplex nx, dcomplex ny, dcomplex nz, double psi, double xi);
	Eigen::Matrix4d GetIntensityMatrix();
	Eigen::Matrix4cd GetAmplitudeMatrix();
	void Solve();
	ComplexVectorMap Sweep(Param sweepParam, Eigen::VectorXd sweepValues);

private:
	double wl;
	double beta;
	vector<Layer> layers;
	vector<vector<string> > names_R;
	vector<vector<string> > names_r;
	Eigen::Matrix4cd A;
	bool solved;
	Eigen::Matrix4d R;
	Eigen::Matrix4cd r;
};
