#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <Eigen/Dense>

namespace TmmModel
{
	//---------------------------------------------------------------
	// Namespaces
	//---------------------------------------------------------------

	using Eigen::Array2cd;
	using Eigen::ArrayXd;
	using Eigen::ArrayXcd;
	using Eigen::VectorXi;
	using Eigen::VectorXd;
	using Eigen::Vector4d;
	using Eigen::Vector4i;
	using Eigen::Vector2d;
	using Eigen::Vector4cd;
	using Eigen::RowVector2d;
	using Eigen::Matrix2cd;
	using Eigen::Matrix3d;
	using Eigen::Matrix3cd;
	using Eigen::Matrix4cd;
	using Eigen::Matrix4d;
	using Eigen::MatrixXd;
	using Eigen::MatrixXcd;

	using std::cout;
	using std::cerr;
	using std::endl;
	using std::abs;
	using std::norm;

	//---------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------

	typedef std::complex<double> dcomplex;
	typedef std::map<std::string, ArrayXcd> ComplexVectorMap;
	typedef std::map<std::string, ArrayXd> DoubleVectorMap;
	
	//---------------------------------------------------------------
	// Functions
	//---------------------------------------------------------------

	double sqr(double a);
	dcomplex sqr(dcomplex a);
	template <typename T> T Interpolate(double x, const ArrayXd & xs, const Eigen::Array<T, Eigen::Dynamic, 1> & ys);
	Matrix3cd RotationSx(double phi);
	Matrix3cd RotationSz(double phi);

	//---------------------------------------------------------------------
	// Enums
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

	enum WaveDirection {
		WD_FORWARD,
		WD_BACKWARD,
		WD_BOTH
	};

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
	// PositionSettings
	//---------------------------------------------------------------------

	struct PositionSettings {
	public:
		PositionSettings();
		PositionSettings(RowVector2d polarization_, int interfaceId_, double distFromInterface_);
		PositionSettings(double polCoef1, double polCoef2, int interfaceId_, double distFromInterface_);
		RowVector2d GetPolarization();
		int GetInterfaceId();
		double GetDistFromInterface();
		bool IsEnabled();

	private:
		bool enabled;
		RowVector2d polarization;
		int interfaceId;
		double distFromInterface;
	};


	//---------------------------------------------------------------------
	// EMFields
	//---------------------------------------------------------------------

	struct EMFields {
	public:
		ArrayXcd E;
		ArrayXcd H;

		EMFields() : E(3), H(3) {}
	};

	//---------------------------------------------------------------------
	// EMFields
	//---------------------------------------------------------------------

	struct EMFieldsList {
	public:
		MatrixXcd E;
		MatrixXcd H;

		EMFieldsList(int size = 0) : E(size, 3), H(size, 3) {}
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
	};


}
