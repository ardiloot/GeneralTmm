#pragma once
#include "Common.h"
#include "Material.h"
#include "Layer.h"
#include "simplex.h"
#include "criteria.h"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

namespace TmmModel
{
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

		EnhFitStuct(Tmm *tmm_, std::vector<Param> optParams_, PositionSettings enhpos_);

		DataType operator()(const ParameterType &params) const
		{
			SetParams(params);
			EMFields r = tmm->CalcFieldsAtInterface(enhPos, WD_BOTH);
			double res = -r.E.matrix().norm();
			return res;
		}
	private:
		Tmm *tmm;
		std::vector<Param> optParams;
		PositionSettings enhPos;
		void SetParams(const ParameterType &params) const;

	};
} // Namespace