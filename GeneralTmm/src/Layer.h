#pragma once
#include "Common.h"
#include "Material.h"

namespace TmmModel {

	//---------------------------------------------------------------------
	// Layer
	//---------------------------------------------------------------------

	class Layer {
		friend class Tmm;

	public:
		Layer(double d_, Material *n_);
		Layer(double d_, Material *nx_, Material *ny_, Material *nz_, double psi_, double xi_);
		void SetParam(Param param, int value);
		void SetParam(Param param, double value);
		void SetParam(Param param, dcomplex value);
		int GetParamInt(Param param) const;
		double GetParamDouble(Param param) const;
		dcomplex GetParamComplex(Param param) const;
		double GetD() const;
		dcomplex GetNx(double wl) const;
		dcomplex GetNy(double wl) const;
		dcomplex GetNz(double wl) const;
		void SolveLayer(double wl, double beta);
		EMFields GetFields(double wl, double beta, double x, Vector4cd coefs, WaveDirection waveDirection) const;

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
}