#pragma once
#include <iostream>
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

	//---------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------

	typedef std::complex<double> dcomplex;
	
	//---------------------------------------------------------------
	// Functions
	//---------------------------------------------------------------
	template <typename T> T Interpolate(double x, const ArrayXd & xs, const Eigen::Array<T, Eigen::Dynamic, 1> & ys);


}
