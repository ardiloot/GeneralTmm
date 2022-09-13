//#define EIGEN_USE_MKL_ALL
#include "tmm.h"
#include <ctime>

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

int main(){
	using namespace TmmModel;

	Eigen::Vector2d pol; pol << 1.0, 0.0;
	PositionSettings ps(pol, -1, 0.0);

	Tmm tmm;
	Material mat0(1.7), mat1(1.0);

	tmm.SetParam(Param(WL), 500e-9);
	tmm.AddIsotropicLayer(INFINITY, &mat0);
	tmm.AddIsotropicLayer(INFINITY, &mat1);

	/*
	std::vector<Param> optParams;
	optParams.push_back(Param(BETA));
	Eigen::VectorXd optInitial(len(optParams));
	optInitial << 0.5;

	tmm.OptimizeEnhancement(optParams, optInitial, ps);
	*/
	clock_t startTime = clock();
	Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(10, 0.0, 1.4);
	Eigen::Map<Eigen::ArrayXd> betasMap(&betas[0], betas.size());

	SweepRes rr = tmm.Sweep(Param(BETA), betasMap);
	cout << rr.mapDouble["R22"] << endl;
	cout << double(clock() - startTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;

	//tmm.SetParam(Param(BETA), 0.5);
	
	//Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(1000000, -1e-6, 1e-6);
	//EMFieldsList r = tmm.CalcFields1D(xs, pol);
	//tmm.CalcFieldsAtInterface(ps);

	system("pause");
	return 0;
}