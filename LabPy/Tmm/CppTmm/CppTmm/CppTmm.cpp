#define EIGEN_USE_MKL_ALL
//#define PYTHON_WRAP

#ifdef PYTHON_WRAP
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "eigen_numpy.h"
using namespace boost::python;
namespace np = boost::numpy;
#endif

#include "tmm.h"
//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------


#ifndef PYTHON_WRAP

int main(){
	Tmm tmm;
	tmm.SetParam(Param(WL), 500e-9);
	tmm.AddIsotropicLayer(INFINITY, 1.5);
	tmm.AddIsotropicLayer(50e-9, 1.6);
	tmm.AddIsotropicLayer(50e-9, 1.7);
	tmm.AddIsotropicLayer(INFINITY, 1.0);

	clock_t startTime = clock();
	ComplexVectorMap rr = tmm.Sweep(Param(BETA), Eigen::VectorXd::LinSpaced(500000, 0.0, 1.4));
	//cout << rr["R22"] << endl;
	//cout << double(clock() - startTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;

	//system("pause");
	return 0;
}

#else

BOOST_PYTHON_MODULE(CppTmm)
{
	np::initialize();
	SetupEigenConverters();

	class_<ComplexVectorMap >("ComplexVectorMap")
		.def(map_indexing_suite<ComplexVectorMap, true>())
	;

	//---------------------------------------------------------------
	// Param Type
	//---------------------------------------------------------------

	enum_<ParamType>("ParamType")
		.value("WL", WL)
		.value("BETA", BETA)
		.value("LAYER_D", LAYER_D)
		.value("LAYER_N", LAYER_N)
		.value("LAYER_NX", LAYER_NX)
		.value("LAYER_NY", LAYER_NY)
		.value("LAYER_NZ", LAYER_NZ)
		.value("LAYER_PSI", LAYER_PSI)
		.value("LAYER_XI", LAYER_XI)
		.value("LAYER_MAT_N", LAYER_MAT_N)
		.value("LAYER_MAT_NX", LAYER_MAT_NX)
		.value("LAYER_MAT_NY", LAYER_MAT_NY)
		.value("LAYER_MAT_NZ", LAYER_MAT_NZ)
		;

	//---------------------------------------------------------------
	// Param
	//---------------------------------------------------------------

	class_<Param>("Param", init<ParamType, int>())
		.def(init<ParamType>())
		.def("GetParamType", &Param::GetParamType)
		.def("GetLayerID", &Param::GetLayerID)
		;

	//---------------------------------------------------------------
	// TMM
	//---------------------------------------------------------------

	void (Tmm::*BoostSetParamsDouble)(Param, double) = &Tmm::SetParam;
	void (Tmm::*BoostSetParamsComplex)(Param, dcomplex) = &Tmm::SetParam;

	//def("F", (void (C::*)(int))&C::F)  
	class_<Tmm>("Tmm")
		.def("SetParam", (BoostSetParamsComplex))
		.def("SetParam", (BoostSetParamsDouble))
		.def("AddIsotropicLayer", &Tmm::AddIsotropicLayer)
		.def("AddLayer", &Tmm::AddLayer)
		.def("Solve", &Tmm::Solve)
		.def("GetIntensityMatrix", &Tmm::GetIntensityMatrix)
		.def("GetAmplitudeMatrix", &Tmm::GetAmplitudeMatrix)
		.def("Sweep", &Tmm::Sweep)
		;

}

#endif

