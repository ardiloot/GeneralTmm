//#define EIGEN_USE_MKL_ALL
#define PYTHON_WRAP

#ifdef PYTHON_WRAP
#include <boost/python.hpp>
#include <boost/numpy.hpp>
//#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "std_map_indexing_suite.hpp"
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
	Eigen::setNbThreads(4);
	//int n = Eigen::nbThreads();
	//cout << "Will use " << n << " threads." << endl;

	Eigen::Vector2d pol; pol << 1.0, 0.0;
	PositionSettings ps(pol, -1, 0.0);

	Tmm tmm;
	tmm.SetParam(Param(WL), 500e-9);
	tmm.AddIsotropicLayer(INFINITY, 1.5);
	tmm.AddIsotropicLayer(50e-9, 1.6);
	tmm.AddIsotropicLayer(50e-9, 1.7);
	tmm.AddIsotropicLayer(INFINITY, 1.0);

	//clock_t startTime = clock();
	tmm.Sweep(Param(BETA), Eigen::VectorXd::LinSpaced(1000000, 0.0, 1.4), ps);
	//cout << rr["R22"] << endl;
	//cout << double(clock() - startTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;

	//tmm.SetParam(Param(BETA), 0.5);
	
	//Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(1000000, -1e-6, 1e-6);
	//EMFieldsList r = tmm.CalcFields1D(xs, pol);
	//tmm.CalcFieldsAtInterface(ps);

	//system("pause");
	return 0;
}

#else

BOOST_PYTHON_MODULE(CppTmm)
{
	np::initialize();
	SetupEigenConverters();

	class_<ComplexVectorMap >("ComplexVectorMap")
		.def(std_map_indexing_suite<ComplexVectorMap, true>())
	;

	class_<DoubleVectorMap >("DoubleVectorMap")
		.def(std_map_indexing_suite<DoubleVectorMap, true>())
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
	// SweepRes
	//---------------------------------------------------------------

	class_<SweepRes>("SweepRes")
		.add_property("resComplex", &SweepRes::GetComplexMap)
		.add_property("resDouble", &SweepRes::GetDoubleMap)
		;

	//---------------------------------------------------------------
	// PositionSettings
	//---------------------------------------------------------------

	class_<PositionSettings>("PositionSettings", init<Eigen::Vector2d, int, double>())
		.def(init<>())
		.add_property("polarization", &PositionSettings::GetPolarization)
		.add_property("interfaceId", &PositionSettings::GetInterfaceId)
		.add_property("distFromInterface", &PositionSettings::GetDistFromInterface)
		.add_property("enabled", &PositionSettings::IsEnabled)
		;

	//---------------------------------------------------------------
	// EMFields
	//---------------------------------------------------------------

	class_<EMFields>("EMFields")
		.add_property("E", &EMFields::GetE)
		.add_property("H", &EMFields::GetH)
		;

	//---------------------------------------------------------------
	// EMFieldsList
	//---------------------------------------------------------------

	class_<EMFieldsList>("EMFieldsList", init<int>())
		.add_property("E", &EMFieldsList::GetE)
		.add_property("H", &EMFieldsList::GetH)
		;

	//---------------------------------------------------------------
	// TMM
	//---------------------------------------------------------------

	void (Tmm::*BoostSetParamsDouble)(Param, double) = &Tmm::SetParam;
	void (Tmm::*BoostSetParamsComplex)(Param, dcomplex) = &Tmm::SetParam;
	SweepRes(Tmm::*BoostSweep1)(Param, Eigen::VectorXd) = &Tmm::Sweep;
	SweepRes(Tmm::*BoostSweep2)(Param, Eigen::VectorXd, PositionSettings) = &Tmm::Sweep;
	

	//def("F", (void (C::*)(int))&C::F)  
	class_<Tmm>("Tmm")
		.def("SetParam", (BoostSetParamsComplex))
		.def("SetParam", (BoostSetParamsDouble))
		.def("AddIsotropicLayer", &Tmm::AddIsotropicLayer)
		.def("AddLayer", &Tmm::AddLayer)
		.def("GetIntensityMatrix", &Tmm::GetIntensityMatrix)
		.def("GetAmplitudeMatrix", &Tmm::GetAmplitudeMatrix)
		.def("Sweep", (BoostSweep1))
		.def("Sweep", (BoostSweep2))
		.def("CalcFields1D", &Tmm::CalcFields1D)
		.def("CalcFieldsAtInterface", &Tmm::CalcFieldsAtInterface)
		;

}

#endif

