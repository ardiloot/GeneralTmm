#include "tmm.h"

namespace TmmModel{

	//---------------------------------------------------------------------
	// Functions
	//---------------------------------------------------------------------

	Eigen::Matrix3cd RotationSx(double phi){
		Eigen::Matrix3cd res;
		res << 1.0, 0.0, 0.0,
			0.0, cos(phi), -sin(phi),
			0.0, sin(phi), cos(phi);
		return res;
	}

	Eigen::Matrix3cd RotationSz(double phi){
		Eigen::Matrix3cd res;
		res << cos(phi), -sin(phi), 0.0,
			sin(phi), cos(phi), 0.0,
			0.0, 0.0, 1.0;
		return res;
	}

	//---------------------------------------------------------------------
	// Param
	//---------------------------------------------------------------------


	Param::Param(ParamType pType_){
		pType = pType_;
		layerId = -1;
	}

	Param::Param(ParamType pType_, int layerId_){
		pType = pType_;
		layerId = layerId_;
	}

	ParamType Param::GetParamType(){
		return pType;
	}

	int Param::GetLayerID(){
		return layerId;
	}


	//---------------------------------------------------------------------
	// Material
	//---------------------------------------------------------------------


	Material::Material(){
		isStatic = true;
		staticN = 1.0;
	}

	Material::Material(dcomplex staticN_){
		isStatic = true;
		staticN = staticN_;
	}

	/*
	Material::Material(boost::python::object &materialClass_){
		isStatic = false;
		staticN = 1.0;
		materialClass = materialClass_;
	}
	*/

	dcomplex Material::n(double wl){
		if (isStatic){
			return staticN;
		}
		else {
			throw std::runtime_error("Material not supported");
			//dcomplex n = boost::python::extract<dcomplex>(materialClass(wl));
			//return n;
		}
	}
	

	bool Material::IsStatic(){
		return isStatic;
	}

} // Namespace