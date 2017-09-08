#pragma once
#include "Common.h"

namespace TmmModel
{

	//---------------------------------------------------------------------
	// Material
	//---------------------------------------------------------------------

	class Material {
	public:
		Material();
		Material(dcomplex staticN_);
		Material(ArrayXd wlsExp_, ArrayXcd nsExp_);
		void SetStatic(dcomplex staticN_);
		dcomplex n(double wl);
		bool IsStatic();

	private:
		bool isStatic;
		dcomplex staticN;
		ArrayXd wlsExp;
		ArrayXcd nsExp;
	};



}