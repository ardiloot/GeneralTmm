#include "Material.h"

TmmModel::Material::Material() {
	SetStatic(1.0);
}

TmmModel::Material::Material(dcomplex staticN_) {
	SetStatic(staticN_);
}

TmmModel::Material::Material(ArrayXd wlsExp_, ArrayXcd nsExp_) : wlsExp(wlsExp_), nsExp(nsExp_) {
	// Copy of wlsExp and nsExp is made intentionally
	isStatic = false;
	staticN = 1.0;

	if (wlsExp.size() != nsExp.size()) {
		throw std::invalid_argument("wls and ns must have the same length");
	}

	if (wlsExp.size() < 2) {
		throw std::invalid_argument("The length of wls and ns must be at least 2");
	}
}

void TmmModel::Material::SetStatic(dcomplex staticN_) {
	isStatic = true;
	staticN = staticN_;
}

TmmModel::dcomplex TmmModel::Material::n(double wl) {
	if (isStatic) {
		return staticN;
	}

	// Interpolate
	dcomplex res = Interpolate(wl, wlsExp, nsExp);
	return res;
}

bool TmmModel::Material::IsStatic() {
	return isStatic;
}
