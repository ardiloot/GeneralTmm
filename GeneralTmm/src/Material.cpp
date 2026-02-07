#include "Material.h"
#include <utility>

namespace TmmModel {

Material::Material() noexcept : isStatic_(true), staticN_(1.0) {}

Material::Material(dcomplex staticN) noexcept : isStatic_(true), staticN_(staticN) {}

Material::Material(ArrayXd wlsExp, ArrayXcd nsExp)
    : isStatic_(false), staticN_(1.0), wlsExp_(std::move(wlsExp)), nsExp_(std::move(nsExp)) {
    if (wlsExp_.size() != nsExp_.size()) {
        throw std::invalid_argument("wls and ns must have the same length");
    }

    if (wlsExp_.size() < 2) {
        throw std::invalid_argument("The length of wls and ns must be at least 2");
    }
}

void Material::SetStatic(dcomplex staticN) noexcept {
    isStatic_ = true;
    staticN_ = staticN;
    wlsExp_.resize(0);
    nsExp_.resize(0);
}

dcomplex Material::n(double wl) const {
    if (isStatic_) {
        return staticN_;
    }

    // Interpolate
    dcomplex res = Interpolate(wl, wlsExp_, nsExp_);
    return res;
}

bool Material::IsStatic() const noexcept {
    return isStatic_;
}

} // namespace TmmModel
