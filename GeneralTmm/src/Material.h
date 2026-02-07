#pragma once
#include "Common.h"

namespace TmmModel {

//---------------------------------------------------------------------
// Material
//---------------------------------------------------------------------

class Material {
public:
    Material() noexcept;
    explicit Material(dcomplex staticN) noexcept;
    Material(ArrayXd wlsExp, ArrayXcd nsExp);
    void SetStatic(dcomplex staticN) noexcept;
    [[nodiscard]] dcomplex n(double wl) const;
    [[nodiscard]] bool IsStatic() const noexcept;

private:
    bool isStatic_;
    dcomplex staticN_;
    ArrayXd wlsExp_;
    ArrayXcd nsExp_;
};


} // namespace TmmModel