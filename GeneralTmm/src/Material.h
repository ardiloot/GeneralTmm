#pragma once
#include "Common.h"

namespace TmmModel {

//---------------------------------------------------------------------
// Material
//---------------------------------------------------------------------

class Material {
public:
    Material() noexcept = default;
    explicit Material(dcomplex staticN) noexcept;
    Material(ArrayXd wlsExp, ArrayXcd nsExp);
    void SetStatic(dcomplex staticN) noexcept;
    [[nodiscard]] dcomplex n(double wl) const;
    [[nodiscard]] bool IsStatic() const noexcept;

private:
    bool isStatic_ = true;
    dcomplex staticN_ = 1.0;
    ArrayXd wlsExp_;
    ArrayXcd nsExp_;
};


} // namespace TmmModel