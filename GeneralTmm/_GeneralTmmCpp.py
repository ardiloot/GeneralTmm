from __future__ import annotations

import numpy as np
import numpy.typing as npt

from GeneralTmm import _GeneralTmmCppExt

__all__ = ["Material", "Tmm"]

# ===============================================================================
# Tmm
# ===============================================================================

Tmm = _GeneralTmmCppExt.Tmm

# ===============================================================================
# Material
# ===============================================================================


class Material(_GeneralTmmCppExt.Material):
    __doc__ = _GeneralTmmCppExt.Material.__doc__

    @staticmethod
    def Static(n: float | complex) -> Material:
        """Helper method to make material with constant refractive index.

        Parameters
        ----------
        n : float or complex
            Constant value for refractive index

        Returns
        -------
        Material
            Material with constant refractive index

        Examples
        --------
        >>> mat = Material.Static(1.5)
        >>> mat.GetN(532e-9)
        1.5

        """
        wls: npt.NDArray[np.float64] = np.array([-1.0, 1.0])
        ns: npt.NDArray[np.complex128] = np.array([n, n], dtype=complex)
        res = Material(wls, ns)
        return res


if __name__ == "__main__":
    pass
