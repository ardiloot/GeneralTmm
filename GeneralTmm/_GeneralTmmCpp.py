import numpy as np
from GeneralTmm import _GeneralTmmCppExt  # @UnresolvedImport

__all__ = ["Material", "Tmm"]

#===============================================================================
# Tmm
#===============================================================================

Tmm = _GeneralTmmCppExt.Tmm

#===============================================================================
# Material
#===============================================================================

class Material(_GeneralTmmCppExt.Material):# @UndefinedVariable
    __doc__ = _GeneralTmmCppExt.Material.__doc__
    
    @staticmethod
    def Static(n):
        """Helper method to make material with constant refractive index.
        
        Parameters
        ----------
        n : float or complex
            Constant value for refractive index
            
        Returns
        -------
        None
        
        Examples
        --------
        >>> mat = Material.Static(1.5)
        >>> mat.GetN(532e-9)
        1.5
        
        """
        wls = np.array([-1.0, 1.0])
        ns = np.array([n, n], dtype = complex)
        res = Material(wls, ns)
        return res
    
    @staticmethod
    def FromLabPy(materialLabPy):
        """Helper method to convert LabPy Material to this Material class.
        As LabPy is currently not a public library, this function has no use
        for most of the users.
        
        Parameters
        ----------
        materialLabPy : :any:`LabPy.Material`
            The instance of LabPy Material
            
        Returns
        -------
        None
        
        """
        if materialLabPy.materialFile == "Static":
            wls = np.array([-1.0, 1.0])
            n = materialLabPy.n + 1.0j * (materialLabPy.k + materialLabPy.kAdditional)
            ns = np.array([n, n], dtype = complex)
        elif materialLabPy.isFormula:
            wls = np.ascontiguousarray(np.linspace(materialLabPy.wlRange[0], materialLabPy.wlRange[1], 500))
            ns = np.ascontiguousarray(materialLabPy(wls))
        else:
            wls = np.ascontiguousarray(materialLabPy.wlExp)
            if materialLabPy.kExp is None:
                ns = np.ascontiguousarray(materialLabPy.nExp, dtype = complex)
            else:
                ns = np.ascontiguousarray(materialLabPy.nExp + 1.0j * materialLabPy.kExp)
            ns += 1.0j * materialLabPy.kAdditional
        res = Material(wls, ns)
        res._materialLabPy = materialLabPy
        return res


if __name__ == "__main__":
    pass