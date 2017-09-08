from GeneralTmm import _GeneralTmmCppExt  # @UnresolvedImport
import numpy as np
import pylab as plt

__all__ = ["Material", "Tmm"]

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

Tmm = _GeneralTmmCppExt.Tmm

if __name__ == "__main__":
    mat0 = Material.Static(1.7)
    mat1 = Material.Static(1.0)
    
    tmm = Tmm()
    tmm.wl = 500e-9
    tmm.beta = 0.0
    tmm.AddIsotropicLayer(float("inf"), mat0)
    tmm.AddIsotropicLayer(float("inf"), mat1)
    
    print(tmm.GetIntensityMatrix())
    print(tmm.GetAmplitudeMatrix())
    
    
    
    
    betas = np.linspace(0, 0.99, 5)
    sr = tmm.Sweep("beta", betas)
    print(sr)
    
    tmm.beta = 0.0
    pol = np.array([1.0, 0.0])
    xs = np.linspace(-1e-6, 1e-6, 100)
    E, H = tmm.CalcFields1D(xs, pol)
    
    #plt.plot(1e6 * xs, E[:, 0])
    #plt.plot(1e6 * xs, E[:, 1])
    #plt.plot(1e6 * xs, E[:, 2])
    #plt.show()