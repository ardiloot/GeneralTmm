import numpy as np
import CppTmm
import LabPy

def ToCppParam(param):
    layerNr = -1
    if param.count("_") == 1:
        layerNr = int(param.split("_")[-1])
    elif param.count('a') > 1:
        raise ValueError("Unknown param %s" % (param))

    if param == "wl":
        return CppTmm.Param(CppTmm.ParamType.WL)
    elif param == "beta":
        return CppTmm.Param(CppTmm.ParamType.BETA)
    elif param == "enhOptRel":
        return CppTmm.Param(CppTmm.ParamType.ENH_OPT_REL)
    elif param == "enhOptMaxIters":
        return CppTmm.Param(CppTmm.ParamType.ENH_OPT_MAX_ITERS)
    elif param == "enhInitialStep":
        return CppTmm.Param(CppTmm.ParamType.ENH_INITIAL_STEP)
    elif param.startswith("d_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_D, layerNr)
    elif param.startswith("n_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_N, layerNr)
    elif param.startswith("nx_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_NX, layerNr)
    elif param.startswith("ny_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_NY, layerNr)
    elif param.startswith("nz_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_NZ, layerNr)
    elif param.startswith("psi_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_PSI, layerNr)
    elif param.startswith("xi_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_XI, layerNr)
    else:
        raise NotImplementedError()

class Tmm(object):

    def __init__(self):
        self._tmm = CppTmm.Tmm()
        self.GetIntensityMatrix = self._tmm.GetIntensityMatrix
        self.GetAmplitudeMatrix = self._tmm.GetAmplitudeMatrix

    def AddIsotropicLayer(self, d, n):
        if type(n) == LabPy.Material:
            self._tmm.AddIsotropicLayerMat(d, n)
        else:
            self._tmm.AddIsotropicLayer(d, n)

    def AddLayer(self, d, nx, ny, nz, psi, xi):
        if type(nx) == LabPy.Material and type(ny) == LabPy.Material and \
        type(nz) == LabPy.Material:
            self._tmm.AddLayerMat(d, nx, ny, nz, psi, xi)
        else:
            self._tmm.AddLayer(d, nx, ny, nz, psi, xi)
            
    def SetParam(self, **kwargs):
        for key, value in kwargs.iteritems():
            p = ToCppParam(key)
            self._tmm.SetParam(p, value)
            
    def SetLayerParam(self, layerId, **kwargs):
        for key, value in kwargs.iteritems():
            self._tmm.SetParam("%s_%d" % (ToCppParam(key), layerId), value)
    
    def Sweep(self, sweepParam, sweepValues, enhPos = None):
        if enhPos == None:
            r = self._tmm.Sweep(ToCppParam(sweepParam), sweepValues)
        else:
            pos = CppTmm.PositionSettings(np.array(enhPos[0]), enhPos[1], enhPos[2])  # @UndefinedVariable
            r = self._tmm.Sweep(ToCppParam(sweepParam), sweepValues, pos)
        
        res = {}
        for k, v in r.resDouble.iteritems():
            res[k] = v[0]
        for k, v in r.resComplex.iteritems():
            res[k] = v[0]
        return res

    def CalcFields1D(self, xs, pol):
        res = self._tmm.CalcFields1D(xs, np.array(pol))
        return res.E, res.H

    def CalcFieldsAtInterface(self, (pol, interface, dist)):
        pos = CppTmm.PositionSettings(np.array(pol), interface, dist)  # @UndefinedVariable
        res = self._tmm.CalcFieldsAtInterface(pos)
        return res.E[0], res.H[0]
    
    def OptimizeEnhancement(self, optParams, optInitials, (pol, interface, dist)):
        pos = CppTmm.PositionSettings(np.array(pol), interface, dist)  # @UndefinedVariable
        params = [ToCppParam(p) for p in optParams]
        res = tmm._tmm.OptimizeEnhancement(params, np.array(optInitials), pos)
        return res
    

if __name__ == "__main__":
    import pylab as plt
    from LabPy import Material
    
    wl = 800e-9
    metalN = Material(r"main/Ag/Johnson")
    
    betas = np.linspace(0.0, 1.4, 1000)
    tmm = Tmm()
    tmm.SetParam(wl = 800e-9)
    tmm.AddIsotropicLayer(float("inf"), 1.5)
    #tmm.AddIsotropicLayer(50e-9, 0.036759 + 5.5698j)
    tmm._tmm.AddIsotropicLayerMat(50e-9, metalN)
    tmm.AddIsotropicLayer(float("inf"), 1.0)
    
    
    tmm.SetParam(enhOptMaxIters = 100)
    tmm.SetParam(enhOptRel = 1e-5)
    tmm.SetParam(enhInitialStep = 1e-9)
    tmm.SetParam(beta = 1.015)
    
    print "opt start"
    optres = tmm.OptimizeEnhancement(["wl"], [800e-9], ((1.0, 0.0), -1, 0.0))
    print "optres", optres
    r = tmm.Sweep("beta", betas, ((1.0, 0.0), -1, 0.0))

    plt.figure()
    plt.subplot(211)
    plt.plot(betas, r["R11"])
    plt.plot(betas, r["R22"])
    plt.subplot(212)
    plt.plot(betas, r["enh"])
    plt.show()
    
    
    
