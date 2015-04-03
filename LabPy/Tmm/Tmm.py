import numpy as np
import CppTmm

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
        self.AddIsotropicLayer = self._tmm.AddIsotropicLayer
        self.AddLayer = self._tmm.AddLayer
        self.GetIntensityMatrix = self._tmm.GetIntensityMatrix
        self.GetAmplitudeMatrix = self._tmm.GetAmplitudeMatrix

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


if __name__ == "__main__":
    import pylab as plt
    betas = np.linspace(0.0, 1.4, 100)
    tmm = Tmm()
    tmm.SetParam(wl = 800e-9)
    tmm.AddIsotropicLayer(float("inf"), 1.5)
    tmm.AddIsotropicLayer(float("inf"), 1.0)
    
    r = tmm.Sweep("beta", betas, ((1.0, 0.0), -1, 0.0))
    tmm.SetParam(n_1 = complex(1.5))
    
    
    plt.figure()
    plt.subplot(211)
    plt.plot(betas, r["R11"])
    plt.plot(betas, r["R22"])
    plt.subplot(212)
    plt.plot(betas, r["enh"])
    plt.show()
    
    
    
