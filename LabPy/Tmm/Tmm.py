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
        return CppTmm.Param(CppTmm.ParamType.WL), "double"
    elif param == "beta":
        return CppTmm.Param(CppTmm.ParamType.BETA), "double"
    elif param == "enhOptRel":
        return CppTmm.Param(CppTmm.ParamType.ENH_OPT_REL), "double"
    elif param == "enhOptMaxIters":
        return CppTmm.Param(CppTmm.ParamType.ENH_OPT_MAX_ITERS), "int"
    elif param == "enhInitialStep":
        return CppTmm.Param(CppTmm.ParamType.ENH_INITIAL_STEP), "double"
    elif param.startswith("d_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_D, layerNr), "double"
    elif param.startswith("n_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_N, layerNr), "complex"
    elif param.startswith("nx_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_NX, layerNr), "complex"
    elif param.startswith("ny_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_NY, layerNr), "complex"
    elif param.startswith("nz_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_NZ, layerNr), "complex"
    elif param.startswith("psi_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_PSI, layerNr), "double"
    elif param.startswith("xi_"):
        return CppTmm.Param(CppTmm.ParamType.LAYER_XI, layerNr), "double"
    else:
        raise NotImplementedError(str(param))

def ToCppWD(waveDirectionStr):
    
    if waveDirectionStr == "both":
        return CppTmm.WaveDirection.WD_BOTH
    elif waveDirectionStr == "forward":
        return CppTmm.WaveDirection.WD_FORWARD
    elif waveDirectionStr == "backward":
        return CppTmm.WaveDirection.WD_BACKWARD
    else:
        raise ValueError("Unknown wave direction: %s" % (waveDirectionStr))

class Tmm(object):

    def __init__(self):
        self._tmm = CppTmm.Tmm()
        self.GetIntensityMatrix = self._tmm.GetIntensityMatrix
        self.GetAmplitudeMatrix = self._tmm.GetAmplitudeMatrix
        self.ClearLayers = self._tmm.ClearLayers
        self._materialsCache = []

    def AddIsotropicLayer(self, d, n):
        if type(n) == LabPy.Material:
            self._tmm.AddIsotropicLayerMat(d, n)
        else:
            self._tmm.AddIsotropicLayer(d, n)
        self._materialsCache.append((d, n))

    def AddLayer(self, d, nx, ny, nz, psi, xi):
        if type(nx) == LabPy.Material and type(ny) == LabPy.Material and \
        type(nz) == LabPy.Material:
            self._tmm.AddLayerMat(d, nx, ny, nz, psi, xi)
        else:
            self._tmm.AddLayer(d, nx, ny, nz, psi, xi)
        self._materialsCache.append((d, nx, ny, nz, psi, xi))
          
    def ClearLayers(self):
        self._tmm.ClearLayers()
        self._materialsCache.clear()
            
    def SetParam(self, **kwargs):
        for key, value in kwargs.iteritems():
            p, _ = ToCppParam(key)
            self._tmm.SetParam(p, value)
            
    def SetLayerParam(self, layerId, **kwargs):
        for key, value in kwargs.iteritems():
            p, _ = ToCppParam("%s_%d" % (key, layerId))
            self._tmm.SetParam(p, value)
            
    def GetParam(self, paramName):
        p, t = ToCppParam(paramName)
        if t == "int":
            return self._tmm.GetParamInt(p)
        elif t == "double":
            return self._tmm.GetParamDouble(p)
        elif t == "complex":
            return self._tmm.GetParamComplex(p)
        else:
            raise NotImplementedError()
            
    def LoadConf(self, conf):
        self.SetParam(wl = conf["wl"], beta = conf["beta"])
        self.ClearLayers()
        for layer in conf["layers"]:
            if layer[0] == "iso":
                mat = LabPy.MaterialFromConf(layer[2])
                self.AddIsotropicLayer(layer[1], mat)
            elif layer[0] == "aniso":
                raise NotImplementedError()
            else:
                raise ValueError("Unknown layer type.")
    
    def Sweep(self, sweepParam, sweepValues, enhPos = None, alphasLayer = -1):
        #enhpos = pol, enhInterface, enhDist
        if enhPos == None:
            r = self._tmm.Sweep(ToCppParam(sweepParam)[0], sweepValues)
        else:
            pos = CppTmm.PositionSettings(np.array(enhPos[0]), enhPos[1], enhPos[2])  # @UndefinedVariable
            r = self._tmm.Sweep(ToCppParam(sweepParam)[0], sweepValues, pos, alphasLayer)
        
        res = {}
        for k, v in r.resDouble.iteritems():
            res[k] = v[0]
        for k, v in r.resComplex.iteritems():
            res[k] = v[0]
        return res

    def CalcFields1D(self, xs, pol, waveDirection = "both"):
        res = self._tmm.CalcFields1D(xs, np.array(pol), ToCppWD(waveDirection))
        return res.E, res.H
    
    def CalcFields2D(self, xs, ys, pol, waveDirection = "both"):
        ky = self.GetParam("beta") * 2.0 * np.pi / self.GetParam("wl")
        phaseY = np.exp(1.0j * ky * ys)
        E1D, H1D = self.CalcFields1D(xs, pol, waveDirection)
        
        E = np.zeros((len(xs), len(ys), 3), dtype = complex)
        H = np.zeros((len(xs), len(ys), 3), dtype = complex)
        for i in range(3):
            E[:, :, i] = np.outer(E1D[:, i], phaseY)
            H[:, :, i] = np.outer(H1D[:, i], phaseY)
        
        return E, H

    def CalcFieldsAtInterface(self, (pol, interface, dist), waveDirection = "both"):
        pos = CppTmm.PositionSettings(np.array(pol), interface, dist)  # @UndefinedVariable
        res = self._tmm.CalcFieldsAtInterface(pos, ToCppWD(waveDirection))
        return res.E[0], res.H[0]
    
    def OptimizeEnhancement(self, optParams, optInitials, (pol, interface, dist)):
        pos = CppTmm.PositionSettings(np.array(pol), interface, dist)  # @UndefinedVariable
        params = [ToCppParam(p)[0] for p in optParams]
        res = self._tmm.OptimizeEnhancement(params, np.array(optInitials), pos)
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
    
    
    
