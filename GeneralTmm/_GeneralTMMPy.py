"""This module contains classes for 4x4 TMM
Hodgkinson, I. J., Kassam, S., & Wu, Q. H. (1997). 
Journal of Computational Physics, 133(1) 75-83

This Python code is only used for testing the C++ code of the same algorithm.

"""

import numpy as np
import math
import warnings
from scipy import optimize

__all__ = ["TmmPy"]

def RotationSx(phi):
    res = np.array([[1.0, 0.0, 0.0], \
                    [0.0, math.cos(phi), -math.sin(phi)], \
                    [0.0, math.sin(phi), math.cos(phi)]])
    return res

def RotationSz(phi):
    res = np.array([[math.cos(phi), -math.sin(phi), 0.0], \
                    [math.sin(phi), math.cos(phi), 0.0], \
                    [0.0, 0.0, 1.0]])
    return res

def Norm(vector):
    if len(vector.shape) > 1:
        if vector.shape[1] != 3:
            raise Exception("Only vectors with length 3 supported.")
        return np.sqrt(abs(vector[:, 0]) ** 2.0 + abs(vector[:, 1]) ** 2.0 +  abs(vector[:, 2]) ** 2.0, dtype = complex).real
    else:        
        if len(vector) != 3:
            raise Exception("Only vectors with length 3 supported.")
        return np.sqrt(abs(vector[0]) ** 2.0 + abs(vector[1]) ** 2.0 + abs(vector[2]) ** 2.0, dtype = complex).real


class _AnisotropicLayer():
    
    def __init__(self, tmm, d, n1, n2, n3, psi, xi):
        self.tmm = tmm
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.psi = psi
        self.xi = xi
        
    def GetConf(self):
        if self.n1 == self.n2 and self.n1 == self.n3:
            return ("iso", self.d, self.n1)
        else:
            return ("aniso", self.d, self.n1, self.n2, self.n3, self.psi, self.xi)
        
    def SetConf(self, **kwargs):
        self.d = kwargs.pop("d", self.d)
        self.n1 = kwargs.pop("n1", self.n1)
        self.n2 = kwargs.pop("n2", self.n2)
        self.n3 = kwargs.pop("n3", self.n3)
        self.psi = kwargs.pop("psi", self.psi)
        self.xi = kwargs.pop("xi", self.xi)
        
    def _GetTangentialFields(self, Ey, Hz, Ez, Hy):
        z0 = 119.9169832 * math.pi
    
        beta = self.tmm.beta
        epsXX = self.epsTensor[0, 0]
        epsXY = self.epsTensor[0, 1]
        epsXZ = self.epsTensor[0, 2]
        
        Ex = -(epsXY * Ey + epsXZ * Ez + beta * z0 * Hz) / epsXX
        Hx = (beta / z0) * Ez
        
        return Ex, Hx
        
    def GetFields(self, x, coefs):
        E = np.array([0.0, 0.0, 0.0], dtype = complex)
        H = np.array([0.0, 0.0, 0.0], dtype = complex)
        for mode in range(4):
            a = self.alpha[mode]
         
            mEy, mHz, mEz, mHy = coefs[mode] * self.F[:, mode]
            mEx, mHx = self._GetTangentialFields(mEy, mHz, mEz, mHy)
            
            phase = np.exp(1.0j * self.tmm.k0 * a * x)
            
            E[0] += mEx * phase
            E[1] += mEy * phase
            E[2] += mEz * phase
            H[0] += mHx * phase
            H[1] += mHy * phase
            H[2] += mHz * phase
        
        return E, H
    
        
    def Solve(self, beta):
        self._CalcEpsilonMatrix()
        self._SolveEigenFunction(beta)
        self._SolvePhaseMatrix()
        self._SolveFieldTransferMatrix()
            
    
    def _CalcEpsilonMatrix(self):
                
        self.epsTensorCrystal = np.array([[self.n1 ** 2.0, 0.0, 0.0], \
                                          [0.0, self.n2 ** 2.0, 0.0], \
                                          [0.0, 0.0, self.n3 ** 2.0]], dtype = complex)
        
        
        
        self.epsTensor = np.dot(RotationSx(self.xi), \
                                np.dot(RotationSz(self.psi), \
                                np.dot(self.epsTensorCrystal, \
                                np.dot(RotationSz(-self.psi), RotationSx(-self.xi))))) 

        

    def _SolveEigenFunction(self, beta):
        z0 = 119.9169832 * math.pi
    
        epsXX = self.epsTensor[0, 0]
        epsYY = self.epsTensor[1, 1]
        epsZZ = self.epsTensor[2, 2]
        epsXY = self.epsTensor[0, 1]
        epsXZ = self.epsTensor[0, 2]
        epsYZ = self.epsTensor[1, 2]
        
        #print
        #print "Solve Eigen Function", beta
        
        #print "epsTensor"
        #print self.epsTensor
        
        mBeta = np.zeros((4, 4), dtype = complex)
        mBeta[0, 0] = -beta * epsXY / epsXX
        mBeta[0, 1] = z0 - (z0 * beta ** 2.0) / epsXX
        mBeta[0, 2] = -beta * epsXZ / epsXX
        mBeta[0, 3] = 0.0
        mBeta[1, 0] = epsYY / z0 - (epsXY ** 2.0) / (z0 * epsXX)
        mBeta[1, 1] = (-beta * epsXY) / epsXX
        mBeta[1, 2] = epsYZ / z0 - (epsXY * epsXZ) / (z0 * epsXX)
        mBeta[1, 3] = 0.0
        mBeta[2, 0] = 0.0
        mBeta[2, 1] = 0.0
        mBeta[2, 2] = 0.0
        mBeta[2, 3] = -z0
        mBeta[3, 0] = (-epsYZ / z0) + (epsXY * epsXZ) / (z0 * epsXX)
        mBeta[3, 1] = beta * epsXZ / epsXX
        mBeta[3, 2] = (beta ** 2.0) / z0 + (epsXZ ** 2.0) / (z0 * epsXX) - epsZZ / z0
        mBeta[3, 3] = 0.0
        values, vectors = np.linalg.eig(mBeta)

        
        poyntingX = np.zeros((4), dtype = float)
        for i in range(4):
            poyntingX[i] = 0.5 * (vectors[0, i] * np.conj(vectors[1, i]) - \
                                  vectors[2, i] * np.conj(vectors[3, i])).real
        
        
        forward, backward = [], []
        for i in range(4):
            movingForward = False
            if abs(poyntingX[i]) > 1e-10:
                movingForward = poyntingX[i] > 0.0
            else:
                movingForward = values[i].imag > 0.0
            
            if movingForward:
                forward.append(i)
            else:
                backward.append(i)
            
        if len(forward) != 2:
            print("ns", self.n1, self.n2, self.n3)
            print("beta", beta)
            print("Values", values)
            print("Poynting", poyntingX)
            print("vectors", vectors)
            raise Exception("Wrong number of forward moving waves: %d" % len(forward))
        
        if abs(values.real[forward[0]] - values.real[forward[1]]) < 1e-10:
            #print "TODDO", values[forward[0]], values[forward[1]]
            pass
        elif values.real[forward[0]] < values.real[forward[1]]:
            forward[0], forward[1] = forward[1], forward[0]
        
        if abs(values.real[backward[0]] - values.real[backward[1]]) < 1e-10:
            #print "TODDO BACK", values[backward[0]], values[backward[1]]
            pass
        elif values.real[backward[0]] > values.real[backward[1]]:
            backward[0], backward[1] = backward[1], backward[0]
        

        option1 = [forward[0], backward[0], forward[1], backward[1]]
        #option2 = [forward[0], backward[1], forward[1], backward[0]]
        order = option1

        self.alpha = values[order]
        self.F = vectors[ :, order]
        self.poynting = poyntingX[order]
        self.invF = np.linalg.inv(self.F)
   

        
    def _SolvePhaseMatrix(self):
        self.phaseMatrix = np.identity(4, dtype = complex)
        
        if self.d == float("inf"):
            return
        
        for i in range(4):
            phi = self.tmm.k0 * self.alpha[i] * self.d
            self.phaseMatrix[i, i] = np.exp(-1.0j * phi)
    
    def _SolveFieldTransferMatrix(self):
        self.M = np.dot(self.F, np.dot(self.phaseMatrix, self.invF))
        
    
class TmmPy():
    
    def __init__(self):
        self.wl = None
        self.beta = None
        self.polarization = None
        self.layers = []
        self.namesr = [["r11", "r12", "t13", "t14"], \
                       ["r21", "r22", "t23", "t24"], \
                       ["t31", "t32", "r33", "r34"], \
                       ["t41", "t42", "r43", "r44"]]
        
        self.namesR = [["R11", "R12", "T13", "T14"], \
                       ["R21", "R22", "T23", "T24"], \
                       ["T31", "T32", "R33", "R34"], \
                       ["T41", "T42", "R43", "R44"]]
        
    def AddIsotropicLayer(self, d, n):
        self.layers.append(_AnisotropicLayer(self, d, n, n, n, 0.0, 0.0))
        
    def AddLayer(self, d, n1, n2, n3, psi, xi):
        self.layers.append(_AnisotropicLayer(self, d, n1, n2, n3, psi, xi))
    
    def GetConf(self):
        layersConf = [layer.GetConf() for layer in self.layers]
        res = {"wl": self.wl, "beta": self.beta, \
               "polarization": self.polarization, "layers": layersConf}    
        return res
    
    def SetConf(self, **kwargs):
        self.wl = kwargs.pop("wl", self.wl)
        self.beta = kwargs.pop("beta", self.beta)
        self.polarization = kwargs.pop("polarization", self.polarization)
        
        # All layer params
        layersConf = kwargs.pop("layers", None)
        if layersConf != None:
            self.layers = []
            for lConf in layersConf:
                if lConf[0] == "iso":
                    _, d, n = lConf
                    self.AddIsotropicLayer(d, n)
                elif lConf[0] == "aniso":
                    _, d, n1, n2, n3, psi, xi = lConf
                    self.AddLayer(d, n1, n2, n3, psi, xi)
                else:
                    raise NotImplemented()
        
        # Layer individual params
        for key, value in list(kwargs.items()):
            if key.find("_") == -1:
                continue
            kwargs.pop(key)
            param, index = key.split("_")
            index = int(index)
            self.layers[index].SetConf(**{param: value})

    
    def SolveFor(self, param, values, polarization = None, enhInterface = None, enhDist = 0.0):
        if polarization != None:
            self.polarization = polarization
            
        res = {}
        for i in range(4):
            for j in range(4):
                res[self.namesr[i][j]] = np.zeros_like(values, dtype = complex)
                res[self.namesR[i][j]] = np.zeros_like(values, dtype = float)
                
        if enhInterface != None:
            res["enh"] = np.zeros_like(values, dtype = float)
                
        for i in range(len(self.layers)):
            res["alphas_%d" % (i)] = np.zeros((len(values), 4), dtype = complex)
                
                
        for i in range(len(values)):
            self.SetConf(**{param: values[i]})
            r, R = self.Solve()
            
            if enhInterface != None: 
                res["enh"][i], _ = self.CalcEnhAtInterface(None, enhInterface, enhDist = enhDist)
    
                
            for j in range(len(self.layers)):
                res["alphas_%d" % (j)][i] = self.layers[j].alpha[:]
    
            #if R[0, 0] > 1.0 + 1e-6:
                #print R

            for j in range(4):
                for k in range(4):
                    res[self.namesr[j][k]][i] = r[j, k]
                    res[self.namesR[j][k]][i] = R[j, k]

        return res

    def Solve(self, wl = None, beta = None):
        if wl == None: wl = self.wl
        if beta == None: beta = self.beta
        self.wl = wl
        self.beta = beta
        self.k0 = 2.0 * math.pi / wl
        
        for layer in self.layers:
            layer.Solve(beta)
            
        # System matrix
        self.A = self.layers[0].invF
        for i in range(1, len(self.layers) - 1):
            self.A = np.dot(self.A, self.layers[i].M)
            
        
            
        self.A = np.dot(self.A, self.layers[-1].F)

        
        # r-matrix
        A = self.A
        r1 = np.array([[0.0, 0.0, -A[0, 0], -A[0, 2]], \
                       [1.0, 0.0, -A[1, 0], -A[1, 2]], \
                       [0.0, 0.0, -A[2, 0], -A[2, 2]], \
                       [0.0, 1.0, -A[3, 0], -A[3, 2]]], dtype = complex)
        r2 = np.array([[-1.0, 0.0, A[0, 1], A[0, 3]], \
                       [0.0, 0.0, A[1, 1], A[1, 3]], \
                       [0.0, -1.0, A[2, 1], A[2, 3]], \
                       [0.0, 0.0, A[3, 1], A[3, 3]]], dtype = complex)

        r = np.dot(np.linalg.inv(r1), r2)
        self.r = r

        pBackward = np.array([self.layers[0].poynting[1], self.layers[0].poynting[3], \
                              self.layers[-1].poynting[1], self.layers[-1].poynting[3]])
 
        pForward = np.array([self.layers[0].poynting[0], self.layers[0].poynting[2], \
                              self.layers[-1].poynting[0], self.layers[-1].poynting[2]])
 
 
        R = np.zeros_like(r, dtype = complex)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            R[0, 0] = abs(r[0, 0]) ** 2.0 * abs(pBackward[0]) / abs(pForward[0])
            R[0, 1] = abs(r[0, 1]) ** 2.0 * abs(pBackward[0]) / abs(pForward[1])
            R[0, 2] = np.NAN #abs(r[0, 2]) ** 2.0 * abs(pBackward[0] / pBackward[2])
            R[0, 3] = np.NAN #abs(r[0, 3]) ** 2.0 * abs(pBackward[0] / pBackward[3])
            R[1, 0] = abs(r[1, 0]) ** 2.0 * abs(pBackward[1]) / abs(pForward[0])
            R[1, 1] = abs(r[1, 1]) ** 2.0 * abs(pBackward[1]) / abs(pForward[1])
            R[1, 2] = np.NAN #abs(r[1, 2]) ** 2.0 * abs(pBackward[1] / pBackward[2])
            R[1, 3] = np.NAN #abs(r[1, 3]) ** 2.0 * abs(pBackward[1] / pBackward[3])
            R[2, 0] = abs(r[2, 0]) ** 2.0 * abs(pForward[2]) / abs(pForward[0])
            R[2, 1] = abs(r[2, 1]) ** 2.0 * abs(pForward[2]) / abs(pForward[1])
            R[2, 2] = np.NAN #abs(r[2, 2]) ** 2.0 * abs(pForward[2] / pBackward[2])
            R[2, 3] = np.NAN #abs(r[2, 3]) ** 2.0 * abs(pForward[2] / pBackward[3])
            R[3, 0] = abs(r[3, 0]) ** 2.0 * abs(pForward[3]) / abs(pForward[0])
            R[3, 1] = abs(r[3, 1]) ** 2.0 * abs(pForward[3]) / abs(pForward[1])
            R[3, 2] = np.NAN #abs(r[3, 2]) ** 2.0 * abs(pForward[3] / pBackward[2])
            R[3, 3] = np.NAN #abs(r[3, 3]) ** 2.0 * abs(pForward[3] / pBackward[3])
            if len(w) > 0:
                print("warning", w)
                print(self.beta)
                print(pBackward)
                print(pForward)
     
        R = R.real
        self.R = R
        
        return r, R
    
    def CalcFields1D(self, xs, polarization = None):
        if polarization != None:
            self.polarization = polarization
        normCoef, coefsAll = self._CalcFieldCoefs()
        layerIndices, layerDs = self._LayerIndices(xs)
    
        resE = np.zeros((len(xs), 3), dtype = complex)
        resH = np.zeros((len(xs), 3), dtype = complex)
        
        for i in range(len(xs)):
            layerId = layerIndices[i]
            coefs = coefsAll[layerId, :]
            
            E, H = self.layers[layerId].GetFields(layerDs[i], coefs)
            resE[i, :] = E
            resH[i, :] = H
    
        resE /= normCoef
        resH /= normCoef
       
        return resE, resH
    
    def CalcEnhAtInterface(self, polarization = None, enhInterface = -1, enhDist = 0.0):
        if polarization != None:
            self.polarization = polarization
            
        if enhInterface < 0:
            layerId = len(self.layers) - 1
        else:
            layerId = enhInterface

        normCoef, coefsAll = self._CalcFieldCoefs()
        E, _ = self.layers[layerId].GetFields(enhDist, coefsAll[layerId, :])
        E /= normCoef
        
        enh = Norm(E)
        return enh, E
    
    def OptimizeForEnhancement(self, optParams, optInitial, polarization = None, enhInterface = -1, enhDist = 0.0):
        def FitFunc(x):
            
            for i in range(len(x)):
                self.SetConf(**{optParams[i]: x[i]})
            self.Solve()
            enh, _ = self.CalcEnhAtInterface(enhInterface = enhInterface, enhDist = enhDist)
            #print x, enh
            return -enh
        
        if polarization != None:
            self.polarization = polarization
        optValues, maxEnh, _, __, ___ = optimize.fmin(FitFunc, optInitial, disp = False, full_output = True)
        maxEnh *= -1
        #print "optValues", optValues
        for i in range(len(optParams)):
            self.SetConf(**{optParams[i]: optValues[i]})
        
        return optValues, maxEnh


    def _CalcFieldCoefs(self, polarization = None):
        if polarization != None:
            self.polarization = polarization
        a1In, a2In = self.polarization
        
        inputFields = np.array([a1In, a2In, 0.0, 0.0], dtype = complex)
        outputFields = np.dot(self.r, inputFields)
        
        Einc, _ = self.layers[0].GetFields(0.0, np.array([a1In, 0.0, a2In, 0.0]))
        
        normCoef = np.sqrt(abs(Einc[0]) ** 2.0 + abs(Einc[1]) ** 2.0 + abs(Einc[2]) ** 2.0, dtype = complex)
        n1 = np.sqrt(self.beta ** 2.0 + self.layers[0].alpha[0] ** 2.0)
        n2 = np.sqrt(self.beta ** 2.0 + self.layers[0].alpha[2] ** 2.0)
        nEff = (a1In * n1.real + a2In * n2.real) / (a1In + a2In) # Maybe not fully correct
        #print "n1", n1, "n2", n2, "nEff", nEff
        normCoef *= math.sqrt(nEff)
        
        coefsSubstrate = np.array([outputFields[2], 0.0, outputFields[3], 0.0])
        mat = self.layers[-1].F

        coefsAll = np.zeros((len(self.layers), 4), dtype = complex)
        
        

        for i in range(len(self.layers)-1, -1, -1):
            mat = np.dot(self.layers[i].M, mat)
            coefsAll[i, :] = np.dot(self.layers[i].invF, np.dot(mat, coefsSubstrate))

        coefsAll[-1, 1] = coefsAll[-1, 3] = 0.0  
        return normCoef, coefsAll
        
    
    def _LayerIndices(self, xs):
        resInd = np.zeros_like(xs, dtype = int)
        resD = np.zeros_like(xs, dtype = float)
        
        curLayer = 0
        curDist = 0.0
        prevDist = 0.0
        
        for i in range(len(xs)):
            while(xs[i] >= curDist):
                curLayer += 1
                prevDist = curDist
                if curLayer >= len(self.layers) - 1:
                    curDist = float("inf")
                    curLayer = len(self.layers) - 1
                curDist += self.layers[curLayer].d
                
            resInd[i] = curLayer
            resD[i] = xs[i] - prevDist
                
        return resInd, resD
        
     
if __name__ == "__main__":
    pass