import numpy as np
import pylab as plt
from LabPy import Material, Core, Plotting
from Tmm import Tmm
plt.rcParams.update(Plotting.Default())

class GaussianTmm(Core.ParamsBaseClass):

    def __init__(self, tmm, **kwargs):
        self._params = ["w0", "nPointsInteg", "th0"]
        self.tmm = tmm
        self.w0 = 3e-6
        self.nPointsInteg = 30
        self.th0 = None
        self.integCriteria = 1e-3
    
        super(self.__class__, self).__init__(**kwargs)
    
    def GaussianCoef(self, ky):
        # Propagating along x-axis
        res = 1.0 / 2.0 / np.sqrt(np.pi) * self.w0 * \
            np.exp(-(ky) ** 2.0 * self.w0 ** 2.0 / 4.0)
        return res
    
    def Sweep(self, paramName, paramValues, enhPos = None, **kwargs):
        self.SetParams(**kwargs)
        
        for i in range(len(paramValues)):
            self.SetParams(**{paramName: paramValues[i]})
            self._PreCalc()
            r = self.tmm.Sweep("beta", self.betas, enhPos = enhPos)
            print r.keys()
            
    
    
    def CalcFields2d(self, xs, zs, pol, **kwargs):
        self.SetParams(**kwargs)
        self._PreCalc()
        
        # Integrate waves
        ESum = np.zeros((len(xs), len(ys), 3), dtype = complex)
        HSum = np.zeros((len(xs), len(ys), 3), dtype = complex)
        ETmpLast, HTmpLast = None, None
        for i in range(len(self.betas)):
            self.tmm.SetParam(beta = self.betas[i])
            ETmp, HTmp = tmm.CalcFields2D(xs, ys, pol)
            if i > 0:
                dthP = (self.phis[i] - self.phis[i - 1]) 
                c = np.cos(self.phis[i]) * dthP * self.k / 2.0
                c1, c2 = self.gaussianCoefs[i - 1], self.gaussianCoefs[i]
                ESum += c * (c1 * ETmpLast + c2 * ETmp)
                HSum += c * (c1 * HTmpLast + c2 * HTmp)
            ETmpLast, HTmpLast = ETmp, HTmp
        
        #plt.figure()
        #plt.plot(np.degrees(self.phis), self.gaussianCoefs.real, "x-")
        #plt.plot(np.degrees(self.phis), self.gaussianCoefs.imag, ".-")
        
        return ESum, HSum
    
    def _PreCalc(self):
        # Params of the first layer
        wl = self.tmm.GetParam("wl").real
        nPrism = self.tmm._materialsCache[0][1](wl).real
        self.k0 = 2.0 * np.pi / wl
        self.k = self.k0 * nPrism
        
        # Calc plane wave expansion coefs
        phiLim  = np.arcsin(2.0 / self.w0 * np.sqrt(-np.log(self.integCriteria)) / self.k) 
        self.phis = np.linspace(-phiLim, phiLim, self.nPointsInteg)
        kxPs, kyPs = np.cos(self.phis) * self.k, np.sin(self.phis) * self.k
        self.kys = kxPs * np.sin(self.th0) + kyPs * np.cos(self.th0)
        self.gaussianCoefs = self.GaussianCoef(kyPs)
        self.betas = (self.kys / self.k0).real
    


if __name__ == "__main__":
    wl = 500e-9
    pol = (1.0, 0.0)
    prismN = Material("Static", n = 1.5)
    substrateN = Material("Static", n = 1.0)
    metalN = Material("main/Au/Johnson")
    metalD = 60e-9
    xs, ys = np.linspace(-10e-6, 10e-6, 500), np.linspace(-10e-6, 10e-6, 501)
    
    tmm = Tmm()
    tmm.SetParam(wl = wl)
    tmm.AddIsotropicLayer(float("inf"), prismN)
    tmm.AddIsotropicLayer(metalD, metalN)
    tmm.AddIsotropicLayer(float("inf"), substrateN)
    
    tmmGauss = GaussianTmm(tmm)
    #E, _ = tmmGauss.CalcFields2d(xs, ys, pol, th0 = np.radians(45.0))
    
    #plt.figure()
    #plt.pcolormesh(1e6 * xs, 1e6 * ys, E[:, :, 0].T.real, rasterized = True)
    #plt.colorbar()
    
    th0s = np.radians(np.linspace(0.0, 50.0, 10))
    tmmGauss.Sweep("th0", th0s)
    
    plt.show()
    