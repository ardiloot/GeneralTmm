import unittest
import numpy as np
from GeneralTmm import Tmm, TmmPy, Material

# Refractive index data of silver from
# P. B. Johnson and R. W. Christy. Optical Constants of the Noble Metals, Phys. Rev. B 6, 4370-4379 (1972)

wlsAg = np.array([4.0e-07,   4.2e-07,   4.4e-07,   4.6e-07, 4.8e-07,   5.0e-07,
    5.2e-07,   5.4e-07, 5.6e-07,   5.8e-07,   6.0e-07,   6.2e-07, 6.4e-07,
    6.6e-07,   6.8e-07,   7.0e-07, 7.2e-07,   7.4e-07,   7.6e-07,   7.8e-07,
    8.0e-07,   8.2e-07,   8.4e-07,   8.6e-07, 8.8e-07,   9.0e-07,   9.2e-07,
    9.4e-07, 9.6e-07,   9.8e-07])
    
nsAg = np.array([0.050 + 2.104j, 0.046 + 2.348j, 0.040 + 2.553j, 0.044 + 2.751j,
    0.050 + 2.948j, 0.050 + 3.131j, 0.050 + 3.316j, 0.057 + 3.505j, 0.057 + 3.679j,
    0.051 + 3.841j, 0.055 + 4.010j, 0.059 + 4.177j, 0.055 + 4.332j, 0.050 + 4.487j,
    0.045 + 4.645j, 0.041 + 4.803j, 0.037 + 4.960j, 0.033 + 5.116j, 0.031 + 5.272j,
    0.034 + 5.421j, 0.037 + 5.570j, 0.040 + 5.719j, 0.040 + 5.883j, 0.040 + 6.048j,
    0.040 + 6.213j, 0.040 + 6.371j, 0.040 + 6.519j, 0.040 + 6.667j, 0.040 + 6.815j,
    0.040 + 6.962j])

class TmmTest(unittest.TestCase):
    
    @classmethod
    def _PrepareTmm(self, wl, layers):
        tmm = Tmm();
        tmm.SetParams(wl = wl)
        for layer in layers:
            if layer[0] == "iso":
                _, d, n = layer
                mat = Material.Static(n)
                tmm.AddIsotropicLayer(d, mat)
            else:
                _, d, n1, n2, n3, psi, xi = layer
                matx = Material.Static(n1)
                maty = Material.Static(n2)
                matz = Material.Static(n3)
                tmm.AddLayer(d, matx, maty, matz, psi, xi)

        # Old Tmm
        oldTmm = TmmPy()
        oldTmm.SetConf(wl = wl)
        for layer in layers:
            if layer[0] == "iso":
                _, d, n = layer
                oldTmm.AddIsotropicLayer(d, n)
            else:
                _, d, n1, n2, n3, psi, xi = layer
                oldTmm.AddLayer(d, n1, n2, n3, psi, xi)
                
        return tmm, oldTmm
        
    @classmethod
    def _testSweep(self, wl, layers, betas, pol, enhInterface, enhDist):
        tmm, oldTmm = self._PrepareTmm(wl, layers)
        res = tmm.Sweep("beta", betas, (pol, enhInterface, enhDist))
        resOld = oldTmm.SolveFor("beta", betas, polarization = pol, \
                                 enhInterface = enhInterface, enhDist = enhDist)
        for k in ["R11", "R22", "R12", "R21", "T31", "T32", "T41", "T42"]:
            np.testing.assert_almost_equal(res[k], resOld[k])
            
        np.testing.assert_almost_equal(res["enh"], resOld["enh"])
    
    @classmethod
    def _testFields(self, wl, beta, layers, xs, pol):
        tmm, oldTmm = self._PrepareTmm(wl, layers)
        
        tmm.beta = beta
        E, H = tmm.CalcFields1D(xs, np.array(pol))
        oldTmm.Solve(wl, beta)
        EOld, HOld = oldTmm.CalcFields1D(xs, pol)

        for i in range(3):
            print("test E", "XYZ"[i])
            np.testing.assert_allclose(abs(E[:, i]), abs(EOld[:, i]), rtol = 1e-7, atol = 1e-15)
            
        for i in range(3):
            print("test H", "XYZ"[i])     
            np.testing.assert_allclose(abs(H[:, i]), abs(HOld[:, i]), rtol = 1e-7, atol = 1e-15)

            
    def testTIRSweep(self):
        for wl in [400e-9, 800e-9]:
            for n1 in [1.2, 1.5]:
                for n2 in [1.0, 2.0]:
                    for pol in [(1.0, 0.0), (0.0, 1.0)]:
                        print("TIR test", wl, n1, n2, pol)
                        layers = [("iso", float("inf"), n1), \
                                  ("iso", float("inf"), n2)]
                        betas = np.linspace(0.0, n1 - 1e-3, 30)
                        self._testSweep(wl, layers, betas, pol, -1, 0.0)

    def testSppSweep(self):
        silver = Material(wlsAg, nsAg)
        for wl in [400e-9, 500e-9, 800e-9]:
            for pol in [(1.0, 0.0), (0.0, 1.0)]:
                print("SPP test", wl, pol)
                layers = [("iso", float("inf"), 1.5), \
                          ("iso", float("inf"), silver(wl)),
                          ("iso", float("inf"), 1.0)]
                betas = np.linspace(0.0, 1.499, 30)
                self._testSweep(wl, layers, betas, pol, -1, 0.0)

    def testAnisoSweep(self):
        wl = 500e-9
        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            for psi in np.linspace(0.0, 2 * np.pi, 5):
                for xi in np.linspace(0.0, np.pi, 5):
                    print("Aniso test", pol, psi, xi)
                    layers = [("iso", float("inf"), 1.5), \
                              ("aniso", float("inf"), 1.76, 1.8, 1.9, psi, xi)]
                    betas = np.linspace(0.0, 1.5 - 1e-3, 30)
                    self._testSweep(wl, layers, betas, pol, -1, 0.0)
                    
    def testAnisoSweep2(self):
        wl = 500e-9
        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            for psi in np.linspace(0.0, 2 * np.pi, 5):
                for xi in np.linspace(0.0, np.pi, 5):
                    print("Aniso test", pol, psi, xi)
                    layers = [("iso", float("inf"), 1.5), \
                              ("aniso", 200e-9, 1.76, 1.8, 1.9, psi, xi),
                              ("iso", float("inf"), 1.5)]
                    betas = np.linspace(0.0, 1.5 - 1e-3, 30)
                    self._testSweep(wl, layers, betas, pol, -1, 0.0)


    def testSppFields(self):
        silver = Material(wlsAg, nsAg)
        beta = 0.5
        xs = np.linspace(-1e-6, 1e-6, 1000)
        for wl in [400e-9, 500e-9, 800e-9]:
            for pol in [(1.0, 0.0), (0.0, 1.0)]:
                print("SPP test", wl, pol)
                layers = [("iso", float("inf"), 1.5), \
                          ("iso", float("inf"), silver(wl)),
                          ("iso", float("inf"), 1.0)]
                self._testFields(wl, beta, layers, xs, pol)
                
    def testAnisoFields(self):
        beta = 0.5
        xs = np.linspace(-1e-6, 1e-6, 100)
        for wl in [400e-9, 500e-9, 800e-9]:
            for psi in np.linspace(0.0, 2 * np.pi, 5):
                for xi in np.linspace(0.0, np.pi, 5):
                    for pol in [(1.0, 0.0), (0.0, 1.0)]:
                        print(wl, psi, xi, pol)
                        layers = [("iso", float("inf"), 1.5), \
                                  ("aniso", float("inf"), 1.76, 1.8, 1.9, psi, xi)]
                        self._testFields(wl, beta, layers, xs, pol)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TmmTest)
    unittest.TextTestRunner().run(suite)