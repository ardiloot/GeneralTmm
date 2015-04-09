import unittest
import numpy as np
from LabPy.Tmm.Tmm import Tmm
from LabPy.Old.GeneralTMM import GeneralTmm  # @UnresolvedImport
from LabPy import Material

class TmmTest(unittest.TestCase):
    
    @classmethod
    def _PrepareTmm(self, wl, layers):
        tmm = Tmm();
        tmm.SetParam(wl = wl)
        for layer in layers:
            if layer[0] == "iso":
                _, d, n = layer
                tmm.AddIsotropicLayer(d, n)
            else:
                _, d, n1, n2, n3, psi, xi = layer
                tmm.AddLayer(d, n1, n2, n3, psi, xi)

        # Old Tmm
        oldTmm = GeneralTmm()
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
        
        tmm.SetParam(beta = beta)
        E, H = tmm.CalcFields1D(xs, pol)
        oldTmm.Solve(wl, beta)
        EOld, HOld = oldTmm.CalcFields1D(xs, pol)

        for i in range(3):
            print "test E", "XYZ"[i]
            np.testing.assert_allclose(abs(E[:, i]), abs(EOld[:, i]), rtol = 1e-7, atol = 1e-15)
            
        for i in range(3):
            print "test H", "XYZ"[i]     
            np.testing.assert_allclose(abs(H[:, i]), abs(HOld[:, i]), rtol = 1e-7, atol = 1e-15)

            
    def testTIRSweep(self):
        for wl in [400e-9, 800e-9]:
            for n1 in [1.2, 1.5]:
                for n2 in [1.0, 2.0]:
                    for pol in [(1.0, 0.0), (0.0, 1.0)]:
                        print "TIR test", wl, n1, n2, pol
                        layers = [("iso", float("inf"), n1), \
                                  ("iso", float("inf"), n2)]
                        betas = np.linspace(0.0, n1 - 1e-3, 30)
                        self._testSweep(wl, layers, betas, pol, -1, 0.0)

    def testSppSweep(self):
        silver = Material(r"main/Ag\Johnson")
        for wl in [400e-9, 500e-9, 800e-9]:
            for pol in [(1.0, 0.0), (0.0, 1.0)]:
                print "SPP test", wl, pol
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
                    print "Aniso test", pol, psi, xi
                    layers = [("iso", float("inf"), 1.5), \
                              ("aniso", float("inf"), 1.76, 1.8, 1.9, psi, xi)]
                    betas = np.linspace(0.0, 1.5 - 1e-3, 30)
                    self._testSweep(wl, layers, betas, pol, -1, 0.0)
                    
    def testAnisoSweep2(self):
        wl = 500e-9
        for pol in [(1.0, 0.0), (0.0, 1.0)]:
            for psi in np.linspace(0.0, 2 * np.pi, 5):
                for xi in np.linspace(0.0, np.pi, 5):
                    print "Aniso test", pol, psi, xi
                    layers = [("iso", float("inf"), 1.5), \
                              ("aniso", 200e-9, 1.76, 1.8, 1.9, psi, xi),
                              ("iso", float("inf"), 1.5)]
                    betas = np.linspace(0.0, 1.5 - 1e-3, 30)
                    self._testSweep(wl, layers, betas, pol, -1, 0.0)


    def testSppFields(self):
        silver = Material(r"main/Ag\Johnson")
        beta = 0.5
        xs = np.linspace(-1e-6, 1e-6, 1000)
        for wl in [400e-9, 500e-9, 800e-9]:
            for pol in [(1.0, 0.0), (0.0, 1.0)]:
                print "SPP test", wl, pol
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
                        print wl, psi, xi, pol
                        layers = [("iso", float("inf"), 1.5), \
                                  ("aniso", float("inf"), 1.76, 1.8, 1.9, psi, xi)]
                        self._testFields(wl, beta, layers, xs, pol)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TmmTest)
    unittest.TextTestRunner().run(suite)