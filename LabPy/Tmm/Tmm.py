import numpy as np
import pylab as plt
from CppTmm import Tmm, Param, ParamType
from LabPy import GeneralTmm
from time import time

betas = np.linspace(0.0, 1.4, 100000)


tmm2 = GeneralTmm()
tmm2.SetConf(wl = 800e-9)
tmm2.AddIsotropicLayer(float("inf"), 1.5)
tmm2.AddIsotropicLayer(50e-9, 0.036759 + 5.5698j)
tmm2.AddIsotropicLayer(float("inf"), 1.0)
start = time()
rr = tmm2.SolveFor("beta", betas)
print "TMM Python time", time() - start

tmm = Tmm()
tmm.SetParam(Param(ParamType.WL), 800e-9)
tmm.SetParam(Param(ParamType.BETA), 0.0)
tmm.AddIsotropicLayer(float("inf"), 1.5)
tmm.AddIsotropicLayer(50e-9, 0.036759 + 5.5698j)
tmm.AddIsotropicLayer(float("inf"), 1.0)

start = time()
aa = tmm.Sweep(Param(ParamType.BETA), betas)
print "TMM Cpp time", time() - start

plt.figure()
plt.plot(betas, aa["R11"][0].real, "-")
#plt.plot(betas, rr["R11"])
plt.show()