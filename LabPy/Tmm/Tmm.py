import numpy as np
import pylab as plt
from CppTmm import Tmm, Param, ParamType  # @UnresolvedImport
from LabPy import GeneralTmm
from time import clock

balance = 300
betas = np.linspace(0.0, 1.4, 3000)
betas2 = np.linspace(0.0, 1.4, len(betas) * balance)

tmm2 = GeneralTmm()
tmm2.SetConf(wl = 800e-9)
tmm2.AddIsotropicLayer(float("inf"), 1.5)
tmm2.AddIsotropicLayer(50e-9, 0.036759 + 5.5698j)
tmm2.AddIsotropicLayer(float("inf"), 1.0)
start = clock()

tmm = Tmm()

tmm.AddIsotropicLayer(float("inf"), 1.5)
tmm.AddIsotropicLayer(50e-9, 0.036759 + 5.5698j)
tmm.AddIsotropicLayer(float("inf"), 1.0)
tmm.SetParam(Param(ParamType.WL), 800e-9)
tmm.SetParam(Param(ParamType.BETA), 0.5)

"""
timeCpp = np.zeros(5)
timePython = np.zeros_like(timeCpp)
for i in range(len(timeCpp)):
    print "Test", i
    start = clock()
    rr = tmm2.SolveFor("beta", betas)
    timePython[i] = clock() - start
    
    start = clock()
    aa = tmm.Sweep(Param(ParamType.BETA), betas2)
    timeTmm = clock() - start
    timeCpp[i] = clock() - start

print "Python"
print timePython
print "mean", np.mean(timePython), "std", np.std(timePython)

print "Cpp"
print timeCpp
print "mean", np.mean(timeCpp), "std", np.std(timeCpp)

print "enh", balance * np.mean(timePython) / np.mean(timeCpp)

plt.figure()
plt.plot(betas2, aa["R11"][0].real, "-")
plt.plot(betas, rr["R11"], "x")


"""

xs = np.linspace(-1e-6, 1e-6, 200)
pol = np.linspace(1.0, 0.0, 2)
pol2 = np.linspace(1.0, 0.0, 2)

print pol.shape

tmm.Solve();
fields = tmm.CalcFields1D(xs, pol)
fields = tmm.CalcFields1D(xs, pol2)

tmm2.Solve(800e-9, 0.5)
EPython, HPython = tmm2.CalcFields1D(xs, (1.0, 0.0))

plt.figure()
for i in range(3):
    plt.plot(1e6 * xs, abs(EPython[:, i]), label = "py,%d" % (i))
    plt.plot(1e6 * xs, abs(fields.E[:, i]), "x", label = "cpp,%d" % (i))
plt.figure()
for i in range(3):
    plt.plot(1e6 * xs, abs(HPython[:, i]), label = "py,%d" % (i))
    plt.plot(1e6 * xs, abs(fields.H[:, i]), "x", label = "cpp,%d" % (i))

    
plt.show()