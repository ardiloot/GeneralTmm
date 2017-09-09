import numpy as np
import pylab as plt
from GeneralTmm import Tmm, Material

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

if __name__ == "__main__":
    # Prepare materials
    prismN = Material.Static(1.5)
    metalN = Material(wlsAg, nsAg)
    substrateN = Material.Static(1.0)
    
    # Parameters
    wl = 950e-9 # wavelength
    metalD = 50e-9 # metal layer thickness
    betas = np.linspace(0.8, 1.2, 200) # effective mode index
    angles = np.arcsin(betas / prismN(wl).real) # angle inside prism
    xs = np.linspace(-0.5e-6, 2e-6, 300) # for plotting fields
    ys = np.linspace(-1e-6, 1e-6, 301) # for plotting fields
    
    # Enhancement measurement polarisation ((1, 0) = p and (0, 1) = s), layerNr
    # and distance from the interface 
    enhPol, enhLayer, enhDist = np.array([1.0, 0.0]), 2, 0.0 
    enhPos = (enhPol, enhLayer, enhDist)
    
    # Init TMM
    tmm = Tmm(wl = wl)
    
    # Add layers
    tmm.AddIsotropicLayer(float("inf"), prismN)
    tmm.AddIsotropicLayer(metalD, metalN)
    tmm.AddIsotropicLayer(float("inf"), substrateN)
    
    # Reflection, transmission, enhancement
    #--------------------------------------------------------------------------- 
    
    # Do calculations
    sr = tmm.Sweep("beta", betas, enhPos = enhPos)
    
    # Plot
    plt.figure(figsize = (8, 5))
    plt.title("Reflection, transmission and enhancement of surface plasmons on Ag")
    plt.plot(np.degrees(angles), sr["R11"], label = r"reflection")
    plt.plot(np.degrees(angles), sr["T31"], label = r"transmission")
    plt.xlabel(r"$\theta$ ($\degree$)")
    plt.ylabel("reflection/transmission")
    plt.legend()
    plt.twinx()
    plt.plot(np.degrees(angles), sr["enh"], "--", color = "red", label = r"enhancement")
    plt.ylabel("enhencement")
    plt.legend()
    
    # Optimize enhancment to find maximal value (important in case of narrow resonances)
    #--------------------------------------------------------------------------- 
    
    # Set simplex optimizer parameters
    tmm.SetParams(enhOptMaxIters = 100)
    tmm.SetParams(enhOptRel = 1e-5)
    tmm.SetParams(enhInitialStep = 1e-3)
    
    # Initial guess from previous calculations
    maxEnhBetaApprox = betas[np.argmax(sr["enh"])]
    
    # Optimize
    maxEnhOpt = tmm.OptimizeEnhancement(["beta"], np.array([maxEnhBetaApprox]), enhPos)
    maxEnhOptBeta = tmm.beta
    
    # Print result
    print("Initial enhancement %.2f" % (np.max(sr["enh"])))
    print(r"Optimized enhancement %.2f at β=%.4f" % (maxEnhOpt, maxEnhOptBeta))
   
    # Calculate and plot 1D fields at the maximum enhancement angle of incidence
    #--------------------------------------------------------------------------- 
    
    tmm.beta = maxEnhOptBeta
    E1D, H1D = tmm.CalcFields1D(xs, enhPol)
    
    plt.figure()
    plt.title("1D fields of surface plasmons on Ag film")
    plt.plot(1e6 * xs, abs(E1D[:, 0]), label = r"|$E_x$|")
    plt.plot(1e6 * xs, abs(E1D[:, 1]), label = r"|$E_y$|")
    plt.plot(1e6 * xs, abs(E1D[:, 2]), label = r"|$E_z$|")
    plt.plot(1e6 * xs, np.linalg.norm(E1D, axis = 1), "--", label = r"|$E$|")
    plt.axvline(0.0, ls = "--", color = "black", lw = 1.0)
    plt.axvline(1e6 * metalD, ls = "--", color = "black", lw = 1.0)
    plt.xlabel("x ($\mu m$)")
    plt.ylabel("Electrical field (V/m)")
    plt.legend()
    
    # Calculate and plot 2D fields at the maximum enhancement angle of incidence
    #--------------------------------------------------------------------------- 
    
    tmm.beta = maxEnhOptBeta
    E2D, H2D = tmm.CalcFields2D(xs, ys, enhPol)
    
    toPlot = [(r"$E_x$ (V/m)", E2D[:, :, 0].real),
              (r"$E_y$ (V/m)", E2D[:, :, 1].real),
              (r"$H_z$ (A/m)", H2D[:, :, 2].real),
              (r"$|E|$ (V/m)", np.linalg.norm(E2D, axis = 2))]
    
    plt.figure()
    plt.suptitle("2D fields of surface plasmons on Ag film")
    for i, (label, data) in enumerate(toPlot):
        plt.subplot(221 + i)
        plt.pcolormesh(1e6 * xs, 1e6 * ys, data.real.T)
        plt.axvline(0.0, ls = "--", color = "white", lw = 0.5)
        plt.axvline(1e6 * metalD, ls = "--", color = "white", lw = 0.5)
        plt.xlabel("x ($\mu m$)")
        plt.ylabel("y ($\mu m$)")
        cbar = plt.colorbar()
        cbar.set_label(label)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.92)
    
    plt.show()