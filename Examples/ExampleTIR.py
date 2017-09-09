import numpy as np
import pylab as plt
from GeneralTmm import Tmm, Material

if __name__ == "__main__":
    # Prepare materials
    prismN = Material.Static(1.5)
    substrateN = Material.Static(1.0)
    
    # Parameters
    wl = 532e-9
    betas = np.linspace(0.0, 1.49, 100)
    angles = np.arcsin(betas / prismN(wl).real)
    
    # Init TMM
    tmm = Tmm()
    tmm.wl = wl
    
    # Add layers
    tmm.AddIsotropicLayer(float("inf"), prismN)
    tmm.AddIsotropicLayer(float("inf"), substrateN)
    
    # Do calculations
    sr = tmm.Sweep("beta", betas)
    
    # Plot
    plt.figure(figsize = (8, 5))
    plt.suptitle("Total internal reflection")
    plt.subplot(121)
    plt.title("Reflection")
    plt.plot(np.degrees(angles), sr["R11"], label = r"p-pol")
    plt.plot(np.degrees(angles), sr["R22"], label = r"s-pol")
    plt.xlabel(r"$\theta$ ($\degree$)")
    plt.legend()
    
    plt.subplot(122)
    plt.title("Transmission")
    plt.plot(np.degrees(angles), sr["T31"], label = r"p-pol")
    plt.plot(np.degrees(angles), sr["T42"], label = r"s-pol")
    plt.xlabel(r"$\theta$ ($\degree$)")
    plt.legend()
    
    plt.show()