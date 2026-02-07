"""Total Internal Reflection (TIR) at a prism/air interface.

Sweeps over incidence angle to show the critical-angle transition
for p- and s-polarizations. Critical angle ≈ 41.8° for glass/air.
"""

import matplotlib.pyplot as plt
import numpy as np

from GeneralTmm import Material, Tmm


def main():
    prismN = Material.Static(1.5)
    substrateN = Material.Static(1.0)

    wl = 532e-9  # green laser
    betas = np.linspace(0.0, 1.49, 100)  # beta = n*sin(theta)
    angles = np.arcsin(betas / prismN(wl).real)

    tmm = Tmm(wl=wl)
    tmm.AddIsotropicLayer(float("inf"), prismN)
    tmm.AddIsotropicLayer(float("inf"), substrateN)
    sr = tmm.Sweep("beta", betas)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Total Internal Reflection", fontsize=14)

    ax1.set_title("Reflection")
    ax1.plot(np.degrees(angles), sr["R11"], label="p-pol")
    ax1.plot(np.degrees(angles), sr["R22"], label="s-pol")
    ax1.set_xlabel(r"$\theta$ ($\degree$)")
    ax1.set_ylabel("Reflectance")
    ax1.legend()

    ax2.set_title("Transmission")
    ax2.plot(np.degrees(angles), sr["T31"], label="p-pol")
    ax2.plot(np.degrees(angles), sr["T42"], label="s-pol")
    ax2.set_xlabel(r"$\theta$ ($\degree$)")
    ax2.set_ylabel("Transmittance")
    ax2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
