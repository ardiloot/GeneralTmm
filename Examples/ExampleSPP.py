"""Surface Plasmon Polariton (SPP) simulation in Kretschmann configuration.

Glass prism | 50 nm Ag film | air.  At the SPP resonance angle the
reflection dips and the E-field is strongly enhanced at the metal surface.

Shows: angular sweeps, enhancement optimization, 1D/2D field maps.
"""

import matplotlib.pyplot as plt
import numpy as np

from GeneralTmm import Material, Tmm

# Refractive index data of silver from
# P. B. Johnson and R. W. Christy. Optical Constants of the Noble Metals, Phys. Rev. B 6, 4370-4379 (1972)

wlsAg = np.array(
    [
        4.0e-07,
        4.2e-07,
        4.4e-07,
        4.6e-07,
        4.8e-07,
        5.0e-07,
        5.2e-07,
        5.4e-07,
        5.6e-07,
        5.8e-07,
        6.0e-07,
        6.2e-07,
        6.4e-07,
        6.6e-07,
        6.8e-07,
        7.0e-07,
        7.2e-07,
        7.4e-07,
        7.6e-07,
        7.8e-07,
        8.0e-07,
        8.2e-07,
        8.4e-07,
        8.6e-07,
        8.8e-07,
        9.0e-07,
        9.2e-07,
        9.4e-07,
        9.6e-07,
        9.8e-07,
    ]
)

nsAg = np.array(
    [
        0.050 + 2.104j,
        0.046 + 2.348j,
        0.040 + 2.553j,
        0.044 + 2.751j,
        0.050 + 2.948j,
        0.050 + 3.131j,
        0.050 + 3.316j,
        0.057 + 3.505j,
        0.057 + 3.679j,
        0.051 + 3.841j,
        0.055 + 4.010j,
        0.059 + 4.177j,
        0.055 + 4.332j,
        0.050 + 4.487j,
        0.045 + 4.645j,
        0.041 + 4.803j,
        0.037 + 4.960j,
        0.033 + 5.116j,
        0.031 + 5.272j,
        0.034 + 5.421j,
        0.037 + 5.570j,
        0.040 + 5.719j,
        0.040 + 5.883j,
        0.040 + 6.048j,
        0.040 + 6.213j,
        0.040 + 6.371j,
        0.040 + 6.519j,
        0.040 + 6.667j,
        0.040 + 6.815j,
        0.040 + 6.962j,
    ]
)


def main():
    prismN = Material.Static(1.5)
    metalN = Material(wlsAg, nsAg)  # Ag, wavelength-dependent
    substrateN = Material.Static(1.0)

    wl = 950e-9  # near-IR
    metalD = 50e-9
    betas = np.linspace(0.8, 1.2, 200)
    angles = np.arcsin(betas / prismN(wl).real)
    xs = np.linspace(-0.5e-6, 2e-6, 300)
    ys = np.linspace(-1e-6, 1e-6, 301)

    # Measure p-polarized enhancement at the metal/air interface
    enhPos = (np.array([1.0, 0.0]), 2, 0.0)  # (polarization, layer, distance)

    tmm = Tmm(wl=wl)
    tmm.AddIsotropicLayer(float("inf"), prismN)
    tmm.AddIsotropicLayer(metalD, metalN)
    tmm.AddIsotropicLayer(float("inf"), substrateN)

    # 1. Reflection, transmission, enhancement vs angle
    sr = tmm.Sweep("beta", betas, enhPos=enhPos)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title("Reflection, transmission and enhancement of surface plasmons on Ag")
    ax1.plot(np.degrees(angles), sr["R11"], label="reflection")
    ax1.plot(np.degrees(angles), sr["T31"], label="transmission")
    ax1.set_xlabel(r"$\theta$ ($\degree$)")
    ax1.set_ylabel("Reflection / Transmission")
    ax1.legend(loc="center left")
    ax1r = ax1.twinx()
    ax1r.plot(np.degrees(angles), sr["enh"], "--", color="red", label="enhancement")
    ax1r.set_ylabel("Enhancement")
    ax1r.legend(loc="center right")
    fig1.tight_layout()

    # 2. Optimize enhancement to find the exact SPP resonance
    tmm.SetParams(enhOptMaxIters=100, enhOptRel=1e-5, enhInitialStep=1e-3)
    maxEnhBetaApprox = betas[np.argmax(sr["enh"])]
    maxEnhOpt = tmm.OptimizeEnhancement(["beta"], np.array([maxEnhBetaApprox]), enhPos)
    maxEnhOptBeta = tmm.beta

    print("Initial enhancement %.2f" % (np.max(sr["enh"])))
    print(r"Optimized enhancement %.2f at β=%.4f" % (maxEnhOpt, maxEnhOptBeta))

    # 3. 1D field profile at the optimal angle
    tmm.beta = maxEnhOptBeta
    E1D, H1D = tmm.CalcFields1D(xs, enhPos[0])

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.set_title("1D field profile of surface plasmons on Ag film")
    ax2.plot(1e6 * xs, abs(E1D[:, 0]), label=r"|$E_x$|")
    ax2.plot(1e6 * xs, abs(E1D[:, 1]), label=r"|$E_y$|")
    ax2.plot(1e6 * xs, abs(E1D[:, 2]), label=r"|$E_z$|")
    ax2.plot(1e6 * xs, np.linalg.norm(E1D, axis=1), "--", label=r"|$E$|")
    ax2.axvline(0.0, ls="--", color="black", lw=1.0)
    ax2.axvline(1e6 * metalD, ls="--", color="black", lw=1.0)
    ax2.set_xlabel(r"x ($\mu m$)")
    ax2.set_ylabel("Electric field (V/m)")
    ax2.legend()
    fig2.tight_layout()

    # 4. 2D field map at the optimal angle
    E2D, H2D = tmm.CalcFields2D(xs, ys, enhPos[0])

    toPlot = [
        (r"$E_x$ (V/m)", E2D[:, :, 0].real),
        (r"$E_y$ (V/m)", E2D[:, :, 1].real),
        (r"$H_z$ (A/m)", H2D[:, :, 2].real),
        (r"$|E|$ (V/m)", np.linalg.norm(E2D, axis=2)),
    ]

    fig3 = plt.figure(figsize=(10, 8))
    fig3.suptitle("2D fields of surface plasmons on Ag film", fontsize=14)
    for i, (label, data) in enumerate(toPlot):
        ax = fig3.add_subplot(221 + i)
        ax.pcolormesh(1e6 * xs, 1e6 * ys, data.real.T)
        ax.axvline(0.0, ls="--", color="white", lw=0.5)
        ax.axvline(1e6 * metalD, ls="--", color="white", lw=0.5)
        ax.set_xlabel(r"x ($\mu m$)")
        ax.set_ylabel(r"y ($\mu m$)")
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label(label)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.92)

    plt.show()


if __name__ == "__main__":
    main()
