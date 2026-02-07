"""Surface Plasmon Polariton (SPP) in Kretschmann configuration.

Glass prism | 50 nm Ag | air.  At the SPP resonance angle the
reflection dips and the E-field is strongly enhanced at the metal surface.
"""

import matplotlib.pyplot as plt
import numpy as np

from GeneralTmm import Material, Tmm

# Ag refractive index — Johnson & Christy, Phys. Rev. B 6, 4370 (1972)
wlsAg = np.arange(400e-9, 1000e-9, 20e-9)  # 400–980 nm in 20 nm steps
# fmt: off
nsAg = np.array([
    0.050+2.104j, 0.046+2.348j, 0.040+2.553j, 0.044+2.751j, 0.050+2.948j,
    0.050+3.131j, 0.050+3.316j, 0.057+3.505j, 0.057+3.679j, 0.051+3.841j,
    0.055+4.010j, 0.059+4.177j, 0.055+4.332j, 0.050+4.487j, 0.045+4.645j,
    0.041+4.803j, 0.037+4.960j, 0.033+5.116j, 0.031+5.272j, 0.034+5.421j,
    0.037+5.570j, 0.040+5.719j, 0.040+5.883j, 0.040+6.048j, 0.040+6.213j,
    0.040+6.371j, 0.040+6.519j, 0.040+6.667j, 0.040+6.815j, 0.040+6.962j,
])
# fmt: on


def main():
    prism = Material.Static(1.5)
    metal = Material(wlsAg, nsAg)  # Ag, wavelength-dependent
    substrate = Material.Static(1.0)

    wl = 950e-9  # near-IR
    d_metal = 50e-9
    betas = np.linspace(0.8, 1.2, 200)
    angles = np.arcsin(betas / prism(wl).real)
    xs = np.linspace(-0.5e-6, 2e-6, 300)
    ys = np.linspace(-1e-6, 1e-6, 301)
    enhPos = (np.array([1.0, 0.0]), 2, 0.0)  # p-pol, layer 2, at interface

    tmm = Tmm(wl=wl)
    tmm.AddIsotropicLayer(float("inf"), prism)
    tmm.AddIsotropicLayer(d_metal, metal)
    tmm.AddIsotropicLayer(float("inf"), substrate)

    # Sweep and optimize enhancement
    sr = tmm.Sweep("beta", betas, enhPos=enhPos)
    tmm.SetParams(enhOptMaxIters=100, enhOptRel=1e-5, enhInitialStep=1e-3)
    beta0 = betas[np.argmax(sr["enh"])]
    maxEnh = tmm.OptimizeEnhancement(["beta"], np.array([beta0]), enhPos)
    print("Initial enhancement %.2f" % np.max(sr["enh"]))
    print("Optimized enhancement %.2f at β=%.4f" % (maxEnh, tmm.beta))

    # 1D field profile at the optimal angle
    E1D, _H1D = tmm.CalcFields1D(xs, enhPos[0])

    # Plot
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig1.suptitle("Surface plasmon polaritons on Ag film", fontsize=13)

    ax1.plot(np.degrees(angles), sr["R11"], label=r"R$_{pp}$")
    ax1.plot(np.degrees(angles), sr["T31"], label=r"T$_{pp}$")
    ax1.set_xlabel(r"$\theta$ ($\degree$)")
    ax1.set_ylabel("Intensity")
    ax1.legend(loc="center left")
    ax1.set_title("(a)  Angular sweep")
    ax1r = ax1.twinx()
    ax1r.plot(np.degrees(angles), sr["enh"], "--", color="red", label="enhancement")
    ax1r.set_ylabel("Enhancement")
    ax1r.legend(loc="center right")

    ax2.plot(1e6 * xs, abs(E1D[:, 0]), label=r"|$E_x$|")
    ax2.plot(1e6 * xs, abs(E1D[:, 1]), label=r"|$E_y$|")
    ax2.plot(1e6 * xs, abs(E1D[:, 2]), label=r"|$E_z$|")
    ax2.plot(1e6 * xs, np.linalg.norm(E1D, axis=1), "--", label=r"|$E$|")
    ax2.axvline(0.0, ls="--", color="black", lw=1.0)
    ax2.axvline(1e6 * d_metal, ls="--", color="black", lw=1.0)
    ax2.set_xlabel(r"x ($\mu m$)")
    ax2.set_ylabel("Electric field (V/m)")
    ax2.set_title("(b)  1D field profile at SPP resonance")
    ax2.legend()
    fig1.tight_layout()

    # 2D field map
    E2D, H2D = tmm.CalcFields2D(xs, ys, enhPos[0])

    toPlot = [
        (r"$E_x$ (V/m)", E2D[:, :, 0].real),
        (r"$E_y$ (V/m)", E2D[:, :, 1].real),
        (r"$H_z$ (A/m)", H2D[:, :, 2].real),
        (r"$|E|$ (V/m)", np.linalg.norm(E2D, axis=2)),
    ]

    fig3 = plt.figure(figsize=(10, 7))
    fig3.suptitle("2D fields of surface plasmons on Ag film", fontsize=13)
    for i, (label, data) in enumerate(toPlot):
        ax = fig3.add_subplot(221 + i)
        ax.pcolormesh(1e6 * xs, 1e6 * ys, data.real.T)
        ax.axvline(0.0, ls="--", color="white", lw=0.5)
        ax.axvline(1e6 * d_metal, ls="--", color="white", lw=0.5)
        ax.set_xlabel(r"x ($\mu m$)")
        ax.set_ylabel(r"y ($\mu m$)")
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label(label)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.92)

    plt.show()


if __name__ == "__main__":
    main()
