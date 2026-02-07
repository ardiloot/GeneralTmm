"""Wave plates — polarization rotation with a birefringent layer.

Half-wave plate: retardation π → full p↔s conversion at 45°.
Quarter-wave plate: retardation π/2 → linear to circular at 45°.

The plate angle ξ rotates the crystal fast/slow axes in the yz plane.
"""

import matplotlib.pyplot as plt
import numpy as np

from GeneralTmm import Material, Tmm


def main():
    n_o, n_e = 1.50, 1.60  # ordinary / extraordinary
    wl = 632.8e-9  # He-Ne laser
    dn = n_e - n_o
    d_hwp = wl / (2 * dn)  # half-wave plate thickness
    d_qwp = wl / (4 * dn)  # quarter-wave plate thickness

    air = Material.Static(1.0)
    mat_x = Material.Static(n_o)
    mat_y = Material.Static(n_e)  # optic axis
    mat_z = Material.Static(n_o)

    xi_values = np.linspace(0, np.pi, 300)  # plate rotation angle

    def build_plate(d):
        tmm = Tmm(wl=wl, beta=0.0)  # normal incidence
        tmm.AddIsotropicLayer(float("inf"), air)
        tmm.AddLayer(d, mat_x, mat_y, mat_z, psi=0.0, xi=0.0)
        tmm.AddIsotropicLayer(float("inf"), air)
        return tmm.Sweep("xi_1", xi_values)

    sr_hwp = build_plate(d_hwp)
    sr_qwp = build_plate(d_qwp)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    fig.suptitle("Wave plates: polarization conversion vs plate angle", fontsize=13)

    for ax in (ax1, ax2):
        for deg in [0, 45, 90, 135, 180]:
            ax.axvline(deg, ls="--", color="gray", lw=0.5)

    ax1.plot(np.degrees(xi_values), sr_hwp["T31"], label=r"T$_{pp}$ (p→p)")
    ax1.plot(np.degrees(xi_values), sr_hwp["T41"], label=r"T$_{ps}$ (p→s)")
    ax1.set_ylabel("Transmittance")
    ax1.legend()
    ax1.set_title(
        r"Half-wave plate ($\Delta\phi = \pi$): "
        r"full polarization rotation at $\xi = 45\degree$"
    )

    ax2.plot(np.degrees(xi_values), sr_qwp["T31"], label=r"T$_{pp}$ (p→p)")
    ax2.plot(np.degrees(xi_values), sr_qwp["T41"], label=r"T$_{ps}$ (p→s)")
    ax2.set_xlabel(r"Plate rotation angle $\xi$ ($\degree$)")
    ax2.set_ylabel("Transmittance")
    ax2.legend()
    ax2.set_title(
        r"Quarter-wave plate ($\Delta\phi = \pi/2$): "
        r"linear $\rightarrow$ circular at $\xi = 45\degree$"
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
