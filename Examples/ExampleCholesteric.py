"""Cholesteric liquid crystal — circular-polarization-selective Bragg reflector.

A helical stack of birefringent layers reflects one circular polarization
in a well-defined wavelength band (n_o * p < λ < n_e * p) and transmits
the other.  This is the mechanism behind structurally colored beetle shells
and cholesteric LC displays.
"""

import matplotlib.pyplot as plt
import numpy as np

from GeneralTmm import Material, Tmm


def main():
    n_o, n_e = 1.5, 1.7  # ordinary / extraordinary
    pitch = 350e-9  # helix pitch → Bragg band centered at ~560 nm
    n_layers_per_pitch = 30  # discretization of the continuous helix
    n_pitches = 20
    d_layer = pitch / n_layers_per_pitch

    air = Material.Static(1.0)
    mat_x = Material.Static(n_o)
    mat_y = Material.Static(n_e)  # optic axis
    mat_z = Material.Static(n_o)

    # Build the helical stack: optic axis rotates around x (layer normal)
    tmm = Tmm(wl=550e-9, beta=0.0)  # normal incidence
    tmm.AddIsotropicLayer(float("inf"), air)
    for i in range(n_layers_per_pitch * n_pitches):
        xi = 2.0 * np.pi * i / n_layers_per_pitch
        tmm.AddLayer(d_layer, mat_x, mat_y, mat_z, psi=0.0, xi=xi)
    tmm.AddIsotropicLayer(float("inf"), air)

    wls = np.linspace(400e-9, 750e-9, 1000)
    sr = tmm.Sweep("wl", wls)
    wl_nm = wls * 1e9

    # Bragg band edges
    band_lo = n_o * pitch * 1e9  # 525 nm
    band_hi = n_e * pitch * 1e9  # 595 nm

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    fig.suptitle("Cholesteric LC: circular-polarization-selective reflection", fontsize=13)

    for ax in (ax1, ax2):
        ax.axvspan(band_lo, band_hi, alpha=0.12, color="green", label="Bragg band")

    ax1.plot(wl_nm, sr["R11"], label=r"R$_{pp}$")
    ax1.plot(wl_nm, sr["R12"], label=r"R$_{ps}$")
    ax1.plot(wl_nm, sr["R11"] + sr["R12"], "--", color="black", label="Total (p in)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title("Reflection — one circular polarization is selectively reflected")
    ax1.legend()

    ax2.plot(wl_nm, sr["T31"], label=r"T$_{pp}$")
    ax2.plot(wl_nm, sr["T41"], label=r"T$_{ps}$")
    ax2.plot(wl_nm, sr["T31"] + sr["T41"], "--", color="black", label="Total (p in)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Transmittance")
    ax2.set_title("Transmission — the other circular polarization passes through")
    ax2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
