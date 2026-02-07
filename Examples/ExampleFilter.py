"""Dielectric thin-film optical filters — Bragg mirror and Fabry-Perot bandpass.

Quarter-wave layers of TiO2 (n = 2.30) and SiO2 (n = 1.46) on a BK7
glass substrate at normal incidence.

(a) Bragg mirror: (HL)^7 H gives > 99.8 % reflectance across a ~200 nm
    stop band centered at the design wavelength.
(b) Fabry-Perot bandpass: two Bragg mirrors sandwich a half-wave SiO2
    cavity, producing a narrow transmission peak at 550 nm.
"""

import matplotlib.pyplot as plt
import numpy as np

from GeneralTmm import Material, Tmm


def main():
    wl0 = 550e-9  # design wavelength (green)
    n_H, n_L = 2.30, 1.46  # TiO2 / SiO2
    n_sub = 1.52  # BK7 glass substrate

    d_H = wl0 / (4 * n_H)  # quarter-wave optical thickness
    d_L = wl0 / (4 * n_L)

    H = Material.Static(n_H)
    L = Material.Static(n_L)
    substrate = Material.Static(n_sub)
    air = Material.Static(1.0)

    # --- Bragg mirror: sub | (HL)^N H | air ---
    N_mirror = 7
    tmm_m = Tmm(wl=wl0)
    tmm_m.AddIsotropicLayer(float("inf"), substrate)
    for _ in range(N_mirror):
        tmm_m.AddIsotropicLayer(d_H, H)
        tmm_m.AddIsotropicLayer(d_L, L)
    tmm_m.AddIsotropicLayer(d_H, H)
    tmm_m.AddIsotropicLayer(float("inf"), air)

    wls = np.linspace(350e-9, 800e-9, 1000)
    sr_m = tmm_m.Sweep("wl", wls)
    wl_nm = wls * 1e9

    # --- Fabry-Perot bandpass: sub | (HL)^N H 2L H (LH)^N | air ---
    N_fp = 5
    tmm_fp = Tmm(wl=wl0)
    tmm_fp.AddIsotropicLayer(float("inf"), substrate)
    for _ in range(N_fp):
        tmm_fp.AddIsotropicLayer(d_H, H)
        tmm_fp.AddIsotropicLayer(d_L, L)
    tmm_fp.AddIsotropicLayer(d_H, H)
    tmm_fp.AddIsotropicLayer(2 * d_L, L)  # half-wave spacer (cavity)
    tmm_fp.AddIsotropicLayer(d_H, H)
    for _ in range(N_fp):
        tmm_fp.AddIsotropicLayer(d_L, L)
        tmm_fp.AddIsotropicLayer(d_H, H)
    tmm_fp.AddIsotropicLayer(float("inf"), air)

    sr_fp = tmm_fp.Sweep("wl", wls)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        r"Dielectric thin-film filters  (TiO$_2$ / SiO$_2$,"
        r"  design $\lambda$ = 550 nm)",
        fontsize=13,
    )

    ax1.plot(wl_nm, sr_m["R11"], label=r"R$_{pp}$")
    ax1.plot(wl_nm, sr_m["T31"], label=r"T$_{pp}$")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Intensity")
    ax1.set_title(f"(a)  Bragg mirror — (HL)$^{N_mirror}$H")
    ax1.legend()
    ax1.set_ylim(-0.02, 1.02)

    ax2.plot(wl_nm, sr_fp["T31"], label=r"T$_{pp}$", color="#d62728")
    ax2.plot(wl_nm, sr_fp["R11"], label=r"R$_{pp}$", color="#1f77b4", alpha=0.5)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Intensity")
    ax2.set_title(f"(b)  Fabry-Perot bandpass — (HL)$^{N_fp}$H 2L H(LH)$^{N_fp}$")
    ax2.legend()
    ax2.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
