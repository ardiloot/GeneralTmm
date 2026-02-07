"""Leaky Dyakonov surface plasmon polaritons at a metal-birefringent interface.

ZnSe prism | Ag (60 nm) | KTP crystal at 900 nm.  When the crystal optic
axis tilts past φ_c ≈ 67°, the extraordinary wave propagates and up to
~36 % of the incident p-polarized light tunnels through the metal film.

Reference: Loot & Hizhnyakov, Appl. Phys. A 122, 327 (2016).
"""

import matplotlib.pyplot as plt
import numpy as np

from GeneralTmm import Material, Tmm


def main():
    wl = 900e-9
    n_o, n_e = 1.740, 1.830  # KTP uniaxial approximation at 900 nm
    n_Ag = 0.040 + 6.371j  # Ag (Johnson & Christy interpolation)
    n_prism = 2.5  # ZnSe coupling prism
    d_metal = 60e-9

    # Critical OA angle
    eps_o, eps_e = n_o**2, n_e**2
    g = eps_e / eps_o - 1
    beta_spp = 1.815  # approximate SPP wavevector
    phi_c = np.degrees(np.arccos(np.sqrt((eps_e / beta_spp**2 - 1) / g)))

    prism = Material.Static(n_prism)
    metal = Material.Static(n_Ag)
    mat_x = Material.Static(n_o)
    mat_y = Material.Static(n_e)  # optic axis sits along y at ξ = 0
    mat_z = Material.Static(n_o)

    # θ grid for the Kretschmann scan (β = n_prism sin θ)
    thetas = np.linspace(43, 50, 1000)
    betas = n_prism * np.sin(np.radians(thetas))

    def make_tmm(phi_deg):
        tmm = Tmm(wl=wl, beta=betas[0])
        tmm.AddIsotropicLayer(float("inf"), prism)
        tmm.AddIsotropicLayer(d_metal, metal)
        tmm.AddLayer(float("inf"), mat_x, mat_y, mat_z, psi=0.0, xi=np.radians(phi_deg))
        return tmm

    # --- Panel (a): R_pp vs angle at selected OA angles ---
    phi_sel = [0, 60, 69, 90]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(
        r"Leaky Dyakonov SPP:  ZnSe / Ag (60 nm) / KTP  at  $\lambda$ = 900 nm",
        fontsize=13,
    )

    for phi_deg, c in zip(phi_sel, colors):
        sr = make_tmm(phi_deg).Sweep("beta", betas)
        lw = 1.8 if phi_deg == 69 else 1.0
        ax1.plot(thetas, sr["R11"], color=c, lw=lw, label=rf"$\varphi$ = {phi_deg}°")

    ax1.set_xlabel(r"Incident angle $\theta$ ($\degree$)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title("(a)  SPP dip in Kretschmann reflection")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=8)

    # --- Panel (b): R, T, A vs OA angle at SPP resonance ---
    phis = np.linspace(0, 90, 500)
    R_min = np.zeros(len(phis))
    T_pe = np.zeros(len(phis))
    A_res = np.zeros(len(phis))

    for i, phi_deg in enumerate(phis):
        sr = make_tmm(phi_deg).Sweep("beta", betas)
        idx = np.argmin(sr["R11"])
        R_min[i] = sr["R11"][idx]
        T_pe[i] = sr["T31"][idx]
        A_res[i] = 1.0 - sr["R11"][idx] - sr["R12"][idx] - sr["T31"][idx] - sr["T41"][idx]

    ax2.plot(phis, R_min, label=r"R$_{pp}$", color="#1f77b4")
    ax2.plot(phis, T_pe, label=r"T$_{pe}$  (extraordinary)", color="#d62728")
    ax2.plot(phis, A_res, label=r"A$_p$  (metal absorption)", color="#2ca02c", ls="--")
    ax2.axvline(phi_c, color="gray", ls=":", alpha=0.6, label=rf"$\varphi_c$ ≈ {phi_c:.0f}°")
    ax2.set_xlabel(r"Optic-axis angle $\varphi$ ($\degree$)")
    ax2.set_ylabel("Intensity")
    ax2.set_title("(b)  Leaky transition at SPP resonance")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
