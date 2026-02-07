"""Generate README images by running each example and saving the figures."""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

# Add project root so examples can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def save_open_figures(prefix):
    """Save all open matplotlib figures and close them."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    figs = [plt.figure(n) for n in plt.get_fignums()]
    if len(figs) == 1:
        path = os.path.join(IMAGES_DIR, f"{prefix}.png")
        figs[0].savefig(path, dpi=150, bbox_inches="tight")
        print(f"  {path}")
    else:
        for i, fig in enumerate(figs, 1):
            path = os.path.join(IMAGES_DIR, f"{prefix}_{i}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  {path}")
    plt.close("all")


# Monkey-patch plt.show so examples don't block
plt.show = lambda *args, **kwargs: None

print("Generating README images...")

from Examples.ExampleTIR import main as tir_main  # noqa: E402

tir_main()
save_open_figures("tir_reflection")

from Examples.ExampleSPP import main  # noqa: E402

main()
figs = [plt.figure(n) for n in plt.get_fignums()]
os.makedirs(IMAGES_DIR, exist_ok=True)
for name, fig in zip(["spp_reflection", "spp_fields_1d", "spp_fields_2d"], figs):
    path = os.path.join(IMAGES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  {path}")
plt.close("all")

from Examples.ExampleAnisotropic import main as aniso_main  # noqa: E402

aniso_main()
save_open_figures("anisotropic_wave_plates")

print("Done.")
