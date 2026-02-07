"""Generate README images by running each example and saving the figures."""

import importlib
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

# Add project root so examples can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Monkey-patch plt.show so examples don't block
plt.show = lambda *args, **kwargs: None

# (module path, list of output filenames — one per figure the example creates)
EXAMPLES = [
    ("Examples.ExampleTIR", ["tir_reflection"]),
    ("Examples.ExampleSPP", ["spp_reflection", "spp_fields_2d"]),
    ("Examples.ExampleFilter", ["filter_dielectric"]),
    ("Examples.ExampleAnisotropic", ["anisotropic_wave_plates"]),
    ("Examples.ExampleCholesteric", ["cholesteric_bragg"]),
    ("Examples.ExampleDSPP", ["dspp_leaky"]),
]


def run_and_save(module_path: str, names: list[str]) -> None:
    """Import an example, call its main(), and save all open figures."""
    mod = importlib.import_module(module_path)
    mod.main()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    figs = [plt.figure(n) for n in plt.get_fignums()]
    if len(figs) != len(names):
        raise RuntimeError(f"{module_path}: expected {len(names)} figure(s), got {len(figs)}")
    for fig, name in zip(figs, names):
        path = os.path.join(IMAGES_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  {path}")
    plt.close("all")


if __name__ == "__main__":
    print("Generating README images...")
    for module_path, names in EXAMPLES:
        print(f"  Running {module_path}...")
        run_and_save(module_path, names)
    print("Done.")
