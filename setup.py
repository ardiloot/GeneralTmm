import sys

import eigency
import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

extra_compile_args = []
if sys.platform in ("linux", "darwin"):
    extra_compile_args.append("-std=c++17")
elif sys.platform == "win32":
    extra_compile_args.append("/std:c++17")

extensions = cythonize(
    [
        Extension(
            "GeneralTmm._GeneralTmmCppExt",
            sources=[
                "GeneralTmm/src/GeneralTmm.pyx",
                "GeneralTmm/src/Common.cpp",
                "GeneralTmm/src/Layer.cpp",
                "GeneralTmm/src/Material.cpp",
                "GeneralTmm/src/tmm.cpp",
            ],
            include_dirs=[np.get_include(), "GeneralTmm/src", "GeneralTmm/src/Simplex"] + eigency.get_includes(),
            language="c++",
            extra_compile_args=extra_compile_args,
        ),
    ]
)

setup(
    ext_modules=extensions,
)
