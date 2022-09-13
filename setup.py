import re
import sys
import eigency
import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


__version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    open("GeneralTmm/__init__.py").read()).group(1)

extra_compile_args = []
if sys.platform in ("linux", "darwin"):
    extra_compile_args.append("-std=c++11")

extensions = cythonize([
    Extension(
        "GeneralTmm._GeneralTmmCppExt",
        sources=[
            "GeneralTmm/src/GeneralTmm.pyx",
            "GeneralTmm/src/Common.cpp",
            "GeneralTmm/src/Layer.cpp",
            "GeneralTmm/src/Material.cpp",
            "GeneralTmm/src/tmm.cpp",
        ],
        include_dirs=[
            np.get_include(),
            "GeneralTmm/src",
            "GeneralTmm/src/Simplex"
        ] + eigency.get_includes(),
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
])

setup(
    name="GeneralTmm",
    version=__version__,
    author="Ardi Loot",
    url="https://github.com/ardiloot/GeneralTmm",
    author_email="ardi.loot@outlook.com",
    packages=["GeneralTmm"],
    include_package_data=True,
    ext_modules=extensions,
    install_requires=["numpy", "scipy", "eigency"],
)
