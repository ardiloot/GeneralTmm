from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy
import glob
import re
import eigency

# Optimization flags
copt = {"msvc": ["/openmp", "/arch:SSE2", "/O2", "/Ot", "/MP"],
         "mingw32" : ["-O3", "-fopenmp"]}
lopt = {"mingw32" : ["-fopenmp"] }

class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        if c in lopt:
            for e in self.extensions:
                e.extra_link_args = lopt[c]
        build_ext.build_extensions(self)

# Parse version
__version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    open('GeneralTmm/__init__.py').read()).group(1)

sources = ["GeneralTmm/src/GeneralTmm.pyx"] + \
    glob.glob("GeneralTmm/src/CppTmm/CppTmm/*.cpp")
    
include_dirs = [r"GeneralTmm/src/CppTmm/CppTmm",
                r"GeneralTmm/src/eigen_3.2.4",
                r"GeneralTmm/src/Simplex",
                numpy.get_include()] + \
                eigency.get_includes(include_eigen = False)

ext = Extension("GeneralTmm._GeneralTmmCppExt",
    sources = sources,
    include_dirs = include_dirs,
    language = "c++")

setup(name = "GeneralTmm",
      version = __version__,
      author = "Ardi Loot",
      url = "https://github.com/ardiloot/GeneralTmm",
      author_email = "ardi.loot@outlook.com",
      packages = ["GeneralTmm"],
      cmdclass = {"build_ext": build_ext_subclass},
      ext_modules = [ext],)
