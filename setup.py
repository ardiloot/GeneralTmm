from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import glob
import re

# Optimization flags
copt = {"msvc": ["/O2",],
        "mingw32": ["-O3"],
        "unix": ["-std=c++11", "-O3"]}
lopt = {"mingw32" : [] }

class build_ext_subclass(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        #__builtins__.__NUMPY_SETUP__ = False
        import numpy
        import eigency
        self.include_dirs += [r"GeneralTmm/src/CppTmm/CppTmm",
                r"GeneralTmm/src/eigen_3.2.4",
                r"GeneralTmm/src/Simplex",
                numpy.get_include()] + \
                eigency.get_includes(include_eigen = False)
        
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

ext = Extension("GeneralTmm._GeneralTmmCppExt",
    sources = sources,
    language = "c++")

setup(name = "GeneralTmm",
      version = __version__,
      author = "Ardi Loot",
      url = "https://github.com/ardiloot/GeneralTmm",
      author_email = "ardi.loot@outlook.com",
      packages = ["GeneralTmm"],
      cmdclass = {"build_ext": build_ext_subclass},
      ext_modules = [ext],
      install_requires = ["numpy", "scipy", "eigency"],
)
