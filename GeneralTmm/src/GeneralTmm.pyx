import numpy as np
from libcpp.pair cimport pair
from CppGeneralTmm cimport *

#===============================================================================
# Common
#===============================================================================

cdef enum ParamDatatype:
    PARAM_DT_DOUBLE,
    PARAM_DT_INT,
    PARAM_DT_COMPLEX

cdef pair[ParamCpp, ParamDatatype] ToCppParam(str param) except +: 
    cdef int layerNr = -1
    if param.count("_") == 1:
        layerNr = int(param.split("_")[-1])
    elif param.count("_") > 1:
        raise ValueError("Unknown param: %s" % (param))

    if param == "wl":
        return pair[ParamCpp, ParamDatatype](ParamCpp(WL), PARAM_DT_DOUBLE)
    elif param == "beta":
        return pair[ParamCpp, ParamDatatype](ParamCpp(BETA), PARAM_DT_DOUBLE)
    elif param == "enhOptRel":
        return pair[ParamCpp, ParamDatatype](ParamCpp(ENH_OPT_REL), PARAM_DT_DOUBLE)
    elif param == "enhOptMaxIters":
        return pair[ParamCpp, ParamDatatype](ParamCpp(ENH_OPT_MAX_ITERS), PARAM_DT_INT)
    elif param == "enhInitialStep":
        return pair[ParamCpp, ParamDatatype](ParamCpp(ENH_INITIAL_STEP), PARAM_DT_DOUBLE)
    elif param.startswith("d_"):
        return pair[ParamCpp, ParamDatatype](ParamCpp(LAYER_D, layerNr), PARAM_DT_DOUBLE)
    elif param.startswith("n_"):
        return pair[ParamCpp, ParamDatatype](ParamCpp(LAYER_N, layerNr), PARAM_DT_COMPLEX)
    elif param.startswith("nx_"):
        return pair[ParamCpp, ParamDatatype](ParamCpp(LAYER_NX, layerNr), PARAM_DT_COMPLEX)
    elif param.startswith("ny_"):
        return pair[ParamCpp, ParamDatatype](ParamCpp(LAYER_NY, layerNr), PARAM_DT_COMPLEX)
    elif param.startswith("nz_"):
        return pair[ParamCpp, ParamDatatype](ParamCpp(LAYER_NZ, layerNr), PARAM_DT_COMPLEX)
    elif param.startswith("psi_"):
        return pair[ParamCpp, ParamDatatype](ParamCpp(LAYER_PSI, layerNr), PARAM_DT_DOUBLE)
    elif param.startswith("xi_"):
        return pair[ParamCpp, ParamDatatype](ParamCpp(LAYER_XI, layerNr), PARAM_DT_DOUBLE)
    else:
        raise ValueError("Unknown param: %s" % (param))

#===============================================================================
# Tmm
#===============================================================================

cdef class Tmm:
    cdef TmmCpp *_thisptr
    
    def __cinit__(self):
        self._thisptr = new TmmCpp()
        
    def __dealloc__(self):
        if self._thisptr is not NULL:
            del self._thisptr

    def SetParams(self, **kwargs):
        cdef pair[ParamCpp, ParamDatatype] paramDef
        for name, value in kwargs.items():
            paramDef = ToCppParam(name)
            if paramDef.second == PARAM_DT_DOUBLE:
                self._thisptr.SetParam(paramDef.first, <double>value)
            elif paramDef.second == PARAM_DT_INT:
                self._thisptr.SetParam(paramDef.first, <int>value)
            elif paramDef.second == PARAM_DT_COMPLEX:
                self._thisptr.SetParam(paramDef.first, <double complex>value)                
            else:
                raise ValueError("Unknown param datatype.")
        
    def AddIsotropicLayer(self, double d, double complex n):
        self._thisptr.AddIsotropicLayer(d, n)

    def AddLayer(self, double d, double complex nx, double complex ny, double complex nz, double psi, double xi):
        self._thisptr.AddLayer(d, nx, ny, nz, psi, xi)

    def ClearLayers(self):
        self._thisptr.ClearLayers()
        
    # Getters
    #--------------------------------------------------------------------------- 
    
    @property
    def wl(self):
        return self._thisptr.GetParamDouble(ParamCpp(WL))
    
    @property
    def beta(self):
        return self._thisptr.GetParamDouble(ParamCpp(BETA))
    
    @property
    def enhOptRel(self):
        return self._thisptr.GetParamDouble(ParamCpp(ENH_OPT_REL))
    
    @property
    def enhOptMaxIters(self):
        return self._thisptr.GetParamInt(ParamCpp(ENH_OPT_MAX_ITERS))
    
    @property
    def enhInitialStep(self):
        return self._thisptr.GetParamDouble(ParamCpp(ENH_INITIAL_STEP))
    
    # Setters
    #--------------------------------------------------------------------------- 
    
    @wl.setter
    def wl(self, double value):  # @DuplicatedSignature
        self._thisptr.SetParam(ParamCpp(WL), <double>value)
        
    @beta.setter
    def beta(self, double value):  # @DuplicatedSignature
        self._thisptr.SetParam(ParamCpp(BETA), <double>value)
    
    @enhOptRel.setter
    def enhOptRel(self, double value):  # @DuplicatedSignature
        self._thisptr.SetParam(ParamCpp(ENH_OPT_REL), <double>value)
    
    @enhOptMaxIters.setter
    def enhOptMaxIters(self, int value):  # @DuplicatedSignature
        self._thisptr.SetParam(ParamCpp(ENH_OPT_MAX_ITERS), <int>value)
    
    @enhInitialStep.setter
    def enhInitialStep(self, double value):  # @DuplicatedSignature
        self._thisptr.SetParam(ParamCpp(ENH_INITIAL_STEP), <double>value)
    
    