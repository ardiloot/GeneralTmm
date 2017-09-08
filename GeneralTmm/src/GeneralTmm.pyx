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

cdef WaveDirectionCpp WaveDirectionFromStr(str waveStr) except +:
    if waveStr == "forward":
        return WD_FORWARD
    elif waveStr == "backward":
        return WD_BACKWARD
    elif waveStr == "both":
        return WD_BOTH
    else:
        raise NotImplementedError()


cdef class _SweepRes:
    cdef dict _res;

    def __init__(self):
        self._res = {}
    
    cdef _Init(self, SweepResCpp resCpp):
        cdef map[string, ArrayXcd] mapComplex = resCpp.GetComplexMap()
        cdef map[string, ArrayXd] mapDouble = resCpp.GetDoubleMap()
        
        for kvPair in mapComplex:
            self._res[kvPair.first.decode()] = ndarray_copy(kvPair.second).squeeze()
        for kvPair2 in mapDouble:
            self._res[kvPair2.first.decode()] = ndarray_copy(kvPair2.second).squeeze() 

    def __getitem__(self, index):
        return self._res[index]
    
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self._res)


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

    def GetIntensityMatrix(self):
        return ndarray_copy(self._thisptr.GetIntensityMatrix())
    
    def GetAmplitudeMatrix(self):
        return ndarray_copy(self._thisptr.GetAmplitudeMatrix())
    
    def Sweep(self, paramName, np.ndarray[double, ndim = 1] values, enhPos = None, int alphaLayer = -1):
        cdef pair[ParamCpp, ParamDatatype] paramDef = ToCppParam(paramName)
        cdef PositionSettingsCpp enhPosCpp
        cdef SweepResCpp resCpp
        
        if enhPos is not None:
            enhPosCpp = PositionSettingsCpp(enhPos[0][0], enhPos[0][1], enhPos[1], enhPos[2])
        
        resCpp = self._thisptr.Sweep(paramDef.first, Map[ArrayXd](values), enhPosCpp, <int>alphaLayer)
        res = _SweepRes()
        res._Init(resCpp)
        return res
    
    def CalcFields1D(self, np.ndarray[double, ndim = 1] xs, np.ndarray[double, ndim = 1] polarization, str waveDirectionStr = "both"):
        cdef WaveDirectionCpp waveDirection = WaveDirectionFromStr(waveDirectionStr) 
        cdef EMFieldsListCpp resCpp        
        resCpp = self._thisptr.CalcFields1D(Map[ArrayXd](xs), Map[Array2d](polarization), waveDirection)

        E = ndarray_copy(resCpp.E)
        H = ndarray_copy(resCpp.H)
        return E, H
    
    def CalcFields2D(self, np.ndarray[double, ndim = 1] xs, np.ndarray[double, ndim = 1] ys, np.ndarray[double, ndim = 1] pol, str waveDirectionStr = "both"):
        ky = self.beta * 2.0 * np.pi / self.wl
        phaseY = np.exp(1.0j * ky * ys)
        E1D, H1D = self.CalcFields1D(xs, pol, waveDirectionStr)
        
        E = np.zeros((len(xs), len(ys), 3), dtype = complex)
        H = np.zeros((len(xs), len(ys), 3), dtype = complex)
        for i in range(3):
            E[:, :, i] = np.outer(E1D[:, i], phaseY)
            H[:, :, i] = np.outer(H1D[:, i], phaseY)
        
        return E, H
        
    def CalcFieldsAtInterface(self, enhPos, str waveDirectionStr = "both"):
        cdef WaveDirectionCpp waveDirection = WaveDirectionFromStr(waveDirectionStr)
        cdef PositionSettingsCpp enhPosCpp
        cdef EMFieldsCpp resCpp

        enhPosCpp = PositionSettingsCpp(enhPos[0][0], enhPos[0][1], enhPos[1], enhPos[2])
        resCpp = self._thisptr.CalcFieldsAtInterface(enhPosCpp, waveDirection)
        
        E = ndarray_copy(resCpp.E).squeeze()
        H = ndarray_copy(resCpp.H).squeeze()
        return E, H
        
        
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
    
    