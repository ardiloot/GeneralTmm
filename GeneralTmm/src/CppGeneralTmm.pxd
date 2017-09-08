from libcpp cimport bool
from eigency.core cimport *

#===============================================================================
# tmm.h
#===============================================================================
        
cdef extern from "tmm.h" namespace "TmmModel":
    
    cdef enum ParamTypeCpp "TmmModel::ParamType":
        WL,
        BETA,
        ENH_OPT_REL,
        ENH_OPT_MAX_ITERS,
        ENH_INITIAL_STEP,
        LAYER_D,
        LAYER_N,
        LAYER_NX,
        LAYER_NY,
        LAYER_NZ,
        LAYER_PSI,
        LAYER_XI,
        LAYER_MAT_NX,
        LAYER_MAT_NY,
        LAYER_MAT_NZ,
        NOT_DEFINED
    
    #---------------------------------------------------------------------------
    
    cdef cppclass ParamCpp "TmmModel::Param":
            ParamCpp() except +
            ParamCpp(ParamTypeCpp pType) except +
            ParamCpp(ParamTypeCpp pType_, int layerId_) except +
            #ParamTypeCpp GetParamType() except +
            #int GetLayerID() except +
    
    #--------------------------------------------------------------------------- 

    cdef cppclass TmmCpp "TmmModel::Tmm":
        Tmm() except +
        
        void SetParam(ParamCpp ParamCpp, int value) except +
        void SetParam(ParamCpp ParamCpp, double value) except +
        void SetParam(ParamCpp ParamCpp, double complex value) except +
        
        int GetParamInt(ParamCpp ParamCpp) except +
        double GetParamDouble(ParamCpp ParamCpp) except + 
        double complex GetParamComplex(ParamCpp ParamCpp) except +
        
        void AddIsotropicLayer(double d, double complex n) except +
        #//void AddIsotropicLayer(double d, boost::python::object &materialClass) except +
        void AddLayer(double d, double complex nx, double complex ny, double complex nz, double psi, double xi) except +
        #//void AddLayer(double d, boost::python::object &matX, boost::python::object &matY, boost::python::object &matZ, double psi, double xi) except +
        void ClearLayers() except +
        
        #Matrix4d GetIntensityMatrix() except +
        #Matrix4cd GetAmplitudeMatrix() except +
        #SweepRes Sweep(ParamCpp sweepParamCpp, VectorXd sweepValues, PositionSettings enhpos, int alphasLayer) except +
        #SweepRes Sweep(ParamCpp sweepParamCpp, VectorXd sweepValues) except +
        #EMFieldsList CalcFields1D(VectorXd xs, VectorXd polarization, WaveDirection waveDirection) except +
        #EMFields CalcFieldsAtInterface(PositionSettings pos, WaveDirection waveDirection) except +
        #double OptimizeEnhancement(vector<ParamCpp> optParamCpps, VectorXd optInitial, PositionSettings pos) except +
        #//double OptimizeEnhancementPython(boost::python::list optParams, VectorXd optInitial, PositionSettings pos) except +
