from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from eigency.core cimport *

#===============================================================================
# Common.h
#===============================================================================
        
cdef extern from "Common.h" namespace "TmmModel":
    
    cdef enum ParamTypeCpp "TmmModel::ParamType":
        WL "TmmModel::ParamType::WL",
        BETA "TmmModel::ParamType::BETA",
        ENH_OPT_REL "TmmModel::ParamType::ENH_OPT_REL",
        ENH_OPT_MAX_ITERS "TmmModel::ParamType::ENH_OPT_MAX_ITERS",
        ENH_INITIAL_STEP "TmmModel::ParamType::ENH_INITIAL_STEP",
        LAYER_D "TmmModel::ParamType::LAYER_D",
        LAYER_N "TmmModel::ParamType::LAYER_N",
        LAYER_NX "TmmModel::ParamType::LAYER_NX",
        LAYER_NY "TmmModel::ParamType::LAYER_NY",
        LAYER_NZ "TmmModel::ParamType::LAYER_NZ",
        LAYER_PSI "TmmModel::ParamType::LAYER_PSI",
        LAYER_XI "TmmModel::ParamType::LAYER_XI",
        LAYER_MAT_NX "TmmModel::ParamType::LAYER_MAT_NX",
        LAYER_MAT_NY "TmmModel::ParamType::LAYER_MAT_NY",
        LAYER_MAT_NZ "TmmModel::ParamType::LAYER_MAT_NZ",
        NOT_DEFINED "TmmModel::ParamType::NOT_DEFINED"
    
    #---------------------------------------------------------------------------
        
    cdef enum WaveDirectionCpp "TmmModel::WaveDirection":
        WD_FORWARD "TmmModel::WaveDirection::WD_FORWARD",
        WD_BACKWARD "TmmModel::WaveDirection::WD_BACKWARD",
        WD_BOTH "TmmModel::WaveDirection::WD_BOTH"
    
    #---------------------------------------------------------------------------
    
    cdef cppclass ParamCpp "TmmModel::Param":
            ParamCpp() except +
            ParamCpp(ParamTypeCpp pType) except +
            ParamCpp(ParamTypeCpp pType, int layerId) except +
    
    #--------------------------------------------------------------------------- 
    
    cdef cppclass PositionSettingsCpp "TmmModel::PositionSettings":
        PositionSettingsCpp() except +
        PositionSettingsCpp(double polCoef1, double polCoef2, int interfaceId, double distFromInterface) except +
            
    #--------------------------------------------------------------------------- 

    cdef cppclass SweepResCpp "TmmModel::SweepRes":
        map[string, ArrayXcd] mapComplex
        map[string, ArrayXd] mapDouble
    
    #---------------------------------------------------------------------------
    
    cdef cppclass EMFieldsCpp "TmmModel::EMFields":
        ArrayXcd E
        ArrayXcd H
    
    #---------------------------------------------------------------------------
    
    cdef cppclass EMFieldsListCpp "TmmModel::EMFieldsList":
        MatrixXcd E
        MatrixXcd H
        
#===============================================================================
# Material.h
#===============================================================================
        
cdef extern from "Material.h" namespace "TmmModel":        
    
    cdef cppclass MaterialCpp "TmmModel::Material":
        MaterialCpp() except +
        MaterialCpp(Map[ArrayXd] & wlsExp, Map[ArrayXcd] & nsExp) except +
        double complex n(double wl) except +
            
#===============================================================================
# tmm.h
#===============================================================================
        
cdef extern from "tmm.h" namespace "TmmModel":

    cdef cppclass TmmCpp "TmmModel::Tmm":
        Tmm() except +
        
        void SetParam(ParamCpp ParamCpp, int value) except +
        void SetParam(ParamCpp ParamCpp, double value) except +
        void SetParam(ParamCpp ParamCpp, double complex value) except +
        
        int GetParamInt(ParamCpp ParamCpp) except +
        double GetParamDouble(ParamCpp ParamCpp) except + 
        double complex GetParamComplex(ParamCpp ParamCpp) except +
        
        void AddIsotropicLayer(double d, MaterialCpp *mat) except +
        void AddLayer(double d, MaterialCpp *matx, MaterialCpp *maty, MaterialCpp *matz, double psi, double xi) except +
        void ClearLayers() except +
        
        Matrix4d GetIntensityMatrix() except +
        Matrix4cd GetAmplitudeMatrix() except +
        SweepResCpp Sweep(ParamCpp sweepParamCpp, Map[ArrayXd] sweepValues, PositionSettingsCpp enhpos, int alphasLayer) except +
        
        EMFieldsListCpp CalcFields1D(Map[ArrayXd] xs, Map[Array2d] polarization, WaveDirectionCpp waveDirection) except +
        EMFieldsCpp CalcFieldsAtInterface(PositionSettingsCpp pos, WaveDirectionCpp waveDirection) except +
        double OptimizeEnhancement(vector[ParamCpp] optParams, Map[ArrayXd] optInitial, PositionSettingsCpp pos) except +
        

