import numpy as np
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from CppGeneralTmm cimport *

#===============================================================================
# Common
#===============================================================================

cdef enum ParamDatatype:
    PARAM_DT_DOUBLE,
    PARAM_DT_INT,
    PARAM_DT_COMPLEX

cdef pair[ParamCpp, ParamDatatype] ToCppParam(str param) except +:
    """ToCppParam(param)

    Converts the name of the param (e.g wl, beta, ...) to C++ param class.

    Parameters
    ----------
    param : str
        The name of the parameter.

    Returns
    -------
    tuple of ParamCpp and ParamDatatype
        C++ class describing the parameter and its datatype

    """
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
    """WaveDirectionFromStr(waveStr)

    Helper function to convert wave direction string to WaveDirectionCpp enum.

    Parameters
    ----------
    waveStr : str
        Wave direction string (forward, backward or both)

    Returns
    -------
    WaveDirectionCpp
        Enum indicating the direction.

    """
    if waveStr == "forward":
        return WD_FORWARD
    elif waveStr == "backward":
        return WD_BACKWARD
    elif waveStr == "both":
        return WD_BOTH
    else:
        raise NotImplementedError()


cdef class _SweepRes:
    """_SweepRes()

    Helper class to hold the results of the :any:`Tmm.Sweep` method. The
    __getitem__ method is overloaded to return the results stored in the
    internal dictionary. For full details of the calculated quantities see the
    publication Hodgkinson, I. J., Kassam, S., & Wu, Q. H. (1997) Journal of
    Computational Physics, 133(1) 75-83

    Attributes
    ----------
    ["R11"] : ndarray of floats
        The intensity reflection coefficient of p-polarized wave (1 == p).
    ["R22"] : ndarray of floats
        The intensity reflection coefficient of s-polarized wave (2 == s).
    ["R12"] : ndarray of floats
        The anisotropic reflection intensity coefficient. Incident light is
        s-polarized (2) and the measured field is p-polarized(1).
    ["R21"] : ndarray of floats
        The anisotropic reflection intensity coefficient. Incident light is
        p-polarized (1) and the measured field is s-polarized(2).
    ["T31"] : ndarray of floats
        The intensity transmission coefficient of p-polarized wave (1 & 3 == p).
    ["T42"] : ndarray of floats
        The intensity transmission coefficient of s-polarized wave (2 & 4 == s).
    ["T32"] : ndarray of floats
        The anisotropic transmission intensity coefficient. Incident light is
        s-polarized (2) and the measured field is p-polarized (3).
    ["T41"] : ndarray of floats
        The anisotropic transmission intensity coefficient. Incident light is
        p-polarized (1) and the measured field is s-polarized (4).
    ["rXY"] : ndarray of complex
        The amplitude reflection coefficients similarly to intensity coefficients.
        The X denotes measured field polarization (1..4) and Y incident
        polarization (1..4)
    ["tXY"] : ndarray of complex
        The amplitude transmission coefficients similarly to intensity coefficients.
        The X denotes measured field polarization (1..4) and Y incident
        polarization (1..4)
    ["enh"] : ndarray of floats
        The enhancement of electric field norm in specified position in
        comparison to the incident field norm in vacuum.
    ["enhEx"] : ndarray of floats
        The enhancement of electric field x-component amplitude in
        specified position in comparison to the incident field norm in vacuum.
    ["enhEy"] : ndarray of floats
        The enhancement of electric field y-component amplitude in
        specified position in comparison to the incident field norm in vacuum.
    ["enhEz"] : ndarray of floats
        The enhancement of electric field z-component amplitude in
        specified position in comparison to the incident field norm in vacuum.

    """
    cdef dict _res;

    def __init__(self):
        self._res = {}

    cdef _Init(self, SweepResCpp resCpp):
        cdef map[string, ArrayXcd] mapComplex = resCpp.mapComplex
        cdef map[string, ArrayXd] mapDouble = resCpp.mapDouble

        for kvPair in mapComplex:
            self._res[kvPair.first.decode()] = ndarray_copy(kvPair.second).squeeze()
        for kvPair2 in mapDouble:
            self._res[kvPair2.first.decode()] = ndarray_copy(kvPair2.second).squeeze()

    def __getitem__(self, index):
        return self._res[index]

    def __contains__(self, key):
        return key in self._res

    def __iter__(self):
        return iter(self._res)

    def keys(self):
        return self._res.keys()

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self._res)


#===============================================================================
# Material
#===============================================================================

cdef class Material:
    """Material(wls, ns)

    This class describes the optical parameters of medium. Default
    constructor takes arrays of wavelengths and complex refractive indices and
    does linear interpolation. For shortcut __call__(wl) is defined as GetN(wl).

    Parameters
    ----------
    wls : ndarray of floats
        Array of wavelengths (m)
    ns : ndarray of complex floats
        Corresponding complex refractive indices to the wls array.

    """
    cdef MaterialCpp *_thisptr
    cdef readonly object chi2

    def __cinit__(self, np.ndarray[double, ndim = 1] wls, np.ndarray[double complex, ndim = 1] ns):
        # Copy is made in c++
        self._thisptr = new MaterialCpp(Map[ArrayXd](wls), Map[ArrayXcd](ns))

    def __dealloc__(self):
        del self._thisptr

    def __call__(self, wl): # For compatibility
        return self.GetN(wl)

    def GetN(self, wl):
        """GetN(wl)

        Returns refractive index of material at specified wavelength.

        Parameters
        ----------
        wl : float
            Wavelength in meters.

        Returns
        -------
        complex
            Complex refractive index at wavelength wl.

        Examples
        --------
        >>> wls = np.array([400e-9, 600e-9, 800e-9], dtype = float)
        >>> ns = np.array([1.5 + 0.3j, 1.7 + 0.2j, 1.8 + 0.1j], dtype = complex)
        >>> mat = Material(wls, ns)
        >>> mat(600e-9)
        1.7 + 0.2j

        """
        return self._thisptr.n(wl)


#===============================================================================
# Tmm
#===============================================================================

cdef class Tmm:
    """Tmm(**kwargs)

    This is main class for 4x4 matrix anisotropic transfer-matrix method (TMM).
    Allows to calculate both isotropic and anisotropic mediums. In case of
    anisotropic mediums the layer is specified by the thickness, by three
    refractive indices (along each axis) and by two angles denoting the alignment
    of crystallographic axes with structure axes. This class is wrapper for the
    code written in C++. This implementation is based on the publication
    Hodgkinson, I. J., Kassam, S., & Wu, Q. H. (1997). Journal of Computational
    Physics, 133(1) 75-83.

    Parameters
    ----------
    **kwargs : dict
        All parameters are passed to :any:`Tmm.SetParams`.

    Attributes
    ----------
    wl : float
        Wavelength in meters.
    beta : float
        Effective mode index. Determines the angle of incidence.
    enhOptRel : float
        The relative error termination condition for optimization.
    enhOptMaxIters : int
        Maximum number of optimization iterations.
    enhInitialStep : float
        Initial step for optimization.

    """
    cdef TmmCpp *_thisptr
    cdef readonly list materialsCache

    def __cinit__(self, **kwargs):
        self._thisptr = new TmmCpp()
        self.materialsCache = []
        self.SetParams(**kwargs)

    def __dealloc__(self):
        if self._thisptr is not NULL:
            del self._thisptr

    def SetParams(self, **kwargs):
        """SetParams(**kwargs)

        Convenience function to set multiple parameters at once. For the list
        of parameters see function :any:`ToCppParam`

        Parameters
        ----------
        **kwargs : dict
            Param and value pairs to set.

        """
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

    def AddIsotropicLayer(self, double d, Material mat):
        """AddIsotropicLayer(d, mat)

        Adds isotropic layer to transfer-matrix method.

        Parameters
        ----------
        d : float
            The thickness of the layer. First and last layers should have
            thickness equal to float("inf").
        mat : :any:`Material`
            Instance of the :any:`Material` class describing the refractive
            index dispersion over wavelength.

        """
        self._thisptr.AddIsotropicLayer(d, mat._thisptr)
        self.materialsCache.append(mat) # Avoids dealloc

    def AddLayer(self, double d, Material matx, Material maty, Material matz, double psi, double xi):
        """AddLayer(d, mat)

        Adds anisotropic layer to transfer-matrix method.

        Parameters
        ----------
        d : float
            The thickness of the layer. First and last layers should have
            thickness equal to float("inf").
        matx : :any:`Material`
            Instance of the :any:`Material` class describing the refractive
            index dispersion over wavelength in x-direction.
        maty : :any:`Material`
            Instance of the :any:`Material` class describing the refractive
            index dispersion over wavelength in y-direction.
        matz : :any:`Material`
            Instance of the :any:`Material` class describing the refractive
            index dispersion over wavelength in z-direction.
        psi : float
            The rotational angle around the z-axis for alignment of the material.
            Measured in radians.
        xi : float
            The rotational angle around the x-axis for alignment of the material.
            Measured in radians.

        """
        self._thisptr.AddLayer(d, matx._thisptr, maty._thisptr, matz._thisptr, psi, xi)
        self.materialsCache.append(matx) # Avoids dealloc
        self.materialsCache.append(maty) # Avoids dealloc
        self.materialsCache.append(matz) # Avoids dealloc

    def ClearLayers(self):
        """ClearLayers()

        Clears all the layers from the structure.

        """

        self.materialsCache.clear()
        self._thisptr.ClearLayers()

    def GetIntensityMatrix(self):
        """GetIntensityMatrix()

        Solves the system and returns the intensity matrix defined by Eq. 20
        in I. J. Hodgkinson, et al (1997). Contains all intensity reflection and
        transmission coefficients. See also :any:`_SweepRes`.

        Returns
        -------
        ndarray of floats
            Has shape (4, 4) and contains the intensity matrix coefficients.

        """
        return ndarray_copy(self._thisptr.GetIntensityMatrix())

    def GetAmplitudeMatrix(self):
        """GetAmplitudeMatrix()

        Solves the system and returns the amplitude matrix defined by Eq. 19
        in I. J. Hodgkinson, et al (1997). See also :any:`_SweepRes`.

        Returns
        -------
        ndarray of complex floats
            Has shape (4, 4) and contains the amplitude matrix coefficients.

        """
        return ndarray_copy(self._thisptr.GetAmplitudeMatrix())

    def Sweep(self, paramName, np.ndarray[double, ndim = 1] values, enhPos = None, int alphaLayer = -1):
        """Sweep(paramName, values, enhPos = None, alphaLayer = -1)

        Convenience function for solving system for multiple values of a single
        parameter (given by :any:`paramName`).

        Parameters
        ----------
        paramName : str
            The name of parameter. See :any:`Tmm.SetParams` for the list of all
            possible parameter names.
        values : ndarray of floats
            The list of values for parameter to solve for.
        enhPos : tuple
            Tuple structured like ((polCoef1, polCoef2), layerNr, distance).
            If this parameter is not None, then the field enhancement will be
            calculated in layer numbered by `layerNr` at distance `distance`
            from the interface. The variables (polCoef1, polCoef2) define the
            polarization for the field enhancement calculation (in comparison to
            the excitation in vacuum) and (1.0, 0.0) corresponds to p-polarization
            and (0.0, 1.0) corresponds to s-polarization.
        alphaLayer: int
            Computes the perpendicular component of the wavevectors in layer
            nr `alphaLayer`.

        Returns
        -------
        :any:`_SweepRes`
            Helper class containing all the results of the sweep.

        """
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
        """CalcFields1D(xs, polarization, waveDirection = "both")

        Calculates the complex electric and magnetic fields along the x-axis.

        Parameters
        ----------
        xs : ndarray of floats
            Position array where to calculate the fields.
        polarization : ndarray of floats (2,)
            Contains the polarization coefficients of the excitation.
        waveDirection : str {"both", "forward", "backward"}
            Allows to select the output fields.

        Returns
        -------
        tuple (E, H) of ndarrays of complex floats
           Variable E contains the electric fields and has shape (N, 3),
           where N is the length of `xs` array and 3 correspond to x-, y- and
           z-direction. Variable H contains magnetic fields in similar manner.

        """
        cdef WaveDirectionCpp waveDirection = WaveDirectionFromStr(waveDirectionStr)
        cdef EMFieldsListCpp resCpp
        resCpp = self._thisptr.CalcFields1D(Map[ArrayXd](xs), Map[Array2d](polarization), waveDirection)

        E = ndarray_copy(resCpp.E)
        H = ndarray_copy(resCpp.H)
        return E, H

    def CalcFields2D(self, np.ndarray[double, ndim = 1] xs, np.ndarray[double, ndim = 1] ys, np.ndarray[double, ndim = 1] pol, str waveDirectionStr = "both"):
        """CalcFields2D(xs, ys, polarization, waveDirection = "both")

        Calculates the complex electric and magnetic fields along in xy-plane in
        the rectangular grid following from `xs` and `ys`.

        Parameters
        ----------
        xs : ndarray of floats
            Position array where to calculate the fields in x-direction.
        ys : ndarray of floats
            Position array where to calculate the fields in y-direction.
        polarization : ndarray of floats (2,)
            Contains the polarization coefficients of the excitation.
        waveDirection : str {"both", "forward", "backward"}
            Allows to select the output fields.

        Returns
        -------
        tuple (E, H) of ndarrays of complex floats
           Variable E contains the electric fields and has shape (N, M, 3),
           where N is the length of `xs` array, M is the length of `ys`
           array and 3 correspond to x-, y- and z-direction. Variable H
           contains magnetic fields in similar manner.

        """
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
        """CalcFieldsAtInterface(enhPos, waveDirectionStr = "both")

        Calculates electric and magnetic fields in one point defined by
        `enhPos`.

        Parameters
        ----------
        enhPos : tuple
            Tuple structured like ((polCoef1, polCoef2), layerNr, distance).
            If this parameter is not None, then the field enhancement will be
            calculated in layer numbered by `layerNr` at distance `distance`
            from the interface. The variables (polCoef1, polCoef2) define the
            polarization for the field enhancement calculation (in comparison to
            the excitation in vacuum) and (1.0, 0.0) corresponds to p-polarization
            and (0.0, 1.0) corresponds to s-polarization.
        waveDirection : str {"both", "forward", "backward"}
            Allows to select the output fields.

        Returns
        -------
        tuple (E, H) of ndarrays of complex
            Variable E is ndarray of complex floats (length 3) containing
            field x-, y- and z-component. Variable H contains magnetic fields
            in similar manner.

        """
        cdef WaveDirectionCpp waveDirection = WaveDirectionFromStr(waveDirectionStr)
        cdef PositionSettingsCpp enhPosCpp
        cdef EMFieldsCpp resCpp

        enhPosCpp = PositionSettingsCpp(enhPos[0][0], enhPos[0][1], enhPos[1], enhPos[2])
        resCpp = self._thisptr.CalcFieldsAtInterface(enhPosCpp, waveDirection)

        E = ndarray_copy(resCpp.E).squeeze()
        H = ndarray_copy(resCpp.H).squeeze()
        return E, H

    def OptimizeEnhancement(self, list optParams, np.ndarray[double, ndim = 1] optInitials, enhpos):
        """OptimizeEnhancement(optParams, optInitials, enhpos)

        Function for optimizing structure params for maximal field enhancement in
        one point (defined by enhpos). See also parameters for optimization:
        `enhOptRel`, `enhOptMaxIters` and `enhInitialStep`.

        Parameters
        ----------
        optParams: list of str
            List of parameter names to optimize.
        optInitials: ndarray of float
            Contains initial guesses for parameters in `optParams`.
        enhPos : tuple
            Tuple structured like ((polCoef1, polCoef2), layerNr, distance).
            The field enhancement will be calculated in layer numbered by
            `layerNr` at distance `distance` from the interface. The variables
            (polCoef1, polCoef2) define the polarization for the field
            enhancement calculation (in comparison to the excitation in vacuum):
            (1.0, 0.0) corresponds to p-polarization and (0.0, 1.0) corresponds to
            s-polarization.

        Returns
        -------
        float
            The optimized field enhancement. All the parameters will be assigned
            their optimal values and could be thus accessed in usual way.

        """
        cdef PositionSettingsCpp enhPosCpp
        cdef vector[ParamCpp] paramVec
        (pol, interface, dist) = enhpos
        enhPosCpp = PositionSettingsCpp(pol[0], pol[1], interface, dist)

        for paramName in optParams:
            paramVec.push_back(ToCppParam(paramName).first)

        res = self._thisptr.OptimizeEnhancement(paramVec, Map[ArrayXd](optInitials), enhPosCpp)
        return res


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
