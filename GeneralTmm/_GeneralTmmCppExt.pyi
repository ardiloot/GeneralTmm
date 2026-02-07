"""Type stubs for the C++ extension module."""

from typing import Any

import numpy as np
import numpy.typing as npt

class Material:
    """Optical material defined by wavelength-dependent refractive index.

    Takes arrays of wavelengths and complex refractive indices and performs
    linear interpolation.  A shortcut ``mat(wl)`` is equivalent to
    ``mat.GetN(wl)``.

    Parameters
    ----------
    wls : ndarray of floats
        Array of wavelengths (m).
    ns : ndarray of complex floats
        Corresponding complex refractive indices.
    """

    def __init__(
        self,
        wls: npt.NDArray[np.floating[Any]],
        ns: npt.NDArray[np.complexfloating[Any, Any]],
    ) -> None: ...
    def __call__(self, wl: float) -> complex:
        """Shortcut for :meth:`GetN`."""
        ...
    def GetN(self, wl: float) -> complex:
        """Return the complex refractive index at the given wavelength.

        Parameters
        ----------
        wl : float
            Wavelength in meters.

        Returns
        -------
        complex
            Complex refractive index at *wl*.
        """
        ...

class _SweepRes:
    """Container for the results returned by :meth:`Tmm.Sweep`.

    Behaves like a read-only mapping.  Access individual result arrays by
    name, e.g. ``res["Rpp"]``.
    """

    def __getitem__(self, index: str) -> npt.NDArray[Any]: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self) -> Any: ...
    def keys(self) -> Any: ...

class Tmm:
    """4x4 anisotropic transfer-matrix method (TMM) solver.

    Supports both isotropic and anisotropic media.  Anisotropic layers are
    specified by three refractive indices (along each axis) and two angles
    denoting the alignment of crystallographic axes with the structure axes.

    This class wraps a C++ implementation based on:
    Hodgkinson, I. J., Kassam, S., & Wu, Q. H. (1997).
    *Journal of Computational Physics*, 133(1), 75–83.

    Parameters
    ----------
    **kwargs : dict
        All keyword arguments are forwarded to :meth:`SetParams`.

    Attributes
    ----------
    wl : float
        Wavelength in meters.
    beta : float
        Effective mode index (determines the angle of incidence).
    enhOptRel : float
        Relative-error termination condition for optimisation.
    enhOptMaxIters : int
        Maximum number of optimisation iterations.
    enhInitialStep : float
        Initial step size for optimisation.
    """

    def __init__(self, **kwargs: float | int | complex) -> None: ...
    def SetParams(self, **kwargs: float | int | complex) -> None:
        """Set one or more solver parameters at once.

        Parameters
        ----------
        **kwargs : dict
            Parameter name / value pairs.  Accepted names include ``"wl"``,
            ``"beta"``, ``"enhOptRel"``, ``"enhOptMaxIters"``, and
            ``"enhInitialStep"``.
        """
        ...
    def AddIsotropicLayer(self, d: float, material: Material) -> None:
        """Append an isotropic layer to the structure.

        Parameters
        ----------
        d : float
            Layer thickness in meters.  Use ``float("inf")`` for the first
            and last (semi-infinite) layers.
        material : Material
            Material describing the refractive-index dispersion.
        """
        ...
    def AddLayer(
        self,
        d: float,
        materialX: Material,
        materialY: Material,
        materialZ: Material,
        psi: float = ...,
        xi: float = ...,
    ) -> None:
        """Append an anisotropic layer to the structure.

        Parameters
        ----------
        d : float
            Layer thickness in meters.  Use ``float("inf")`` for semi-infinite
            layers.
        materialX : Material
            Refractive-index dispersion along the x-axis.
        materialY : Material
            Refractive-index dispersion along the y-axis.
        materialZ : Material
            Refractive-index dispersion along the z-axis.
        psi : float
            Rotation angle around the z-axis (radians).
        xi : float
            Rotation angle around the x-axis (radians).
        """
        ...
    def ClearLayers(self) -> None:
        """Remove all layers from the structure."""
        ...
    def GetIntensityMatrix(self) -> npt.NDArray[np.float64]:
        """Solve and return the 4x4 intensity matrix.

        The intensity matrix is defined by Eq. 20 in Hodgkinson *et al.*
        (1997) and contains all intensity reflection and transmission
        coefficients.

        Returns
        -------
        ndarray of float, shape (4, 4)
        """
        ...
    def GetAmplitudeMatrix(self) -> npt.NDArray[np.complex128]:
        """Solve and return the 4x4 amplitude matrix.

        The amplitude matrix is defined by Eq. 19 in Hodgkinson *et al.*
        (1997).

        Returns
        -------
        ndarray of complex, shape (4, 4)
        """
        ...
    def Sweep(
        self,
        paramName: str,
        values: npt.NDArray[np.floating[Any]],
        enhPos: tuple[tuple[float, float], int, float] | None = ...,
        alphaLayer: int = ...,
    ) -> _SweepRes:
        """Solve the system for multiple values of a single parameter.

        Parameters
        ----------
        paramName : str
            Parameter name (see :meth:`SetParams`).
        values : ndarray of floats
            Array of values to sweep over.
        enhPos : tuple, optional
            ``((polCoef1, polCoef2), layerNr, distance)`` specifying the
            position and polarisation for field-enhancement calculation.
            ``(1.0, 0.0)`` is p-polarisation; ``(0.0, 1.0)`` is
            s-polarisation.
        alphaLayer : int, optional
            If non-negative, compute the perpendicular wavevector component
            in this layer.

        Returns
        -------
        _SweepRes
            Mapping-like object containing all sweep results.
        """
        ...
    def CalcFields1D(
        self,
        xs: npt.NDArray[np.floating[Any]],
        polarization: npt.NDArray[np.floating[Any]],
        waveDirection: str = ...,
    ) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        """Calculate complex E and H fields along the x-axis.

        Parameters
        ----------
        xs : ndarray of floats
            Positions at which to evaluate the fields.
        polarization : ndarray of floats, shape (2,)
            Polarisation coefficients of the excitation.
        waveDirection : {"both", "forward", "backward"}
            Select which wave components to include.

        Returns
        -------
        E : ndarray of complex, shape (N, 3)
            Electric-field components (x, y, z) at each position.
        H : ndarray of complex, shape (N, 3)
            Magnetic-field components (x, y, z) at each position.
        """
        ...
    def CalcFields2D(
        self,
        xs: npt.NDArray[np.floating[Any]],
        ys: npt.NDArray[np.floating[Any]],
        polarization: npt.NDArray[np.floating[Any]],
        waveDirection: str = ...,
    ) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        """Calculate complex E and H fields on a 2-D xy grid.

        Parameters
        ----------
        xs : ndarray of floats
            Positions along the x-axis.
        ys : ndarray of floats
            Positions along the y-axis.
        polarization : ndarray of floats, shape (2,)
            Polarisation coefficients of the excitation.
        waveDirection : {"both", "forward", "backward"}
            Select which wave components to include.

        Returns
        -------
        E : ndarray of complex, shape (N, M, 3)
            Electric-field components at each grid point.
        H : ndarray of complex, shape (N, M, 3)
            Magnetic-field components at each grid point.
        """
        ...
    def CalcFieldsAtInterface(
        self,
        enhPos: tuple[tuple[float, float], int, float],
        waveDirection: str = ...,
    ) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        """Calculate E and H fields at a single interface point.

        Parameters
        ----------
        enhPos : tuple
            ``((polCoef1, polCoef2), layerNr, distance)`` — the layer and
            distance from the interface at which to evaluate fields, plus the
            polarisation used for the enhancement calculation.
        waveDirection : {"both", "forward", "backward"}
            Select which wave components to include.

        Returns
        -------
        E : ndarray of complex, shape (3,)
            Electric-field components (x, y, z).
        H : ndarray of complex, shape (3,)
            Magnetic-field components (x, y, z).
        """
        ...
    def OptimizeEnhancement(
        self,
        optParams: list[str],
        optInitials: npt.NDArray[np.floating[Any]],
        enhPos: tuple[tuple[float, float], int, float],
    ) -> float:
        """Optimise structure parameters for maximal field enhancement.

        See also :attr:`enhOptRel`, :attr:`enhOptMaxIters`, and
        :attr:`enhInitialStep`.

        Parameters
        ----------
        optParams : list of str
            Parameter names to optimise.
        optInitials : ndarray of floats
            Initial guesses for each parameter.
        enhPos : tuple
            ``((polCoef1, polCoef2), layerNr, distance)`` defining the
            evaluation point and polarisation.

        Returns
        -------
        float
            The optimised field enhancement.  After the call, all parameters
            are set to their optimal values.
        """
        ...
    @property
    def wl(self) -> float:
        """Wavelength in meters."""
        ...
    @wl.setter
    def wl(self, value: float) -> None: ...
    @property
    def beta(self) -> float:
        """Effective mode index (determines the angle of incidence)."""
        ...
    @beta.setter
    def beta(self, value: float) -> None: ...
    @property
    def enhOptRel(self) -> float:
        """Relative-error termination condition for optimisation."""
        ...
    @enhOptRel.setter
    def enhOptRel(self, value: float) -> None: ...
    @property
    def enhOptMaxIters(self) -> int:
        """Maximum number of optimisation iterations."""
        ...
    @enhOptMaxIters.setter
    def enhOptMaxIters(self, value: int) -> None: ...
    @property
    def enhInitialStep(self) -> float:
        """Initial step size for optimisation."""
        ...
    @enhInitialStep.setter
    def enhInitialStep(self, value: float) -> None: ...
