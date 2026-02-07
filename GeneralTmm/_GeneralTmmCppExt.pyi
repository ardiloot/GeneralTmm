"""Type stubs for the C++ extension module."""

from typing import Any

import numpy as np
import numpy.typing as npt

class Material:
    """Material class for optical parameters."""

    def __init__(
        self, wls: npt.NDArray[np.float64], ns: npt.NDArray[np.complex128]
    ) -> None: ...
    def __call__(self, wl: float) -> complex: ...
    def GetN(self, wl: float) -> complex: ...

class _SweepRes:
    """Helper class to hold the results of the Tmm.Sweep method."""

    def __getitem__(self, index: str) -> npt.NDArray[Any]: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self) -> Any: ...
    def keys(self) -> Any: ...

class Tmm:
    """Transfer matrix method solver."""

    def __init__(self, wl: float = ...) -> None: ...
    def AddIsotropicLayer(self, d: float, material: Material) -> None: ...
    def AddLayer(
        self,
        d: float,
        materialX: Material,
        materialY: Material,
        materialZ: Material,
        psi: float = ...,
        xi: float = ...,
    ) -> None: ...
    def Sweep(
        self,
        paramName: str,
        values: npt.NDArray[np.float64],
        enhPos: tuple[tuple[int, int], float, float] | None = ...,
        alphaLayer: int = ...,
    ) -> _SweepRes: ...
    def GetIntensityMatrix(self) -> npt.NDArray[np.float64]: ...
    def GetAmplitudeMatrix(self) -> npt.NDArray[np.complex128]: ...
    def CalcFields1D(
        self,
        xs: npt.NDArray[np.float64],
        polarization: npt.NDArray[np.float64],
        waveDirection: str = ...,
    ) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]: ...
    @property
    def wl(self) -> float: ...
    @wl.setter
    def wl(self, value: float) -> None: ...
    @property
    def beta(self) -> float: ...
    @beta.setter
    def beta(self, value: float) -> None: ...
