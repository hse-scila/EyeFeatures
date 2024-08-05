import pandas as pd
import numpy as np
from typing import List, Literal, Union, Tuple

from scipy.signal import savgol_filter, iirfilter, firwin, lfilter
from eyetracking.preprocessing.base import BaseSmoothingPreprocessor


# ======== SMOOTHING PREPROCESSORS ========
class SavGolFilter(BaseSmoothingPreprocessor):
    """
    Savitzkiy-Golay filter. 'x' and 'y' directions are filtered independently, time is ignored.
    Parameters are passed to `scipy.signal.savgol_filter`.
    """
    def __init__(
        self,
        x: str,
        y: str,
        t: str = None,
        pk: List[str] = None,
        window_length: int = 11,
        polyorder: int = 2,
        **savgol_kw
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.wl = window_length
        self.po = polyorder
        self.savgol_kw = savgol_kw

    def _check_params(self):
        m = "Savitzkiy-Golay filter"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")

    def _preprocess(self, X: pd.DataFrame):
        # TODO 2D SG filter: https://github.com/espdev/sgolay2
        if self.pk:
            X = X.drop(self.pk, axis=1)

        X_filt = X.copy()
        if len(X_filt) >= self.wl:
            X_filt[self.x] = savgol_filter(x=X[self.x].values, window_length=self.wl, polyorder=self.po,
                                           **self.savgol_kw)
            X_filt[self.y] = savgol_filter(x=X[self.y].values, window_length=self.wl, polyorder=self.po,
                                           **self.savgol_kw)
        return X_filt


class FIRFilter(BaseSmoothingPreprocessor):
    """
    FIR filter. Convolution with RIR kernel along 'x' and 'y'. `kwargs` are passed to `scipy.signal.firwin`
    to determine the kernel.
    :param mode: parameter of `scipy.signal.fftconvolve`.

    Default values are taken from https://arxiv.org/pdf/2303.02134.
    """
    def __init__(
        self,
        x: str,
        y: str,
        t: str = None,
        pk: List[str] = None,
        numtaps: int = 81,
        fs: int = 250,
        cutoff: Union[float, Tuple[float, ...]] = 100,
        pass_zero: Literal[False, True, "bandpass", "lowpass", "highpass", "bandstop"] = False,
        mode: Literal["valid", "full", "same"] = "valid",
        **fir_kw
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.mode = mode
        self.numtaps = numtaps
        self.cutoff = cutoff
        self.pass_zero = pass_zero
        self.fs = fs
        self.fir_kw = fir_kw

    def _check_params(self):
        m = "FIR filter"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")

    def _preprocess(self, X: pd.DataFrame):
        if self.pk:
            X = X.drop(self.pk, axis=1)

        X_filt = X.copy()
        if len(X) > self.numtaps:
            kernel = firwin(numtaps=self.numtaps, cutoff=self.cutoff, pass_zero=self.pass_zero, fs=self.fs, **self.fir_kw)
            x_filt = np.convolve(X[self.x].values.ravel(), kernel, mode=self.mode)
            y_filt = np.convolve(X[self.y].values.ravel(), kernel, mode=self.mode)

            # slice part of array based on convolution mode
            k, h = len(kernel) - 1, (len(kernel) - 1) // 2
            if self.mode == "valid":
                X_filt = X[h + (k % 2):-h]
            elif self.mode == "same":
                pass
            else:           # "full"
                x_filt = x_filt[h + k % 2:-h]
                y_filt = y_filt[h + k % 2:-h]

            X_filt.loc[:, self.x] = x_filt
            X_filt.loc[:, self.y] = y_filt

        return X_filt


class IIRFilter(BaseSmoothingPreprocessor):
    """
    IIR filter. Convolution with IIR kernel along 'x' and 'y'. `kwargs` are passed to `scipy.signal.iirfilter`
    to determine the kernel.
    :param mode: parameter of `scipy.signal.fftconvolve`.

    Default values are taken from https://arxiv.org/pdf/2303.02134.
    """
    def __init__(
        self,
        x: str,
        y: str,
        t: str = None,
        pk: List[str] = None,
        N: int = 7,
        Wn: Union[int, Tuple[int, int]] = 0.5,
        **iir_kw
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.N = N
        self.Wn = Wn
        self.iir_kw = iir_kw

    def _check_params(self):
        m = "IIR filter"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.iir_kw.get('output', 'ba') == 'ba', "Only 'output'='ba' is supported."

    def _preprocess(self, X: pd.DataFrame):
        if self.pk:
            X = X.drop(self.pk, axis=1)

        if self.iir_kw.get('ftype') is None:
            self.iir_kw['ftype'] = 'butter'

        if self.iir_kw.get('btype') is None:
            self.iir_kw['btype'] = 'highpass'

        b, a = iirfilter(N=self.N, Wn=self.Wn, **self.iir_kw, output='ba')
        x_filt = lfilter(b, a, X[self.x].values)[len(b) - 1:]  # "valid" in np.convolve
        y_filt = lfilter(b, a, X[self.y].values)[len(b) - 1:]  # "valid" in np.convolve

        k, h = len(a) - 1, (len(a) - 1) // 2
        X_filt = X[h + (k % 2):-h]

        X_filt.loc[:, self.x] = x_filt
        X_filt.loc[:, self.y] = y_filt
        return X_filt
