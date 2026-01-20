from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import firwin, iirfilter, lfilter, savgol_filter

from eyefeatures.preprocessing.base import BaseSmoothingPreprocessor


# ======== SMOOTHING PREPROCESSORS ========
class SavGolFilter(
    BaseSmoothingPreprocessor
):  # TODO 2D SG filter: https://github.com/espdev/sgolay2
    """Savitzky-Golay filter. 'x' and 'y' directions are filtered independently,
    time is ignored. Parameters are passed to `scipy.signal.savgol_filter`.

    Notes:
        Default values are taken from https://arxiv.org/pdf/2303.02134.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str = None,
        pk: list[str] = None,
        window_length: int = 11,
        polyorder: int = 2,
        **savgol_kw,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.wl = window_length
        self.po = polyorder
        self.savgol_kw = savgol_kw

    def _check_params(self):
        m = "Savitzkiy-Golay filter"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pk:
            X = X.drop(self.pk, axis=1)

        X_filt = X.copy()
        if len(X_filt) >= self.wl:
            X_filt[self.x] = savgol_filter(
                x=X[self.x].values,
                window_length=self.wl,
                polyorder=self.po,
                **self.savgol_kw,
            )
            X_filt[self.y] = savgol_filter(
                x=X[self.y].values,
                window_length=self.wl,
                polyorder=self.po,
                **self.savgol_kw,
            )
        return X_filt


class FIRFilter(BaseSmoothingPreprocessor):  # TODO 2D version?
    """FIR filter. Convolution with FIR kernel along 'x' and 'y'.
    `kwargs` are passed to `scipy.signal.firwin` to determine the kernel.

    Args:
        mode: parameter of `scipy.signal.fftconvolve`.

    Notes:
        Default values are taken from https://arxiv.org/pdf/2303.02134.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str = None,
        pk: list[str] = None,
        numtaps: int = 81,
        fs: int = 250,
        cutoff: float | tuple[float, ...] = 100,
        pass_zero: Literal[
            False, True, "bandpass", "lowpass", "highpass", "bandstop"
        ] = False,
        mode: Literal["valid", "full", "same"] = "valid",
        **fir_kw,
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

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pk:
            X = X.drop(self.pk, axis=1)

        X_filt = X.copy()
        if len(X) > self.numtaps:
            kernel = firwin(
                numtaps=self.numtaps,
                cutoff=self.cutoff,
                pass_zero=self.pass_zero,
                fs=self.fs,
                **self.fir_kw,
            )
            x_filt = np.convolve(X[self.x].values.ravel(), kernel, mode=self.mode)
            y_filt = np.convolve(X[self.y].values.ravel(), kernel, mode=self.mode)

            # slice part of array based on convolution mode
            k, h = len(kernel) - 1, (len(kernel) - 1) // 2
            if self.mode == "valid":
                X_filt = X[h + (k % 2) : -h]
            elif self.mode == "same":
                pass
            else:  # "full"
                x_filt = x_filt[h + k % 2 : -h]
                y_filt = y_filt[h + k % 2 : -h]

            X_filt.loc[:, self.x] = x_filt
            X_filt.loc[:, self.y] = y_filt

        return X_filt


class IIRFilter(BaseSmoothingPreprocessor):  # TODO 2D version?
    """IIR filter. Convolution with IIR kernel along 'x' and 'y'.
    `kwargs` are passed to `scipy.signal.iirfilter` to determine the kernel.

    Args:
        mode: parameter of `scipy.signal.fftconvolve`.

    Notes:
        Default values are taken from https://arxiv.org/pdf/2303.02134.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str = None,
        pk: list[str] = None,
        N: int = 7,
        Wn: int | tuple[int, int] = 0.5,
        **iir_kw,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.N = N
        self.Wn = Wn
        self.iir_kw = iir_kw

    def _check_params(self):
        m = "IIR filter"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert (
            self.iir_kw.get("output", "ba") == "ba"
        ), "Only 'output'='ba' is supported."

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pk:
            X = X.drop(self.pk, axis=1)

        if self.iir_kw.get("ftype") is None:
            self.iir_kw["ftype"] = "butter"

        if self.iir_kw.get("btype") is None:
            self.iir_kw["btype"] = "highpass"

        b, a = iirfilter(N=self.N, Wn=self.Wn, **self.iir_kw, output="ba")
        x_filt = lfilter(b, a, X[self.x].values)[len(b) - 1 :]  # "valid" in np.convolve
        y_filt = lfilter(b, a, X[self.y].values)[len(b) - 1 :]  # "valid" in np.convolve

        k, h = len(a) - 1, (len(a) - 1) // 2
        X_filt = X[h + (k % 2) : -h]

        X_filt.loc[:, self.x] = x_filt
        X_filt.loc[:, self.y] = y_filt
        return X_filt


class WienerFilter(BaseSmoothingPreprocessor):
    """Wiener filter. Applied independently along 'x' and 'y' axis.

    Args:
        K: estimate of ratio of noise-to-signal spectra. If 'auto', such value of K
           on grid [-1e-3, 1e-3] with 2e3 values is chosen to minimize PSNR.
        sigma: std of Gaussian filter.
        size: length of Gaussian filter.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str = None,
        pk: list[str] = None,
        K: float | Literal["auto"] = 4.3e-5,
        sigma: float = 0.2,
        size: int = 11,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.K = K
        self.sigma = sigma
        self.size = size

    def _check_params(self):
        assert self.K == "auto" or isinstance(self.K, float)

    @staticmethod
    def _gaussian_kernel(size, sigma):
        if size % 2 == 0:
            x = (np.arange(-size / 2, size / 2, 1) + 0.5).ravel()
        else:
            x = np.arange(-size // 2 + 1, size // 2 + 1, 1).ravel()

        # get 1D kernel using Gaussian PDF
        kernel = (
            1
            / (2 * np.pi * np.square(sigma))
            * np.exp(-np.square(x) / (2 * np.square(sigma)))
        )
        # normalize kernel
        kernel = kernel / np.sum(kernel)
        return kernel

    @staticmethod
    def _compute_psnr(seq1, seq2):
        """Peak-Signal-to-Noise Ratio. Must be applied to
        normalized coordinates with maximum value of 1.
        """
        mse = np.sum(np.square(seq1 - seq2)) / len(seq1)
        psnr = 20 * np.log10(2 / np.sqrt(mse))  # 2 = length of [-1, 1]
        return psnr

    @staticmethod
    def _kernel_fft(kernel: np.array, length: int):
        diff = length - len(kernel.ravel())
        l_pad = diff // 2
        r_pad = diff - l_pad

        # pad the kernel
        kernel_padded = np.pad(kernel, pad_width=(l_pad, r_pad))
        # IFFT for center-based kernel
        kernel_adjusted = np.fft.ifftshift(kernel_padded)
        # FFT to adjust kernel to the correct position
        kernel_transformed = np.fft.fft(kernel_adjusted)
        return kernel_transformed

    def _wiener_filter(self, seq: np.array, kernel: np.array, K):
        H = self._kernel_fft(kernel, seq.shape[-1])
        G = np.fft.fft(seq)
        H_conj = np.conjugate(H)

        F_est = (H_conj * G) / (H * H_conj + K)
        f_est = np.fft.ifft(F_est)

        return np.abs(f_est)  # take modulus of complex numbers

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pk:
            X = X.drop(self.pk, axis=1)

        # check if input has enough length
        if len(X) <= self.size:
            return X

        # define sequences for Wiener filter
        x_path, y_path = X[self.x].values, X[self.y].values
        # get 1D Gaussian kernel
        kernel = self._gaussian_kernel(self.size, self.sigma)

        # find optimal value for K
        if self.K == "auto":
            best_psnr = -np.inf
            best_K = 0

            for K in list(np.linspace(-1e-3, 1e-3, 2001)):
                x_filt = self._wiener_filter(x_path, kernel=kernel, K=K)
                y_filt = self._wiener_filter(y_path, kernel=kernel, K=K)

                psnr = self._compute_psnr(x_filt, y_filt)
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_K = K

            self.K = best_K

        x_filt = self._wiener_filter(x_path, kernel=kernel, K=self.K)
        y_filt = self._wiener_filter(y_path, kernel=kernel, K=self.K)

        X_filt = X.copy()
        X_filt.loc[:, self.x] = x_filt
        X_filt.loc[:, self.y] = y_filt
        return X_filt
