"""Shared helpers: interpolation and resampling of spectra, template RV fits, planet RVs, CCF trend."""
import os

import numpy as np
import psutil
import torch
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize_scalar

from data import SpectralData

MASKED_WEIGHT = 1e-12   # weight of masked pixels (effectively zero, but safe to invert)
MIN_WEIGHT = 4e4        # interpolate_to_input_grid masks pixels with lower weight (noise > 0.5%)
T0 = 800.0   # reference time of the planet phases [JD - 2459000]
SHUTDOWN = 800.0     # 2022 NEID shutdown: the CCF trend is fitted separately before and after
EDGE = 10   # pixels at either end of the grid left out of the fit


# From spender (https://github.com/pmelchior/spender, spender/util.py, MIT License)
def interp1d(x, y, xnew):
    """Linear interpolation of `y(x)` at `xnew`

    Beyond the range of `x`, the function is linearly extrapolated from the first or
    last interval. Replaces `torchinterp1d.interp1d` with the same behavior, but
    without a custom backward pass so that autograd computes the gradients.

    Parameters
    ----------
    x: `torch.tensor`, shape (L, ) or (N, L)
        Sorted sample coordinates
    y: `torch.tensor`, shape (L, ) or (N, L)
        Sample values
    xnew: `torch.tensor`, shape (M, ) or (N, M)
        Coordinates to interpolate at

    Returns
    -------
    ynew: `torch.tensor`, shape (N, M)
        Interpolated values, with the dtype of `y`
    """
    x, y, xnew = torch.atleast_2d(x), torch.atleast_2d(y), torch.atleast_2d(xnew)
    N = max(x.shape[0], y.shape[0], xnew.shape[0])
    x, y, xnew = x.expand(N, -1), y.expand(N, -1), xnew.expand(N, -1)

    # index of the left neighbor, clamped to the first and last interval. Search in
    # the common dtype: casting xnew to lower precision can move it across a node
    dtype = torch.promote_types(x.dtype, xnew.dtype)
    ind = torch.searchsorted(x.to(dtype).contiguous(), xnew.to(dtype).contiguous()) - 1
    ind = ind.clamp(0, x.shape[1] - 2)

    eps = torch.finfo(y.dtype).eps
    slopes = (y[:, 1:] - y[:, :-1]) / (eps + x[:, 1:] - x[:, :-1])
    ynew = y.gather(1, ind) + slopes.gather(1, ind) * (xnew - x.gather(1, ind))
    return ynew.to(y.dtype)


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))   # respects the CPUs given to this job
    except AttributeError:
        return os.cpu_count() or 1


def mem_report():
    """Print free CPU memory and, if available, free memory on each GPU."""
    print(f"CPU RAM free: {psutil.virtual_memory().available / 1e9:.1f} GB")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU {i} ... Mem free: {free / 1e6:.0f}MB / {total / 1e6:.0f}MB")


# -------------------------------------------------------------- resampling --

def _bad_pixels(spectrum, weight, template, out_of_range):
    """Mask unusable pixels after linear interpolation onto the model grid."""
    # Zero-flux pixels are at most half the template after linear
    # interpolation; cut at 0.6 for safety.
    bad = (spectrum < template * 0.6) | (template == 0) | (spectrum == 0) | (weight < 1.0)
    # also mask +/- 1 pixel around them
    bad |= torch.roll(bad, -1, dims=1)
    bad |= torch.roll(bad, +1, dims=1)
    return bad | out_of_range


def interpolate_to_input_grid_raw(batch, instrument, template_raw, polyb=0.):
    """Resample raw spectra of one order onto the model wavelength grid.

    Wavelengths are shifted by the barycentric velocity and by the fractional
    wavelength correction `polyb`. NaN weights in `batch` are replaced in place.

    Returns (spectrum, weight, ssbrv, jd); bad pixels get zero flux and
    MASKED_WEIGHT.
    """
    wave_raw, spec_raw, w_raw, ssbrv, jd = batch
    w_raw[torch.isnan(w_raw)] = MASKED_WEIGHT

    wave_obs = instrument.wave_obs
    template = template_raw.repeat(len(spec_raw), 1)

    z = ssbrv / instrument.c
    wave = wave_raw + wave_raw * polyb + wave_raw * z

    spectrum = interp1d(wave, spec_raw, wave_obs).float()
    weight = interp1d(wave, w_raw, wave_obs).float()

    wmin = wave.min(dim=1, keepdim=True)[0]
    wmax = wave.max(dim=1, keepdim=True)[0]
    bad = _bad_pixels(spectrum, weight, template, (wave_obs < wmin) | (wave_obs > wmax))

    weight[bad] = MASKED_WEIGHT
    spectrum[bad] = 0.0
    return spectrum, weight, ssbrv, jd


def interpolate_to_input_grid(batch, instrument, template_input, extra_rv=0.):
    """Doppler-shift residual spectra that are already on the model grid.

    `batch[1]` holds residuals (spectrum - template), `batch[2]` their weights.
    The spectra are shifted by `extra_rv` (m/s) and resampled onto the grid.

    Returns (residual, weight); bad pixels and pixels with weight below
    MIN_WEIGHT get zero residual and MASKED_WEIGHT.
    """
    _, spec_raw, w, _, _ = batch

    n_batch = len(spec_raw)
    template = template_input.repeat(n_batch, 1)
    wave_obs = instrument.wave_obs[0]
    wave_raw = wave_obs.repeat(n_batch, 1)

    z = extra_rv / instrument.c
    wave = wave_raw + wave_raw * z

    spectrum = interp1d(wave, spec_raw + template, wave_obs)
    weight = interp1d(wave, w, wave_obs)

    wmin = wave.min(dim=1)[0].unsqueeze(1)
    wmax = wave.max(dim=1)[0].unsqueeze(1)
    bad = _bad_pixels(spectrum, weight, template, (wave_obs < wmin) | (wave_obs > wmax))
    bad |= weight < MIN_WEIGHT

    spec_input = (spectrum - template).float()
    weight[bad] = MASKED_WEIGHT
    spec_input[bad] = 0.0
    return spec_input, weight.float()


def model_input(batch, instrument, template, apparent_rv):
    """Encoder input of a batch of merged spectra: (residual spectra, weights).

    The residuals (spectrum - template, 0 where there is no data) are shifted
    to zero apparent RV.
    """
    _, flux, w, _, ids = batch
    residual = flux - template
    residual[w < 1] = 0
    spec_input, w = interpolate_to_input_grid((None, residual, w, None, None), instrument, template,
                                              extra_rv=-apparent_rv(ids))
    return spec_input, w


# ------------------------------------------------------------- RV level --

class TemplateFit:
    """Fit the apparent RV of spectra against a template (cubic spline through it).

    Spectra are on the template's wavelength grid, or on their own grid `wave`.
    The fit uses the pixels inside the range of the template's data, without
    EDGE pixels at either end of the spectrum.
    """

    def __init__(self, wave, template):
        self.wave = np.asarray(wave, dtype=np.float64)
        has_data = template != 0   # the template is 0 where no spectrum has good data
        # cubic spline with not-a-knot ends, as scipy.interpolate.interp1d(kind="cubic")
        self.spline = make_interp_spline(self.wave[has_data], template[has_data], k=3)
        self.range = self.wave[has_data][0], self.wave[has_data][-1]
        self.pixels = self.fit_pixels(self.wave)

    def fit_pixels(self, wave):
        """Indices of the pixels of a spectrum with wavelengths `wave` used in the fit."""
        inside = (wave > self.range[0]) & (wave < self.range[1])
        inside[:EDGE] = inside[-EDGE:] = False
        return np.flatnonzero(inside)

    def chi2(self, v, wave, flux, weight):
        """chi^2 of spectrum (flux, weight) at wavelengths wave against the template shifted by v [m/s]."""
        return np.sum(weight * (flux - self.spline(wave / (1 + v / SpectralData.c))) ** 2)

    def fit(self, flux, weight, wave=None):
        """Apparent RV [m/s] and chi^2 per pixel with good data (weight > 1).

        wave: wavelengths of the spectrum (default: the template's grid).
        """
        pixels = self.pixels if wave is None else self.fit_pixels(np.asarray(wave, dtype=np.float64))
        wave = self.wave if wave is None else np.asarray(wave, dtype=np.float64)
        data = wave[pixels], flux[pixels], weight[pixels]
        result = minimize_scalar(self.chi2, bracket=(-1.0, 1.0), args=data)
        if not result.success:
            raise RuntimeError(f"apparent RV fit failed: {result.message}")
        return result.x, result.fun / max(np.sum(data[2] > 1), 1)


def planet_rv(t, amplitude, period, phase):
    """Circular-orbit RV [m/s] at times t [JD - 2459000]; phase in cycles at T0."""
    return amplitude * np.sin(2 * np.pi * ((t - T0) / period + phase))


def ccf_trend(t, v, indicators):
    """Linear fit of v to the CCF indicators (plus a constant), before and after the shutdown."""
    trend = np.zeros_like(v)
    for segment in (t < SHUTDOWN, t >= SHUTDOWN):
        if segment.any():
            X = np.hstack([np.ones((segment.sum(), 1)), indicators[segment]])
            coeffs, *_ = np.linalg.lstsq(X, v[segment], rcond=None)
            trend[segment] = X @ coeffs
    return trend
