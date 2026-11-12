#!/usr/bin/env python
"""Measure the CCF activity indicators of every merged spectrum.

Each merged spectrum y is cross-correlated with the line profile of the
template T, Doppler-shifted to 201 velocities from -15 to +15 km/s:

    CCF(v) = sum_j y_j (1 - T(wave_j (1 - v/c)))

with T the cubic spline through the template. Pixels without data are filled
by cubic interpolation of the spectrum; the sum is not weighted. Each CCF is
normalized to its maximum, and a Gaussian a exp(-(v - mu)^2 / 2 sigma^2) + c is
fitted within +/-5 km/s. The bisector span is the mean bisector velocity at
10-40% of the CCF depth below its top minus that at 60-90%. These indicators
give v_trend in rv_decomposition.py and the detrended RVs of the traditional
candidate search; the RVs themselves come from measure_apparent_rv.py.

Usage (from the repository root):
    ./measure_ccf_indicators.py --dir data/

Input, from merge_orders.py, in --merged (default: {dir}/merged):
    {tag}_{when}_{k}.pt, {tag}_{when}-template.pt and {tag}_{when}-param.pt
Output: every observation in {tag}_{when}-param.pt gets
    "v_ccf"       Gaussian center [m/s]
    "ccf_depth"   Gaussian depth -a, relative to the CCF maximum
    "ccf_fwhm"    Gaussian FWHM [km/s]
    "ccf_bis"     bisector span [m/s]
    "ccf_offset"  Gaussian offset c
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch
from scipy.interpolate import interp1d, make_interp_spline
from scipy.optimize import curve_fit

from data import SpectralData, load_batch, save_params
from util import available_cpus

C_KMS = SpectralData.c / 1e3                 # speed of light [km/s]
VELOCITIES = np.linspace(-15, 15, 201)    # CCF velocity grid [km/s]
FIT_RANGE = 5.0                           # Gaussian fit within +/- this velocity [km/s]
TOP, BOTTOM = (0.1, 0.4), (0.6, 0.9)      # bisector ranges, as fractions of the CCF depth
N_LEVELS = 20                             # bisector levels per range
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def spline(x, y, fill):
    """Cubic not-a-knot spline through (x, y), as interp1d(kind="cubic"); `fill` outside x."""
    s = make_interp_spline(x, y, k=3)

    def evaluate(xnew):
        out = np.full(xnew.shape, fill, dtype=np.float64)
        inside = (xnew >= x[0]) & (xnew <= x[-1])
        out[inside] = s(xnew[inside])
        return out
    return evaluate


def shifted_line_profiles(wave, template, velocities=VELOCITIES):
    """1 - template, Doppler-shifted to every velocity, shape (n_velocity, n_pix); 0 outside."""
    wmin = wave.min() / (1 + velocities.min() / C_KMS)
    wmax = wave.max() / (1 + velocities.max() / C_KMS)
    good = (template > 0) & (wave > wmin) & (wave < wmax)   # template 0: no data
    profile = spline(wave[good], 1 - template[good], fill=0.0)
    return np.stack([profile(wave * (1 - v / C_KMS)) for v in velocities])


def gaussian(v, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((v - mu) / sigma) ** 2) + c


def bisector_span(ccf, velocities=VELOCITIES):
    """Mean bisector velocity in the TOP depth range minus that in the BOTTOM range [km/s]."""
    depth = 1 - ccf.min()   # the CCF is normalized to its maximum
    blue, red = velocities <= 0, velocities > 0
    left = interp1d(ccf[blue], velocities[blue])
    right = interp1d(ccf[red], velocities[red])

    def bisector(fractions):
        levels = 1 - np.linspace(*fractions, N_LEVELS) * depth
        return np.mean(0.5 * (left(levels) + right(levels)))
    return bisector(TOP) - bisector(BOTTOM)


_wave = None   # merged wavelength grid in every worker process


def _init_worker(wave):
    global _wave
    _wave = wave


def fill_gaps(spectrum):
    """Spectrum with pixels without data (weight < 1) filled by cubic interpolation; 1 outside."""
    flux, weight = spectrum
    good = weight >= 1
    return spline(_wave[good], flux[good].astype(np.float64), fill=1.0)(_wave)


def ccf_indicators(ccf):
    """(v_ccf [m/s], depth, FWHM [km/s], bisector span [m/s], offset) of one normalized CCF."""
    near = np.abs(VELOCITIES) <= FIT_RANGE
    v, y = VELOCITIES[near], ccf[near]
    p0 = [y.min() - y.max(), v[np.argmin(y)], 4.0, y.max()]
    (a, mu, sigma, c), _ = curve_fit(gaussian, v, y, p0=p0)
    return mu * 1e3, -a, FWHM_PER_SIGMA * abs(sigma), bisector_span(ccf) * 1e3, c


def measure(spectra, profiles, pool, chunksize):
    """Indicators of a list of (flux, weight) spectra, shape (n_spectra, 5)."""
    filled = np.array(pool.map(fill_gaps, spectra, chunksize=chunksize))
    ccf = filled @ profiles.T
    ccf /= ccf.max(axis=1, keepdims=True)
    return np.array(pool.map(ccf_indicators, ccf, chunksize=chunksize))


KEYS = ["v_ccf", "ccf_depth", "ccf_fwhm", "ccf_bis", "ccf_offset"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", default=os.environ.get("DATA_DIR"),
                        help="directory with the prepared data (default: $DATA_DIR)")
    parser.add_argument("--merged", help="directory with the merged spectra (default: {dir}/merged)")
    parser.add_argument("--tag", default="newprod_merged", help="name prefix of the merged spectra")
    parser.add_argument("--when", default="full", help="dataset suffix")
    parser.add_argument("--processes", type=int, default=available_cpus(),
                        help="worker processes (default: all available CPUs)")
    args = parser.parse_args()
    if args.merged is None:
        if args.dir is None:
            parser.error("pass --dir or --merged, or set DATA_DIR")
        args.merged = os.path.join(args.dir, "merged")

    stem = f"{args.tag}_{args.when}"
    wave, template = [t[0].numpy().astype(np.float64) for t in load_batch(os.path.join(args.merged, f"{stem}-template.pt"))[:2]]
    param_file = os.path.join(args.merged, f"{stem}-param.pt")
    params = torch.load(param_file, weights_only=False)
    name_of = {np.float32(m["timestamp"]): n for n, m in params.items() if n != "info"}
    files = SpectralData.list_batches(args.merged, stem)
    if not files:
        raise SystemExit(f"no merged spectra {stem}_*.pt in {args.merged}")
    profiles = shifted_line_profiles(wave, template)

    print(f"measuring {len(files)} batch files with {args.processes} processes")
    values = {}
    with Pool(args.processes, initializer=_init_worker, initargs=(wave,)) as pool:
        for filename in files:
            batch = load_batch(filename)
            spectra = [(s[1].numpy(), s[2].numpy()) for s in batch]
            result = measure(spectra, profiles, pool, max(1, len(spectra) // (4 * args.processes)))
            for s, row in zip(batch, result):
                values[name_of[s[4].numpy()[0]]] = row
            print(f"{filename}: {len(batch)} spectra")

    missing = [n for n in params["info"]["sample_names"] if n not in values]
    if missing:
        raise SystemExit(f"{len(missing)} observations in {param_file} have no merged spectrum")
    for name, row in values.items():
        params[name].update({key: float(x) for key, x in zip(KEYS, row)})
    save_params(params, param_file)

    table = np.array(list(values.values()))
    print(f"saved {', '.join(KEYS)} for {len(table)} spectra to {param_file}")
    for key, column in zip(KEYS, table.T):
        print(f"  {key:10s} median {np.median(column):10.4f}  RMS {column.std():.4f}")
