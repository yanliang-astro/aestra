#!/usr/bin/env python
"""Traditional RV-level search for candidate planet periods, for one injected planet.

A planet with semi-amplitude K, period P and phase phi is added to the apparent
RVs (measure_apparent_rv.py),

    v_meas = v_app + K sin(2 pi ((t - t0) / P + phi)),   t0 = 800 (JD 2459800),

with t = JD - 2459000. v_meas is detrended with a linear fit to the CCF
indicators (measure_ccf_indicators.py), separately before and after the 2022
shutdown (t = 800), and Lomb-Scargle periodograms of the detrended RVs are
searched in several windows:

    data                                 periods [d]   peaks
    t < 500, 500-800, 800-1200, 1200-1600   1.7-10        10 each
    t < 800, t > 800                     1.7-150       40 each
    all                                  90-400        40

Peaks are the strongest periodogram points, skipping 1- and 0.5-day aliases and
points close to a stronger peak. A sinusoid is then fitted to all detrended RVs
at every peak period (amplitude, period and phase free, with a soft 10% prior
on the period), and the candidates are sorted by amplitude, keeping only the
first of candidates within 2% in period. rv_decomposition.py fits these
candidates, block by block, jointly with the activity model.

Usage (from the repository root), with the default example planet
(K = 0.5 m/s, P = 40 d, phase 0; set --amp, --period, --phase for others, --amp 0 for none):
    ./search_candidates.py --dir data/ candidates.txt

Input: the merged param file with v_app and the CCF indicators, in --merged
    (default: {dir}/merged)
Output: text file of candidates, one "K [m/s]  P [d]  phase" per line, sorted
    by K; phase in cycles at t0
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import minimize

from data import load_rvs
from util import T0, available_cpus, ccf_trend, planet_rv

# Lomb-Scargle windows: data with lo < t < hi, periods pmin-pmax [d], peaks kept
WINDOWS = [((-np.inf, 500), (1.7, 10), 10), ((500, 800), (1.7, 10), 10),
           ((800, 1200), (1.7, 10), 10), ((1200, 1600), (1.7, 10), 10),
           ((-np.inf, 800), (1.7, 150), 40), ((800, np.inf), (1.7, 150), 40),
           ((-np.inf, np.inf), (90, 500), 40)]
MAX_PERIOD = 400.0   # longest candidate period [d]
PERIOD_PRIOR = 0.1   # width of the soft prior on the fitted period, relative to the peak period


def periodogram_peaks(t, v, pmin, pmax, n_peaks, frac=0.1, delta=0.1):
    """Periods and powers of the n_peaks strongest Lomb-Scargle points in pmin-pmax.

    Points within 0.02 d of 1 d or 0.5 d are skipped, as are points within
    min(frac, max(0.05, 2P/1200)) in relative period, or delta in days, of a
    stronger point already kept; periods of MAX_PERIOD and longer are dropped.
    """
    freq, power = LombScargle(t, v).autopower(minimum_frequency=1 / pmax, maximum_frequency=1 / pmin,
                                              samples_per_peak=10)
    period = 1 / freq
    kept, kept_power = [], []
    for i in np.argsort(power)[::-1]:
        p = period[i]
        if abs(p - 1) < 0.02 or abs(p - 0.5) < 0.02:
            continue
        if kept:
            k = np.array(kept)
            if np.min(np.abs(k - p) / p) < min(frac, max(0.05, 2 * p / 1200)) or np.min(np.abs(k - p)) < delta:
                continue
        kept.append(p)
        kept_power.append(power[i])
        if len(kept) >= n_peaks:
            break
    kept, kept_power = np.array(kept), np.array(kept_power)
    short = kept < MAX_PERIOD
    return kept[short], kept_power[short]


def sinusoid_chi2(params, t, v, period0):
    """Mean squared residual of a sinusoid, plus the soft prior on its period."""
    amplitude, period, phase = params
    return np.mean((v - planet_rv(t, amplitude, period, phase)) ** 2) + (period / period0 - 1) ** 2 / PERIOD_PRIOR ** 2


def fit_sinusoid(period, t, v):
    """(K, P, phase) of the sinusoid fitted to v, starting from the best point of a coarse grid at `period`."""
    grid = [(a, period, ph) for ph in np.arange(0, 1, 0.02) for a in np.arange(0.1, 0.8, 0.1)]
    start = min(grid, key=lambda p: np.mean((v - planet_rv(t, *p)) ** 2))
    return minimize(sinusoid_chi2, start, args=(t, v, period), method="Nelder-Mead", tol=1e-6).x


def unique_periods(periods, frac=0.02):
    """Mask of periods not within a fraction frac of an earlier one."""
    kept = np.zeros_like(periods)
    for i, p in enumerate(periods):
        if np.abs(kept / p - 1).min() >= frac:
            kept[i] = p
    return kept > 0


_rvs = None   # (t, detrended v) in every worker process


def _init_worker(t, v):
    global _rvs
    _rvs = (t, v)


def _fit(period):
    return fit_sinusoid(period, *_rvs)


def find_candidates(t, v_meas, indicators, processes=1, verbose=False):
    """Candidates (K, P, phase), sorted by K, from the detrended v_meas."""
    t = np.asarray(t, dtype=np.float64)
    v = v_meas - ccf_trend(t, v_meas, indicators)
    periods, powers = [], []
    for (lo, hi), (pmin, pmax), n_peaks in WINDOWS:
        inside = (t > lo) & (t < hi)
        if inside.sum() < 10:
            continue
        p, power = periodogram_peaks(t[inside], v[inside], pmin, pmax, n_peaks)
        periods.append(p)
        powers.append(power)
        if verbose:
            print(f"t in ({lo}, {hi}), P {pmin}-{pmax} d: {len(p)} peaks, strongest {p[:3].round(3)}")
    periods = np.concatenate(periods)[np.argsort(np.concatenate(powers))[::-1]]

    with Pool(processes, initializer=_init_worker, initargs=(t, v)) as pool:
        fits = np.array(pool.map(_fit, periods))
    fits = fits[np.argsort(fits[:, 0])[::-1]]
    fits[:, 2] %= 1
    return fits[unique_periods(fits[:, 1])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("outfile", help="output text file of candidates")
    parser.add_argument("--dir", default=os.environ.get("DATA_DIR"),
                        help="directory with the prepared data (default: $DATA_DIR)")
    parser.add_argument("--merged", help="directory with the merged spectra (default: {dir}/merged)")
    parser.add_argument("--tag", default="newprod_merged", help="name prefix of the merged spectra")
    parser.add_argument("--when", default="full", help="dataset suffix")
    parser.add_argument("-a", "--amp", type=float, default=0.5,
                        help="injected semi-amplitude [m/s] (default: 0.5; 0: no planet)")
    parser.add_argument("-per", "--period", type=float, default=40.0, help="injected period [d] (default: 40)")
    parser.add_argument("-phase", "--phase", type=float, default=0.0,
                        help="injected phase [cycles] at t0 (default: 0)")
    parser.add_argument("--processes", type=int, default=available_cpus(),
                        help="worker processes (default: all available CPUs)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the peaks of every window")
    args = parser.parse_args()
    if args.merged is None:
        if args.dir is None:
            parser.error("pass --dir or --merged, or set DATA_DIR")
        args.merged = os.path.join(args.dir, "merged")

    _, t, v_app, indicators = load_rvs(os.path.join(args.merged, f"{args.tag}_{args.when}-param.pt"))
    if indicators is None:
        raise SystemExit("no CCF indicators: run measure_ccf_indicators.py first")
    t64 = t.astype(np.float64)
    v_meas = v_app + (planet_rv(t64, args.amp, args.period, args.phase) if args.amp else 0.0)
    candidates = find_candidates(t64, v_meas, indicators, args.processes, args.verbose)
    injection = f"injected K = {args.amp} m/s, P = {args.period} d, phase = {args.phase}"
    np.savetxt(args.outfile, candidates, fmt="%.6f",
               header=f"{injection}\nK [m/s]  P [d]  phase [cycles at t0 = {T0:g}]")
    print(f"{injection}: {len(candidates)} candidates, saved to {args.outfile}")
    for K, P, phase in candidates[:10]:
        print(f"  P = {P:9.3f} d   K = {K:.3f} m/s   phase = {phase:.3f}")
