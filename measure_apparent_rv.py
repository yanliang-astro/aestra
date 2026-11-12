#!/usr/bin/env python
"""Measure the apparent radial velocity of every merged spectrum by template fitting.

The merged spectra (merge_orders.py) are in the barycentric frame, but each
exposure still carries an apparent Doppler shift at the m/s level, from stellar
activity and any planets. For a trial velocity v, the merged template is shifted
with a cubic spline, and the apparent RV v_app of a spectrum is the v that
minimizes the weighted chi^2 between the spectrum and the shifted template:

    chi^2(v) = sum_j w_j * (y_j - T(wave_j / (1 + v/c)))^2

where T is the cubic spline through the template. Evaluating T at
wave / (1 + v/c) is the same as interpolating the template from the shifted
grid wave * (1 + v/c), since spline interpolation does not depend on the scale
of the x-axis. The training removes v_app from each spectrum before it is
encoded.

Usage (from the repository root):
    ./measure_apparent_rv.py --dir data/

Input, from merge_orders.py, in --merged (default: {dir}/merged):
    {tag}_{when}_{k}.pt, {tag}_{when}-template.pt and {tag}_{when}-param.pt
Output: every observation in {tag}_{when}-param.pt gets
    "v_app"     apparent RV [m/s] relative to the template
    "chi2_app"  chi^2 at v_app per pixel with good data (weight > 1)
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch

from data import SpectralData, load_batch, save_params
from util import TemplateFit, available_cpus

_fitter = None   # one TemplateFit per worker process


def _init_worker(wave, template):
    global _fitter
    _fitter = TemplateFit(wave, template)


def _fit_one(spectrum):
    return _fitter.fit(*spectrum)


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
    wave, template = [t[0].numpy() for t in load_batch(os.path.join(args.merged, f"{stem}-template.pt"))[:2]]
    param_file = os.path.join(args.merged, f"{stem}-param.pt")
    params = torch.load(param_file, weights_only=False)
    name_of = {np.float32(m["timestamp"]): n for n, m in params.items() if n != "info"}
    files = SpectralData.list_batches(args.merged, stem)
    if not files:
        raise SystemExit(f"no merged spectra {stem}_*.pt in {args.merged}")

    print(f"fitting {len(files)} batch files with {args.processes} processes")
    v_app = {}
    with Pool(args.processes, initializer=_init_worker, initargs=(wave, template)) as pool:
        for filename in files:
            batch = load_batch(filename)
            spectra = [(s[1].numpy(), s[2].numpy()) for s in batch]
            fits = pool.map(_fit_one, spectra, chunksize=max(1, len(spectra) // (4 * args.processes)))
            for s, (v, chi2) in zip(batch, fits):
                v_app[name_of[s[4].numpy()[0]]] = (v, chi2)
            print(f"{filename}: {len(batch)} spectra")

    missing = [n for n in params["info"]["sample_names"] if n not in v_app]
    if missing:
        raise SystemExit(f"{len(missing)} observations in {param_file} have no merged spectrum")
    for name, (v, chi2) in v_app.items():
        params[name]["v_app"] = float(v)
        params[name]["chi2_app"] = float(chi2)
    save_params(params, param_file)

    v = np.array([v for v, _ in v_app.values()])
    chi2 = np.array([c for _, c in v_app.values()])
    print(f"saved v_app for {len(v)} spectra to {param_file}: "
          f"RMS {v.std():.3f} m/s, median chi^2 per pixel {np.median(chi2):.2f}")
