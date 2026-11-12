#!/usr/bin/env python
"""Split the apparent RVs into stellar activity and planets (stage 2), for one injected planet.

A planet with semi-amplitude K, period P and phase phi is added to the apparent
RVs (measure_apparent_rv.py):

    v_meas = v_app + K sin(2 pi ((t - t0) / P + phi)),   t0 = 800 (JD 2459800)

where t = JD - 2459000. v_meas is then decomposed as

    v_meas = f(z) + sum_k K_k sin(2 pi ((t - t0) / P_k + phi_k)) + v_trend

f is the activity RV estimator, a small MLP of the stellar latents z
(export_latents.py), which stay fixed. The sum runs over candidate signals with
fixed periods P_k and phases phi_k; their amplitudes K_k are fitted jointly with
f by minimizing 0.05 * sum (v_meas - f(z) - planets - v_trend)^2.

Candidates are fitted in four period blocks, separately, so that candidates of
similar period compete for the signal:

    block    periods [d]   candidates   v_trend
    inner    P < 10        10           CCF indicators
    middle   10-100        10           CCF indicators
    outer    100-250       3            0
    further  P > 250       1            0

The candidates come from the traditional RV-level search of search_candidates.py,
run on v_meas, or from a --candidates file of the same format. Each block keeps
the first candidates in its period range (the list is sorted by amplitude), after
dropping candidates within 10% in period (P/1200 for long periods) of an earlier
one. v_trend is a linear fit of v_meas to the CCF depth, FWHM, bisector span and
offset (measure_ccf_indicators.py), done separately before and after the 2022
shutdown (t = 800).

Usage (from the repository root), with the default example planet
(K = 0.5 m/s, P = 40 d, phase 0; set --amp, --period, --phase for others, --amp 0 for none):
    ./rv_decomposition.py --dir data/ -v stellar_model/stellar_s20-latents.pt recovery.pt

Input: the latents file (export_latents.py) and the merged param file with v_app
    and the CCF indicators, in --merged (default: {dir}/merged); optionally a text
    file of candidates, one "K [m/s]  P [d]  phase" per line
Output: a dict with the injection, the RVs, the candidates and, for every block,
    the fitted candidates "solution" (K, P, phase), v_act, v_planets, v_trend, the
    model state and the loss history
"""
import argparse
import os
import time

import numpy as np
import torch
from torch import nn, optim

from data import load_rvs
from search_candidates import find_candidates
from models import ActivityEstimator, PlanetRV
from util import T0, available_cpus, ccf_trend, planet_rv

BLOCKS = {           # period range [d], candidates kept, subtract v_trend
    "inner": (0.0, 10.0, 10, True),
    "middle": (10.0, 100.0, 10, True),
    "outer": (100.0, 250.0, 3, False),
    "further": (250.0, np.inf, 1, False),
}
# Columns of the saved loss history, the layout of the other training scripts;
# only "velocity reg." is used here.
LOSS_NAMES = ["fit", "similarity", "redshift", "consistency", "velocity reg.", "flexibility"]
VELOCITY = 4


def unique_periods(periods, basefrac=0.1, timespan=1200):
    """Mask of periods not within a fraction max(basefrac, P/timespan) of an earlier one (0.1 above 100 d)."""
    kept = np.zeros_like(periods)
    for i, p in enumerate(periods):
        frac = 0.1 if p > 100 else max(basefrac, p / timespan)
        if np.abs(kept / p - 1).min() >= frac:
            kept[i] = p
    return kept > 0


def select_candidates(candidates, block):
    """Candidates (K, P, phase) of one period block, in the order of the candidate file.

    Candidates need a positive initial amplitude: the planet model only fits those.
    """
    pmin, pmax, n_max, _ = BLOCKS[block]
    periods = candidates[:, 1]
    keep = unique_periods(periods) & (periods > pmin) & (periods < pmax) & (candidates[:, 0] > 0)
    return candidates[keep][:n_max]


class Observations:
    """Time, apparent RV, latents (and CCF indicators) of every merged spectrum."""

    def __init__(self, latent_file, param_file):
        latents = torch.load(latent_file)
        self.names, self.t, self.v_app, self.indicators = load_rvs(param_file)
        if self.names != latents["names"] or not np.array_equal(self.t, latents["timestamp"].numpy()):
            raise ValueError(f"{latent_file} and {param_file} hold different observations")
        self.latents = latents["latents"]


class RVDecomposition(nn.Module):
    """Activity RV estimator plus sinusoidal planets with fixed periods and phases."""

    def __init__(self, n_latent, candidates, n_hidden=(128, 8)):
        super().__init__()
        self.activity_estimator = ActivityEstimator(n_latent, n_hidden=n_hidden)
        self.doppler_model = PlanetRV(candidates)

    def forward(self, s, t):
        """(v_act, v_planets), each of shape (B, 1), for latents s and times t of shape (B, 1)."""
        return self.activity_estimator(s), self.doppler_model(t).sum(dim=0)[:, None]

    def solution(self):
        """Fitted (K, P, phase) of every candidate; negative K are flipped by half a cycle."""
        m = self.doppler_model
        K, P, phase = m.get_amplitudes().detach(), m.get_periods(), m.get_phases().clone()
        flip = K < 0
        phase[flip] = (phase[flip] + 0.5) % 1
        return torch.stack([K.abs(), P, phase], dim=1).cpu()


def train(model, s, t, v_meas, v_trend, n_epoch, lr=2e-3, batch_size=500, n_batch=1, verbose=False):
    """Fit the activity estimator and the planet amplitudes; returns the loss history.

    s, t, v_meas, v_trend hold all observations (on the model's device); every
    epoch draws n_batch random batches of batch_size observations (n_batch=None:
    all observations once).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)
    history = np.zeros((2, 1, n_epoch, len(LOSS_NAMES)))

    for epoch in range(n_epoch):
        model.train()
        n_sample = 0
        batches = torch.randperm(len(t)).split(batch_size)
        for idx in batches[:n_batch]:
            idx = idx.to(t.device)
            v_act, v_planets = model(s[idx], t[idx])
            resid = v_meas[idx] - v_act - v_trend[idx] - v_planets
            loss = 0.05 * torch.sum(resid ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            history[0, 0, epoch, VELOCITY] += loss.item()
            n_sample += len(idx)
        history[0, 0, epoch] /= n_sample
        scheduler.step()
        if verbose and (epoch % 100 == 0 or epoch == n_epoch - 1):
            print(f"====> Epoch: {epoch}  loss: {history[0, 0, epoch, VELOCITY]:.4f}")
    return history


def run_block(block, obs, v_meas, candidates, args, device, seed):
    """Train one period block; returns its results, or None without candidates in range."""
    selected = select_candidates(candidates, block)
    if len(selected) == 0:
        print(f"\n[{block}] no candidates in range, skipped")
        return None
    use_trend = BLOCKS[block][3] and not args.no_trend
    if use_trend and obs.indicators is None:
        raise SystemExit(f"[{block}] needs the CCF indicators for v_trend: "
                         "run measure_ccf_indicators.py first, or pass --no-trend")
    v_trend = ccf_trend(obs.t.astype(np.float64), v_meas, obs.indicators) if use_trend else np.zeros_like(v_meas)
    print(f"\n[{block}] {len(selected)} candidates, v_trend {'on' if use_trend else 'off'}")

    torch.manual_seed(seed)
    model = RVDecomposition(obs.latents.shape[1], selected, n_hidden=args.hidden).to(device)
    as_column = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)[:, None]
    s, t = obs.latents.to(device), as_column(obs.t)
    v_meas_t, v_trend_t = as_column(v_meas), as_column(v_trend)
    history = train(model, s, t, v_meas_t, v_trend_t, args.iteration, lr=args.rate,
                    batch_size=args.batch_size, n_batch=args.batch_number or None, verbose=args.verbose)

    model.eval()
    with torch.no_grad():
        v_act, v_planets = model(s, t)
    solution = model.solution()
    for K, P, phase in solution[torch.argsort(solution[:, 0], descending=True)].tolist():
        print(f"  P = {P:9.3f} d   K = {K:.3f} m/s   phase = {phase:.3f}")
    rms = (v_meas_t - v_act - v_trend_t - v_planets).std().item()
    print(f"  residual RMS {rms:.3f} m/s (v_meas RMS {v_meas.std():.3f} m/s)")
    return {"candidates": torch.as_tensor(selected), "solution": solution,
            "v_trend": torch.as_tensor(v_trend), "v_act": v_act[:, 0].cpu(), "v_planets": v_planets[:, 0].cpu(),
            "model": {k: v.cpu() for k, v in model.state_dict().items()}, "losses": torch.as_tensor(history)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("latents", help="stellar latents from export_latents.py")
    parser.add_argument("outfile", help="output file")
    parser.add_argument("--dir", default=os.environ.get("DATA_DIR"),
                        help="directory with the prepared data (default: $DATA_DIR)")
    parser.add_argument("--merged", help="directory with the merged spectra (default: {dir}/merged)")
    parser.add_argument("--tag", default="newprod_merged", help="name prefix of the merged spectra")
    parser.add_argument("--when", default="full", help="dataset suffix")
    parser.add_argument("--candidates", help="text file of candidates: K [m/s], P [d], phase "
                        "(default: run the search of search_candidates.py)")
    parser.add_argument("-a", "--amp", type=float, default=0.5,
                        help="injected semi-amplitude [m/s] (default: 0.5; 0: no planet)")
    parser.add_argument("-per", "--period", type=float, default=40.0, help="injected period [d] (default: 40)")
    parser.add_argument("-phase", "--phase", type=float, default=0.0,
                        help="injected phase [cycles] at t0 (default: 0)")
    parser.add_argument("--block", action="append", choices=list(BLOCKS), dest="blocks",
                        help="period block to fit, repeatable (default: all four)")
    parser.add_argument("--no-trend", action="store_true", help="no v_trend in the inner and middle blocks")
    parser.add_argument("-b", "--batch_size", type=int, default=500, help="observations per batch")
    parser.add_argument("-l", "--batch_number", type=int, default=1, help="batches per epoch (0: all)")
    parser.add_argument("-r", "--rate", type=float, default=2e-3, help="peak learning rate")
    parser.add_argument("-it", "--iteration", type=int, default=1000, help="number of epochs per block")
    parser.add_argument("--hidden", type=int, nargs=2, default=[128, 8], metavar=("N1", "N2"),
                        help="hidden layer sizes of the activity RV estimator (default: 128 8)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (block i uses seed + i)")
    parser.add_argument("--processes", type=int, default=available_cpus(),
                        help="worker processes of the candidate search (default: all available CPUs)")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU")
    parser.add_argument("-v", "--verbose", action="store_true", help="print progress")
    args = parser.parse_args()
    if args.merged is None:
        if args.dir is None:
            parser.error("pass --dir or --merged, or set DATA_DIR")
        args.merged = os.path.join(args.dir, "merged")
    args.blocks = args.blocks or list(BLOCKS)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    obs = Observations(args.latents, os.path.join(args.merged, f"{args.tag}_{args.when}-param.pt"))
    v_inject = planet_rv(obs.t.astype(np.float64), args.amp, args.period, args.phase) if args.amp else 0.0
    v_meas = obs.v_app + v_inject
    print(f"{len(obs.t)} observations, injected K = {args.amp} m/s, P = {args.period} d, phase = {args.phase}")
    if args.candidates:
        candidates = np.loadtxt(args.candidates, ndmin=2)
        if candidates.shape[1] < 3:
            raise SystemExit(f"{args.candidates}: expected columns K [m/s], P [d], phase")
        print(f"{len(candidates)} candidates from {args.candidates}")
    else:
        if obs.indicators is None:
            raise SystemExit("the candidate search needs the CCF indicators: run measure_ccf_indicators.py first")
        candidates = find_candidates(obs.t.astype(np.float64), v_meas, obs.indicators, args.processes)
        print(f"{len(candidates)} candidates from the RV-level search")

    t_start = time.time()
    blocks = {}
    for i, block in enumerate(BLOCKS):
        if block in args.blocks:
            result = run_block(block, obs, v_meas, candidates[:, :3], args, device, args.seed + i)
            if result is not None:
                blocks[block] = result

    torch.save({"injection": {"amplitude": args.amp, "period": args.period, "phase": args.phase, "t0": T0},
                "timestamp": torch.as_tensor(obs.t), "names": obs.names,
                "v_app": torch.as_tensor(obs.v_app), "v_meas": torch.as_tensor(v_meas),
                "candidates": torch.as_tensor(candidates[:, :3]), "blocks": blocks, "settings": vars(args)},
               args.outfile)
    print(f"\nsaved {args.outfile} ({time.time() - t_start:.1f} s)")
