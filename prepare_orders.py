#!/usr/bin/env python
"""Prepare the per-order spectra of the NEID solar data from NEID Level-2 files.

Observations: the quality-cut list (default data/NEID_HIGH_QUALITY_OBSNAME.txt;
paper Sec. 2) with JD < --jd_max whose L2 files are in --l2, shuffled with
--seed, first --n_sample (or an ordered list of observation names, --sample).

For every observation and echelle order (array index N = 173 - DRP order):
  - keep the free-spectral-range pixels of the NEID FSR mask;
  - divide the flux by the blaze and by its median; weights 1/variance, with
    pixels without flux and isolated single-pixel spikes set to (almost) zero;
  - reject the spectrum if it has more than 100 negative-flux pixels or its
    order's CCF RV is more than 50 m/s from the mean of its batch.
Each order's wavelengths are then shifted by a bulk RV offset, one before and
one after the 2022 shutdown: the median RV of the spectra against a first
template of the order (two passes: template, RVs, offsets, final template).
The template of an order is the median of its first 1000 spectra, shifted by
the barycentric correction and interpolated onto the order's grid from
orders.config.

Usage (from the repository root):
    ./prepare_orders.py --l2 /path/to/L2 --fsr_mask /path/to/neidMaster_FSR_Mask20210218_v002.fits --dir data/

Output in --dir, for every order N (with --data newprod_order --when full):
    {data}{N}_{when}_{k}.pt       batches of spectra: lists of per-spectrum tuples
                                  (wavelengths, flux, weight, SSBRV [m/s], id = JD - 2459000)
    {data}{N}_{when}-template.pt  [wave, flux, weight, 0, 888], each of shape (1, n_pix)
    {data}{N}_{when}-param.pt     metadata of every observation and run info
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import torch
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

from data import DEFAULT_ORDERS, SpectralData, load_batch
from util import TemplateFit, available_cpus

FIRST_ORDER = 173        # NEID DRP order of array index 0 (index N: order 173 - N)
SSBRV_OFFSET = 0.8       # [km/s] added to the NEID barycentric correction
JD0 = 2459000.0          # spectrum id = JD - JD0
SHUTDOWN_JD = 2459797.5  # 2022 Aug 1: bulk offsets are separate before and after
LARGE_ERROR = 1e6        # flux error of pixels without data (weight 1e-12)
MAX_NEGATIVE = 100       # spectra with more negative-flux pixels are rejected
MAX_CCF_OFFSET = 50.0    # [m/s] spectra with a CCF RV farther from the batch mean are rejected
N_TEMPLATE = 1000        # spectra in the median template
HEADER_KEYS = ("OBSJD", "DATE-OBS", "AIRMASS", "E_VER")
# HDUs of NEID L2 files
SCIFLUX, SCIVAR, SCIWAVE, TELLURIC, CCFS, SCIBLAZE = 1, 4, 7, 10, 12, 15


# ------------------------------------------------------------ inputs --

def order_grid(config, order):
    """Wavelength grid of an order from orders.config: pixel steps from a quadratic polynomial."""
    _, wave_min, _, *poly, n_pix = config[order]
    grid = np.zeros(int(n_pix))
    grid[0] = wave_min
    for j in range(1, len(grid)):
        grid[j] = grid[j - 1] + np.polyval(poly, grid[j - 1])
    return grid


def load_fsr_mask(filename):
    """Pixels of every order outside the free spectral range (True = drop), shape (122, 9216)."""
    with fits.open(filename) as hdul:
        return np.array(hdul[0].data, dtype=bool)


def select_observations(obslist, l2_dir, jd_max, seed, n_sample=None):
    """Names of the listed observations with JD < jd_max and an L2 file in l2_dir, shuffled, first n_sample."""
    table = np.loadtxt(obslist, dtype={"names": ("filename", "jd", "ccfrv"), "formats": ("S53", "f8", "f8")})
    names = np.array([os.path.basename(x.decode()) for x in table["filename"][table["jd"] < jd_max]])
    sel = np.nonzero(np.isin(names, os.listdir(l2_dir)))[0]
    np.random.RandomState(seed).shuffle(sel)
    return list(names[sel[:n_sample]])


def read_l2(filename, keep):
    """FSR pixels (wave, flux, variance, blaze) of the orders in `keep` and the metadata of one L2 file.

    keep: {order: boolean pixel mask}. Returns (spectra by order, shared metadata, metadata by order).
    """
    with fits.open(filename) as hdul:
        header, telluric, ccf = hdul[0].header, hdul[TELLURIC].header, hdul[CCFS].header
        data = [hdul[i].data for i in (SCIWAVE, SCIFLUX, SCIVAR, SCIBLAZE)]
        # native byte order, but the stored precision (variance and blaze are float32)
        spectra = {o: [d[o][mask].astype(d.dtype.newbyteorder("=")) for d in data] for o, mask in keep.items()}
        info = {key: header[key] for key in HEADER_KEYS}
        info.update({key: telluric[key] for key in ("ZENITH", "WVAPOR")})
        info["CCFRVMOD"] = ccf["CCFRVMOD"] * 1e3
        info["DVRMSMOD"] = ccf["DVRMSMOD"] * 1e3
        info["timestamp"] = np.float32(info["OBSJD"] - JD0)
        per_order = {o: {"SSBRV": (header[f"SSBRV{FIRST_ORDER - o:03d}"] + SSBRV_OFFSET) * 1e3,
                         "CCFRV": ccf[f"CCFRV{FIRST_ORDER - o:03d}"] * 1e3} for o in keep}
    return spectra, info, per_order


# ----------------------------------------------------- one spectrum --

def deepest_pixels(wave, spectrum, num, min_separation):
    """(index, depth) of the deepest pixels of spectrum at least min_separation apart, sorted by index.

    Depth is 1 - spectrum / (99% quantile).
    """
    depth = 1 - spectrum / np.quantile(spectrum, 0.99)
    picked = []
    for i in np.argsort(depth)[::-1]:
        if all(abs(wave[i] - wave[j]) > min_separation for j in picked):
            picked.append(i)
            if len(picked) == num:
                break
    return sorted(zip(picked, depth[picked]))


def single_pixel_spikes(wave, spectrum, error, sigma=1.5, snr=3, radius=2):
    """Mask of isolated outliers: the 10 largest deviations from a smoothed spectrum (+/- 1 pixel)
    that are significant at snr and do not sit at the bottom of a stellar line."""
    deviation = np.abs(spectrum - gaussian_filter1d(spectrum, sigma))
    n = len(spectrum)
    known_bad = set(np.flatnonzero(error > 1))
    spikes = []
    for i, amp in deepest_pixels(wave, 1 - deviation, num=10, min_separation=0.1):
        if amp / error[i] < snr or i in known_bad:
            continue
        nearby = np.sort(spectrum[max(0, i - radius):min(i + radius, n - 1)])
        if nearby[0] < 0.8 and nearby[1] - nearby[0] < 0.1:   # bottom of a stellar line
            continue
        spikes.extend([i - 1, i, i + 1])
    mask = np.zeros(n, dtype=bool)
    mask[[i for i in spikes if 0 <= i < n]] = True
    return mask


def normalize(flux, variance, blaze, wave):
    """Blaze-corrected flux normalized by its median, and its error; LARGE_ERROR where unusable."""
    bad = np.isnan(flux) | np.isnan(blaze) | (flux <= 0.0)
    if bad.all():
        raise ValueError("no valid pixels")
    spectrum = np.zeros_like(flux)
    error = np.full_like(flux, LARGE_ERROR)
    spectrum[~bad] = flux[~bad] / blaze[~bad]
    norm = np.quantile(spectrum[~bad], 0.5)
    spectrum[~bad] /= norm
    # errors at the precision of the variance (float32 in NEID L2 files)
    error[~bad] = (variance[~bad] ** 0.5 / (norm * blaze[~bad])).astype(variance.dtype)
    error[single_pixel_spikes(wave, spectrum, error)] = LARGE_ERROR
    return spectrum, error


# ----------------------------------------------- pass 1: L2 -> batches --

def save_batch(filename, wave, flux, error, ssbrv, ids):
    """Per-spectrum tuples (wave, flux, weight, ssbrv, id); pixels with weight < 1 get zero flux."""
    weight = error ** -2.0
    flux = np.where(weight < 1.0, 0.0, flux)
    # torch.tensor copies each row: a view would make torch.save write the whole array
    batch = [(torch.tensor(w, dtype=torch.float64), torch.tensor(f, dtype=torch.float32),
              torch.tensor(ww, dtype=torch.float32), torch.tensor([v], dtype=torch.float64),
              torch.tensor([i], dtype=torch.float32))
             for w, f, ww, v, i in zip(wave, flux, weight, ssbrv, ids)]
    torch.save(batch, filename)


def prepare_batch(k, names, args, keep):
    """Read the L2 files of batch k and write its per-order batch files (without bulk offset).

    Returns {order: [(name, metadata), ...]} of the spectra kept in each order.
    """
    rows = {o: [] for o in keep}
    for name in names:
        try:
            spectra, info, per_order = read_l2(os.path.join(args.l2, name), keep)
        except (OSError, KeyError, ValueError, IndexError) as error:
            print(f"skipping {name}: {error}")
            continue
        for o, (wave, flux, variance, blaze) in spectra.items():
            try:
                spectrum, error = normalize(flux, variance, blaze, wave)
            except ValueError as e:
                print(f"order {o}: skipping {name}: {e}")
                continue
            if np.sum(spectrum < 0.0) > MAX_NEGATIVE:
                print(f"order {o}: skipping {name}: {np.sum(spectrum < 0.0)} negative pixels")
                continue
            rows[o].append((name, {**info, **per_order[o]}, wave, spectrum, error))

    kept = {}
    for o, items in rows.items():
        if items:
            ccfrv = np.array([meta["CCFRV"] for _, meta, *_ in items])
            far = np.abs(ccfrv - ccfrv.mean()) > MAX_CCF_OFFSET
            for (name, *_), ccf in zip(items, ccfrv - ccfrv.mean()):
                if abs(ccf) > MAX_CCF_OFFSET:
                    print(f"order {o}: skipping {name}: CCF RV {ccf:.1f} m/s from the batch mean")
            items = [item for item, f in zip(items, far) if not f]
        kept[o] = [(name, meta) for name, meta, *_ in items]
        if items:
            _, metas, wave, spectrum, error = zip(*items)
            save_batch(batch_file(args, o, k), np.array(wave), np.array(spectrum), np.array(error),
                       [m["SSBRV"] for m in metas], [m["timestamp"] for m in metas])
    print(f"batch {k}: {len(names)} observations")
    return kept


def batch_file(args, order, k):
    return os.path.join(args.dir, f"{args.data}{order}_{args.when}_{k}.pt")


# ------------------------------------ pass 2: bulk offset and template --

def median_template(files, wave_obs, n_template=N_TEMPLATE):
    """Median flux and weight of the first n_template spectra, shifted by SSBRV, on the grid wave_obs."""
    flux, weight = [], []
    for filename in files:
        for wave, f, w, ssbrv, _ in load_batch(filename):
            if len(flux) == n_template:
                break
            wave, f, w = wave.numpy(), f.numpy(), w.numpy()
            wave = wave + wave * ssbrv.numpy() / SpectralData.c
            good = w > 1.0
            inside = (wave_obs > wave[good].min()) & (wave_obs < wave[good].max())
            flux_i, weight_i = np.zeros(len(wave_obs)), np.full(len(wave_obs), 1e-12)
            flux_i[inside] = np.interp(wave_obs[inside], wave[good], f[good])
            weight_i[inside] = np.interp(wave_obs[inside], wave, w)
            flux_i[weight_i < 1.0] = 0.0
            flux.append(flux_i)
            weight.append(weight_i)
    return np.median(flux, axis=0), np.median(weight, axis=0)


def template_rvs(files, wave_obs, template):
    """RV [m/s] of every spectrum against the template, and its id."""
    fitter = TemplateFit(wave_obs, template)
    rvs, ids = [], []
    for filename in files:
        for wave, f, w, ssbrv, i in load_batch(filename):
            wave = wave.numpy() * (1.0 + ssbrv.numpy() / SpectralData.c)
            rvs.append(fitter.fit(f.numpy(), w.numpy(), wave)[0])
            ids.append(i.item())
    return np.array(rvs), np.array(ids)


def finish_order(order, files, wave_obs):
    """Bulk offsets, shifted wavelengths and final template of one order.

    Returns (template, template weight, (offset before, offset after) [m/s], {id: RV}).
    """
    template, _ = median_template(files, wave_obs)
    rvs, ids = template_rvs(files, wave_obs, template)
    jd = ids + JD0
    offsets = (np.median(rvs[jd < SHUTDOWN_JD]) if np.any(jd < SHUTDOWN_JD) else 0.0,
               np.median(rvs[jd >= SHUTDOWN_JD]) if np.any(jd >= SHUTDOWN_JD) else 0.0)
    for filename in files:
        batch = load_batch(filename)
        shifted = []
        for wave, f, w, ssbrv, i in batch:
            offset = offsets[0] if i.item() + JD0 < SHUTDOWN_JD else offsets[1]
            wave = wave.numpy()
            shifted.append((torch.tensor(wave + (-offset / SpectralData.c) * wave, dtype=torch.float64), f, w, ssbrv, i))
        torch.save(shifted, filename)
    template, template_w = median_template(files, wave_obs)
    print(f"order {order}: bulk offsets {offsets[0]:.2f} m/s (before), {offsets[1]:.2f} m/s (after)")
    return template, template_w, offsets, dict(zip(ids.astype(np.float32), rvs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--l2", required=True, help="directory with the NEID L2 files")
    parser.add_argument("--fsr_mask", required=True, help="NEID FSR mask, neidMaster_FSR_Mask20210218_v002.fits")
    parser.add_argument("--dir", default=os.environ.get("DATA_DIR"),
                        help="output directory (default: $DATA_DIR)")
    parser.add_argument("--orders", type=int, nargs="+", default=DEFAULT_ORDERS,
                        help="order indices (default: 31-74 without 69 and 70)")
    parser.add_argument("--config", default="orders.config", help="wavelength grids of the orders")
    parser.add_argument("--obslist", default="data/NEID_HIGH_QUALITY_OBSNAME.txt",
                        help="quality-cut observation list: L2 path, JD, CCF RV per line")
    parser.add_argument("--jd_max", type=float, default=2460600.0, help="latest JD (default: 2460600)")
    parser.add_argument("--n_sample", type=int, help="number of observations (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the selection")
    parser.add_argument("--sample", help="text file of observation names to use instead, in order")
    parser.add_argument("--data", default="newprod_order", help="output name prefix")
    parser.add_argument("--when", default="full", help="output name suffix")
    parser.add_argument("--batch_size", type=int, default=500, help="spectra per batch file")
    parser.add_argument("--processes", type=int, default=min(8, available_cpus()),
                        help="worker processes (default: up to 8; each needs ~2 GB)")
    args = parser.parse_args()
    if args.dir is None:
        parser.error("pass --dir or set DATA_DIR")
    os.makedirs(args.dir, exist_ok=True)

    if args.sample:
        names = [line.split()[0] for line in open(args.sample) if line.strip() and not line.startswith("#")]
    else:
        names = select_observations(args.obslist, args.l2, args.jd_max, args.seed, args.n_sample)
    print(f"{len(names)} observations, {len(args.orders)} orders")
    fsr = load_fsr_mask(args.fsr_mask)
    keep = {o: ~fsr[o] for o in sorted(set(args.orders))}
    config = np.loadtxt(args.config)
    batches = np.array_split(names, np.arange(0, len(names), args.batch_size)[1:])

    with Pool(args.processes) as pool:
        kept = pool.starmap(prepare_batch, [(k, list(b), args, keep) for k, b in enumerate(batches)])
        orders = list(keep)
        files = {o: [batch_file(args, o, k) for k in range(len(batches)) if kept[k][o]] for o in orders}
        grids = {o: order_grid(config, o) for o in orders}
        finished = pool.starmap(finish_order, [(o, files[o], grids[o]) for o in orders])

    for o, (template, template_w, offsets, rvs) in zip(orders, finished):
        stem = os.path.join(args.dir, f"{args.data}{o}_{args.when}")
        torch.save([torch.from_numpy(grids[o][None]).double(), torch.from_numpy(template[None]).float(),
                    torch.from_numpy(template_w[None]).float(), torch.zeros((1, 1), dtype=torch.float64),
                    torch.full((1, 1), 888.0)], f"{stem}-template.pt")
        params = {name: {**meta, "v_template": rvs[np.float32(meta["timestamp"])]}
                  for k in range(len(batches)) for name, meta in kept[k][o]}
        params["info"] = {"sample_names": [n for n in names if n in params], "files": files[o], "order": o,
                          "baseline": template, "baseline_w": template_w, "bulk_offset": offsets}
        torch.save(params, f"{stem}-param.pt")
        print(f"order {o}: {len(params) - 1} spectra in {len(files[o])} batch files, saved {stem}-template.pt and -param.pt")
