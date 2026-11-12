#!/usr/bin/env python
"""Remove tellurics and continuum from every order and merge the orders into stellar 1D spectra.

For each order, the sky model (tellurics_training.py) is evaluated on every
spectrum, and its telluric lines and continuum are divided out. Telluric lines
are only removed inside the order's sky mask (calculate_skymask.py); elsewhere
the model's telluric component is ignored. What remains is the stellar spectrum,
including its variability. Within each batch file, pixels without data, without
light, or more than 1% above the median are rejected, and the weights include
the local scatter of the cleaned spectra.

The orders are then joined into one stellar spectrum per observation on a common
wavelength grid: overlapping wavelengths come from the redder order, and only
observations present in every order are kept. The template is the per-pixel
median over all merged spectra. To keep memory bounded, every batch file is
cleaned twice: once per order for the template, once per batch for the output.

Usage (from the repository root):
    ./merge_orders.py --dir data/                  # orders 31-68 and 71-74
    ./merge_orders.py --dir data/ --orders 51 52 --models "skymodel/blue_order{order}_{when}.pt"

Input, for every order N:
    {dir}/{data}{N}_{when}-template.pt, batch files {dir}/{data}{N}_{when}_{k}.pt
    and {dir}/{data}{N}_{when}-param.pt; the sky model (--models, default
    skymodel/sky_order{N}_{when}.pt) and its sky mask from calculate_skymask.py
    (--masks, default skymask/{data}{N}_{when}-skymask.pt)
Output, in --outdir (default: {dir}/merged):
    {outtag}_{when}_{k}.pt       lists of per-spectrum tuples (0, flux, weight, SSBRV, id);
                                 0 stands for the wavelengths, shared by all spectra
    {outtag}_{when}-template.pt  [wave, flux, weight, 0, 888], each of shape (1, n_pix)
    {outtag}_{when}-param.pt     per-observation metadata and run info
"""
import argparse
import glob
import os
import re
import warnings

import numpy as np
import torch

from data import DEFAULT_ORDERS, SpectralData, load_batch
from models import load_sky_model
from util import interpolate_to_input_grid_raw

META_KEYS = ["OBSJD", "DATE-OBS", "AIRMASS", "WVAPOR", "E_VER", "ZENITH", "CCFRVMOD", "DVRMSMOD", "timestamp"]
MIN_FLUX = 0.05          # reject pixels where the batch median is darker than this
MAX_EXCESS = 0.01        # reject pixels this far above the batch median
REJECTED_WEIGHT = 1e-6   # weight of rejected pixels
SCATTER_WINDOW = 6       # pixels in the running window of the local scatter


# ------------------------------------------------------------- one order --

def local_scatter_weight(resid, valid, window=SCATTER_WINDOW):
    """Inverse variance of resid over a running window, from the valid pixels only.

    Windows with 3 or fewer valid pixels get a weight of 1e-12. The result is
    padded at both ends to the length of resid.
    """
    resid, valid = resid.float(), valid.float()

    def running_sum(x):
        c = torch.cumsum(x, dim=1)
        return c[:, window:] - c[:, :-window]

    s1, s2, n = running_sum(resid), running_sum(resid ** 2), running_sum(valid)
    var = torch.full_like(s1, 1e12)
    good = n > 3
    var[good] = s2[good] / n[good] - (s1[good] / n[good]) ** 2
    pad = window // 2
    var = torch.cat([var[:, :1].repeat(1, pad), var, var[:, -1:].repeat(1, pad)], dim=1)
    return 1.0 / var


@torch.no_grad()
def clean_batch(model, instrument, batch, template, skymask):
    """Divide the sky model's telluric lines and continuum out of a batch of spectra.

    Telluric lines are only divided out where skymask is True. Returns the
    stellar flux and weights on the order's wavelength grid, SSBRV and ids.
    """
    wave_obs = instrument.wave_obs
    polyb = model.evaluate_wavelength_polynomial(batch[0], batch[4])
    spectrum, w, ssbrv, ids = interpolate_to_input_grid_raw(batch, instrument, template, polyb=polyb)
    spec_input = spectrum - template
    spec_input[w < 1] = 0
    s = model.encode(spec_input)
    z = ssbrv / instrument.c
    lines, continuum, _, _ = model._forward(s, z, wave_obs, skymask=skymask)
    telluric = model.transform((1 - lines) * (1 + continuum), z, wave_obs)
    flux = (spectrum - 1e-3 * s[:, -1:]) / telluric   # also undo the model's additive offset

    median = torch.median(flux, dim=0)[0]
    resid = flux - median
    bad = (w < 1) | (median < MIN_FLUX) | (resid > MAX_EXCESS)
    resid[bad] = 0
    flux[bad] = 0
    w[bad] = REJECTED_WEIGHT
    w = 1 / (1 / local_scatter_weight(resid, w > 1) + 1 / w)
    return flux, w, ssbrv, ids


class Order:
    """Data, sky model and sky mask of one order."""

    def __init__(self, order, args, device):
        self.order = order
        self.device = device
        self.tag = f"{args.data}{order}_{args.when}"
        wave, template = load_batch(os.path.join(args.dir, f"{self.tag}-template.pt"))[:2]
        self.wave = wave.numpy().reshape(-1)
        self.template = template.to(device)
        self.instrument = SpectralData(wave.to(device))
        fmt = dict(order=order, when=args.when, data=args.data, dir=args.dir.rstrip("/"))
        self.model = load_sky_model(args.models.format(**fmt), self.instrument, device)
        self.skymask = load_batch(args.masks.format(**fmt)).bool().reshape(-1).to(device)
        if len(self.skymask) != len(self.wave):
            raise ValueError(f"order {order}: sky mask and wavelength grid differ in length")
        self.params = torch.load(os.path.join(args.dir, f"{self.tag}-param.pt"), weights_only=False)
        self.files = {}   # batch files {tag}_{k}.pt by k
        for name in glob.glob(os.path.join(args.dir, f"{self.tag}_*.pt")):
            match = re.fullmatch(rf"{re.escape(self.tag)}_(\d+)\.pt", os.path.basename(name))
            if match:
                self.files[int(match.group(1))] = name

    def ids(self, k):
        """Observation ids in batch file k, read without loading the spectra."""
        return {s[4].numpy()[0] for s in torch.load(self.files[k], mmap=True)}

    def clean(self, k):
        """Stellar (flux, weight, ssbrv, ids) of batch file k, as numpy arrays."""
        batch = [torch.stack(field).to(self.device) for field in zip(*load_batch(self.files[k]))]
        flux, w, ssbrv, ids = clean_batch(self.model, self.instrument, batch, self.template, self.skymask)
        return flux.cpu().numpy(), w.cpu().numpy(), ssbrv.cpu().numpy(), ids[:, 0].cpu().numpy()


# ------------------------------------------------------------ all orders --

def overlap_masks(waves):
    """Pixels kept from each order (sorted by wavelength): those bluer than the next order."""
    return [w < waves[i + 1].min() if i + 1 < len(waves) else np.ones_like(w, dtype=bool)
            for i, w in enumerate(waves)]


def select_observations(orders):
    """Observations to merge, as {k: sorted ids}, and the name of every id.

    An observation is merged if all orders list it in their param file and hold
    it in their batch file k (the batch files of all orders are cut from the same
    observation list). Observations are identified by their timestamp.
    """
    first = orders[0].params
    listed = set.intersection(*(set(o.params["info"]["sample_names"]) for o in orders))
    name_of = {np.float32(first[n]["timestamp"]): n for n in listed}
    if len(name_of) != len(listed):
        raise ValueError("observation timestamps are not unique")
    in_files = [{k: o.ids(k) for k in o.files} for o in orders]
    ks = sorted(set.intersection(*(set(o.files) for o in orders)))
    merged = {k: sorted(set(name_of).intersection(*(ids[k] for ids in in_files))) for k in ks}
    n_merged = sum(len(ids) for ids in merged.values())
    everywhere = set(name_of).intersection(*(set().union(*ids.values()) for ids in in_files))
    print(f"{len(orders)} orders: {len(listed)} observations listed in every order, "
          f"{n_merged} in the batch files of every order")
    if len(everywhere) > n_merged:
        print(f"WARNING: {len(everywhere) - n_merged} observations are in different batch files "
              "in different orders and are left out")
    return merged, name_of


def median_template(flux, weight):
    """Per-pixel median flux and weight over the good pixels (flux > 0, weight > 1)."""
    good = (flux > 0) & (weight > 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns: no good pixel
        template = np.nanmedian(np.where(good, flux, np.nan), axis=0)
        template_w = np.nanmedian(np.where(good, weight, np.nan), axis=0)
    return np.nan_to_num(template), np.nan_to_num(template_w)


def merged_template(orders, keep, merged):
    """Template of the merged spectra, order by order from all merged observations."""
    ids = list(set().union(*merged.values()))
    templates = []
    for o, kept in zip(orders, keep):
        flux, weight = [], []
        for k in sorted(merged):
            f, w, _, i = o.clean(k)
            rows = np.isin(i, ids)
            flux.append(f[rows][:, kept])
            weight.append(w[rows][:, kept])
        templates.append(median_template(np.concatenate(flux), np.concatenate(weight)))
        print(f"order {o.order}: template from {sum(len(f) for f in flux)} spectra")
    return np.concatenate([t[0] for t in templates]), np.concatenate([t[1] for t in templates])


def write_merged(orders, keep, k, ids, filename):
    """Clean batch file k of every order and save the merged spectra of ids."""
    flux, weight = [], []
    for o, kept in zip(orders, keep):
        f, w, ssbrv, i = o.clean(k)
        row_of = {v: r for r, v in enumerate(i)}
        rows = [row_of[v] for v in ids]
        flux.append(f[rows][:, kept])
        weight.append(w[rows][:, kept])
    ssbrv = ssbrv[rows]   # SSBRV of the reddest order
    # torch.tensor copies each row: a view would make torch.save write the whole batch array
    batch = [(torch.zeros(1, dtype=torch.float64), torch.tensor(f, dtype=torch.float32),
              torch.tensor(w, dtype=torch.float32), torch.tensor(v, dtype=torch.float64),
              torch.tensor([i], dtype=torch.float32))
             for f, w, v, i in zip(np.concatenate(flux, axis=1), np.concatenate(weight, axis=1), ssbrv, ids)]
    torch.save(batch, filename)
    print(f"saved {filename} ({len(batch)} spectra, {len(batch[0][1])} pixels)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", default=os.environ.get("DATA_DIR"),
                        help="directory with the prepared data (default: $DATA_DIR)")
    parser.add_argument("--orders", type=int, nargs="+", default=DEFAULT_ORDERS,
                        help="echelle orders (default: 31-74 without 69 and 70)")
    parser.add_argument("--data", default="newprod_order", help="dataset name prefix")
    parser.add_argument("--when", default="full", help="dataset suffix")
    parser.add_argument("--models", default="skymodel/sky_order{order}_{when}.pt",
                        help="sky model path; {dir}, {order}, {when} and {data} are filled in")
    parser.add_argument("--masks", default="skymask/{data}{order}_{when}-skymask.pt",
                        help="sky mask path from calculate_skymask.py; {dir}, {order}, {when} and {data} are filled in")
    parser.add_argument("--outdir", help="output directory (default: {dir}/merged)")
    parser.add_argument("--outtag", default="newprod_merged", help="output name prefix")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU")
    args = parser.parse_args()
    if args.dir is None:
        parser.error("pass --dir or set DATA_DIR")
    outdir = args.outdir or os.path.join(args.dir, "merged")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    orders = [Order(o, args, device) for o in sorted(set(args.orders))]
    if any(a.wave.min() >= b.wave.min() for a, b in zip(orders, orders[1:])):
        raise ValueError("orders must be sorted by wavelength")
    keep = overlap_masks([o.wave for o in orders])
    wave = np.concatenate([o.wave[kept] for o, kept in zip(orders, keep)])
    merged, name_of = select_observations(orders)
    merged = {k: ids for k, ids in merged.items() if ids}

    template, template_w = merged_template(orders, keep, merged)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"{args.outtag}_{args.when}")
    filenames = [f"{stem}_{k}.pt" for k in merged]
    for k, filename in zip(merged, filenames):
        write_merged(orders, keep, k, merged[k], filename)

    torch.save([torch.from_numpy(wave[None]).double(), torch.from_numpy(template[None]).float(),
                torch.from_numpy(template_w[None]).float(), torch.zeros((1, 1), dtype=torch.float64),
                torch.full((1, 1), 888.0)], f"{stem}-template.pt")
    names = [name_of[i] for k in merged for i in merged[k]]
    first = orders[0].params
    params = {n: {key: first[n][key] for key in META_KEYS} for n in names}
    params["info"] = {"sample_names": names, "order": [o.order for o in orders], "files": filenames,
                      "wave_obs": wave[None], "baseline": template, "baseline_w": template_w}
    torch.save(params, f"{stem}-param.pt")
    print(f"saved {stem}-template.pt and {stem}-param.pt ({len(names)} spectra, {len(wave)} pixels)")
