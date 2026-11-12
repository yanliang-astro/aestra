#!/usr/bin/env python
"""Build the telluric (sky-line) mask for one echelle order from a trained sky model.

The sky model (tellurics_training.py) splits every spectrum into telluric lines,
stellar activity and continuum. This script evaluates it on all spectra of the
order, correlates the telluric absorption at each pixel with a telluric proxy,
and masks the wavelengths around the most strongly correlated lines. Orders that
end bluer than 5015 A (models.TELLURIC_MIN_WAVE) have no telluric lines: their
mask is all False, and the model is not evaluated.

Usage (from the repository root):
    ./calculate_skymask.py skymodel/sky_order51_full.pt --dir data/ --plot

Input:  {dir}/{data}{N}_{when}-template.pt and batches {data}{N}_{when}_*.pt;
        {dir}/{data}{N}_{when}-param.pt only if the water-vapor proxy is used
Output: {outdir}/{data}{N}_{when}-skymask.pt, a boolean tensor over the model
        wavelength grid (True = telluric pixel to mask), and with --plot a
        diagnostic figure next to it
"""
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from data import SpectralData, load_batch
from models import TELLURIC_MIN_WAVE, has_tellurics, load_sky_model
from util import interpolate_to_input_grid_raw

N_RENORM = 5            # passes of telluric-continuum renormalization
LSF_MIN_PEAK = 0.135    # below this, or for a multi-peaked LSF, use NEID water vapor as the proxy
MAX_LINES = 200         # number of candidate lines
MIN_SEPARATION = 0.2    # minimum distance between candidate lines [Angstrom]
MIN_DEPTH = 0.90        # keep candidates whose normalized correlation exceeds this
MASK_HALF_WIDTH = 0.2   # masked on each side of a kept line [Angstrom]


def is_single_peaked(kernel, rel_height=0.1):
    """True if the kernel has exactly one local maximum above rel_height * its highest peak."""
    k = np.ravel(kernel)
    peaks = np.flatnonzero((k[1:-1] > k[:-2]) & (k[1:-1] > k[2:])) + 1
    if len(peaks) == 0:
        return False
    return np.sum(k[peaks] >= rel_height * k[peaks].max()) == 1


@torch.no_grad()
def evaluate_sky(model, instrument, loader, template, device):
    """Evaluate the sky model on every spectrum.

    The model spectrum is transform((1 - absorption) * (1 + continuum)) * (spec_rest + stellar).
    Returns a dict of (n_spec, n_pix) arrays "absorption", "stellar", "continuum"
    and "bad" (pixel without data, weight < 1), plus the id/timestamp "ids" of
    every spectrum, sorted by timestamp.
    """
    out = {"absorption": [], "stellar": [], "continuum": [], "bad": [], "ids": []}
    for batch in loader:
        batch = [item.to(device) for item in batch]
        polyb = model.evaluate_wavelength_polynomial(batch[0], batch[4])
        spectrum, w, ssbrv, _ = interpolate_to_input_grid_raw(batch, instrument, template, polyb=polyb)
        spec_input = spectrum - template
        spec_input[w < 1] = 0
        s = model.encode(spec_input)
        lines, continuum, stellar, _ = model._forward(s, ssbrv / instrument.c, instrument.wave_obs)
        out["absorption"].append(lines.squeeze(1).cpu().numpy())
        out["stellar"].append(stellar.cpu().numpy())
        out["continuum"].append(continuum.squeeze(1).cpu().numpy())
        out["bad"].append((w < 1).cpu().numpy())
        out["ids"].append(batch[4][:, 0].cpu().numpy())
    out = {key: np.concatenate(item) for key, item in out.items()}
    # Sort by timestamp: the file order from glob depends on the file system, and
    # the summation order shifts the correlations at the 1e-5 level.
    order = np.argsort(out["ids"], kind="stable")
    return {key: item[order] for key, item in out.items()}


def renormalize(transmission, n_iter=N_RENORM):
    """Divide each spectrum by its median over the telluric continuum.

    The continuum is the half of the pixels with the lowest scatter across
    spectra, re-selected on every pass. Returns the renormalized transmission
    and the per-pixel scatter of the last pass (measured before its division).
    """
    y = transmission.copy()
    for _ in range(n_iter):
        scatter = y.std(axis=0)
        continuum = scatter < np.quantile(scatter, 0.5)
        y /= np.median(y[:, continuum], axis=1, keepdims=True)
    return y, scatter


def load_water_vapor(param_file, ids):
    """NEID water vapor (WVAPOR) of each spectrum, matched by timestamp."""
    params = torch.load(param_file, weights_only=False)
    by_time = {np.float32(params[name]["timestamp"]): params[name]["WVAPOR"]
               for name in params["info"]["sample_names"]}
    return np.array([by_time[np.float32(t)] for t in ids])


def correlation_coefficients(x, y):
    """Pearson correlation of x (n_spec,) with every row of y (n_pix, n_spec)."""
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
    return ys @ xs / len(x)


def find_deepest_lines(wave, spectrum, num_lines, min_separation):
    """Greedily pick the deepest features of `spectrum` at least min_separation apart.

    Depth is 1 - spectrum / (99% quantile). Returns [(wavelength, depth)] sorted
    by wavelength.
    """
    depth = 1 - spectrum / np.quantile(spectrum, 0.99)
    picked = []
    for i in np.argsort(depth)[::-1]:
        if all(abs(wave[i] - wave[j]) > min_separation for j in picked):
            picked.append(i)
            if len(picked) == num_lines:
                break
    return sorted(zip(wave[picked], depth[picked]))


def mask_lines(wave, line_waves, half_width):
    """Boolean mask that is True within half_width of any of line_waves."""
    mask = np.zeros(len(wave), dtype=bool)
    for w0 in line_waves:
        mask |= np.abs(wave - w0) <= half_width
    return mask


def plot_skymask(wave, transmission, stellar, continuum, bad, spec_rest, corr, skymask, line_waves,
                 title, filename, n_show=200, edge=20, max_bad=0.5):
    """Diagnostic figure: sky-model components of a sample of spectra and the mask.

    Telluric transmission is colored (plasma) by each spectrum's depth at the deepest
    telluric line (lowest median transmission). The stellar variability and the
    continuum are colored (Spectral) by the stellar flux (spec_rest + stellar) at the most
    variable stellar line: the pixel with the largest 16-84% scatter of the
    stellar component, ignoring `edge` pixels at each end and pixels without
    data (`bad`) in more than `max_bad` of the spectra, where the model is
    unconstrained. Those pixels are also left out of the stellar panel.
    The telluric mask is shaded in the telluric and correlation panels.
    """
    rows = np.random.default_rng(0).choice(len(transmission), min(n_show, len(transmission)), replace=False)
    tel_pix = np.argmin(np.median(transmission, axis=0))
    tel_depth = 1 - transmission[:, tel_pix]
    no_data = bad.mean(axis=0) > max_bad
    star_scatter = np.diff(np.quantile(stellar, [0.16, 0.84], axis=0), axis=0)[0] / 2
    star_scatter[:edge] = star_scatter[-edge:] = 0
    star_scatter[no_data] = 0
    star_pix = np.argmax(star_scatter)
    star_flux = spec_rest[star_pix] + stellar[:, star_pix]

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, width_ratios=[80, 1])
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[i, 0], sharex=axes[0]) for i in range(1, 4)]

    def draw(ax, spectra, value, cmap):
        """Plot spectra[rows] colored by value, highest value drawn last (on top)."""
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(*np.quantile(value, [0.01, 0.99]))
        for i in rows[np.argsort(value[rows])]:
            ax.plot(wave, spectra[i], color=cmap(norm(value[i])), lw=0.5)
        ax.set_ylim(np.nanquantile(spectra[rows], [0.0005, 0.9995]))
        return plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    def mark(ax, pix):
        ax.plot(wave[pix], 1, "v", color="#0b0b0b", ms=6, clip_on=False, transform=ax.get_xaxis_transform())

    angstrom = r"$\mathrm{\AA}$"
    sm = draw(axes[0], transmission, tel_depth, "plasma")
    mark(axes[0], tel_pix)
    axes[0].set_ylabel("telluric transmission")
    fig.colorbar(sm, cax=fig.add_subplot(grid[0, 1]),
                 label=f"depth of deepest\ntelluric line ({wave[tel_pix]:.2f} {angstrom})")

    sm = draw(axes[1], np.where(no_data, np.nan, stellar), star_flux, "Spectral")
    draw(axes[2], 1 + continuum, star_flux, "Spectral")
    for ax in axes[1:3]:
        mark(ax, star_pix)
    axes[1].set_ylabel("stellar variability")
    axes[2].set_ylabel("continuum")
    fig.colorbar(sm, cax=fig.add_subplot(grid[1:3, 1]),
                 label=f"stellar flux at the most variable\nstellar line ({wave[star_pix]:.2f} {angstrom})")

    axes[3].plot(wave, corr, color="#0b0b0b", lw=0.8)
    axes[3].set_ylim(0, 1.05)
    axes[3].set_ylabel("correlation with proxy")
    axes[3].set_xlabel(f"wavelength [{angstrom}]")

    for ax in (axes[0], axes[3]):
        ax.fill_between(wave, 0, 1, where=skymask, transform=ax.get_xaxis_transform(),
                        color="#e1e0d9", lw=0, zorder=-10)
        for w0 in line_waves:
            ax.axvline(w0, color="#898781", lw=0.5, ls="--", zorder=-5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[0].set_xlim(wave[0], wave[-1])
    axes[0].set_title(title, loc="left", fontsize=10)
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return wave[star_pix], star_flux


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="sky model checkpoint, named *_order{N}_*.pt")
    parser.add_argument("--dir", default=os.environ.get("DATA_DIR"),
                        help="directory with the prepared data (default: $DATA_DIR)")
    parser.add_argument("--data", default="newprod_order", help="dataset name prefix")
    parser.add_argument("--when", default="full", help="dataset suffix")
    parser.add_argument("--outdir", default="skymask", help="output directory")
    parser.add_argument("--batch_size", type=int, default=500, help="spectra per model evaluation")
    parser.add_argument("--plot", action="store_true", help="save a diagnostic figure")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU")
    args = parser.parse_args()
    if args.dir is None:
        parser.error("pass --dir or set DATA_DIR")

    match = re.search(r"_order(\d+)_", os.path.basename(args.model))
    if match is None:
        parser.error(f"cannot read the order number from {args.model}")
    tag = f"{args.data}{match.group(1)}_{args.when}"
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    template_data = load_batch(os.path.join(args.dir, f"{tag}-template.pt"))
    wave_obs = template_data[0].to(device)
    template = template_data[1].to(device)
    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, f"{tag}-skymask.pt")
    if not has_tellurics(wave_obs):
        torch.save(torch.zeros(wave_obs.shape[-1], dtype=torch.bool), outfile)
        print(f"{tag} ends at {wave_obs.max():.1f} A, bluer than {TELLURIC_MIN_WAVE:g} A: "
              f"no telluric lines, saved an all-False mask to {outfile}")
        sys.exit()
    instrument = SpectralData(wave_obs)
    model = load_sky_model(args.model, instrument, device)
    loader = SpectralData.get_data_loader(args.dir, tag, batch_size=args.batch_size, shuffle=False)

    sky = evaluate_sky(model, instrument, loader, template, device)
    ids = sky["ids"]
    transmission, scatter = renormalize(1 - sky["absorption"])
    print(f"{tag}: {len(transmission)} spectra, {transmission.shape[1]} pixels")

    lsf = model.lines_act(model.lsf_kernel).detach().cpu().numpy().ravel()
    single_peak = is_single_peaked(lsf)
    if lsf.max() < LSF_MIN_PEAK or not single_peak:
        proxy_name = "NEID water vapor"
        proxy = load_water_vapor(os.path.join(args.dir, f"{tag}-param.pt"), ids)
    else:
        proxy_name = "empirical (most variable pixel)"
        proxy = 1 - transmission[:, np.argmax(scatter)]
    print(f"proxy: {proxy_name} (LSF max {lsf.max():.3f}, single peak: {single_peak})")

    corr = correlation_coefficients(proxy, (1 - transmission).T)
    if not np.all(np.isfinite(corr)):
        raise ValueError(f"{np.sum(~np.isfinite(corr))} pixels have an undefined correlation")
    corr = gaussian_filter1d(corr, 1)
    corr[corr < 0] = 0

    wave = wave_obs.cpu().numpy().reshape(-1)
    lines = find_deepest_lines(wave, 1 - corr, MAX_LINES, MIN_SEPARATION)
    line_waves = [w0 for w0, depth in lines if depth > MIN_DEPTH]
    skymask = mask_lines(wave, line_waves, MASK_HALF_WIDTH)
    print(f"lines: {len(line_waves)}, mask fraction: {skymask.mean():.2f}")

    torch.save(torch.from_numpy(skymask), outfile)
    print(f"saved {outfile}")

    if args.plot:
        figfile = os.path.join(args.outdir, f"{tag}-skymask.png")
        title = (f"{tag}: {len(line_waves)} telluric lines, {skymask.mean():.0%} of pixels masked"
                 f" (proxy: {proxy_name})")
        spec_rest = model.spec_rest.detach().cpu().numpy().reshape(-1)
        star_wave, star_flux = plot_skymask(wave, transmission, sky["stellar"], sky["continuum"], sky["bad"], spec_rest,
                                            corr, skymask, line_waves, title, figfile)
        print(f"most variable stellar line: {star_wave:.2f} A, flux {star_flux.min():.4f}..{star_flux.max():.4f}")
        print(f"saved {figfile}")
