#!/usr/bin/env python
"""Train the telluric (sky) model for one echelle order.

The model (models.TelluricModel) splits every spectrum into telluric
lines, stellar variability and continuum. skytrain.sh trains it in two stages:
  1. --consistency: warm-up that only minimizes the scatter of the residual
     spectra (data - template) after the learned wavelength correction;
  2. the full fit: weighted chi^2 of the model spectrum plus regularization.

Usage (from the repository root):
    ./tellurics_training.py --order 51 --dir data/ --batch_size 1000 --rate 2e-2 \\
        --iteration 1000 --consistency newprod_order skymodel/sky_order51_full.pt

Input:  {dir}/{data}{order}_{when}-template.pt and batches {data}{order}_{when}_*.pt
Output: checkpoint {"model": [state_dict], "losses": array (2, 1, n_epoch, 6)}
"""
import argparse
import os
import time

import numpy as np
import torch
from accelerate import Accelerator
from torch import optim

from data import SpectralData, load_batch
from models import TelluricModel, has_tellurics
from util import interpolate_to_input_grid_raw, mem_report

# Columns of the saved loss history. The layout is kept from earlier versions so
# that old checkpoints can be resumed; this model only fills "fit",
# "consistency" and "flexibility". In the fit stage, the "consistency" column
# holds the continuum-scatter penalty instead. The validation half stays zero.
LOSS_NAMES = ["fit", "similarity", "redshift", "consistency", "velocity reg.", "flexibility"]
FIT, CONSISTENCY, FLEXIBILITY = 0, 3, 5

CONTINUUM_MAX_STD = 0.005   # continuum scatter allowed before it is penalized
CHECKPOINT_EVERY = 100      # epochs between checkpoints


def get_losses(model, instrument, batch, template, skymask=None, consistency=False):
    """Losses of one batch, as a length-6 list in the LOSS_NAMES layout.

    consistency: warm-up loss only, the scatter across the batch of the
    residual spectra after the learned wavelength correction.
    """
    losses = [0.0] * len(LOSS_NAMES)
    wave_obs = instrument.wave_obs

    # shift to the (apparent) stellar restframe, the frame of wave_obs
    polyb = model.evaluate_wavelength_polynomial(batch[0], batch[4])
    spectrum, w, ssbrv, jd = interpolate_to_input_grid_raw(batch, instrument, template, polyb=polyb)
    spec_input = spectrum - template
    spec_input[w < 1] = 0
    batch_size = jd.shape[0]

    if consistency:
        losses[CONSISTENCY] = 1e2 * torch.sum(spec_input.std(dim=0).pow(2)) * batch_size
        return losses

    s = model.encode(spec_input)
    lines, continuum, y_act, spec_model = model._forward(s, ssbrv / instrument.c, wave_obs, skymask=skymask)

    # weighted chi^2 per spectrum, normalized by the number of unmasked pixels
    chi2 = torch.sum(w * (spectrum - spec_model).pow(2), dim=-1) / torch.sum(w > 1, dim=-1)
    losses[FIT] = torch.sum(chi2)

    # penalize continuum scatter across the batch above CONTINUUM_MAX_STD
    continuum_std = continuum.std(dim=0).squeeze(0)
    penalty = torch.zeros_like(continuum_std)
    high = continuum_std > CONTINUUM_MAX_STD
    penalty[high] = 1e3 * (continuum_std[high] - CONTINUUM_MAX_STD)
    losses[CONSISTENCY] = batch_size * torch.mean(penalty)

    # keep the components small where they are not needed
    quiet = (y_act.std() < 0.002) & (spectrum / (1 - lines.squeeze(1)) > 0.9)
    flex = 10 * torch.sum(y_act[quiet].pow(2))
    flex += batch_size * torch.sum(continuum.mean(dim=0).pow(2))
    flex += 2e-2 * torch.sum(lines[lines < 0.05].pow(2))
    losses[FLEXIBILITY] = flex
    return losses


def checkpoint(accelerator, model, outfile, losses):
    """Save the model state and the loss history (not the optimizer state)."""
    accelerator.save({"model": [accelerator.unwrap_model(model).state_dict()], "losses": losses}, outfile)


def load_checkpoint(filename, model):
    """Load the model state from a checkpoint; returns the loss history."""
    # weights_only=False: the checkpoint also holds the loss history (numpy array)
    state = torch.load(filename, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"][0], strict=True)
    return state["losses"]


def train(model, instrument, loader, template, skymask, n_epoch, outfile, losses=None,
          lr=1e-3, n_batch=None, consistency=False, verbose=False):
    """Train for n_epoch epochs, continuing the loss history `losses` if given.

    Each epoch takes at most n_batch batches from the loader. The learning rate
    follows a one-cycle schedule over the n_epoch new epochs; resuming restarts
    the schedule and the Adam state, since neither is saved.
    """
    params = list(model.parameters())
    if verbose:
        print("model parameters:", sum(p.numel() for p in params if p.requires_grad))
        mem_report()

    optimizer = optim.Adam([{"params": params, "lr": lr}], eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator()
    model, instrument, loader, optimizer = accelerator.prepare(model, instrument, loader, optimizer)
    template = template.to(accelerator.device)
    if verbose:
        print("device:", accelerator.device)

    # loss history: (train/valid, 1, epoch, loss term), continued after earlier runs
    start = 0
    if losses is not None:
        losses = losses[:, :, np.sum(losses[0][0], axis=1) > 0, :]
        start = losses.shape[2]
    history = np.zeros((2, 1, start + n_epoch, len(LOSS_NAMES)))
    if start:
        history[:, :, :start] = losses

    for epoch in range(start, start + n_epoch):
        model.train()
        instrument.train()
        n_sample = 0
        for k, batch in enumerate(loader):
            batch_losses = get_losses(model, instrument, batch, template, skymask=skymask,
                                      consistency=consistency)
            accelerator.backward(sum(batch_losses))
            accelerator.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            history[0, 0, epoch] += [l.item() if torch.is_tensor(l) else l for l in batch_losses]
            n_sample += len(batch[0])
            if n_batch is not None and k == n_batch - 1:
                break
        history[0, 0, epoch] /= n_sample
        scheduler.step()

        if verbose:
            print(f"====> Epoch: {epoch}")
            print("TRAINING Losses:", history[0, 0, epoch])
        if epoch % CHECKPOINT_EVERY == 0 or epoch == start + n_epoch - 1:
            checkpoint(accelerator, model, outfile, history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data", help="dataset name prefix, e.g. newprod_order")
    parser.add_argument("outfile", help="output checkpoint")
    parser.add_argument("-o", "--order", type=int, required=True, help="echelle order")
    parser.add_argument("-dir", "--dir", default=os.environ.get("DATA_DIR"),
                        help="directory with the prepared data (default: $DATA_DIR)")
    parser.add_argument("-when", "--when", default="full", help="dataset suffix")
    parser.add_argument("-n", "--latents", type=int, default=1, help="number of telluric latents")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("-l", "--batch_number", type=int, default=None, help="batches per epoch (default: all)")
    parser.add_argument("-r", "--rate", type=float, default=1e-3, help="peak learning rate")
    parser.add_argument("-it", "--iteration", type=int, default=100000, help="number of epochs")
    parser.add_argument("-c", "--consistency", action="store_true", help="warm-up stage: consistency loss only")
    parser.add_argument("-C", "--clobber", action="store_true", help="continue training an existing outfile")
    parser.add_argument("-shuffle", "--shuffle", action="store_true", help="shuffle batch files and spectra")
    parser.add_argument("--detect_anomaly", action="store_true",
                        help="enable autograd anomaly detection (debugging, ~20%% slower)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print progress")
    args = parser.parse_args()
    if args.dir is None:
        parser.error("pass --dir or set DATA_DIR")
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit(f"{args.outfile} exists: set -C/--clobber to continue training it")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    tag = f"{args.data}{args.order}_{args.when}"
    wave_obs, template = load_batch(os.path.join(args.dir, f"{tag}-template.pt"))[:2]
    instrument = SpectralData(wave_obs)
    model = TelluricModel(wave_obs, template.float().detach().clone(), instrument, n_latent=args.latents)
    loader = SpectralData.get_data_loader(args.dir, tag, batch_size=args.batch_size, shuffle=args.shuffle)
    # Orders that end bluer than TELLURIC_MIN_WAVE (5015 A) have no telluric lines: all are masked.
    skymask = torch.full((wave_obs.shape[-1],), has_tellurics(wave_obs), dtype=torch.bool)

    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print(f"continuing {args.outfile}")
        losses = load_checkpoint(args.outfile, model)

    t0 = time.time()
    train(model, instrument, loader, template, skymask, args.iteration, args.outfile, losses=losses,
          lr=args.rate, n_batch=args.batch_number, consistency=args.consistency, verbose=args.verbose)
    if args.verbose:
        print(f"--- {time.time() - t0:.1f} seconds ---")
