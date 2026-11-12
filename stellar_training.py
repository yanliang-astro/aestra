#!/usr/bin/env python
"""Train the stellar activity encoder-decoder on the merged spectra (stage 1).

Every merged spectrum (merge_orders.py) is shifted by minus its apparent RV
(measure_apparent_rv.py), so that it holds line-shape changes but no bulk
Doppler shift. The encoder compresses each shifted residual spectrum
(spectrum - template) into n_latent numbers, the latent vector of the stellar
activity state, and a single-layer decoder maps it back to the full residual
spectrum. The loss is the weighted chi^2 of this reconstruction; the template
stays fixed. Splitting the RVs into activity and planets is a later stage.

The defaults are the settings of the paper (20 latents, 10000 epochs of one
batch of 250 spectra, peak learning rate 3e-3).

Usage (from the repository root):
    ./stellar_training.py --dir data/ -v stellar_model/stellar_s20.pt

Input, in --merged (default: {dir}/merged):
    {tag}_{when}_{k}.pt and {tag}_{when}-template.pt from merge_orders.py,
    {tag}_{when}-param.pt with the apparent RVs from measure_apparent_rv.py
Output: checkpoint {"model": state_dict of SpectrumAutoencoder, "losses": array (2, 1, n_epoch, 6)}
"""
import argparse
import os
import time

import numpy as np
import torch
from accelerate import Accelerator
from torch import optim

from data import ApparentRV, SpectralData, load_batch
from models import build_stellar_model, load_stellar_checkpoint
from util import mem_report, model_input

# Columns of the saved loss history, the layout of tellurics_training.py and of
# the RV decomposition stage, which continues this history; only "fit" is used here.
LOSS_NAMES = ["fit", "similarity", "redshift", "consistency", "velocity reg.", "flexibility"]
FIT = 0

CHECKPOINT_EVERY = 50   # epochs between checkpoints


def get_losses(model, instrument, batch, template, apparent_rv):
    """Losses of one batch of merged spectra, as a length-6 list in the LOSS_NAMES layout."""
    losses = [0.0] * len(LOSS_NAMES)
    spec_input, w = model_input(batch, instrument, template, apparent_rv)

    s = model.encode(spec_input)
    model_resid = model.decoder.spec_rest + model.decode(s) - template
    model_resid[w < 1] = 0
    losses[FIT] = 2 * model._loss(spec_input, w, model_resid, individual=True).sum()
    return losses


def checkpoint(accelerator, model, outfile, losses):
    """Save the model state and the loss history (not the optimizer state)."""
    accelerator.save({"model": accelerator.unwrap_model(model).state_dict(), "losses": losses}, outfile)


def train(model, instrument, loader, template, apparent_rv, n_epoch, outfile, losses=None,
          lr=1e-3, n_batch=None, verbose=False):
    """Train encoder and decoder for n_epoch epochs, continuing the loss history `losses` if given.

    Each epoch takes at most n_batch batches from the loader. The learning rate
    follows a one-cycle schedule over the n_epoch new epochs; resuming restarts
    the schedule and the Adam state, since neither is saved.
    """
    model.decoder.spec_rest.requires_grad_(False)   # the template stays fixed
    params = [p for part in (model.encoder, model.decoder) for p in part.parameters() if p.requires_grad]
    if verbose:
        print("model parameters:", sum(p.numel() for p in params))
        mem_report()

    optimizer = optim.Adam(params, lr=lr, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator()
    model, loader, optimizer = accelerator.prepare(model, loader, optimizer)
    template = template.to(accelerator.device)
    apparent_rv = apparent_rv.to(accelerator.device)
    if verbose:
        print("device:", accelerator.device)

    # loss history: (train/valid, 1, epoch, loss term), continued after earlier runs
    start = 0
    if losses is not None:
        losses = losses[:, :, np.sum(losses[0][0], axis=1) != 0, :]
        start = losses.shape[2]
    history = np.zeros((2, 1, start + n_epoch, len(LOSS_NAMES)))
    if start:
        history[:, :, :start] = losses

    for epoch in range(start, start + n_epoch):
        model.train()
        n_sample = 0
        for k, batch in enumerate(loader):
            batch_losses = get_losses(model, instrument, batch, template, apparent_rv)
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
            print(f"====> Epoch: {epoch}  fit loss: {history[0, 0, epoch, FIT]:.4f}")
        if epoch % CHECKPOINT_EVERY == 0 or epoch == start + n_epoch - 1:
            checkpoint(accelerator, model, outfile, history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("outfile", help="output checkpoint")
    parser.add_argument("-dir", "--dir", default=os.environ.get("DATA_DIR"),
                        help="directory with the prepared data (default: $DATA_DIR)")
    parser.add_argument("--merged", help="directory with the merged spectra (default: {dir}/merged)")
    parser.add_argument("--tag", default="newprod_merged", help="name prefix of the merged spectra")
    parser.add_argument("-when", "--when", default="full", help="dataset suffix")
    parser.add_argument("-n", "--latents", type=int, default=20, help="number of stellar latents")
    parser.add_argument("-b", "--batch_size", type=int, default=250, help="batch size")
    parser.add_argument("-l", "--batch_number", type=int, default=1,
                        help="batches per epoch (0: all)")
    parser.add_argument("-r", "--rate", type=float, default=3e-3, help="peak learning rate")
    parser.add_argument("-it", "--iteration", type=int, default=10000, help="number of epochs")
    parser.add_argument("-C", "--clobber", action="store_true", help="continue training an existing outfile")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                        help="draw batch files at random and shuffle their spectra (default: on)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print progress")
    args = parser.parse_args()
    if args.merged is None:
        if args.dir is None:
            parser.error("pass --dir or --merged, or set DATA_DIR")
        args.merged = os.path.join(args.dir, "merged")
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit(f"{args.outfile} exists: set -C/--clobber to continue training it")

    stem = f"{args.tag}_{args.when}"
    wave, template = load_batch(os.path.join(args.merged, f"{stem}-template.pt"))[:2]
    model = build_stellar_model(wave, template, args.latents)
    instrument = model.encoder.instrument
    loader = SpectralData.get_data_loader(args.merged, stem, batch_size=args.batch_size, shuffle=args.shuffle)
    apparent_rv = ApparentRV(os.path.join(args.merged, f"{stem}-param.pt"))

    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print(f"continuing {args.outfile}")
        losses = load_stellar_checkpoint(args.outfile, model)
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)

    t0 = time.time()
    train(model, instrument, loader, template, apparent_rv, args.iteration, args.outfile, losses=losses,
          lr=args.rate, n_batch=args.batch_number or None, verbose=args.verbose)
    if args.verbose:
        print(f"--- {time.time() - t0:.1f} seconds ---")
