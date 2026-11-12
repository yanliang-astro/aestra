#!/usr/bin/env python
"""Encode every merged spectrum with a trained stage-1 model and save the stellar latents.

The spectra are prepared exactly as in stellar_training.py (residual to the
template, shifted by minus the apparent RV) and passed through the encoder. The
latent vectors describe the activity state of every exposure; the RV
decomposition (rv_decomposition.py) keeps them fixed.

Usage (from the repository root):
    ./export_latents.py --dir data/ stellar_model/stellar_s20.pt

Input, in --merged (default: {dir}/merged): the merged spectra, template and
    param file (with v_app), as for stellar_training.py
Output (default: next to the model, {model}-latents.pt): a dict with
    "timestamp"  (N,) float32, the spectrum ids, sorted
    "latents"    (N, n_latent) float32
    "names"      observation names, in the same order
    "model"      path of the stage-1 checkpoint
"""
import argparse
import os

import numpy as np
import torch

from data import ApparentRV, SpectralData, load_batch
from models import load_stellar_model
from util import model_input


@torch.no_grad()
def encode_files(model, files, template, apparent_rv, device, batch_size=250):
    """Latents and ids of all spectra in the merged batch files."""
    model.eval()
    instrument = model.encoder.instrument
    latents, ids = [], []
    for filename in files:
        spectra = load_batch(filename)
        for start in range(0, len(spectra), batch_size):
            batch = [torch.stack(field).to(device) for field in zip(*spectra[start:start + batch_size])]
            spec_input, _ = model_input(batch, instrument, template, apparent_rv)
            latents.append(model.encode(spec_input).cpu())
            ids.append(batch[4][:, 0].cpu())
        print(f"{filename}: {len(spectra)} spectra")
    return torch.cat(latents), torch.cat(ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="stage-1 checkpoint from stellar_training.py")
    parser.add_argument("-o", "--outfile", help="output file (default: {model}-latents.pt)")
    parser.add_argument("--dir", default=os.environ.get("DATA_DIR"),
                        help="directory with the prepared data (default: $DATA_DIR)")
    parser.add_argument("--merged", help="directory with the merged spectra (default: {dir}/merged)")
    parser.add_argument("--tag", default="newprod_merged", help="name prefix of the merged spectra")
    parser.add_argument("--when", default="full", help="dataset suffix")
    parser.add_argument("-b", "--batch_size", type=int, default=250, help="spectra per encoder pass")
    parser.add_argument("--cpu", action="store_true", help="run on the CPU")
    args = parser.parse_args()
    if args.merged is None:
        if args.dir is None:
            parser.error("pass --dir or --merged, or set DATA_DIR")
        args.merged = os.path.join(args.dir, "merged")
    outfile = args.outfile or f"{os.path.splitext(args.model)[0]}-latents.pt"
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    stem = f"{args.tag}_{args.when}"
    wave, template = load_batch(os.path.join(args.merged, f"{stem}-template.pt"))[:2]
    model, _ = load_stellar_model(args.model, wave, template)
    model.to(device)
    param_file = os.path.join(args.merged, f"{stem}-param.pt")
    apparent_rv = ApparentRV(param_file).to(device)
    files = SpectralData.list_batches(args.merged, stem)
    if not files:
        raise SystemExit(f"no merged spectra {stem}_*.pt in {args.merged}")

    latents, ids = encode_files(model, files, template.to(device), apparent_rv, device, args.batch_size)
    order = torch.argsort(ids)
    latents, ids = latents[order], ids[order]
    params = torch.load(param_file, weights_only=False)
    name_of = {np.float32(m["timestamp"]): n for n, m in params.items() if n != "info"}
    names = [str(name_of[i]) for i in ids.numpy()]
    torch.save({"timestamp": ids, "latents": latents, "names": names, "model": args.model}, outfile)
    print(f"saved {outfile}: {len(ids)} spectra x {latents.shape[1]} latents")
