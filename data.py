"""Spectral data: the wavelength grid, batch files and param files of a data set.

A data set is a template file {select}-template.pt, batch files
{select}_{k}.pt, each a list of per-spectrum tuples
(wavelengths, flux, weight, SSBRV, id) written with torch.save, and a param
file {select}-param.pt with the metadata of every observation.
"""
import glob
import os
import random
import re
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

DEFAULT_ORDERS = [o for o in range(31, 75) if o not in (69, 70)]   # order indices of the paper
TREND_INDICATORS = ("ccf_depth", "ccf_fwhm", "ccf_bis", "ccf_offset")   # written by measure_ccf_indicators.py


def load_batch(filename):
    """Load a batch or template file written with torch.save."""
    return torch.load(filename, map_location="cpu")


class BatchedFilesDataset(IterableDataset):
    """Stream the spectra of a list of batch files, one file at a time.

    Each file holds a list of per-spectrum tuples. One pass over the dataset
    reads len(file_list) files; a DataLoader on top of it forms batches that can
    span several files.

    shuffle: draw every file at random (with replacement) instead of in order.
    shuffle_instance: shuffle the spectra within each file.
    Both use Python's random module.

    After https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """

    def __init__(self, file_list, load_fct=load_batch, shuffle=False, shuffle_instance=False):
        if not file_list:
            raise ValueError("file list is empty")
        self.file_list = file_list
        self.load_fct = load_fct
        self.shuffle = shuffle
        self.shuffle_instance = shuffle_instance

    def process_data(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.file_list) - 1)
        data = self.load_fct(self.file_list[idx])
        if self.shuffle_instance:
            random.shuffle(data)
        yield from data

    def __iter__(self):
        return chain.from_iterable(map(self.process_data, range(len(self.file_list))))


class SpectralData(nn.Module):
    """Wavelength grid wave_obs of a set of spectra, and the speed of light c [m/s].

    The grid is a buffer, so it moves to the device of the model that holds it.
    """
    c = 299792458.0   # speed of light [m/s]

    def __init__(self, wave_obs):
        super().__init__()
        self.register_buffer("wave_obs", wave_obs)

    @staticmethod
    def list_batches(directory, select):
        """Batch files {select}_{k}.pt in directory, sorted by k."""
        files = {}
        for name in glob.glob(os.path.join(directory, f"{select}_*.pt")):
            match = re.fullmatch(rf"{re.escape(select)}_(\d+)\.pt", os.path.basename(name))
            if match:
                files[int(match.group(1))] = name
        return [files[k] for k in sorted(files)]

    @classmethod
    def get_data_loader(cls, directory, select, batch_size=200, shuffle=True):
        """DataLoader over the spectra of the batch files {select}_{k}.pt.

        shuffle: draw the batch files at random (with replacement) and shuffle
        the spectra within each file; otherwise read all files in order.
        """
        files = cls.list_batches(directory, select)
        if not files:
            raise FileNotFoundError(f"no batch files {select}_*.pt in {directory}")
        data = BatchedFilesDataset(files, load_batch, shuffle=shuffle, shuffle_instance=shuffle)
        return DataLoader(data, batch_size=batch_size)


def save_params(params, filename):
    """Rewrite a param file in one step (no half-written file if interrupted)."""
    tmp = filename + ".tmp"
    torch.save(params, tmp)
    os.replace(tmp, filename)


def load_rvs(param_file):
    """Names, t [JD - 2459000], v_app [m/s] and CCF indicators (None if missing), sorted by t."""
    params = torch.load(param_file, weights_only=False)
    names = [str(n) for n in params if n != "info"]
    missing = [n for n in names if "v_app" not in params[n]]
    if missing:
        raise SystemExit(f"{len(missing)} observations have no v_app: run measure_apparent_rv.py first")
    t = np.array([params[n]["timestamp"] for n in names], dtype=np.float32)
    order = np.argsort(t, kind="stable")
    names = [names[i] for i in order]
    v_app = np.array([params[n]["v_app"] for n in names])
    indicators = None
    if all(key in params[names[0]] for key in TREND_INDICATORS):
        indicators = np.array([[params[n][key] for key in TREND_INDICATORS] for n in names])
    return names, t[order], v_app, indicators


class ApparentRV:
    """Apparent RVs from measure_apparent_rv.py, looked up by spectrum id (timestamp)."""

    def __init__(self, param_file):
        params = torch.load(param_file, weights_only=False)
        names = [n for n in params if n != "info"]
        missing = [n for n in names if "v_app" not in params[n]]
        if missing:
            raise SystemExit(f"{len(missing)} observations in {param_file} have no v_app: "
                             "run measure_apparent_rv.py first")
        t = np.array([params[n]["timestamp"] for n in names], dtype=np.float32)
        order = np.argsort(t)
        if np.any(np.diff(t[order]) == 0):
            raise ValueError(f"observation timestamps in {param_file} are not unique")
        self.t = torch.from_numpy(t[order])
        self.v = torch.tensor([params[names[i]]["v_app"] for i in order], dtype=torch.float64)

    def to(self, device):
        self.t, self.v = self.t.to(device), self.v.to(device)
        return self

    def __call__(self, ids):
        """v_app [m/s], shape (B, 1), of the spectra with ids of shape (B, 1)."""
        i = torch.searchsorted(self.t, ids[:, 0]).clamp(max=len(self.t) - 1)
        if not torch.equal(self.t[i], ids[:, 0]):
            raise KeyError("spectrum without an apparent RV in the param file")
        return self.v[i].unsqueeze(1)
