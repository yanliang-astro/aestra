import glob, os, urllib.request
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchinterp1d import Interp1d
import astropy.io.fits as fits
import astropy.table as aTable
from functools import partial

from instrument import Instrument
from util import BatchedFilesDataset, load_batch, cubic_transform

class Synthetic(Instrument):
    _wave_obs = torch.arange(5372.52,5476.09,0.01, dtype=torch.double)
    wave_rest = torch.arange(5372.51, 5476.10,0.01, dtype=torch.double)
    c = 299792458. # m/s

    def __init__(self, lsf=None, calibration=None):
        super().__init__(Synthetic._wave_obs, lsf=lsf, calibration=calibration)

    @classmethod
    def get_data_loader(cls, dir, select=None, which=None, tag=None, batch_size=30, shuffle=False):
        files = cls.list_batches(dir, select=select, 
                                 which=which, tag=tag)
        if which in ["train", "valid"]:
            subset = slice(0,4)
        else:
            subset = None
        load_fct = partial(load_batch, subset=subset)
        data = BatchedFilesDataset(files, load_fct, shuffle=shuffle)
        return DataLoader(data, batch_size=batch_size)

    @classmethod
    def list_batches(cls, dir, select=None, which=None, tag=None):
        if tag is None:tag = "chunk50"
        if select is None:select = cls.__mro__[0].__name__
        filename = f"{select}{tag}_*.pkl"
        batch_files = glob.glob(dir + "/" + filename)
        batches = [item for item in batch_files if not "copy" in item]

        NBATCH = len(batches)
        train_batches = batches#[:int(0.9*NBATCH)]
        #valid_batches = batches[int(0.9*NBATCH):int(0.95*NBATCH)]
        valid_batches = test_batches = batches

        if which == "test": return test_batches
        elif which == "valid": return valid_batches
        elif which == "train": return train_batches
        else: return batches

    @classmethod
    def save_batch(cls, dir, batch, select=None, tag=None, counter=None):
        if tag is None:
            tag = f"chunk{len(batch[-1])}"
        if select is None:select = cls.__mro__[0].__name__
        if counter is None:
            counter = ""
        filename = os.path.join(dir, f"{select}{tag}_{counter}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(batch, f)

    @classmethod
    def save_in_batches(cls, dir, files, select=None, tag=None, batch_size=30):
        N = len(files)
        idx = np.arange(0, N, batch_size)
        batches = np.array_split(files, idx[1:])
        for counter, ids_ in zip(idx, batches):
            print (f"saving batch {counter} / {N}")
            print("batch size:",len(ids_))
            batch = cls.make_batch(ids_)
            cls.save_batch(dir, batch, select, tag=tag, counter=counter)

    @classmethod
    def augment_spectra(cls,batch,noise=True,ratio=0.20):
        spec, w, _, ID = batch[:4]
        batch_size, spec_size = spec.shape
        device = spec.device
        wave_obs = cls._wave_obs.to(device)

        # uniform distribution of redshift offsets
        z_lim = 2e-8 # 6 m/s
        z_offset = z_lim*(torch.rand(batch_size,1, device=device)-0.5)

        wave_redshifted = wave_obs - wave_obs*z_offset

        # redshift interpolation
        spec_new = cubic_transform(wave_obs, spec, wave_redshifted)
        w_new = cubic_transform(wave_obs, w, wave_redshifted)

        if noise:
            sigma = (w_new.mean(dim=-1)**(-0.5))[:,None]
            spec_noise = sigma*torch.normal(mean=0,std=1.0,size=spec.shape,
                                            device=device)
            noise_mask = torch.rand(spec.shape).to(device)>ratio
            spec_noise[noise_mask]=0
            spec_new += spec_noise
            w_new = 1 / (1 / (w_new) + spec_noise**2)
        if spec.dtype==torch.float32:spec_new=spec_new.float()
        # ensure extrapolated values have zero weights
        wmin = wave_obs.min()
        wmax = wave_obs.max()
        out = (wave_redshifted<wmin)|(wave_redshifted>wmax)
        spec_new[out] = 1
        w_new[out] = 0
        return spec_new, w_new, _, z_offset
