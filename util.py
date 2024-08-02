#!/usr/bin/env python
# coding: utf-8
import io, os, sys, time, random
import numpy as np
import pickle
from scipy.special import gamma
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from itertools import chain
import pickle, humanize, psutil, GPUtil, io, random
from astropy.io import fits
from torchinterp1d import Interp1d
from torchcubicspline import natural_cubic_spline_coeffs
from astropy.timeseries import LombScargle
from spender_model import SpectrumAutoencoder

# Normalize x to range [-1, 1] based on x_ref
def normalize_x(x,x_ref):
    x_min = x_ref.min()
    x_max = x_ref.max()
    x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
    return x_normalized

# Function to fit the cubic model using matrix inversion
def fit_cubic_via_matrix_inversion(x_in, y, x_ref):
    N_batch, N_data = y.shape
    x = normalize_x(x_in,x_ref)
    # Create Vandermonde matrix (N_data, 4)
    X = torch.stack([x**3, x**2, x, torch.ones_like(x)], dim=-1)  
    X_T = X.transpose(0, 1)  # Transpose of X (4, N_data)

    # Perform batch-wise matrix multiplication and inversion
    XTX = X_T @ X  # (4, 4)
    XTX_inv = torch.inverse(XTX)  # (4, 4)
    XTy = torch.einsum('ij,bj->bi', X_T, y)  # (N_batch, 4)
    # Calculate coefficients using the normal equation
    coefficients = torch.einsum('ij,bj->bi', XTX_inv, XTy)  # (N_batch, 4)
    return coefficients

# Function to evaluate the cubic model using the fitted coefficients
def evaluate_cubic(x_in, coefficients, x_ref):
    x = normalize_x(x_in,x_ref)
    # Create Vandermonde matrix for evaluation points (N_data, 4)
    X_eval = torch.stack([x**3, x**2, x, torch.ones_like(x)], dim=-1)
    # Perform batched polynomial evaluation
    y_eval = torch.einsum('bi,ij->bj', coefficients, X_eval.transpose(0, 1))
    return y_eval

def load_master_fsr_mask():
    filename = "neidMaster_FSR_Mask20210218_v002.fits"
    hdulist = fits.open(filename)
    header = hdulist[0].header
    fsr_mask = hdulist[0].data
    fsr_mask = np.array(fsr_mask,dtype=bool)
    return fsr_mask

def normalize_residual_cubic(wave_obs,spectrum,weight,template):
    # normalize input residual spectrum to zero
    x = wave_obs.float()
    spec_input = spectrum-template
    n_order = wave_obs.shape[0]
    for o in range(n_order):
        bad = torch.any(weight[:,o,:]<1.0, dim=0)
        coefficients = fit_cubic_via_matrix_inversion(x[o,~bad], spec_input[:,o,~bad],x[o])
        y_eval = evaluate_cubic(x[o], coefficients,x[o])
        spec_input[:,o,:] -= y_eval
    spec_input[weight<1.0] = 0
    return spec_input

def normalize_residual(spectrum,weight,template):
    bad = weight<1.0
    # normalize input residual spectrum to zero
    spec_input = spectrum-template
    spec_input[bad] = 0
    spec_mean = spec_input.sum(dim=2)/(~bad).sum(dim=2)
    spec_input -= spec_mean[:,:,None]
    spec_input[bad] = 0
    return spec_input

def divide_sky_model(spec,w,spec_sky,fringe_spec,template):
    spec /= spec_sky # divide our telluric model
    spec -= fringe_spec # subtract our fringe model
    spec_input = normalize_residual(spec,w,template)
    return spec_input

def interpolate_to_input_grid(batch,instrument,template_data,skymask=None,telluric_raw=1,aug=False,planetary_rv=0):
    wave_raw,spec_raw,w_raw,ssbrv = batch[:4]
    wave_obs = instrument.wave_obs
    n_order,n_spec = wave_obs.shape
    n_batch = spec_raw.shape[0]
    device = wave_obs.device
    template = template_data[1].repeat(n_batch,1,1)

    if skymask==None:skymask=wave_obs<0
    # produce augmentation data -- inject rv offset
    if aug:
        z_lim = 5e-8 # 15 m/s
        #z_lim = 1e-8 # 3 m/s
        z_offset = z_lim*(torch.rand(n_batch,1, device=device)-0.5)
    else: z_offset = 0

    # total rv = ssbrv + injected planetary_rv + rv offset
    z = (ssbrv+planetary_rv)/instrument.c + z_offset

    spectrum = torch.zeros((n_batch,n_order,n_spec),device=device)
    weight = torch.zeros((n_batch,n_order,n_spec),device=device)
    wave = wave_raw + wave_raw * z[:,:,None]

    out = torch.zeros_like(spectrum,dtype=bool)
    for i in range(n_order):
        spectrum[:,i,:] = Interp1d()(wave[:,i,:], spec_raw[:,i,:], wave_obs[i])
        weight[:,i,:] = Interp1d()(wave[:,i,:], w_raw[:,i,:], wave_obs[i])
        wmin = wave[:,i,:].min(dim=1)[0]
        wmax = wave[:,i,:].max(dim=1)[0]
        out_ = (wave_obs[i]<wmin.unsqueeze(1))|(wave_obs[i]>wmax.unsqueeze(1))
        out[:,i,:] = out_
    ill = (template==0)|(spectrum==0)|(weight<1.0)
    # mask out +/- 1 pixel of bad input data (zero flux)
    bad = (spectrum<(template*0.6))|ill
    # after linear interpolation, bad flux values are at most half of template values
    # cut at 0.6 for safety
    bad |= torch.roll(bad, -1, dims=2)
    bad |= torch.roll(bad, +1, dims=2)
    bad |= out

    if aug:
        sigma = 0.5*(weight[~bad].mean())**(-0.5)
        spec_noise = sigma*torch.normal(mean=0,std=1.0,size=spectrum.shape,
                                        device=device)
        spectrum += spec_noise

    weight[bad] = 1e-12
    spectrum[bad] = 0.0
    weight[:,skymask] = 1e-12
    spectrum[:,skymask] = 0
    return spectrum, weight, z_offset


def merge_batch(file_batches):
    waves = [];spectra = [];weights = []
    ssbrv = [];specid = []
    for batchname in file_batches:
        print("batchname:",batchname)
        batch = load_batch(batchname)
        waves.append(batch[0])
        spectra.append(batch[1])
        weights.append(batch[2])
        ssbrv.append(batch[3])
        specid.append(batch[4])
    waves =  torch.cat(waves,axis=0)
    spectra = torch.cat(spectra,axis=0)
    weights = torch.cat(weights,axis=0)
    ssbrv = torch.cat(ssbrv,axis=0)
    specid = torch.cat(specid,axis=0)
    print("waves:",waves.shape,"spectra:",spectra.shape,
          "w:",weights.shape,"ssbrv:",ssbrv.shape,
          "specid:",specid.shape)
    return waves,spectra, weights, ssbrv, specid

def load_model(path, instrument, device):
    mdict = torch.load(path, map_location=device)
    model_dict = mdict['model'][0]
    wave_rest = model_dict['decoder.wave_rest']
    spec_rest = model_dict['decoder.spec_rest']
    n_latent = 3#len(model_dict['encoder.mlp.mlp.9.bias'])

    model = SpectrumAutoencoder(instrument,
                                wave_rest=wave_rest,
                                spec_rest=spec_rest,
                                n_latent=n_latent,
                                normalize=False)
    model.load_state_dict(mdict["model"][0],strict=False)
    model.to(device)
    model.eval()
    return model,mdict["losses"],n_latent

def simulate_planet(t,amp=1,period=0.11,t0=0):
    phase = ((t/period)-t0)%1
    v_planet = amp*torch.sin(2*np.pi*phase)[:,None]
    return phase,v_planet

def cubic_evaluate(coeffs, tnew):
    t = coeffs[0]
    a,b,c,d = [item.squeeze(-1) for item in coeffs[1:]]
    maxlen = b.size(-1) - 1
    index = torch.bucketize(tnew, t) - 1
    index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
    # will never access the last element of self._t; this is correct behaviour
    fractional_part = tnew - t[index]

    batch_size, spec_size = tnew.shape
    batch_ind = torch.arange(batch_size,device=tnew.device)
    batch_ind = batch_ind.repeat((spec_size,1)).T

    inner = c[batch_ind, index] + d[batch_ind, index] * fractional_part
    inner = b[batch_ind, index] + inner * fractional_part
    return a[batch_ind, index] + inner * fractional_part

def cubic_transform(xrest, yrest, wave_shifted):
    coeffs = natural_cubic_spline_coeffs(xrest, yrest.unsqueeze(-1))
    out = cubic_evaluate(coeffs, wave_shifted)
    return out

def moving_mean(x,y,w=None,n=20,skip_weight=True):
    dx = (x.max()-x.min())/n
    xgrid = np.linspace(x.min(),x.max(),n+2)
    xgrid = xgrid[1:-1]
    ygrid = np.zeros_like(xgrid)
    delta_y = np.zeros_like(xgrid)
    non_zero = ygrid>-np.inf
    for i,xmid in enumerate(xgrid):
        mask = x>(xmid-dx)
        mask *= x<(xmid+dx)
        if mask.sum()<5:
            non_zero[i] = False
            continue
        if skip_weight:
            ygrid[i] = np.mean(y[mask])
            delta_y[i] = y[mask].std()/np.sqrt(mask.sum())
        else:
            if w[mask].sum()==0:
                print(w[mask])
            ygrid[i] = np.average(y[mask],weights=w[mask])
            delta_y[i] = np.sqrt(np.cov(y[mask], aweights=w[mask]))/np.sqrt(mask.sum())
    return xgrid[non_zero],ygrid[non_zero],delta_y[non_zero]

def moving_median(x,y,n=20):
    dx = (x.max()-x.min())/n
    xgrid = np.linspace(x.min(),x.max(),n+2)
    xgrid = xgrid[1:-1]
    ygrid = np.zeros_like(xgrid)
    delta_y = np.zeros_like(xgrid)
    for i,xmid in enumerate(xgrid):
        mask = x>(xmid-dx)
        mask *= x<(xmid+dx)
        ygrid[i] = np.median(y[mask])
        delta_y[i] = y[mask].std()/np.sqrt(mask.sum())
    return xgrid,ygrid,delta_y

def plot_fft(timestamp,signals,fname,labels,period=100,fs=14,period_max = 1000):
    cs = ["k","b","r"]
    alphas = [1,1,1,0.7]
    lw = [2,2,2,2]
    fig,ax = plt.subplots(figsize=(5,3),constrained_layout=True)

    for i,ts in enumerate(signals[:len(cs)]):
        frequency, power = LombScargle(timestamp, ts).autopower()
        p_axis = 1.0/frequency
        # Plot the result
        ax.plot(p_axis,power, c=cs[i],lw=lw[i],label="%s"%(labels[i]), alpha=alphas[i])
        mask = p_axis<period_max
        rank = np.argsort(power[mask])[::-1]
        print(labels[i],"peaks:",p_axis[mask][rank[:5]])
    pmax = power[mask].max()
    ax.set_xlim(1,299)
    ax.set_ylim(0,0.12)
    ax.set_xlabel('Period [days]');ax.set_ylabel('Power')
    ax.axvline(period,ls="--",c="grey",zorder=-10,label="$P_{true}$")
    ax.legend(fontsize=fs)
    plt.savefig("[%s]periodogram.png"%fname,dpi=300)
    #with open("results-%s.pkl"%fname,"wb")  as f:
    #    pickle.dump(signals,f)
    #    pickle.dump(labels,f)
    return

def plot_sphere(pos,radius,ax,c="grey",alpha=0.5,zorder=0):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = radius* np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    # Plot the surface
    ax.plot_surface(x, y, z, alpha=alpha, zorder=zorder,color=c)
    return

def density_plot(points,bins=30):
    x,y,z = points
    fig, ax = plt.subplots()
    density,X,Y,_ = ax.hist2d(x, y, bins=bins)
    #print("X,Y",X,Y)
    X, Y = np.meshgrid(X[1:],Y[1:])
    mesh_dict = {"XY":[X,Y,density]}
    return mesh_dict

def visualize_encoding(points,points_aug,color_target,v_name,
                       radius=0,tag=None,cmap="viridis"):

    axis_mean = points.mean(axis=1,keepdims=True)
    axis_std = points.std(axis=1,keepdims=True)
    points -= axis_mean
    points /= axis_std

    points_aug -= axis_mean
    points_aug /= axis_std

    rand = np.random.randint(points.shape[1],size=(points.shape[1]))
    print("rand:",rand.shape)
    N = len(rand)
    dist = ((points-points[:,rand])**2).sum(axis=0)
    dist_aug = ((points-points_aug)**2).sum(axis=0)

    print("random pairs: %.5f"%dist.mean(),dist.shape)
    print("augment pairs: %.5f"%dist_aug.mean(),dist_aug.shape)

    bins = np.logspace(-4,1,20)
    fig,ax = plt.subplots(figsize=(4,2.5),constrained_layout=True)
    _=ax.hist(dist,label=r"$\langle \Delta s_{rand} \rangle $: %.3f"%dist.mean(),
              color="b",bins=bins,log=False,histtype="stepfilled",alpha=0.7)
    _=ax.hist(dist_aug,label=r"$\langle \Delta s_{aug} \rangle$: %.3f"%dist_aug.mean(),
              color="r",bins=bins,log=False,histtype="stepfilled",alpha=0.7)
    ax.legend(loc=2);ax.set_xlabel("latent distance $\Delta s$");ax.set_ylabel("N")
    ax.set_xscale('log')
    plt.savefig("[%s]histogram.png"%tag,dpi=300)

    import matplotlib.colors

    elev=20;azim=130; dtr = np.pi/180.0
    viewpoint = np.array([np.cos(elev*dtr)*np.cos(azim*dtr),
                          np.cos(elev*dtr)*np.sin(azim*dtr),
                          np.sin(elev*dtr)])
    dist = 8
    viewpoint *= dist
    print("viewpoint:",viewpoint.shape,"points:",points.shape)
    depth = ((points-viewpoint[:,None])**2).sum(axis=0)**0.5
    depth /= depth.min()
    size = 40/depth**2+5
    #colors = points[0] 
    colors = color_target
    vmin,vmax=np.quantile(color_target,[0.05,0.95])
    print("colors:",colors.min(),colors.max())
    print("color_target:",color_target.min(),color_target.max())
    #print("depth:",depth.shape)
    #print(size.min(),size.mean(),size.max())
    # 3D rendering


    fig = plt.figure(figsize = (10, 8))
    ax = plt.axes(projection ="3d")
    # Add x, y gridlines
    pic = ax.scatter(points[0], points[1], points[2], s=size, marker="o",
                     alpha=1,c=colors,cmap=cmap,vmin=vmin,vmax=vmax)

    xlim=(-3, 3)
    ylim=(-3, 3)
    zlim=(-3, 5)

    ms=5;c="darkgrey"
    ax.scatter(points[0], points[1],[zlim[0]]*N,s=ms,c=c,alpha=1)
    ax.scatter(points_aug[0], points_aug[1],[zlim[0]]*N,
               s=ms,c=c,alpha=1)
    ms=5;c="grey"
    ax.scatter(points[0],[ylim[0]]*N, points[2],s=ms,c=c,alpha=1,zorder=-10)
    ax.scatter(points_aug[0],[ylim[0]]*N, points_aug[2],
               s=ms,c=c,alpha=1,zorder=-10)

    pos = [0,4,0]
    fs = 20
    # plot a sphere
    #if radius > 0:plot_sphere(pos,radius,ax,alpha=0.5,zorder=0)
    ax.set_proj_type('persp', focal_length=0.5)
    ax.set_xlabel("$s_1$",fontsize=fs)
    ax.set_ylabel("$s_2$",fontsize=fs)
    ax.set_zlabel("$s_3$",fontsize=fs)
    ax.xaxis.labelpad=-10
    ax.yaxis.labelpad=-10
    ax.zaxis.labelpad=-10

    ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([])
    ax.view_init(elev=elev,azim=azim,roll=0)
    ax.dist=dist
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
    cbar = fig.colorbar(pic, ax=ax,location = 'top', pad=0.0, shrink=0.4)
    #cbar.ax.set_xticks([])
    #cbar.ax.set_xticklabels([-2,-1,0,1],fontsize=12)
    cbar.set_label(v_name,fontsize=16,labelpad=10)
    #ax.set_aspect('equal')
    #plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.98)
    plt.savefig("[%s]3D.png"%tag,dpi=300)
    return

############ Functions for creating batched files ###############
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_batch(batch_name, subset=None):
    with open(batch_name, 'rb') as f:
        if torch.cuda.is_available():
            batch = pickle.load(f)
        else:
            batch = CPU_Unpickler(f).load()

    if subset is not None:
        return batch[subset]
    return batch

# based on https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
class BatchedFilesDataset(IterableDataset):

    def __init__(self, file_list, load_fct, shuffle=False, shuffle_instance=False):
        assert len(file_list), "File list cannot be empty"
        self.file_list = file_list
        self.shuffle = shuffle
        self.shuffle_instance = shuffle_instance
        self.load_fct = load_fct

    def process_data(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.file_list) -1)
        batch_name = self.file_list[idx]
        data = self.load_fct(batch_name)
        data = list(zip(*data))
        if self.shuffle_instance:
            random.shuffle(data)
        for x in data:
            yield x

    def get_stream(self):
        return chain.from_iterable(map(self.process_data, range(len(self.file_list))))

    def __iter__(self):
        return self.get_stream()

    def __len__(self):
        return len(self.file_list)


def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    if torch.cuda.device_count() ==0: return

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
    return


def resample_to_restframe(wave_obs,wave_rest,y,w,z):
    wave_z = (wave_rest.unsqueeze(1)*(1 + z)).T
    wave_obs = wave_obs.repeat(y.shape[0],1)
    # resample observed spectra to restframe
    yrest = Interp1d()(wave_obs, y, wave_z)
    wrest =  Interp1d()(wave_obs, w, wave_z)

    # interpolation = extrapolation outside of observed region, need to mask
    msk = (wave_z<=wave_obs.min())|(wave_z>=wave_obs.max())
    # yrest[msk]=0 # not needed because all spectral elements are weighted
    wrest[msk]=0
    return yrest,wrest

def generate_lines(xrange,max_amp=0.7,width=0.3,n_lines=100):
    amps = np.random.uniform(low=0.01,high=max_amp,size=n_lines)
    sigmas = np.random.normal(loc=width,scale=0.1*width,size=n_lines)
    line_loc = np.random.uniform(low=(xrange[0]+width),high=(xrange[1]-width),size=n_lines)
    sigmas = np.maximum(sigmas,0.01)
    lines = {"loc":line_loc,"amp":amps,"sigma":sigmas}
    return lines

def evaluate_lines(wave,lines,z=0,depth=1,skew=0,broaden=1,window=5):
    abs_lines = np.ones_like(wave)
    line_location = lines["loc"]+lines["loc"]*z
    for i,loc in enumerate(line_location):
        amp,sigma = lines["amp"][i],broaden*lines["sigma"][i]
        mask = (wave>(loc-window*sigma))*(wave<(loc+window*sigma))
        if skew>0:signal = gamma_profile(wave[mask],amp,loc,sigma, skew)
        else:signal = amp*np.exp(-0.5*((wave[mask]-loc)/sigma)**2)
        abs_lines[mask] *= (1-depth*signal)
    return abs_lines

def gauss(x, *p):
    amp, mu, sigma, b = p
    return amp*np.exp(-(x-mu)**2/(2.*sigma**2))+b


def gamma_profile(x, amp, mu, sigma, skew):
    a = 4/skew**2; b=2*a; sigma_0 = a**0.5/b
    mu0 = (a-1)/b
    y = np.zeros_like(x)
    xloc = ((x-mu)/sigma)*sigma_0 + mu0
    mask = xloc>0
    y[mask] = ((xloc[mask])**(a-1))*np.exp(-b*(xloc[mask]))
    y/=y.max()
    return amp*y
