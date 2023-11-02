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
from torchinterp1d import Interp1d
from torchcubicspline import natural_cubic_spline_coeffs
from astropy.timeseries import LombScargle

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
    #wave_shifted = - xobs * z + xobs
    #print("xrest:",xrest.shape,"yrest:",yrest.shape)
    coeffs = natural_cubic_spline_coeffs(xrest, yrest.unsqueeze(-1))
    out = cubic_evaluate(coeffs, wave_shifted)
    #print("out:",out.shape)
    return out

def moving_mean(x,y,w=None,n=20,skip_weight=True):
    dx = (x.max()-x.min())/n
    xgrid = np.linspace(x.min(),x.max(),n+2)
    xgrid = xgrid[1:-1]
    ygrid = np.zeros_like(xgrid)
    delta_y = np.zeros_like(xgrid)
    for i,xmid in enumerate(xgrid):
        mask = x>(xmid-dx)
        mask *= x<(xmid+dx)
        if skip_weight:
            ygrid[i] = np.mean(y[mask])
            delta_y[i] = y[mask].std()/np.sqrt(mask.sum())
        else:
            ygrid[i] = np.average(y[mask],weights=w[mask])
            delta_y[i] = np.sqrt(np.cov(y[mask], aweights=w[mask]))/np.sqrt(mask.sum())
    return xgrid,ygrid,delta_y

'''
def calculate_fft(time,signal):
    time_interval = time[1]-time[0]
    # Perform the FFT
    fft = np.fft.fft(signal)
    # Calculate the frequency axis
    freq_axis = np.fft.fftfreq(len(signal), time_interval)
    real  = freq_axis>0
    p_axis = 1.0/freq_axis[real]
    # Only show the real part of the power spectrum
    power_spectrum = np.real(fft * np.conj(fft))
    power_spectrum /= max(power_spectrum[real])
    return p_axis,power_spectrum[real]
'''
def plot_fft(timestamp,signals,fname,labels,period=100,fs=14):
    cs = ["grey","k","b","r"]
    alphas = [1,1,1,0.7]
    lw = [2,2,2,2]
    fig,ax = plt.subplots(figsize=(4,2.5),constrained_layout=True)
    pmax=0
    for i,ts in enumerate(signals[:len(cs)]):
        if "encode" in labels[i]:continue
        if "doppler" in labels[i]:continue
        frequency, power = LombScargle(timestamp, ts).autopower()
        p_axis = 1.0/frequency
        # Plot the result
        ax.plot(p_axis,power, c=cs[i],lw=lw[i],label="%s"%(labels[i]), alpha=alphas[i])
        if power.max()>pmax: pmax = power.max()
    ax.set_xlim(1,299)
    ax.set_ylim(0,1.1*pmax)
    ax.set_xlabel('Period [days]');ax.set_ylabel('Power')
    ax.axvline(period,ls="--",c="grey",zorder=-10,label="$P_{true}$")
    if "uniform" in fname:
        ax.set_yticks([0.01,0.02,0.03])
        title = r"$\mathbf{Case\ I \ (N=1000)}$"
    elif "dynamic" in fname:
        ax.set_yticks([0.05,0.10,0.15,0.20])
        title = r"$\mathbf{Case\ II \ (N=200)}$"
    else:title="test"
    ax.legend(fontsize=fs,title=title)
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

def visualize_encoding(points,points_aug,v_target,v_name,
                       radius=0,tag=None):

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
    colors = v_target
    vmin,vmax=np.quantile(v_target,[0.05,0.95])
    print("colors:",colors.min(),colors.max())
    print("v_target:",v_target.min(),v_target.max())
    #print("depth:",depth.shape)
    #print(size.min(),size.mean(),size.max())
    # 3D rendering


    fig = plt.figure(figsize = (10, 8))
    ax = plt.axes(projection ="3d")
    # Add x, y gridlines
    pic = ax.scatter(points[0], points[1], points[2], s=size, marker="o",
                     alpha=1,c=colors,cmap="viridis",vmin=vmin,vmax=vmax)

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
    cbar.set_label("$v_{%s}$[m/s]"%v_name,fontsize=16,labelpad=10)
    #ax.set_aspect('equal')
    #plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.98)
    plt.savefig("[%s]R1-3D.png"%tag,dpi=300)
    exit()
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
