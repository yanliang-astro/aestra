#!/usr/bin/env python
# coding: utf-8
import io, os, sys, time, random
import numpy as np
import pickle
import pandas 
import torch
import argparse
import  multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.ndimage import gaussian_filter1d
from matplotlib.cm import get_cmap
from astropy.io import fits
from scipy.interpolate import interp1d,CubicSpline
from scipy.special import gamma
from synthetic_data import Synthetic
from util import moving_mean,plot_fft,mem_report,load_batch


dynamic_dir = "/scratch/gpfs/yanliang/neid-dynamic"
datadir = "/scratch/gpfs/yanliang/NEID-SOLAR"
device =  torch.device("cpu")

colors = ["k",'b','c','m','orange',"gold",'navy',"skyblue"]
n_colors = len(colors)

def get_barycentric_corr_rv(header):
    # Initialize dictionaries to store the Barycentric Corrections
    barycentric_corr_rv = []

    # Loop over the echelle orders
    index = 0
    for i in range(173, 51, -1):
        # Construct the keyword
        keyword_rv = f'SSBRV{i:03d}'

        # Check if the keyword exists in the header and if so, store the value
        if keyword_rv in header:
            barycentric_corr_rv.append(header[keyword_rv])
    barycentric_corr_rv = np.array(barycentric_corr_rv)
    return barycentric_corr_rv

def read_ccf_rv(header):
    # Initialize dictionaries to store the Barycentric Corrections
    ccf_rv = []

    # Loop over the echelle orders
    index = 0
    for i in range(173, 51, -1):
        # Construct the keyword
        keyword_rv = f'CCFRV{i:03d}'

        # Check if the keyword exists in the header and if so, store the value
        if keyword_rv in header:
            ccf_rv.append(header[keyword_rv])
    ccf_rv = np.array(ccf_rv)
    return ccf_rv

def read_single_order(filename,order=0,read_keys=['OBSJD','DATE-OBS']):
    hdulist = fits.open(filename)

    header = hdulist[0].header
    ccf_header = hdulist[12].header
    
    science_wavelength = hdulist[7].data[order]
    science_flux = hdulist[1].data[order]
    science_variance = hdulist[4].data[order]
    science_blaze = hdulist[15].data[order]
    telluric_model = hdulist[10].data[order]

    # Close the FITS file
    hdulist.close()
    
    science = [science_wavelength,science_flux,science_variance]

    SSBRV= get_barycentric_corr_rv(header)
    info_dict = {key:header[key] for key in read_keys}
    info_dict["SSBRV"] = SSBRV[order]
    info_dict["CCFRV"] = read_ccf_rv(ccf_header)[order]
    info_dict["CCFRVMOD"] = ccf_header["CCFRVMOD"]
    return [science,science_blaze,telluric_model],info_dict

def read_fits(filename,read_keys=['OBSJD','DATE-OBS']):
    hdulist = fits.open(filename)

    header = hdulist[0].header
    ccf_header = hdulist[12].header
    
    science_wavelength = hdulist[7].data
    science_flux = hdulist[1].data
    science_variance = hdulist[4].data
    science_blaze = hdulist[15].data
    telluric_model = hdulist[10].data

    # Close the FITS file
    hdulist.close()
    
    science = [science_wavelength,science_flux,science_variance]

    SSBRV= get_barycentric_corr_rv(header)
    info_dict = {key:header[key] for key in read_keys}
    info_dict["SSBRV"] = SSBRV+0.8
    info_dict["CCFRV"] = read_ccf_rv(ccf_header)
    info_dict["CCFRVMOD"] = ccf_header["CCFRVMOD"]
    return [science,science_blaze,telluric_model],info_dict

def get_wavelengths(poly,wave_min,wave_max):
    input_pix = np.arange(n_pix+6)
    input_grid = np.zeros((n_pix+6))
    for j in range(len(input_grid)):
        if j==0:input_grid[j] = wave_min;continue
        local_bin = np.polyval(poly,input_grid[j-1])
        input_grid[j] = input_grid[j-1]+local_bin
    if input_grid[-1]<wave_max:
        print("input_grid too short!!",input_grid[-1],wave_max)
    return input_grid

np.random.seed(0)
torch.manual_seed(0)

sample_names = [i for i in os.listdir(datadir) if "L2_2021" in i]

#"order wave_min wave_max wave_bin"
n_sample = 500
n_order = 122
n_pix = 9216
n_bins = 12000
deg = 2

orders = np.arange(n_order)
wave_min = np.zeros((n_order))
wave_max = np.zeros((n_order))
wave_poly = np.zeros((n_order,deg+1))

good_order = np.ones((n_order))
wave_matrix = np.zeros((n_order,n_sample,n_pix))
wgrid = np.zeros((n_order,n_sample,n_pix-1))
wave_bin = np.zeros((n_order,n_sample,n_pix-1))

#wave_bin = np.zeros((n_order))+np.inf
#orders = [95]

for i,obsname in enumerate(sample_names[:n_sample]):
    data,info_dict = read_fits("%s/%s"%(datadir,obsname))
    science,blaze,telluric = data
    wave = science[0]
    ccfrvs = info_dict["CCFRV"]
    ssbrv = 1e3*info_dict["SSBRV"]
    for o in orders:
        wave_order = wave[o]*(1.0+ssbrv[o]/Synthetic.c)
        wbin = wave_order[1:]-wave_order[:-1]
        wave_bin[o,i] = wbin
        wave_matrix[o,i] = wave_order
        if ccfrvs[o] is None:good_order[o]=0

for o in orders:
    if good_order[o]==0:continue
    wleft = wave_matrix[o,:,:-1].flatten()
    wbin = wave_bin[o].flatten()
    poly,cov = np.polyfit(wleft,wbin,deg=deg,cov=True)

    poly_std = [cov[j][j]**0.5 for j in range(deg+1)]
    wbin_fit = np.polyval(poly,wleft)
    chi = np.mean((wbin-wbin_fit)**2/0.00001**2)
    print(o,"resid chi: %.2f"%chi)
    if chi>100:
        good_order[o]==0
        continue
    #for j in range(deg+1):
    #    print("poly %d: %.2e +/- %.2e"%(j,poly[j],poly_std[j]))
    wave_min[o] = wave_matrix[o,:,0].min()-0.01
    wave_max[o] = wave_matrix[o,:,-1].max()+0.01
    wave_poly[o] = poly

config_file = "new_orders.config"
file_content = ["#order wave_min wave_max wave_poly\n"]
for o in range(n_order):
    polystr = " ".join(["%.7e"%item for item in wave_poly[o]])
    text = "%d %.7f %.7f %s\n"%(o,wave_min[o],wave_max[o],polystr)
    file_content.append(text)
with open(config_file,"w") as f:
    for line in file_content:
        f.writelines(line)
os.system("cat %s"%config_file)

config_data = np.loadtxt(config_file).T
wave_min = config_data[1]
wave_max = config_data[2]
wave_poly = config_data[3:6].T

o = 61
wleft = wave_matrix[o,:,:-1].flatten()
wbin = wave_bin[o].flatten()
wbin_fit = np.polyval(wave_poly[o],wleft)
chi = np.mean((wbin-wbin_fit)**2/0.00001**2)
print("resid chi: %.2f"%chi)

input_grid = get_wavelengths(wave_poly[o],wave_min[o],wave_max[o])
y_offset = 0.001
fig,axs=plt.subplots(ncols=3,figsize=(10, 3),dpi=200,
                     constrained_layout=True)
for i,obsname in enumerate(sample_names[:n_sample]):
    wave_order = wave_matrix[o,i]
    wleft = wave_order[:-1]
    wbin = wave_bin[o,i]
    for ax in axs[:2]:
        ax.plot(wave_order,np.zeros((n_pix))+i*y_offset,".-",c=colors[i%len(colors)])
    axs[2].plot(wleft,wbin,"-",c=colors[i%len(colors)])
for ax in axs[:2]:
    ax.plot(input_grid,np.zeros_like(input_grid)-y_offset,".-",c="grey",label="merge grid")
axs[2].plot(wleft,wbin,"r--",lw=1.0,label="polyfit deg=%d $\chi^2=%.2f$"%(deg,chi))
axs[0].set_xlim(wave_min[o]-0.01,wave_min[o]+0.05)
axs[1].set_xlim(wave_max[o]-0.05,wave_max[o]+0.01)
axs[2].set_ylabel("wave bin ($\AA$)")
for ax in axs:ax.legend(loc=1)
plt.savefig("[config]order%d.png"%o,dpi=200)


