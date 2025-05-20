#!/usr/bin/env python
# coding: utf-8
import io
import os
import sys
import time
import random
import pickle
import argparse
import multiprocessing as mp
import torch

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.ndimage import gaussian_filter1d
from matplotlib.cm import get_cmap
from astropy.io import fits
from scipy.interpolate import interp1d
from synthetic_data import Synthetic
#from torchinterp1d import Interp1d

from util import moving_median,load_batch,merge_batch,load_master_fsr_mask
from scipy.optimize import curve_fit,minimize

dynamic_dir = "/scratch/gpfs/yanliang/neid-dynamic"
datadir = "/scratch/gpfs/yanliang/NEID-SOLAR"
runtime_dir = "/scratch/gpfs/yanliang/neid-dynamic/params/"
device =  torch.device("cpu")

blacklist = [37, 43, 44, 51, 52, 55, 60, 65, 66, 69, 70, 71, 75, 76, 78, 79, 80, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

colors = ["k",'b','c','m','orange',"gold",'navy',"skyblue"]
n_colors = len(colors)

# Gaussian function
def gaussian(x, amplitude, mean, stddev):
    if stddev<0.01:return np.ones_like(x)
    if amplitude>1:amplitude=1
    return np.abs(amplitude) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Function to fit multiple Gaussians
def multi_gaussian(x, *params):
    y = np.ones_like(x)
    for i in range(0, len(params), 3):
        y *= 1-(gaussian(x, params[i], params[i+1], params[i+2]))
    return y

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

def read_fits(filename,order_value,quality_mask,read_keys=['OBSJD','DATE-OBS','AIRMASS']):
    hdulist = fits.open(filename)
    header = hdulist[0].header
    ccf_header = hdulist[12].header
    telluric_header = hdulist[10].header
    
    science_wavelength = hdulist[7].data[order_value][quality_mask]
    science_flux = hdulist[1].data[order_value][quality_mask]
    science_variance = hdulist[4].data[order_value][quality_mask]
    science_blaze = hdulist[15].data[order_value][quality_mask]
    # Close the FITS file
    hdulist.close()
    
    science = [science_wavelength,science_flux,science_variance]
    data = [science,science_blaze]

    SSBRV= get_barycentric_corr_rv(header)
    CCFRV = read_ccf_rv(ccf_header)
    info_dict = {key:header[key] for key in read_keys}
    info_dict.update({key:telluric_header[key] for key in ["ZENITH","WVAPOR"]})

    km_m = 1e3
    info_dict["SSBRV"] = (SSBRV[order_value]+0.8)*km_m
    info_dict["CCFRV"] = CCFRV[order_value]*km_m
    info_dict["CCFRVMOD"] = ccf_header["CCFRVMOD"]*km_m
    info_dict["DVRMSMOD"] = ccf_header["DVRMSMOD"]*km_m
    # time zero point
    info_dict["timestamp"] = np.float32(info_dict["OBSJD"] - 2459000.0)
    return data,info_dict

def redshift_chi(rv,wave_model,yrest,weight_rest,wave_data,ydata,wdata):
    wave_shifted = wave_model*(1 + rv/Synthetic.c)
    bad = yrest==0

    mask = (wave_data>min(wave_shifted[~bad]))&(wave_data<max(wave_shifted[~bad]))
    model_obs = interp1d(wave_shifted[~bad], yrest[~bad])(wave_data[mask])
    loss = np.sum(wdata[mask]* (ydata[mask] - model_obs)**2) / len(ydata[mask])
    return loss

def find_deepest_lines(wave_obs, raw_spectrum, num_lines=30, min_separation=0.10,return_ind=False):
    spectrum = raw_spectrum/np.quantile(raw_spectrum,0.99)
    depth = 1 - spectrum
    sorted_indices = np.argsort(depth)[::-1]  # Indices of depths sorted from largest to smallest

    # Start with the deepest line
    unique_indices = [sorted_indices[0]]

    # Iterate over the sorted indices and select peaks with the required minimum separation
    for index in sorted_indices[1:]:
        # Check if this index is sufficiently far from all previously selected peaks
        if all(np.abs(wave_obs[index] - wave_obs[prev_index]) > min_separation for prev_index in unique_indices):
            unique_indices.append(index)
        # Stop if we've found the required number of lines
        if len(unique_indices) == num_lines:
            break

    # Get the corresponding wavelengths and depths for the unique peaks
    unique_wavelengths = wave_obs[unique_indices]
    unique_depths = depth[unique_indices]

    if return_ind:return sorted(zip(unique_indices, unique_depths), key=lambda x: x[0])
    # Return as a list of tuples sorted by depth
    return sorted(zip(unique_wavelengths, unique_depths), key=lambda x: x[0])

def mask_deepest_lines(wave_obs, lines_to_mask, mask_width=0.15):
    # Copy the spectrum to avoid modifying the original
    skymask = np.zeros_like(wave_obs,dtype=bool)
    # Loop through the lines to mask
    for line in lines_to_mask:
        # Find the indices within the mask width of the line
        mask_indices = np.where(np.abs(wave_obs - line[0]) <= mask_width)[0]
        # Mask the spectrum by setting the depth to 1 (i.e., no absorption)
        skymask[mask_indices] = True
    return skymask

def prepare_spectrum_single_order(obsname,quality_mask,order_value):
    large_number = 1e6
    try:
        data,info_dict = read_fits("%s/%s"%(datadir,obsname),order_value=order_value,quality_mask=quality_mask)
    except:
        print("broken file!",obsname)

    n_spec = quality_mask.sum()
    wavelegth = np.zeros((n_spec))
    spectrum = np.zeros((n_spec))
    spectrum_err = np.zeros((n_spec))

    science,blaze = data
    wave_raw,flux,flux_var = science

    ssbrv = info_dict["SSBRV"]
    jd = info_dict["OBSJD"]

    isnan = np.isnan(flux) | np.isnan(blaze)| (flux<=0.0)

    normflux = np.zeros_like(flux)
    normflux_err = np.zeros_like(flux_var)

    norm = np.quantile(flux[~isnan]/blaze[~isnan],0.5)
    normflux[~isnan] = flux[~isnan]/(norm*blaze[~isnan])
    normflux_err[~isnan] = flux_var[~isnan]**0.5/(norm*blaze[~isnan])
    if normflux.min()<0:
        print("negative flux!",normflux.min(),normflux.max())
    elif blaze.min()<=0.:
        print("blaze nan!",blaze.min(),blaze.max())

    wavelength = wave_raw
    spectrum[~isnan] = normflux[~isnan]
    spectrum_err[~isnan] = normflux_err[~isnan]
    spectrum_err[isnan] = large_number
    spectrum[isnan] = 0.0

    badmask = detect_bad_pixel([wavelength,spectrum,spectrum_err])
    spectrum_err[badmask] = large_number
    return [wavelength,spectrum,spectrum_err],info_dict

def save_batch(batch,filename):
    wave,specs,w,ssbrv,IDs = batch
    wave = torch.from_numpy(wave.astype(np.double))
    spec = torch.from_numpy(specs.astype(np.float32))
    weight = torch.from_numpy(w.astype(np.float32))
    ssbrv = torch.from_numpy(ssbrv.astype(np.double))
    ID = torch.from_numpy(IDs.astype(np.float32))
    
    if ssbrv.ndim==1: ssbrv = ssbrv.unsqueeze(1)
    if ID.ndim==1: ID = ID.unsqueeze(1)

    batch = [wave,spec,weight,ssbrv,ID]
    print("wave:",wave.shape,"spec:",spec.shape,"weight:",weight.shape,
          "ssbrv:",ssbrv.min(),ssbrv.max(),"ID",ID.shape)
    print("saving to %s..."%filename)
    with open(filename, 'wb') as f:
        pickle.dump(batch, f)
    return

def save_auxfile(input_data,filename):
    input_data = torch.from_numpy(input_data.astype(np.double))
    print("input_data",input_data.shape)
    print("saving to %s..."%filename)
    with open(filename, 'wb') as f:
        pickle.dump(input_data, f)
    return
def make_batch(sample_names,order_value,quality_mask,max_neg_flux=100,max_wave_std=0.0005):
    large_number = 1e6
    batch_size = len(sample_names)
    n_spec = quality_mask.sum()

    wavemat =  np.zeros((batch_size,n_spec))
    specmat = np.zeros((batch_size,n_spec))
    errmat = np.zeros((batch_size,n_spec))
    good =  np.ones((batch_size),dtype=bool)

    local_dict = {}
    for i_obs,obsname in enumerate(sample_names):
        data,info_dict = prepare_spectrum_single_order(obsname,quality_mask,order_value)
        wavelength,spectrum,spectrum_err = data
        # negative flux?
        neg = np.sum(spectrum<0.0,axis=-1)
        if neg.sum()>max_neg_flux:
            good[i_obs] = False
            print("negative!!",obsname,neg)
            print("flux: %.2f, %.2f"%(spectrum.min(),spectrum.max()))
        #if spectrum.min()<0.01:good[i_obs] = False
        if not good[i_obs]: continue
        local_dict[obsname] = info_dict
        wavemat[i_obs] = wavelength
        specmat[i_obs] = spectrum
        errmat[i_obs] = spectrum_err

    wave_mean = np.mean(wavemat,axis=0,keepdims=True)
    wave_std = (wavemat-wave_mean).std(axis=-1)
    wh_obs = np.where(wave_std>max_wave_std)
    for i_obs in np.unique(wh_obs):
        #good[i_obs] = False
        print("unusual wave solution: %d, skip..."%i_obs)
    print("wave_std:",wave_std.shape)
    print("where",wh_obs)

    bad = errmat**(-2)<1.0
    print("bad pixels:",(bad.sum()/batch_size))
    print("good:",good.sum())
    specmat[bad] = 0.0
    wavemat=wavemat[good]
    specmat=specmat[good]
    errmat=errmat[good]

    good_dict={}
    for i_obs,obsname in enumerate(sample_names):
        if good[i_obs]:good_dict[obsname] = local_dict[obsname]
    return sample_names[good],wavemat,specmat,errmat,good_dict

def photon_noise(spec_rest,wave_rest,sn):
    A0 = spec_rest*(sn**2)
    dAdl = np.gradient(A0,wave_rest)
    W = (wave_rest**2)*(dAdl)**2/A0
    Ne = A0.sum()
    Q = W.sum()/(Ne**0.5)
    print("W:",W.shape,"Ne:",Ne,"Q:",Q)
    RV_rms = Synthetic.c/(W.sum())**0.5
    return RV_rms

def fit_rv(wave,spec,w,wave_rest,rest_model,weight_model):
    result = scipy.optimize.minimize(redshift_chi,0.0, method='Nelder-Mead',args=(wave_rest,rest_model,weight_model,wave,spec,w,))
    label = "RV_fit=%.2f $\chi^2$:%.2f"%(result.x,result.fun)
    return result.x, result.fun, label

def get_timeseries(neid_dict,colname,keys):
    vector = np.array([neid_dict[key][colname] for key in keys])
    print(colname,vector.shape)
    return vector

def velocity_label(velocity,label):
    quantiles=[0.16,0.50,0.84]
    q1,q2,q3 = np.quantile(velocity,quantiles)
    val = "${%.2f}^{+%.2f}_{-%.2f}$"%(q2,q3-q2,q2-q1)
    vlabel = "%s = %s [m/s]"%(label,val)
    return vlabel

def make_batch_worker(batch_id, order_value, quality_mask, batch_name, neid_dict):
    batch_id,wavemat,specmat,errmat,sub_dict = make_batch(batch_id,order_value,quality_mask)
    ssbrvs = get_timeseries(sub_dict,'SSBRV',batch_id).T
    timestamp = get_timeseries(sub_dict,'timestamp',batch_id)
    save_batch([wavemat,specmat,errmat**(-2),ssbrvs,timestamp],batch_name)
    print("good spectra: %d"%len(batch_id))
    neid_dict.update(sub_dict)
    return 0

def fit_rv_worker(wave,spec,w,wave_baseline,baseline,baseline_w,mdict,obsname,order):
    v_template,base_chi,message = fit_rv(wave,spec,w,wave_baseline,baseline,baseline_w)
    summary = {"v_template":v_template[0],"chi_template":base_chi}
    mdict["%s"%(obsname)]=summary
    return 0

def process_task(args, mdict):
    # Unpack arguments
    wave, specs, weights, wave_baseline, baseline, baseline_w, obsname, order = args
    # Your existing task logic with the managed dictionary
    fit_rv_worker(wave, specs, weights, wave_baseline, baseline, baseline_w, mdict, obsname, order)
    n_items = len(mdict)
    if n_items%500==0: 
        print("mdict:",n_items)
    return

def wrap_data(sample_names,datatag,batch_size,order_value,quality_mask):
    idx = np.arange(0, len(sample_names), batch_size)
    batches = np.array_split(sample_names, idx[1:])

    file_batches = ["%s/%s_%d.pkl"%(dynamic_dir,datatag,k) for k in range(len(batches))]

    general_info = {"sample_names":sample_names,
                    "files":file_batches,
                    "order":order_value}

    process_list = []
    manager = mp.Manager()
    mdict = manager.dict()
    for k in range(len(batches)):
        batch_name = file_batches[k]
        #if os.path.isfile(batch_name):
        #    print(batch_name,"file exists! continue...")
        #    continue
        batch_id = batches[k]
        print ("saving batch  %d / %d"%(k,len(file_batches)))    
        work_p = mp.Process(target=make_batch_worker,
                            args=(batch_id, order_value, quality_mask, batch_name, mdict))
        process_list.append(work_p)

    for i_start in range(0, len(process_list), num_cores):
        print("[wrap_data]Currently running #%i - #%i"%(i_start, min(i_start+num_cores,len(process_list))))
        running_list = process_list[i_start:i_start+num_cores]
        [p.start() for p in running_list]
        [p.join()  for p in running_list]


    neid_dict = {k:v for k,v in mdict.items()}
    # remove poor-quality spectra
    sample_names = [i for i in sample_names if i in neid_dict]
    print("sample_names:",len(sample_names))

    general_info["sample_names"] = sample_names
    neid_dict.update({"info":general_info})
    with open(f'{runtime_dir}/{datatag}-param.pkl',"wb") as f:
        pickle.dump(neid_dict,f)
    return sample_names

def interpolate_to_input_grid(batch,instrument,template_raw):
    wave_raw,spec_raw,w,ssbrv,jd = [item.numpy() for item in batch]
    
    n_batch,n_spec = spec_raw.shape
    template = template_raw[None,:]*np.ones((n_batch,1))
    wave_obs = instrument.wave_obs.numpy()
    z = (ssbrv)/instrument.c
    wave = wave_raw + wave_raw * z
    #out = torch.zeros_like(spectrum,dtype=bool)

    spectrum = np.zeros((n_batch,wave_obs.shape[1]))
    weight = np.zeros((n_batch,wave_obs.shape[1]))
    for i in range(n_batch):
        spectrum[i] = interp1d(wave[i], spec_raw[i],bounds_error=False,fill_value=0)(wave_obs)
        weight[i] = interp1d(wave[i], w[i],bounds_error=False,fill_value=0)(wave_obs)
    wmin = np.min(wave, axis=1, keepdims=True)
    wmax = np.max(wave, axis=1, keepdims=True)

    out_ = (wave_obs < wmin) | (wave_obs > wmax)
    out = out_

    ill = (template == 0) | (spectrum == 0) | (weight < 1.0)

    # mask out +/- 1 pixel of bad input data (zero flux)
    bad = (spectrum < (template * 0.6)) | ill

    # after linear interpolation, bad flux values are at most half of template values
    # cut at 0.6 for safety
    bad |= np.roll(bad, -1, axis=1)
    bad |= np.roll(bad, +1, axis=1)
    bad |= out

    weight[bad] = 1e-12
    spectrum[bad] = 0.0
    return spectrum, weight, ssbrv,jd

def calculate_template_spectrum(datatag,wave_obs):
    print(f'Loading from {runtime_dir}/{datatag}-param.pkl')
    with open(f'{runtime_dir}/{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)

    # calculate v_template and chi_template
    file_batches = neid_dict["info"]["files"]
    order_value = neid_dict["info"]["order"]
    batch = merge_batch(file_batches)
    
    waves,specs,weights,ssbrvs,ids = [item.numpy() for item in batch]
    print("file_batches:",file_batches)
    
    # interpolate to homogeneous grid - calculate the template spectrum
    print("native grid",waves.shape,"input grid",wave_obs.shape)
    n_template = min(1000,len(specs))
    n_pix = len(wave_obs)

    input_flux = np.zeros((n_template,n_pix))
    input_weight = np.zeros((n_template,n_pix))

    for i in range(n_template):
        wave_raw = waves[i]
        ssbrv = ssbrvs[i]
        flux = specs[i]
        weight = weights[i]
        wave = wave_raw + wave_raw*(ssbrv)/Synthetic.c
        good = (weights[i]>1.0)
        inbound = (wave_obs>min(wave[good]))&(wave_obs<max(wave[good]))

        input_flux[i][inbound] = interp1d(wave[good], flux[good], kind='linear')(wave_obs[inbound])
        input_weight[i][inbound] = interp1d(wave, weight, kind='linear')(wave_obs[inbound])
        input_weight[i][~inbound] = 1e-12

        bad = input_weight[i]<1.0
        input_flux[i][bad] = 0.0

    template = np.median(input_flux,axis=0)
    template_w = np.median(input_weight,axis=0)
    dispersion =  np.zeros((n_pix))

    for i in range(n_pix):
        flux = input_flux[:,i]
        non_zero = flux>0
        if non_zero.sum()==0:continue
        dispersion[i] = np.std(flux[non_zero])

    '''
    fig,ax=plt.subplots(figsize=(16,4),constrained_layout=True)
    for i in range(n_template):
        mask = input_flux[i]>0
        ax.plot(wave_obs[mask],input_flux[i][mask],"k-",lw=1,alpha=0.1,drawstyle="steps-mid")
    ax.plot(wave_obs,template,"r-",lw=1,drawstyle="steps-mid",label="template")
    ax.plot(wave_obs,dispersion,"c-",lw=1,drawstyle="steps-mid",label="dispersion")
    ax.legend()
    #ax.set_xlim(wave_obs[300]-2,wave_obs[300]+2)
    plt.savefig("[%s]template.png"%datatag,dpi=200)
    '''

    template_name="%s/%s.pkl"%(dynamic_dir,datatag+"-template")
    save_batch([wave_obs[None,:],template[None,:],template_w[None,:],
                np.array([0]),np.array([888])],template_name)
    neid_dict["info"].update({"baseline":template,"baseline_w":template_w})
    with open(f'{runtime_dir}/{datatag}-param.pkl',"wb") as f:
        pickle.dump(neid_dict,f)
    return 

def calculate_v_template(datatag,wave_obs):
    print(f'Loading from {runtime_dir}/{datatag}-param.pkl')
    with open(f'{runtime_dir}/{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)

    sample_names = neid_dict["info"]["sample_names"]
    # calculate v_template and chi_template
    file_batches = neid_dict["info"]["files"]
    template = neid_dict["info"]["baseline"]
    template_w = neid_dict["info"]["baseline_w"]
    order_value = neid_dict["info"]["order"]

    batch = merge_batch(file_batches)
    waves,specs,weights,ssbrvs,ids = [item.numpy() for item in batch]
    print("ssbrvs [m/s]:",ssbrvs.min(),ssbrvs.max())
    #process_list = []
    manager = mp.Manager()
    mdict = manager.dict()
    # Use a pool of workers
    pool_size = num_cores  # Number of processes in the pool
    pool = mp.Pool(pool_size)
    tasks = []
    for i_epoch,obsname in enumerate(sample_names):
        # skip existing entries
        #if 'v_template' in neid_dict[obsname]:continue
        z_ssb = ssbrvs[i_epoch]/Synthetic.c
        # shift to stellar restframe
        wave_stellar = waves[i_epoch]*(1.0+z_ssb)
        task_args = (wave_stellar, specs[i_epoch], weights[i_epoch], wave_obs,template,template_w, obsname, order_value)
        tasks.append(task_args)

    for i,task in enumerate(tasks):
        pool.apply_async(process_task, args=(task, mdict))
    # Close and join the pool
    pool.close()
    pool.join()

    for k in sample_names:
        key = "%s"%(k)
        if key in mdict:neid_dict[k].update(mdict[key])
        else:print("%s missing..."%key)

    print("Saving to %s-param.pkl..."%datatag)
    with open(f'{runtime_dir}/{datatag}-param.pkl',"wb") as f:
        pickle.dump(neid_dict,f)
    return 

def get_wavelengths(poly,wave_min,wave_max,n_pix=9216):
    input_pix = np.arange(n_pix)
    input_grid = np.zeros((n_pix))
    for j in range(len(input_grid)):
        if j==0:input_grid[j] = wave_min;continue
        local_bin = np.polyval(poly,input_grid[j-1])
        input_grid[j] = input_grid[j-1]+local_bin
    if input_grid[-1]<wave_max:
        print("input_grid too short!!",input_grid[-1],wave_max)
    return input_grid

def get_order_wavelengths(target_order):
    config_data = np.loadtxt("orders.config").T
    wave_min = config_data[1]
    wave_max = config_data[2]
    wave_poly = config_data[3:6].T
    wave_bins = np.array(config_data[6],dtype=int)
    o = target_order
    wave_obs = get_wavelengths(wave_poly[o],wave_min[o],wave_max[o],
                               n_pix=wave_bins[o])
    return wave_obs

def tensor2array(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else: return tensor.detach().numpy()

def detect_bad_pixel(data,sigma=1.5,snr=3,radius=2):
    wavelength,spectrum,spectrum_err = data
    spec_smooth = gaussian_filter1d(spectrum, sigma)
    ydiff = np.abs(spectrum-spec_smooth)
    badmask = np.zeros(spectrum.shape,dtype=bool)
    n_spec = spectrum.shape[-1]

    candidates = find_deepest_lines(wavelength, 1-ydiff, num_lines=10,min_separation=0.1, return_ind=True)
    known_bad = np.where(spectrum_err>1)[0]
    valid = []
    for ind,amp in candidates:
        sn = amp/spectrum_err[ind]
        if sn<snr:continue
        if ind in known_bad:continue
        # check if coincide with a stellar line
        nearby = spectrum[max(0,ind-radius):min(ind+radius,n_spec-1)]
        sorted_ind = np.argsort(nearby)
        fluxes = nearby[sorted_ind]
        if (fluxes[0]<0.8)&((fluxes[1]-fluxes[0])<0.1):
            continue
        valid.extend([ind-1,ind,ind+1])
    valid = [item for item in valid if item>=0 and item<n_spec]
    badmask[valid] = True
    #print("badmask:",badmask.sum())
    return badmask

def preview_spectrum(obsname,order_value,quality_mask):
    data,info_dict = prepare_spectrum_single_order(obsname,quality_mask,order_value)
    wavelength,spectrum,spectrum_err = data

    ylim=[0,max(1.2,spectrum.max())]
    fig, ax = plt.subplots(figsize=(15,3),dpi=200,constrained_layout=True)

    ax.set_title("Order %d"%order_value)
    ax.plot(wavelength,spectrum,"k-",lw=1,label="data",drawstyle="steps-mid")
    #ax.plot(wavelength[i],spec_smooth[i],"b-",lw=1,label="smooth",drawstyle="steps-mid")
    #ax.plot(wavelength[i],ydiff[i],"r-",lw=1,drawstyle="steps-mid")
    ax.fill_between(wavelength,spectrum-spectrum_err,
                    spectrum+spectrum_err,step="mid",
                    color="k",alpha=0.3,zorder=-10)
    ax.legend()
    ax.set_ylim(ylim)
    plt.savefig("[%s]single-obs.png"%tag)
    return

def initialize_restframe_model(wave_obs,template,weight):
    wave_rest = wave_obs
    spec_rest = np.ones_like(template)
    spec_rest[weight>1.0]=template[weight>1.0]
    init_rest = np.array([wave_rest,spec_rest])
    print("init_rest:",init_rest.shape)
    return init_rest

def plot_skymask(input_wave,y_sky,dispersion,wavemean,skymask):
    fig, axs = plt.subplots(figsize=(12,8),nrows=n_order,constrained_layout=True)
    for o in range(n_order):
        ax=axs[o]
        for i in range(20):
            ax.plot(input_wave[o],1-y_sky[i][o],"k-",alpha=0.3,lw=1)
        ax.plot(input_wave[o],1-dispersion[o],"r-")
        for line in lines:
            ax.axvline(x=line[0],ymin=0,ymax=1.2,ls="--",color="b")
        ax.fill_between(wavemean[o],0,1.1,where=skymask[o],color="b",alpha=0.3)
        where=input_wave[o][2000]
        ax.set_ylim(0.9,1.01)
        ax.set_xlim(where-10,where+10)
    plt.savefig("skymask.png",dpi=200)
    return

def print_string(vname,v,mode="1"):
    if mode=="1":
        quantiles = [v.min(),v.max(),v.mean()]
        string = " ".join(["%.2f"%item for item in quantiles])
    if mode=="2":
        string =  "%.3f +/- %.3f m/s"%(v.mean(),v.std())
    print("%s: %s"%(vname,string))
    return

def plot_sample_selection(axs, timestamp,v_template,v_ccf,fit_chi,order, 
                          grey_color = "grey", bright_color = "k"):
    ax=axs[0]
    ax.scatter(NEID_JD,NEID_CCFRV,c="grey",s=5,label="all (N=%d)"%len(NEID_JD))
    img = ax.scatter(NEID_JD[SELECT],NEID_CCFRV[SELECT],s=5,label="selected (N=%d)"%len(SELECT))
    ax.legend(loc="upper left")
    ax.set_xlabel("JD")
    ax.set_ylabel("NEID Solar RV [km/s]")
    ax=axs[1]
    ax.scatter(timestamp,v_template,c=bright_color,s=3,
               label="$v_{template}$ RMS = %.2f m/s"%(v_template.std()))
    ax.scatter(timestamp,v_ccf,c=grey_color,s=3,label="$v_{CCF}$ RMS = %.2f m/s"%(v_ccf.std()))

    ax.legend(title="Discrepancy RMS = %.2f m/s"%template_ccf_offset.std())
    ax.set_xlabel("JD")
    ax.set_ylabel("RV [m/s]")
    ax=axs[2]
    ax.scatter(timestamp,fit_chi,c=bright_color,s=3,
               label="Order %d $\chi^2_{template}$"%(order))
    ax.legend()
    ax.set_xlabel("JD")
    ax.set_ylabel("$\chi^2_r$")
    #plt.savefig("[%s]sample-selection.png"%datatag,dpi=300)
    #plt.clf()
    return

def moving_std_weight_batch(data,valid_mask,window_size=6):
    # Ensure data is a 2D numpy array
    data = np.asarray(data)  # Input shape (B, N), where B = batch size, N = sequence length

    # Compute cumulative sums for the data and its squares along axis=1
    cumsum = np.cumsum(data, axis=1)
    cumsum2 = np.cumsum(data**2, axis=1)
    # Compute cumulative sum for the valid mask
    valid_cumsum = np.cumsum(valid_mask, axis=1)

    # Compute moving sums using valid cumulative sums
    moving_sum = cumsum[:, window_size:] - cumsum[:, :-window_size]
    moving_sum2 = cumsum2[:, window_size:] - cumsum2[:, :-window_size]
    valid_count = valid_cumsum[:, window_size:] - valid_cumsum[:, :-window_size]

    good = valid_count>3
    var = np.zeros_like(moving_sum)

    # Calculate moving mean and mean of squares
    var[good] = moving_sum2[good]/valid_count[good]
    var[good] -= (moving_sum[good]/valid_count[good])**2

    # Handle edge cases where std might be zero
    large_value = 1e12
    var[~good] = large_value

    # Pad the result to match the input size
    pad_left = np.full((data.shape[0], window_size // 2), var[:, 0:1])  # Pad with first std value
    pad_right = np.full((data.shape[0], window_size // 2), var[:, -1:])  # Pad with last std value
    var = np.concatenate((pad_left, var, pad_right), axis=1)
    return 1/var


def plot_continuum(spec,w,spec_input,f_continuum,b_poly,template):
    y_before = spec - template
    #y_after = y_recover - template
    mask = w>1
    y_before[~mask] = 0
    yerr = np.zeros_like(w)
    yerr[mask] = w[mask]**(-0.5)
    yerr[~mask] = 1e6

    x = np.arange(len(template))    
    loss = w*spec_input**2
    print("loss:",loss[np.argsort(loss)[::-1]][:10])
    fig,axs = plt.subplots(figsize=(10,5),nrows=2,constrained_layout=True)
    #plt.plot(wave_obs,template,"-",color="grey")
    
    ax = axs[0]
    ax.fill_between(x,y_before-yerr,y_before+yerr,color="grey",step="mid")
    ax.plot(x,y_before,"k-",drawstyle="steps-mid",
            label=f"residual loss = {np.mean(w*y_before**2):.4f}")
    ax.plot(x[mask],f_continuum*template[mask]+b_poly,"r-",label="model")
    ax.plot(x[mask],f_continuum+b_poly,c="cyan",label="recovered continuum")
    ax=axs[1]
    ax.plot(x,spec_input,"-",c="grey",drawstyle="steps-mid",label=f"loss = { loss.mean():.4f}")
    ax.plot(x,gaussian_filter1d(spec_input,2),"k-")
    for ax in axs:
        #ax.set_xlim(1000,1500)
        ax.set_ylim(y_before.min(),y_before.max())
        ax.set_ylabel("flux");ax.legend(loc=3)
    plt.savefig("[continuum]test.png",dpi=200)
    exit()

# Define the objective function
def continuum_loss(param, y_perturb, weight, y_quiet, smooth_radius = 12, full=False):
    b = param
    b_poly = b

    y_recover = np.zeros_like(y_perturb)
    x = np.linspace(-1,1,len(y_perturb))

    mask = weight>1
    y_smooth = gaussian_filter1d((y_perturb-y_quiet)[mask],smooth_radius)
    y_template_smooth = gaussian_filter1d(y_quiet[mask],smooth_radius)
    f_continuum = (y_smooth - b) / y_template_smooth

    y_recover[mask] = (y_perturb[mask] - b_poly) / (1 + f_continuum)
    y_resid =  y_recover - y_quiet

    loss = np.sum(weight*y_resid**2)/(weight>1).sum()
    #print(mask.sum(),"b:",b,"loss:",loss)
    if full: return loss, y_recover, f_continuum, b_poly
    return loss

def continuum_worker(batch_name,save_name,template):
    batch = load_batch(batch_name)
    spec, w_raw, ssbrv,jd = interpolate_to_input_grid(batch,instrument,template)
    spec_input = spec - template
    w_std = moving_std_weight_batch(spec-template,w_raw>1)
    w = 1/(1/w_std+1/w_raw)
    for i in range(spec_input.shape[0]):
        #if i<=10:continue
        result = minimize(continuum_loss,[0.0001], args=(spec[i],w[i],template))
        pbest = result.x
        loss,y_recover,f_continuum,b_poly = continuum_loss(pbest,spec[i],w[i],template,full=True)
        spec_input[i,w[i]>1] = (y_recover - template)[w[i]>1]
        #plot_continuum(spec[i],w[i],spec_input[i],f_continuum,b_poly,template)
    save_batch([np.zeros_like(jd),spec_input,w,ssbrv,jd],save_name)
    return

def correct_for_continuum(datatag,instrument,template_data):
    print(f'Loading from {runtime_dir}{datatag}-param.pkl')
    with open(f'{runtime_dir}{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)
    file_batches = neid_dict["info"]["files"]
    template = neid_dict["info"]["baseline"]
    print("template:",template.shape)
    wave_obs = tensor2array(template_data[0])[0]
    print("wave_obs:",wave_obs.shape)
    print("file_batches:",file_batches)

    process_list = []
    manager = mp.Manager()
    mdict = manager.dict()
    for k,batch_name in enumerate(file_batches):
        print("Loading %s..."%batch_name)
        datatag = os.path.basename(batch_name).rsplit('.', 1)[0]
        save_name = f"{dynamic_dir}/processed/{datatag}.pkl"
        #continuum_worker(batch_name,save_name,template)
        print ("saving batch  %d / %d"%(k,len(file_batches)))    
        work_p = mp.Process(target=continuum_worker,
                            args=(batch_name,save_name,template))
        process_list.append(work_p)

    print("process_list",process_list)
    for i_start in range(0, len(process_list), num_cores):
        print("[continuum]Currently running #%i - #%i"%(i_start, min(i_start+num_cores,len(process_list))))
        running_list = process_list[i_start:i_start+num_cores]
        [p.start() for p in running_list]
        [p.join()  for p in running_list]
    return

def daily_average_spectrum(datatag):
    print(f'Loading from {runtime_dir}{datatag}-param.pkl')
    with open(f'{runtime_dir}{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)
    file_batches = neid_dict["info"]["files"]
    template = neid_dict["info"]["baseline"]
    save_name = f"{dynamic_dir}/daily/{datatag}_0.pkl"

    processed = []
    for batch_name in file_batches:
        tag = os.path.basename(batch_name).rsplit('.', 1)[0]
        processed.append(f"{dynamic_dir}/processed/{tag}.pkl")

    print("processed:",processed)
    batch = merge_batch(processed)
    _,spec_input,w,ssbrv,jd = [item.numpy() for item in batch]
    w[w==0] = 1e-12
    dates = np.round(jd,0)
    uniq_dates = np.unique(dates)
    n_days = len(uniq_dates)
    print("jd:",jd.shape,"dates:",n_days,"spec_input:",spec_input.shape)
    print("ssbrv:",ssbrv.shape,ssbrv.min(),ssbrv.max())
    spec_day = np.zeros((n_days,spec_input.shape[1]))
    w_day = np.zeros((n_days,spec_input.shape[1]))
    ssbrv_day = np.zeros((n_days,1))
    jd_day = np.zeros((n_days,1))
    for i,day in enumerate(uniq_dates):
        mask = np.where(dates==day)[0]
        spec_day[i] = np.sum(w[mask]*spec_input[mask],axis=0)/np.sum(w[mask],axis=0)
        w_day[i] = np.sum(w[mask],axis=0)

        w_indiv = w[mask].mean(axis=-1)
        ssbrv_day[i] = np.sum(w_indiv*ssbrv[mask,0])/np.sum(w_indiv)
        jd_day[i] = np.sum(w_indiv*jd[mask,0])/np.sum(w_indiv)

    save_batch([np.zeros_like(jd_day),spec_day,w_day,ssbrv_day,jd_day],save_name)
    return

np.random.seed(0)
torch.manual_seed(0)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your script')

# Define optional arguments with default values
parser.add_argument('-t', '--tag', help='Tag description', default='test')
parser.add_argument('-when', '--when', help='Before or after the fire', default='a')
parser.add_argument('-n', '--samples', type=int, help='Number of samples', default=100)
parser.add_argument('-batch', '--batch_size', type=int, help='Batch size', default=500)
parser.add_argument('-cpu', '--num_cores', type=int, help='Number of CPU cores', default=10)
parser.add_argument('-load', '--load_data', action='store_true', help='Load data')
#parser.add_argument('-o','--orders', nargs='+', help='<Required> Orders', required=True)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
tag = args.tag
n_sample = args.samples
batch_size = args.batch_size
num_cores = args.num_cores
load_data = args.load_data

#ORDERS = [30,40,50,53,54,56,57,58]
#ORDERS = [int(o) for o in args.orders]
#ORDERS = np.arange(65,100)
#ORDERS = np.arange(41,100)
#ORDERS = np.arange(15,20)
ORDERS = [50,51]
ORDERS = [i for i in ORDERS if not i in blacklist]
print("ORDERS:",ORDERS)
print(" ".join([str(i) for i in ORDERS[1::2]]))

n_order = len(ORDERS)
fsr_mask = load_master_fsr_mask()
quality_mask = ~fsr_mask

input_wave = [get_order_wavelengths(o) for o in ORDERS]
for i,order in enumerate(ORDERS):
    print("order %d: wave_obs: %d, quality_mask:%d"%(order,len(input_wave[i]),quality_mask[order].sum()))

file_path = "/scratch/gpfs/yanliang/headers/NEID_QUIET_OBSNAME.txt"
# Load the data from the text file
data = np.loadtxt(file_path, dtype={'names': ('filename', 'jd', 'ccfrv'), 'formats': ('S30', 'f8', 'f8')})

if args.when =="a":
    suffix = "after"
    goodmask = (data['jd']>2459920.0)
    #goodmask = (data['jd']>2459920.0) & (data['jd']<2459940.0)
elif args.when =="b":
    suffix = "before"
    goodmask = (data['jd']<2459800.0)
else:
    print("invalid argument %s"%args.when)
    exit()
# Extract columns into separate arrays
NEID_JD = data['jd'][goodmask]
NEID_FILENAMES = np.array([x.decode('utf-8') for x in data['filename'][goodmask]])
NEID_CCFRV = data['ccfrv'][goodmask]

existing_files = os.listdir(datadir)
print("existing files:",len(existing_files))
print("qualified files:",len(NEID_FILENAMES))

# Find the indices of available_names in full_names
sel = np.nonzero(np.in1d(NEID_FILENAMES, existing_files))[0]
print("total number:",len(sel))
np.random.shuffle(sel)
SELECT = sel[:n_sample]

#for order in ORDERS:
#    preview_spectrum("neidL2_20211109T204628.fits",
#                     order,quality_mask=quality_mask[order])

raw_sample_names = list(NEID_FILENAMES[SELECT])
print("Selected samples:",len(raw_sample_names))

for i,order in enumerate(ORDERS):    
    datatag = "%s_order%d_%s"%(tag,order,suffix)
    if not load_data:
        print("wrapping order:",order)
        wrap_data(raw_sample_names,datatag,batch_size,order,quality_mask[order])
        calculate_template_spectrum(datatag,input_wave[i])
        ##calculate_v_template(datatag,input_wave[i])
        template_data = load_batch("%s/%s-template.pkl"%(dynamic_dir,datatag))
        instrument = Synthetic(template_data[0])
        correct_for_continuum(datatag,instrument,template_data)
        #daily_average_spectrum(datatag)
        exit()

        # remove the pre-post fire wavelength calib offset
        #print("Correcting for the bulk v offset...")
        #correct_v_bulk_offset(datatag)
        # redo template caculation
        #calculate_template_spectrum(datatag,input_wave[i])
        # redo v offest calculation
        #calculate_v_template(datatag,input_wave[i])

    print("Loading from %s-param.pkl"%datatag)
    with open(f'{runtime_dir}/{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)
    sample_names = neid_dict["info"]["sample_names"]
    timestamp = get_timeseries(neid_dict,'timestamp',sample_names)
    jds = get_timeseries(neid_dict,'OBSJD',sample_names)
    v_template = get_timeseries(neid_dict,'v_template',sample_names)
    fit_chi = get_timeseries(neid_dict,'chi_template',sample_names)
    ccfrv = get_timeseries(neid_dict,'CCFRV',sample_names)
    water_vapor = get_timeseries(neid_dict,'WVAPOR',sample_names)
    
    v_ccf = ccfrv-np.median(ccfrv)
    template_ccf_offset = v_template-v_ccf

    print("\nOrder %d:"%order)
    print_string("base_chi",fit_chi)
    if fit_chi.mean()>2:
        print("Mean $\chi^2 > 10$, skip order %d..."%order)
        blacklist.append(order)
        title_color = "r"
    else: title_color = "k"

    print_string("$v_{CCF}$",v_ccf,mode="2")
    print_string("$v_{template}$",(v_template),mode="2")
    print_string("$v_{template}-v_{CCF}$",
                 template_ccf_offset,mode="2")
    v_template -= np.median(v_template)

    fig,axs = plt.subplots(ncols=4,nrows=2,figsize=(15,6),constrained_layout=True)

    scale = fit_chi.max()/water_vapor.max()
    axs[1,2].plot(timestamp,scale*water_vapor,".",c="lavender",
                  label="Water Vapor",zorder=-10)
    plot_sample_selection(axs[1,:3], timestamp,v_template,v_ccf,fit_chi,order) 

    baseline = neid_dict["info"]["baseline"]
    baseline_w = neid_dict["info"]["baseline_w"]
    wave_obs = input_wave[i]

    print("\n\ngood spectra: %d/%d"%(len(sample_names),len(sel)))
    sn = baseline/baseline_w**(-0.5)
    print("sn:",sn.min(),sn.max(),"mean sn:",sn.mean())
    good = sn>1
    RV_limit = photon_noise(baseline[good],wave_obs[good],sn[good])
    print("Order %d RV_limit: %.2f m/s \n"%(order,RV_limit))

    file_batches = neid_dict["info"]["files"]

    print("file_batches:",file_batches)
    # load generated data
    batch = merge_batch(file_batches[:1])
    wave_raw,spec_raw,weights,ssbrvs,ids = [item.numpy() for item in batch]

    n_epoch,N_SPEC = spec_raw.shape
    fit_chi = fit_chi[:n_epoch]
    water_vapor = water_vapor[:n_epoch]

    wave_mean = np.median(wave_raw,axis=0)
    wave_std = np.std(wave_raw,axis=0)

    ax = axs[1,3]
    for i in range(min(100,n_epoch)):
        ax.plot(wave_mean,np.abs(wave_raw[i]-wave_mean),"k-",alpha=0.1,zorder=-20)
    if wave_std.max()>0.01:c="r"
    else: c="b"
    ax.plot(wave_mean,wave_std,label="order %d"%order,c=c)
    ax.legend(ncols=2,loc="upper left")
    ax.set_xlabel("wavelength")
    ax.set_ylabel("wavelength dispersion")

    rank = np.argsort(water_vapor)[::-1]
    i_plots = rank[::20]#list(rank[:2])+list(rank[-2:])
    i_plots = i_plots[:5]

    cmap = get_cmap('plasma')
    cmin,cmax = min(water_vapor),max(water_vapor)
    colors =[cmap((ii-cmin)/(cmax-cmin)) for ii in water_vapor]

    for ax in axs[0, :]:fig.delaxes(ax)
    ax = fig.add_subplot(2, 1, 1)  # Add a new subplot spanning the second row
    for i_obs,obsname in enumerate(sample_names):
        if not i_obs in i_plots:continue
        ccfrv = v_ccf[i_obs]
        yoffset = 0#ccfrv
        date_obs = neid_dict[obsname]['DATE-OBS']
        date = date_obs[5:10]
        time = date_obs[11:16]

        snr = spec_raw[i_obs]/(weights[i_obs]**(-0.5))

        text = "%.2f $v_{CCF}$:%.2f m/s $\chi^2=%.2f$"%(neid_dict[obsname]['timestamp'],ccfrv,fit_chi[i_obs])
        print(text,obsname)
        ax.plot(wave_raw[i_obs], spec_raw[i_obs],drawstyle="steps-mid",alpha=1.0,lw=1,
                c=colors[i_obs],label=text)
        ax.set_xlabel("Raw wavelength ($\AA$)")
        ax.set_ylabel("normalized flux")
        ax.legend(loc="lower left",ncols=2)
        ax.set_title("Order %d %s the fire"%(order,suffix),color=title_color)
    plt.tight_layout()
    plt.savefig(f'./[{datatag}]diagnostic.png',dpi=300)
    plt.clf()
    
print("blacklist",blacklist)



