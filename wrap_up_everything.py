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
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.ndimage import gaussian_filter1d
from matplotlib.cm import get_cmap
from astropy.io import fits
from scipy.interpolate import interp1d
from synthetic_data import Synthetic
from spender_model import TelluricModel
#from torchinterp1d import Interp1d

from util import moving_median,load_batch,merge_batch,load_master_fsr_mask,interpolate_to_input_grid_raw
from scipy.optimize import curve_fit,minimize

dynamic_dir = "/scratch/gpfs/JNWINN/yanliang/neid-production"
#datadir = "/scratch/gpfs/yanliang/NEID-SOLAR"
datadir = "/scratch/gpfs/JNWINN/yanliang/NEID-DRP1p4"
runtime_dir = f"{dynamic_dir}/params/"
device =  torch.device("cpu")

blacklist = [29,69,70]

colors = ["k",'b','c','m','orange',"gold",'navy',"skyblue"]
n_colors = len(colors)

# Gaussian function
def gaussian(x, amplitude, mean, stddev):
    if stddev<0.01:return np.ones_like(x)
    if amplitude>1:amplitude=1
    return np.abs(amplitude) * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Define a Gaussian function
def gaussian_ccf_chi(params,x, y_obs,full=False,y_err=1e-3):
    a, mu, sigma, c = params
    y_gauss = a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c
    chi = (y_obs-y_gauss)**2/y_err**2
    if full:return y_gauss
    return chi.mean()

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

def read_fits(filename,order_value,quality_mask,read_keys=['OBSJD','DATE-OBS','AIRMASS','E_VER']):
    hdulist = fits.open(filename)
    header = hdulist[0].header
    ccf_header = hdulist[12].header
    telluric_header = hdulist[10].header

    science_wavelength = hdulist[7].data[order_value][quality_mask]
    science_flux = hdulist[1].data[order_value][quality_mask]
    science_variance = hdulist[4].data[order_value][quality_mask]
    science_blaze = hdulist[15].data[order_value][quality_mask]


    # just the lines, no continuum
    #telluric_flux = hdulist[10].data[order_value][quality_mask]
    #print('\n telluric_flux:',telluric_flux.shape)
    #telluric_model = telluric_flux[:,0]#*telluric_flux[:,1]
    telluric_model = np.ones_like(science_flux)
    # Close the FITS file
    hdulist.close()
    
    science = [science_wavelength,science_flux,science_variance]
    data = [science,science_blaze,telluric_model]

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

def load_model(mainfile, instrument):
    device = instrument.wave_obs.device
    mdict = torch.load(mainfile, map_location=device)
    wave_rest =  mdict['model'][0]['wave_rest']
    spec_rest =  mdict['model'][0]['spec_rest']
    bias = mdict['model'][0]['encoder.mlp.mlp.9.bias']

    model = TelluricModel(wave_rest,spec_rest,instrument)
    model.load_state_dict(mdict['model'][0], strict=False)
    model.to(device)
    losses = mdict['losses']
    return model, losses

def evaluate_sky_model(model,batch,template=None,skymask=None):
    instrument = model.instrument
    polyb = model.evaluate_wavelength_polynomial(batch[0],batch[4])
    print(f"\nWavelength shift:{polyb.std(dim=1).max()*instrument.c:.3f} m/s \n")
    wave_obs = instrument.wave_obs

    # shift to the stellar restframe - frame of wave_obs
    spectrum, w, ssbrv,jd = interpolate_to_input_grid_raw(batch,instrument,template,polyb=polyb)
    spec_input = spectrum - template
    spec_input[w<1] = 0

    wave_raw,spec_raw,w_raw,ssbrv,jd = batch
    z_sky = (ssbrv)/instrument.c
    s_sky = model.encode(spec_input)

    lines,continuum,y_act,spec_model =  model._forward(s_sky,z_sky,wave_obs,skymask=skymask)

    telluric_spec = (1 - lines) * (1 + continuum)
    telluric_spec = model.transform(telluric_spec, z_sky, wave_obs)  

    aux = s_sky[:, -1:]
    clean_spec = (spectrum - 1e-3 * aux)/telluric_spec

    spec_avg = torch.median(clean_spec,dim=0)[0]
    spec_resid = clean_spec - spec_avg
    bad = (w<1)|(spec_avg<0.05)|(spec_resid>0.01)

    spec_resid[bad] = 0
    clean_spec[bad] = 0
    w[bad] = 1e-6

    w_std = moving_std_weight_batch_torch(spec_resid,w>1)
    w_new = 1/(1/w_std+1/w)

    new_batch = [clean_spec, w_new, ssbrv,jd]
    new_batch = [tensor2array(item) for item in new_batch]
    
    if spec_resid.max()<0.01:    return new_batch
    print("spec_resid:",spec_resid.min(),spec_resid.max())
    spec_resid = tensor2array(spec_resid)
    print("skymask fraction:",skymask.sum()/len(skymask))
    print("count_zero:",(lines==0).sum()/(lines.shape[0]*lines.shape[-1]))
    wave_raw,spec_raw,w_raw = [tensor2array(item) for item in batch[:3]]
    clean_spec, w_new, ssbrv,jd = new_batch
    wave_obs = tensor2array(wave_obs[0])
    template = tensor2array(template[0])
    spec_input = tensor2array(spec_input)
    fig,axs = plt.subplots(figsize=(18,4),nrows=2,constrained_layout=True)
    axs[0].plot(wave_obs,template,'k-',lw=1,alpha=1)
    for i in range(clean_spec.shape[0]):
        count_bad = (w_new[i]<1).sum()
        #print("count_bad:",count_bad)
        #print("spec input w:",(w[i]<1).sum())
        #print("w_raw:",(w_raw[i]<1).sum())
        #axs[0].plot(spec_input[i],'k-',lw=1,alpha=1)
        axs[0].plot(wave_raw[i],spec_raw[i],'-',lw=1,alpha=1)
        axs[1].plot(wave_obs,spec_resid[i],'-',lw=1,alpha=1)
        #break
    axs[0].set_ylabel("raw flux")
    axs[1].set_ylabel("telluric corrected")
    #axs[1].plot(wave_obs,clean_spec[where],'r-',lw=1,alpha=1)
    #for ax in axs:ax.set_ylim(0.86,1.05);
    #for ax in axs: ax.set_xlim(5670,560)
    plt.savefig("test.png",dpi=200)
    exit()
    #'''
    return


def redshift_chi(rv,wave_model,yrest,weight_rest,wave_data,ydata,wdata):
    wave_shifted = wave_model*(1.0 + rv/Synthetic.c)
    bad = yrest==0

    mask = (wave_data>min(wave_shifted[~bad]))&(wave_data<max(wave_shifted[~bad]))
    mask[:10] = False
    mask[-10:] = False
    model_obs = interp1d(wave_shifted[~bad], yrest[~bad], kind='cubic')(wave_data[mask])
    loss = np.sum(wdata[mask]* (ydata[mask] - model_obs)**2) / len(ydata[mask])
    #print("rv:",rv,"loss:",loss,"mask:",mask.sum())
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

def prepare_spectrum_single_order(obsname,quality_mask,order_value,telluric=False):
    large_number = 1e6
    #data,info_dict = read_fits("%s/%s"%(datadir,obsname),order_value=order_value,quality_mask=quality_mask)
    
    try:
        data,info_dict = read_fits("%s/%s"%(datadir,obsname),order_value=order_value,quality_mask=quality_mask)
    except:
        print("broken file!",obsname)

    n_spec = quality_mask.sum()
    wavelegth = np.zeros((n_spec))
    spectrum = np.zeros((n_spec))
    spectrum_err = np.zeros((n_spec))

    science,blaze,telluric_model = data
    wave_raw,flux,flux_var = science

    ssbrv = info_dict["SSBRV"]
    jd = info_dict["OBSJD"]

    isnan = np.isnan(flux) | np.isnan(blaze)| (flux<=0.0)

    normflux = np.zeros_like(flux)
    normflux_err = np.zeros_like(flux_var)
    

    normflux[~isnan] = flux[~isnan]/(telluric_model[~isnan]*blaze[~isnan])
    norm = np.quantile(normflux[~isnan],0.5)
    normflux[~isnan] /= norm
    normflux_err[~isnan] = flux_var[~isnan]**0.5/(norm*telluric_model[~isnan]*blaze[~isnan])

    #norm = np.quantile(flux[~isnan]/blaze[~isnan],0.5)
    #normflux[~isnan] = flux[~isnan]/(norm*blaze[~isnan])
    #normflux_err[~isnan] = flux_var[~isnan]**0.5/(norm*blaze[~isnan])
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
    if telluric:info_dict['telluric_model'] = telluric_model
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
def make_batch(sample_names,order_value,quality_mask,bulk_offset,max_neg_flux=100,max_wave_std=0.0005):
    large_number = 1e6
    batch_size = len(sample_names)
    n_spec = quality_mask.sum()

    wavemat =  np.zeros((batch_size,n_spec))
    specmat = np.zeros((batch_size,n_spec))
    errmat = np.zeros((batch_size,n_spec))
    good =  np.ones((batch_size),dtype=bool)

    shift_z = -bulk_offset[:,None]/Synthetic.c
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

    ccfrv = [local_dict[k]['CCFRV'] for k in local_dict]
    ccfrv_mean = np.mean(ccfrv)
    #print("ccfrv:",ccfrv)
    for i_obs,obsname in enumerate(sample_names):
        ccfrv_i = local_dict[obsname]['CCFRV']
        if np.abs(ccfrv_i-ccfrv_mean)>50:
            good[i_obs] = False
            print("large CCF RV!!",obsname,(ccfrv_i-ccfrv_mean))
    
    wavemat += shift_z*wavemat
    wave_mean = np.mean(wavemat,axis=0,keepdims=True)
    wave_std = (wavemat-wave_mean).std(axis=-1)
    #wh_obs = np.where(wave_std>max_wave_std)
    #for i_obs in np.unique(wh_obs):
        #good[i_obs] = False
    #    print("unusual wave solution: %d, skip..."%i_obs)
    print("wave_std:",wave_std.shape)

    bad = errmat**(-2)<1.0
    print("bad pixels per spec:",(bad.sum()/batch_size))
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
    result = scipy.optimize.minimize(redshift_chi,0.05, method='Nelder-Mead',args=(wave_rest,rest_model,weight_model,wave,spec,w,),tol=1e-8)
    label = "RV_fit=%.2f $\chi^2$:%.2f"%(result.x,result.fun)

    # not converged... try again
    if np.abs(result.x)<1e-4:
        chi_previous = result.fun
        result = scipy.optimize.minimize(redshift_chi,-0.05, method='Nelder-Mead',args=(wave_rest,rest_model,weight_model,wave,spec,w,),tol=1e-8)
        label = "RV_fit=%.2f $\chi^2$:%.2f"%(result.x,result.fun)
        if not result.fun<chi_previous:
            print("Strange behavior..",label)
            label += "Failed"
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

def make_batch_worker(batch_id, order_value, quality_mask, batch_name, bulk_offset, neid_dict):
    batch_id,wavemat,specmat,errmat,sub_dict = make_batch(batch_id,order_value,quality_mask,bulk_offset)
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

def remove_bulk_offset(datecodes,tref,vref,
                       utc_cut='20220801',jd_cut=2459797.5):
    bulk_offset = np.zeros(len(datecodes))
    ealier = np.array([d < utc_cut for d in datecodes])
    bulk_offset[ealier] = np.median(vref[tref<jd_cut])
    bulk_offset[~ealier] = np.median(vref[tref>=jd_cut])

    #plt.plot(tref,vref,'k.')
    #plt.axhline(bulk_offset.min(),color="b")
    #plt.axhline(bulk_offset.max(),color="r")
    #plt.savefig("test.png")
    return bulk_offset

def wrap_data(sample_names,datatag,diag_tag,batch_size,order_value,quality_mask):
    idx = np.arange(0, len(sample_names), batch_size)
    batches = np.array_split(sample_names, idx[1:])

    file_batches = ["%s/%s_%d.pkl"%(dynamic_dir,datatag,k) for k in range(len(batches))]

    with open(f'{runtime_dir}/{diag_tag}-param.pkl',"rb") as f:
        ref_dict = pickle.load(f)

    sname = ref_dict['info']['sample_names']
    tref = np.array([ref_dict[k]['OBSJD'] for k in sname])
    vref = np.array([ref_dict[k]['v_template'] for k in sname])

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
        date_code = [item[7:15] for item in batch_id]
        bulk_offset = remove_bulk_offset(date_code,tref,vref)

        print ("saving batch  %d / %d"%(k,len(file_batches)))    
        work_p = mp.Process(target=make_batch_worker,
                            args=(batch_id, order_value, quality_mask, batch_name, bulk_offset, mdict))
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

def calculate_template_spectrum(datatag,wave_obs,subdir=""):
    print(f'Loading from {runtime_dir}/{datatag}-param.pkl')
    with open(f'{runtime_dir}/{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)

    order_value = neid_dict["info"]["order"]
    
    file_batches = []
    for k,batch_name in enumerate(neid_dict["info"]["files"]):
        basename = os.path.basename(batch_name).rsplit('.', 1)[0]
        fname = f"{dynamic_dir}/{subdir}{basename}.pkl"
        if os.path.isfile(fname):
            file_batches.append(fname)

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

    template_name="%s/%s%s.pkl"%(dynamic_dir,subdir,datatag+"-template")
    save_batch([wave_obs[None,:],template[None,:],template_w[None,:],
                np.array([0]),np.array([888])],template_name)
    neid_dict["info"].update({f"{subdir}baseline":template,f"{subdir}baseline_w":template_w})
    with open(f'{runtime_dir}/{datatag}-param.pkl',"wb") as f:
        pickle.dump(neid_dict,f)
    return

def update_template_spectrum(datatag,wave_obs):
    print(f'Loading from {runtime_dir}/{datatag}-param.pkl')
    with open(f'{runtime_dir}/{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)
    file_batches = neid_dict["info"]["files"]
    order_value = neid_dict["info"]["order"]
    
    processed = []
    for k,batch_name in enumerate(file_batches):
        basename = os.path.basename(batch_name).rsplit('.', 1)[0]
        fname = f"{dynamic_dir}/processed/{basename}.pkl"
        if os.path.isfile(fname):
            processed.append(fname)
    print("processed:",processed)
    batch = merge_batch(processed)
    _,spectrum,weights,ssbrvs,ids = [item.numpy() for item in batch]
    
    n_spec = spectrum.shape[-1]
    template = np.median(spectrum,axis=0)
    template_w = np.median(weights,axis=0)
    dispersion =  np.zeros((n_spec))

    for i in range(n_spec):
        flux = spectrum[:,i]
        non_zero = flux>0
        if non_zero.sum()==0:continue
        dispersion[i] = np.std(flux[non_zero])

    '''
    fig,ax=plt.subplots(figsize=(16,4),constrained_layout=True)
    for i in range(50):
        mask = spectrum[i]>0
        ax.plot(wave_obs[mask],spectrum[i][mask],"k-",lw=1,alpha=0.1,drawstyle="steps-mid")
    ax.plot(wave_obs,template,"r-",lw=1,drawstyle="steps-mid",label="template")
    ax.plot(wave_obs,dispersion,"c-",lw=1,drawstyle="steps-mid",label="dispersion")
    ax.legend()
    #ax.set_xlim(wave_obs[300]-2,wave_obs[300]+2)
    plt.savefig("[%s]template.png"%datatag,dpi=200)
    '''

    template_name=f"{dynamic_dir}/processed/{datatag}-template.pkl"
    save_batch([wave_obs[None,:],template[None,:],template_w[None,:],
                np.array([0]),np.array([888])],template_name)
    neid_dict["info"].update({"processed_baseline":template,
                              "processed_baseline_w":template_w})
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

def v_template_consistency(datatag,wave_obs):
    print(f'Loading from {runtime_dir}/{datatag}-param.pkl')
    with open(f'{runtime_dir}/{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)

    sample_names = neid_dict["info"]["sample_names"]
    # calculate v_template and chi_template
    file_batches = neid_dict["info"]["files"]
    template = neid_dict["info"]["processed_baseline"]
    template_w = neid_dict["info"]["baseline_w"]
    order_value = neid_dict["info"]["order"]
    
    
    processed = []
    for k,batch_name in enumerate(file_batches[:1]):
        basename = os.path.basename(batch_name).rsplit('.', 1)[0]
        fname = f"{dynamic_dir}/processed/{basename}.pkl"
        processed.append(fname)
    print("processed:",processed)

    batch = merge_batch(processed[:1])
    _,specs,weights,ssbrvs,ids = [item.numpy() for item in batch]

    v_extra = np.linspace(-50,50,len(specs))
    
    v_extra = v_extra

    ref_dict = {}
    inject_dict = {}
    for i_epoch in range(len(v_extra)):
        # no injection
        args = (wave_obs, specs[i_epoch], weights[i_epoch], wave_obs, template, template_w, f"{i_epoch}", order_value)
        process_task(args,ref_dict)
        
        # inject v_extra
        wave_inject = wave_obs*(1+v_extra[i_epoch]/Synthetic.c)
        args = (wave_inject, specs[i_epoch], weights[i_epoch], wave_obs, template, template_w, f"{i_epoch}", order_value)
        process_task(args,inject_dict)
    
    v_ref = np.array([ref_dict[key]['v_template'] for key in ref_dict])
    v_inject = np.array([inject_dict[key]['v_template'] for key in ref_dict])
    chi_ref = np.array([ref_dict[key]['chi_template'] for key in ref_dict])
    chi_inejct = np.array([inject_dict[key]['chi_template'] for key in ref_dict])

    quantile = np.quantile(v_ref,[0.16,0.84])
    print("quantile:",quantile)
    print(f"v_ccf rms: {0.5*(quantile[1]-quantile[0]):.3f} m/s")
    print(f"chi : {chi_ref.mean():.3f}")

    fig,axs=plt.subplots(figsize=(8,3),ncols=2,constrained_layout=True)
    ax=axs[0]
    ax.plot(v_extra,(v_inject-v_ref),"k.")
    ax.plot(v_extra,v_extra,"r--",label="x=y")
    ax.legend()
    ax.set_xlabel("Injected Velocity [m/s]")
    ax.set_ylabel("CCF Recovered Velocity [m/s]")
    ax=axs[1]
    xrange = [chi_ref.min(),chi_ref.max()]
    ax.plot(chi_ref,chi_inejct,"k.")
    ax.plot(xrange,xrange,"r--",label="x=y")
    ax.legend()
    ax.set_xlabel("$\chi_r$")
    ax.set_ylabel("$\chi_r$ With Injected Velocity")
    plt.savefig("v_template_check.png",dpi=200)
    plt.clf()
    exit()
    return

def multiple_order_v_template(tag,suffix):
    params_file = f'{runtime_dir}{tag}_{suffix}-param.pkl'
    print(f'Loading from {params_file}')
    with open(params_file,"rb") as f:
        neid_dict = pickle.load(f)
    sample_names = neid_dict["info"]["sample_names"]
    #print(neid_dict['info']['files'])
    # calculate v_template and chi_template
    file_batches = neid_dict["info"]["files"]
    wave_obs = neid_dict["info"]["wave_obs"][0]
    template = neid_dict["info"]["baseline"]
    template_w = neid_dict["info"]["baseline_w"]
    order_value = neid_dict["info"]["order"]

    start = 0
    for j,file in enumerate(file_batches):
        basename = os.path.basename(file)
        merge_file = f"{dynamic_dir}/merge/{basename}"
        batch = merge_batch([merge_file])
        _,specs,weights,ssbrvs,ids = [item.numpy() for item in batch]

        manager = mp.Manager()
        mdict = manager.dict()
        # Use a pool of workers
        pool_size = num_cores  # Number of processes in the pool
        pool = mp.Pool(pool_size)
        tasks = []

        end = start + len(ids)
        for iloc,i_epoch in enumerate(np.arange(start,end)):
            obsname = sample_names[i_epoch]
            # skip existing entries
            if 'v_template' in neid_dict[obsname]:continue
            # already in stellar restframe
            task_args = (wave_obs, specs[iloc], weights[iloc], wave_obs,template,template_w, obsname, order_value)
            tasks.append(task_args)
        start = end

        for i,task in enumerate(tasks):
            pool.apply_async(process_task, args=(task, mdict))
        # Close and join the pool
        pool.close()
        pool.join()

        print(j,"mdict:",mdict)
        for k in sample_names:
            if k in mdict:neid_dict[k].update(mdict[k])
            #else:print("%s missing..."%k)

        print(f"Saving to {params_file}...")
        with open(params_file,"wb") as f:
            pickle.dump(neid_dict,f)
    return 

def multiple_order_v_template_consistency(tag,suffix):
    params_file = f'{runtime_dir}{tag}_{suffix}-param.pkl'
    print(f'Loading from {params_file}')
    with open(params_file,"rb") as f:
        neid_dict = pickle.load(f)
    sample_names = neid_dict["info"]["sample_names"]
    print(neid_dict['info']['files'])
    print("sample_names:",len(sample_names))
    # calculate v_template and chi_template
    file_batches = neid_dict["info"]["files"]
    wave_obs = neid_dict["info"]["wave_obs"][0]
    template = neid_dict["info"]["baseline"]
    template_w = neid_dict["info"]["baseline_w"]
    order_value = neid_dict["info"]["order"]

    ccf_files = []
    for k,batch_name in enumerate(file_batches):
        print("Loading %s..."%batch_name)
        datatag = os.path.basename(batch_name).rsplit('.', 1)[0]
        ccf_files.append(f"{dynamic_dir}/ccf_info/{datatag}.pkl")

    print("file_batches:",ccf_files)
    #print("wave_obs:",wave_obs.shape,"template:",template.shape)

    batch = merge_batch(ccf_files[:1])
    ccf_info,specs,weights,ssbrvs,ids = [item.numpy() for item in batch]

    print("ccf_info:",ccf_info.shape)
    v_template = ccf_info[:,0]
    v_ccf = ccf_info[:,1]
    times = ids

    fig,ax=plt.subplots(figsize=(8,3),constrained_layout=True)
    ax.plot(times,v_ccf,'k.',label=f"v_ccf RMS = {v_ccf.std():.3f}")
    ax.plot(times,v_template,'r.',label=f"v_template RMS = {v_template.std():.3f}")
    ax.legend()
    plt.savefig("test.png",dpi=200)
    exit()
    
    v_extra = np.linspace(-10,10,len(specs))
    v_extra = v_extra

    ref_dict = {}
    inject_dict = {}
    for i_epoch in range(len(v_extra)):
        # no injection
        args = (wave_obs, specs[i_epoch], weights[i_epoch], wave_obs, template, template_w, f"{i_epoch}", order_value)
        process_task(args,ref_dict)
        
        # inject v_extra
        wave_inject = wave_obs*(1+v_extra[i_epoch]/Synthetic.c)
        args = (wave_inject, specs[i_epoch], weights[i_epoch], wave_obs, template, template_w, f"{i_epoch}", order_value)
        process_task(args,inject_dict)
    
    v_ref = np.array([ref_dict[key]['v_template'] for key in ref_dict])
    v_inject = np.array([inject_dict[key]['v_template'] for key in ref_dict])
    chi_ref = np.array([ref_dict[key]['chi_template'] for key in ref_dict])
    chi_inejct = np.array([inject_dict[key]['chi_template'] for key in ref_dict])

    quantile = np.quantile(v_ref,[0.16,0.84])
    print("quantile:",quantile)
    print(f"v_ref rms: {0.5*(quantile[1]-quantile[0]):.3f} m/s")
    v_resid = v_inject-v_ref-v_extra
    print(f"v_difference rms: {v_resid.std():.3e} m/s")
    print(f"chi : {chi_ref.mean():.3f}")

    fig,axs=plt.subplots(figsize=(8,3),ncols=2,constrained_layout=True)
    ax=axs[0]
    ax.plot(v_extra,(v_inject-v_ref),"k.")
    ax.plot(v_extra,v_extra,"r--",label="x=y")
    ax.legend()
    ax.set_xlabel("Injected Velocity [m/s]")
    ax.set_ylabel("CCF Recovered Velocity [m/s]")
    ax=axs[1]
    xrange = [chi_ref.min(),chi_ref.max()]
    ax.plot(chi_ref,chi_inejct,"k.")
    ax.plot(xrange,xrange,"r--",label="x=y")
    ax.legend()
    ax.set_xlabel("$\chi_r$")
    ax.set_ylabel("$\chi_r$ With Injected Velocity")
    plt.savefig("v_template_check.png",dpi=200)
    plt.clf()
    exit()
    return

def update_v_template(datatag,wave_obs):
    print(f'Loading from {runtime_dir}/{datatag}-param.pkl')
    with open(f'{runtime_dir}/{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)

    sample_names = neid_dict["info"]["sample_names"]
    # calculate v_template and chi_template
    file_batches = neid_dict["info"]["files"]
    template = neid_dict["info"]["processed_baseline"]
    template_w = neid_dict["info"]["baseline_w"]
    order_value = neid_dict["info"]["order"]

    processed = []
    for k,batch_name in enumerate(file_batches):
        basename = os.path.basename(batch_name).rsplit('.', 1)[0]
        fname = f"{dynamic_dir}/processed/{basename}.pkl"
        processed.append(fname)
    print("processed:",processed)

    batch = merge_batch(processed)
    _,specs,weights,ssbrvs,ids = [item.numpy() for item in batch]

    manager = mp.Manager()
    mdict = manager.dict()
    # Use a pool of workers
    pool_size = num_cores  # Number of processes in the pool
    pool = mp.Pool(pool_size)
    tasks = []
    for i_epoch,obsname in enumerate(sample_names):
        # skip existing entries
        #if 'v_template' in neid_dict[obsname]:continue
        # already in stellar restframe
        task_args = (wave_obs, specs[i_epoch], weights[i_epoch], wave_obs,template,template_w, obsname, order_value)
        tasks.append(task_args)

    for i,task in enumerate(tasks):
        pool.apply_async(process_task, args=(task, mdict))
    # Close and join the pool
    pool.close()
    pool.join()


    mdict_update = {}
    for k in mdict:
        mdict_update[k] = {}
        for col in mdict[k]:
            mdict_update[k][f"{col}_processed"]=mdict[k][col]    

    for k in sample_names:
        if k in mdict_update:neid_dict[k].update(mdict_update[k])
        else:print("%s missing..."%k)

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

def preview_spectrum(ax,obsname,order_value,quality_mask):
    data,info_dict = prepare_spectrum_single_order(obsname,quality_mask,order_value,telluric=True)
    wavelength,spectrum,spectrum_err = data

    spec_telluric = info_dict['telluric_model']
    #spec_telluric /= spec_telluric.max()
    print("order",order_value,"spec_telluric",spec_telluric.min(),spec_telluric.max())
    
    #spec_telluric = 1-(10*(1-spec_telluric))
    print(spec_telluric.shape)

    #telluric = info_dict['telluric_model']
    ylim=[0.8,max(1.05,spectrum.max())]

    ax.set_title("Order %d"%order_value)
    ax.plot(wavelength,spectrum,"k-",lw=1,label="corrected",drawstyle="steps-mid")
    ax.plot(wavelength,spec_telluric,"b-",lw=1,label="telluric",drawstyle="steps-mid")
    #ax.plot(wavelength,spectrum/spec_telluric,"r-",lw=1,label="correction",drawstyle="steps-mid")
    #ax.plot(wavelength[i],ydiff[i],"r-",lw=1,drawstyle="steps-mid")
    ax.fill_between(wavelength,spectrum-spectrum_err,
                    spectrum+spectrum_err,step="mid",
                    color="k",alpha=0.3,zorder=-10)
    ax.legend(loc=3)
    ax.set_xlim(wavelength[0],wavelength[-1])
    ax.set_ylim(ylim)
    return

def initialize_restframe_model(wave_obs,template,weight):
    wave_rest = wave_obs
    spec_rest = np.ones_like(template)
    spec_rest[weight>1.0]=template[weight>1.0]
    init_rest = np.array([wave_rest,spec_rest])
    print("init_rest:",init_rest.shape)
    return init_rest

def print_string(vname,v,mode="1"):
    if mode=="1":
        quantiles = [v.min(),v.max(),v.mean()]
        string = " ".join(["%.2f"%item for item in quantiles])
    if mode=="2":
        string =  "%.3f +/- %.3f m/s"%(v.mean(),v.std())
    print("%s: %s"%(vname,string))
    return

def plot_sample_selection(axs, timestamp,v_template,v_ccf,our_ccf,fit_chi,order, 
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
    ax.scatter(timestamp,v_ccf,c=grey_color,s=2,label="$v_{CCF,NEID}$ RMS = %.2f m/s"%(v_ccf.std()))
    ax.scatter(timestamp,our_ccf,c='r',s=2,label="$v_{CCF,Clean}$ RMS = %.2f m/s"%(our_ccf.std()))

    ax.legend(title="Discrepancy RMS = %.2f m/s"%(v_template-our_ccf).std())
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

def moving_std_weight_batch_torch(data, valid_mask, window_size=6):
    # Input: data and valid_mask are tensors of shape (B, N)
    # Ensure float dtype
    data = data.float()
    valid_mask = valid_mask.float()

    # Compute cumulative sums for data, data², and mask
    cumsum = torch.cumsum(data, dim=1)
    cumsum2 = torch.cumsum(data ** 2, dim=1)
    valid_cumsum = torch.cumsum(valid_mask, dim=1)

    # Compute moving sums using cumulative sum trick
    moving_sum = cumsum[:, window_size:] - cumsum[:, :-window_size]
    moving_sum2 = cumsum2[:, window_size:] - cumsum2[:, :-window_size]
    valid_count = valid_cumsum[:, window_size:] - valid_cumsum[:, :-window_size]

    # Initialize variance tensor
    var = torch.zeros_like(moving_sum)
    good = valid_count > 3

    # Compute variance where valid
    var[good] = moving_sum2[good] / valid_count[good]
    var[good] -= (moving_sum[good] / valid_count[good]) ** 2

    # Handle invalid cases with large value
    large_value = 1e12
    var[~good] = large_value

    # Pad on both sides to match original sequence length
    pad_left = var[:, 0:1].repeat(1, window_size // 2)
    pad_right = var[:, -1:].repeat(1, window_size // 2)
    var_padded = torch.cat([pad_left, var, pad_right], dim=1)

    return 1.0 / var_padded

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
def continuum_loss(param, y_perturb, weight, y_quiet, smooth_radius = 15, full=False):
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
    spec_input[w<1] = 0
    #'''
    for i in range(spec_input.shape[0]):
        #if i<=10:continue
        result = minimize(continuum_loss,[0.0001], args=(spec[i],w[i],template))
        pbest = result.x
        loss,y_recover,f_continuum,b_poly = continuum_loss(pbest,spec[i],w[i],template,full=True)
        spec_input[i,w[i]>1] = (y_recover - template)[w[i]>1]
        #plot_continuum(spec[i],w[i],spec_input[i],f_continuum,b_poly,template)
    #'''
    save_batch([np.zeros_like(jd),spec_input,w,ssbrv,jd],save_name)
    return


def correct_for_telluric_lines(datatag,instrument,template_data,skymodel,skymask):
    print("\n\n Correcting for telluric lines and continuum...")
    params_file = f'{runtime_dir}{datatag}-param.pkl'
    print(f'Loading from {params_file}')
    with open(params_file,"rb") as f:
        neid_dict = pickle.load(f)
    file_batches = neid_dict["info"]["files"]
    template = template_data[1]
    print("template:",template.shape)
    print("file_batches:",file_batches)
    
    for k,batch_name in enumerate(file_batches):
        print("Loading %s..."%batch_name)
        datatag = os.path.basename(batch_name).rsplit('.', 1)[0]
        save_name = f"{dynamic_dir}/processed/{datatag}.pkl"

        batch = load_batch(batch_name)
        batch = [item.to(device=device) for item in batch]
        print("batch_name:",batch_name)
        spectra, w, ssbrv, jd = evaluate_sky_model(skymodel,batch,template,skymask)
        save_batch([np.zeros_like(jd),spectra,w,ssbrv,jd],save_name)
    return

def merge_multiple_orders(outtag,ORDERS,suffix,sample_names,batch_size,tag="newprod"):
    templates = []
    skymasks = []
    merge_dict = {}

    info_keys = ['OBSJD','DATE-OBS','AIRMASS','WVAPOR','E_VER','ZENITH',
                 'CCFRVMOD','DVRMSMOD','timestamp']


    idx = np.arange(0, len(sample_names), batch_size)
    batches = np.array_split(sample_names, idx[1:])
    order_dict = {}
    print("batches:",len(batches))

    intersect_mask = np.array([True]*len(sample_names))
    
    for i,order in enumerate(ORDERS):
        order_dict[order] = {}
        datatag = f"{tag}_order{order}_{suffix}"
        template_name = f"{dynamic_dir}/{datatag}-template.pkl"
        skymask_file = f"{dynamic_dir}/skymask/{datatag}-skymask.pkl"

        params_file = f'{runtime_dir}{datatag}-param.pkl'
        print(f'Loading from {params_file}')
        with open(params_file,"rb") as f:
            neid_dict = pickle.load(f)

        samples_i = neid_dict['info']['sample_names']
        mask_i = np.isin(sample_names,samples_i,assume_unique=True)
        intersect_mask &= mask_i
        print("order:",order,"samples:",len(samples_i),
              "intersect_mask",intersect_mask.sum())
        order_dict[order]['samples_i'] = samples_i

        template_data = load_batch(template_name)
        template_data = [item.to(device=device) for item in template_data]
        templates.append(template_data)
        skymasks.append(load_batch(skymask_file).bool())

        for obsname in sample_names:
            if not obsname in neid_dict:continue
            select = {key:neid_dict[obsname][key] for key in info_keys}
            if not obsname in merge_dict:merge_dict[obsname] = select

    intersect_names = np.array(sample_names)[intersect_mask]
    merge_dict['info'] = {'sample_names':intersect_names,"order":ORDERS}

    #'''
    for k in range(len(batches)):
        k_names = batches[k]
        for o in ORDERS:
            in_order = np.isin(k_names,order_dict[o]['samples_i'])
            order_k_mask = np.isin(k_names[in_order],intersect_names)
            order_dict[o][k] = order_k_mask
            print(f"batch {k} order {o} mask shape: {order_k_mask.shape} mask sum: {order_k_mask.sum()}")
    #'''
    # update spectral template
    combined_wave_obs = torch.cat([item[0] for item in templates],dim=1)
    print("combined_wave_obs:",combined_wave_obs.shape)
    combined_wave_obs = tensor2array(combined_wave_obs)

    wavelist = [tensor2array(item[0]) for item in templates]
    wavemasks = [
        (w < np.min(wavelist[i+1])) if i < len(wavelist)-1 else np.ones_like(w, bool) 
        for i, w in enumerate(wavelist)
    ]

    wavemasks = np.concatenate(wavemasks,axis=1)[0]
    combined_wave_obs = combined_wave_obs[:,wavemasks]
    print("combined_wave_obs",combined_wave_obs.shape)


    filenames = []
    for k in range(len(batches)):
        save_name = f"{dynamic_dir}/merge/{outtag}_{suffix}_{k}.pkl"
        filenames.append(save_name)
        if os.path.isfile(save_name):
            print(f"{save_name} already exists...")
            continue
        combined_spectra = []
        combined_w = []
        combined_jd = []
        for i,order in enumerate(ORDERS):
            datatag = f"{tag}_order{order}_{suffix}"
            batch_name = f"{dynamic_dir}/{datatag}_{k}.pkl"
            skymodel_file = f"skymodel/blue_order{order}_full.pt"

            print("batch_name:",batch_name)
            wave_obs,template = templates[i][:2]
            instrument = Synthetic(wave_obs)

            skymodel,losses = load_model(skymodel_file,instrument)
            batch = load_batch(batch_name)
            batch = [item.to(device=device) for item in batch]

            out_batch = evaluate_sky_model(skymodel,batch,template,skymasks[i])
            mask_ok = order_dict[order][k]
            spectra, w, ssbrv, jd = [item[mask_ok] for item in out_batch]

            combined_spectra.append(spectra)
            combined_w.append(w)
            combined_jd.append(jd)

        combined_spectra = np.concatenate(combined_spectra,axis=1)
        combined_w = np.concatenate(combined_w,axis=1)
        combined_spectra = combined_spectra[:,wavemasks]
        combined_w = combined_w[:,wavemasks]
        print("combined_spectra:",combined_spectra.shape,
              "combined_w:",combined_w.shape,"jd:",jd.shape)
        print(f"saving to {save_name}...")
        save_batch([np.zeros_like(jd),combined_spectra,combined_w,ssbrv,jd],save_name)


    print("filenames:",filenames)
    batch = merge_batch(filenames[:1])
    _,spectrum,weights,ssbrvs,ids = [item.numpy() for item in batch]
    print("spectrum:",spectrum.shape)

    n_spec = spectrum.shape[-1]
    template = np.zeros((n_spec))
    template_w = np.zeros((n_spec))
    dispersion =  np.zeros((n_spec))

    for i in range(n_spec):
        flux = spectrum[:,i]
        w = weights[:,i]
        good = (flux>0)&(w>1)
        if good.sum()==0:continue
        template[i] = np.median(flux[good])
        template_w[i] = np.median(w[good])
        dispersion[i] = np.std(flux[good])

    
    merge_dict['info']['files'] = filenames
    merge_dict['info']['wave_obs'] = combined_wave_obs
    merge_dict['info']['baseline'] = template
    merge_dict['info']['baseline_w'] = template_w

    params_file = f'{runtime_dir}{outtag}_{suffix}-param.pkl'
    print(f"Saving to {params_file}...")
    with open(params_file,"wb") as f:
        pickle.dump(merge_dict,f)

    template_name=f"{dynamic_dir}/merge/{outtag}_{suffix}-template.pkl"
    save_batch([combined_wave_obs,template[None,:],template_w[None,:],
                np.array([0]),np.array([888])],template_name)
    #'''
    fig,ax=plt.subplots(figsize=(16,3),constrained_layout=True)
    for i in range(50):
        mask = spectrum[i]>0
        ax.plot(combined_wave_obs[0][mask],spectrum[i][mask],"k-",lw=1,alpha=0.1)
    ax.plot(combined_wave_obs[0],template,"r-",lw=1,drawstyle="steps-mid",label="template")
    #ax.plot(combined_wave_obs[0],template_w/template_w.max(),"b-",lw=0.5,drawstyle="steps-mid",label="weights")
    ax.plot(combined_wave_obs[0],100*dispersion,"c-",lw=1,drawstyle="steps-mid",label="100xdispersion")
    ax.legend(loc="lower right")
    #ax.set_xlim(5452.5,5460)
    plt.savefig("[%s]template.png"%outtag,dpi=200)
    #'''
    return

def compute_ccf(spec_tensor, w_tensor, wavelength, template, 
                velocity_grid=np.linspace(-15, 15, 201),v_extra=None):

    spectra = tensor2array(spec_tensor)
    weights = tensor2array(w_tensor)

    N_spectra, N_pixels = spectra.shape

    # Initialize CCF matrix
    ccf_matrix = np.zeros((N_spectra, len(velocity_grid)))
    # Speed of light in km/s
    c = 299792.458

    if v_extra is None: v_extra=np.zeros((N_spectra))

    wmin = wavelength.min()/(1 + velocity_grid.min() / c)
    wmax = wavelength.max()/(1 + velocity_grid.max() / c)
    goodmask = (template<1)&(wavelength>wmin)&(wavelength<wmax)

    template_interp = interp1d(wavelength[goodmask], template[goodmask], kind='cubic', bounds_error=False, fill_value=0.0)

    shifted_template = np.zeros((len(velocity_grid),N_pixels))
    for i, velocity in enumerate(velocity_grid):
        # Doppler shift the template wavelengths
        shifted_wavelength = wavelength * (1 - velocity / c)
        shifted_template[i] = template_interp(shifted_wavelength)

    # Compute cross-correlation for each spectrum
    for j in range(N_spectra):
        #valid = (weights[j]>1)&(shifted_template!=1)
        bad = weights[j]<1
        spectra_shifted = interp1d(wavelength[~bad]* (1.0+v_extra[j]/c), spectra[j][~bad], kind='cubic', bounds_error=False, fill_value=1.0)(wavelength)
        for i in range(len(velocity_grid)):
            ccf_matrix[j, i] = np.sum(spectra_shifted * shifted_template[i])
    # Normalize CCF to range [0, 1]
    ccf_matrix = (ccf_matrix) / (
        np.max(ccf_matrix, axis=1, keepdims=True)
    )
    return ccf_matrix, velocity_grid

def fit_gaussian_ccf(velocity_grid, ccf):
    """
    Fit a Gaussian to the CCF to determine the velocity offset.
    
    Parameters:
        velocity_grid (numpy.ndarray): 1D array of velocity values (km/s).
        ccf (numpy.ndarray): 1D array of normalized CCF values.
    
    Returns:
        popt (tuple): Best-fit parameters (amplitude, mean, sigma, offset).
        velocity_offset (float): The best-fit Gaussian center (km/s).
    """
    # Initial guesses
    a_init = np.min(ccf)-np.max(ccf)  # Amplitude (min of CCF)
    mu_init = velocity_grid[np.argmin(ccf)] # Initial guess for center
    sigma_init = 4  # Rough estimate of width
    c_init = np.max(ccf)  # Baseline

    p0 = [a_init, mu_init, sigma_init, c_init]

    # Fit Gaussian to CCF
    #popt, pcov = curve_fit(gaussian_ccf, velocity_grid, ccf, p0=p0)
    res = minimize(gaussian_ccf_chi, p0, method='Nelder-Mead', args=(velocity_grid,ccf),tol=1e-7)
    popt = res.x
    chi = res.fun
    # Extract velocity offset (Gaussian mean)
    velocity_offset = popt[1]
    '''
    print("velocity_offset:",velocity_offset*1e3,"chi:",chi)
    y_gauss = gaussian_ccf_chi(popt,velocity_grid,ccf,full=True)
    y0 = gaussian_ccf_chi(p0,velocity_grid,ccf,full=True)
    plt.plot(velocity_grid,ccf,'k-')
    plt.plot(velocity_grid,y0,'b-',lw=1)
    plt.plot(velocity_grid,y_gauss,'r-',lw=1)
    plt.axvline(popt[1],ls='--',color="grey",label=f"v={popt[1]*1e3:.2f}m/s")
    plt.legend()
    plt.savefig("test.png",dpi=200)
    exit()
    '''
    return popt, velocity_offset, chi

def compute_bisspan(velocity_grid,ccf_matrix):
    v_ccf = np.zeros((ccf_matrix.shape[0],2))
    params =  np.zeros((ccf_matrix.shape[0],4))
    mask = np.abs(velocity_grid)<=5.0
    for i,ccf in enumerate(ccf_matrix):
        popt, velocity_offset, chi = fit_gaussian_ccf(velocity_grid[mask], ccf[mask])
        v_ccf[i][0] = velocity_offset*1e3
        v_ccf[i][1] = chi
        params[i] = popt
    #v_ccf[:,0] -= v_ccf[:,0].mean()

    pos = velocity_grid>0
    depth = ccf_matrix.max()-ccf_matrix.min()
    depths_top = 1-np.linspace(0.1, 0.4, 20)*depth
    depths_bottom = 1-np.linspace(0.6, 0.9, 20)*depth
    bisspan = np.zeros((ccf_matrix.shape[0]))

    for i,ccf in enumerate(ccf_matrix):
        left_half = interp1d(ccf[~pos],velocity_grid[~pos],kind="linear")
        right_half =  interp1d(ccf[pos],velocity_grid[pos],kind="linear")

        v_top = np.array([np.mean([left_half(x),right_half(x)]) for x in depths_top])
        v_bottom = np.array([np.mean([left_half(x),right_half(x)]) for x in depths_bottom])
        bisspan[i] = np.mean(v_top)-np.mean(v_bottom)
    return params,v_ccf,bisspan

def check_ccf_consistency(spectra,w,wave_obs,template):
    v_extra = np.linspace(-0.05,0.05,len(spectra))
    ccf_matrix, velocity_grid = compute_ccf(spectra, w, wave_obs, 1-template)
    params,v_ccf,bisspan = compute_bisspan(velocity_grid,ccf_matrix)
    
    ccf_matrix, velocity_grid = compute_ccf(spectra, w, wave_obs, 1-template,v_extra=v_extra)
    params,v_ccf_inject,bisspan_inject = compute_bisspan(velocity_grid,ccf_matrix)

    quantile = np.quantile(v_ccf[:,0],[0.16,0.84])
    print("quantile:",quantile)
    print(f"v_ccf rms: {0.5*(quantile[1]-quantile[0]):.3f} m/s")
    print(f"chi : {v_ccf[:,1].mean():.3f}")

    v_extra*=1e3
    bisspan*=1e3
    bisspan_inject*=1e3

    fig,axs=plt.subplots(figsize=(8,3),ncols=2,constrained_layout=True)
    ax=axs[0]
    ax.plot(v_extra,(v_ccf_inject[:,0]-v_ccf[:,0]),"k.")
    ax.plot(v_extra,v_extra,"r--",label="x=y")
    ax.legend()
    ax.set_xlabel("Injected Velocity [m/s]")
    ax.set_ylabel("CCF Recovered Velocity [m/s]")
    ax=axs[1]
    xrange = [bisspan.min(),bisspan.max()]
    ax.plot(bisspan,bisspan_inject,"k.")
    ax.plot(xrange,xrange,"r--",label="x=y")
    ax.legend()
    ax.set_xlabel("BISSPAN [m/s]")
    ax.set_ylabel("BISSPAN with Injected Velocity [m/s]")
    plt.savefig("bisspan.png",dpi=200)
    plt.clf()
    exit()
    return

def ccf_worker(batch_name,save_name,wave_obs,template,timetable):
    batch = load_batch(batch_name)
    print("[ccf worker]batch_name:",batch_name)
    #_,spec_input,w,ssbrv,jd = batch
    #spectra = tensor2array(spec_input) + template
    _,spectra,w,ssbrv,jd = batch

    v_template = [timetable[t]["v_template"] for t in tensor2array(jd[:,0])]
    v_template = np.array(v_template)
    #check_ccf_consistency(spectra,w,wave_obs,template)

    print("compute_ccf...")
    ccf_matrix, velocity_grid = compute_ccf(spectra, w, wave_obs, 1-template)
    params,v_ccf,bisspan = compute_bisspan(velocity_grid,ccf_matrix)
    print(f"v_ccf RMS:{v_ccf[:,0].std():.3f} m/s  chi:{v_ccf[:,1].mean():.2f}")
    #v_template = np.zeros_like(v_ccf[:,0])
    print(f"v_template RMS:{v_template.std():.3f} m/s ")
    print(f"Difference RMS:{(v_template-v_ccf[:,0]).std():.3f} m/s ")

    depth,_,sigma,c = params.T

    input_features = np.vstack((v_template,v_ccf[:,0],depth,bisspan,sigma,c)).T
    print("saving to %s..."%save_name)
    input_features = torch.from_numpy(input_features.astype(np.double))
    template_new = torch.from_numpy((template).astype(np.float32))[None,:]
    spec_input = spectra - template_new
    spec_input[w<1] = 0
    print("spec_input:",spec_input.shape)
    with open(save_name, 'wb') as f:
        pickle.dump([input_features,spec_input,w,ssbrv,jd], f)
    return


def add_ccf_trad_indicators(datatag):
    print("\n\nAdd CCF INFO...")
    params_file = f'{runtime_dir}{datatag}-param.pkl'
    print(f'Loading from {params_file}')
    with open(params_file,"rb") as f:
        neid_dict = pickle.load(f)
    file_batches = neid_dict["info"]["files"]
    template = neid_dict["info"]['baseline']

    print("keys:",neid_dict["info"].keys())
    wave_obs = neid_dict["info"]["wave_obs"][0]
    sample_names = neid_dict["info"]["sample_names"]

    # Create a time-to-parameters mapping
    timetable = {np.float32(neid_dict[item]["timestamp"]): neid_dict[item] for item in sample_names}

    #template = neid_dict["info"]['baseline']
    print("template:",template.shape)
    print("wave_obs:",wave_obs.shape)
    print("file_batches:",file_batches)

    v_template = get_timeseries(neid_dict,'v_template',sample_names)
    print_string("$v_{template}$",v_template,mode="2")

    process_list = []
    manager = mp.Manager()
    mdict = manager.dict()
    ccf_files = []
    for k,batch_name in enumerate(file_batches):
        print("Loading %s..."%batch_name)
        datatag = os.path.basename(batch_name).rsplit('.', 1)[0]
        #load_name = f"{dynamic_dir}/processed/{datatag}.pkl"
        load_name = f"{dynamic_dir}/merge/{datatag}.pkl"
        save_name = f"{dynamic_dir}/ccf_info/{datatag}.pkl"

        ccf_files.append(save_name)
        if os.path.isfile(save_name): 
            print(save_name,"file exists")
            continue
        print ("saving batch  %d / %d"%(k,len(file_batches))) 
        work_p = mp.Process(target=ccf_worker,
                            args=(load_name,save_name,wave_obs,template,timetable))
        process_list.append(work_p)

    neid_dict["info"]["ccf_files"] = ccf_files
    print(f"Saving to {params_file}...")
    with open(params_file,"wb") as f:
        pickle.dump(neid_dict,f)

    num_cores = 10
    print("process_list",process_list)
    for i_start in range(0, len(process_list), num_cores):
        print("[ccf]Currently running #%i - #%i"%(i_start, min(i_start+num_cores,len(process_list))))
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
parser.add_argument('-when', '--when', help='Before or after the fire', default='full')
parser.add_argument('-n', '--samples', type=int, help='Number of samples', default=100)
parser.add_argument('-batch', '--batch_size', type=int, help='Batch size', default=500)
parser.add_argument('-cpu', '--num_cores', type=int, help='Number of CPU cores', default=80)
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



#ORDERS = np.arange(65,100)
#ORDERS = np.arange(41,100)
ORDERS = np.arange(31,75)
ORDERS = [i for i in ORDERS if not i in blacklist]

#ORDERS = np.arange(31,33) # test
#ORDERS = np.arange(49,56) # safe
#ORDERS = np.arange(45,49)# safeblue
#ORDERS = np.arange(31,36)# extremblue

print("ORDERS:",ORDERS)
print(" ".join([str(i) for i in ORDERS[1::2]]))

n_order = len(ORDERS)
fsr_mask = load_master_fsr_mask()
quality_mask = ~fsr_mask

input_wave = [get_order_wavelengths(o) for o in ORDERS]
for i,order in enumerate(ORDERS):
    print("order %d: wave_obs: %d, quality_mask:%d"%(order,len(input_wave[i]),quality_mask[order].sum()))

file_path = "NEID_HIGH_QUALITY_OBSNAME.txt"
# Load the data from the text file
data = np.loadtxt(file_path, dtype={'names': ('filename', 'jd', 'ccfrv'), 'formats': ('S53', 'f8', 'f8')})

if args.when =="a":
    suffix = "after"
    goodmask = (data['jd']>2459920.0)
    #goodmask = (data['jd']>2459920.0) & (data['jd']<2459940.0)
elif args.when =="b":
    suffix = "before"
    goodmask = (data['jd']<2459800.0)
elif args.when =="full":
    suffix = "full"
    goodmask = (data['jd']<2460600.0)
else:
    print("invalid argument %s"%args.when)
    exit()

# Extract columns into separate arrays
NEID_JD = data['jd'][goodmask]
NEID_FILENAMES = np.array([os.path.basename(x.decode('utf-8')) for x in data['filename'][goodmask]])
NEID_CCFRV = data['ccfrv'][goodmask]

existing_files = os.listdir(datadir)
print("existing files:",len(existing_files))
print("qualified files:",len(NEID_FILENAMES))
print("NEID_FILENAMES:",NEID_FILENAMES[:5])


# Find the indices of available_names in full_names
sel = np.nonzero(np.in1d(NEID_FILENAMES, existing_files))[0]
print("total number:",len(sel))
np.random.shuffle(sel)
SELECT = sel[:n_sample]
raw_sample_names = list(NEID_FILENAMES[SELECT])
print("Selected samples:",len(raw_sample_names))

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#merge_multiple_orders(tag,ORDERS,suffix,raw_sample_names,batch_size)
multiple_order_v_template(tag,suffix)
#add_ccf_trad_indicators(f"{tag}_{suffix}")
#multiple_order_v_template_consistency(tag,suffix)

exit()
'''
fig, axs = plt.subplots(figsize=(15,2*n_order),nrows=n_order,dpi=200,constrained_layout=True)
for i,order in enumerate(ORDERS):
    preview_spectrum(axs[i],'neidL2_20210710T210857.fits',
                     order,quality_mask=quality_mask[order])
    plt.savefig("[%s]single-obs.png"%tag)
exit()
'''

#torch.cuda.set_device('cuda:1')
#device = torch.device('cuda')

for i,order in enumerate(ORDERS):    
    datatag = "%s_order%d_%s"%(tag,order,suffix)
    diag_tag = "test_order%d_%s"%(order,suffix)
    if not load_data:
        print("wrapping order:",order)
        wrap_data(raw_sample_names,datatag,diag_tag,batch_size,order,quality_mask[order])
        calculate_template_spectrum(datatag,input_wave[i])
        # skip v_template calculation
        continue
        calculate_v_template(datatag,input_wave[i])

    print("Loading from %s-param.pkl"%datatag)
    with open(f'{runtime_dir}/{datatag}-param.pkl',"rb") as f:
        neid_dict = pickle.load(f)
    sample_names = neid_dict["info"]["sample_names"]
    timestamp = get_timeseries(neid_dict,'timestamp',sample_names)
    jds = get_timeseries(neid_dict,'OBSJD',sample_names)
    #print("info",neid_dict["data"].keys())
    print("jds:",jds.max())
    
    SSBRV = get_timeseries(neid_dict,'SSBRV',sample_names)
    CCFRV = get_timeseries(neid_dict,'CCFRV',sample_names)
    water_vapor = get_timeseries(neid_dict,'WVAPOR',sample_names)
    print("SSBRV:",SSBRV.min(),SSBRV.max())

    CCFRV = CCFRV-np.mean(CCFRV)
    print("\nOrder %d:"%order)
    #print_string("base_chi",fit_chi)
    title_color = "k"

    if not "ccf_files" in neid_dict["info"]:
        file_batches = neid_dict["info"]["files"]
        print("file_batches:",file_batches)
        # load generated data
        batch = merge_batch(file_batches)
        wave_raw,spec_raw,weights,ssbrvs,ids = [item.numpy() for item in batch]
        
        print("wave_raw nan:",np.isnan(wave_raw).sum())
        print("spec_raw nan:",np.isnan(spec_raw).sum())
        print("weights nan:",np.isnan(weights).sum())
        exit()
        wave_mean = np.mean(wave_raw,axis=0,keepdims=True)
        wave_std = (wave_raw-wave_mean).std(axis=0)
        wave_mean = wave_mean[0]


        v_template = get_timeseries(neid_dict,'v_template',sample_names)
        fit_chi = get_timeseries(neid_dict,'chi_template',sample_names)

        print_string("$v_{CCF}$",CCFRV,mode="2")
        print_string("$v_{template}$",v_template,mode="2")
        
        #bulk_offset = np.median(v_template[timestamp<800])-np.median(v_template[timestamp>800])
        #print(f"bulk_offset: {bulk_offset:.2f}m/s")
        
        fig,axs = plt.subplots(nrows=2,ncols=4,figsize=(15,6),constrained_layout=True)
        for ax in axs[0, :]:fig.delaxes(ax)
        ax = fig.add_subplot(2, 1, 1) 
        cdata = fit_chi
        rank = np.argsort(cdata)[::-1]
        i_plots = rank[::(len(rank)//7)]
        cmap = get_cmap('plasma')
        cmin,cmax = min(cdata),max(cdata)
        colors =[cmap((ii-cmin)/(cmax-cmin)) for ii in cdata]

        for i_obs,obsname in enumerate(sample_names):
            if not i_obs in i_plots:continue
            ccfrv = CCFRV[i_obs]
            yoffset = 0#ccfrv
            date_obs = neid_dict[obsname]['DATE-OBS']
            date = date_obs[5:10]
            time = date_obs[11:16]

            err = weights[i_obs]**(-0.5)
            snr = spec_raw[i_obs]/err
            text = "%.2f $v_{CCF}$:%.2f m/s $\chi^2=%.2f$"%(neid_dict[obsname]['timestamp'],ccfrv,fit_chi[i_obs])
            print(text,obsname)
            
            ax.plot(wave_raw[i_obs],spec_raw[i_obs],drawstyle="steps-mid",alpha=1.0,lw=1,c=colors[i_obs],label=text)
            ax.fill_between(wave_raw[i_obs],spec_raw[i_obs]-err,spec_raw[i_obs]+err,color=colors[i_obs],alpha=0.2,step="mid")
            ax.set_xlabel("Raw wavelength ($\AA$)")
            ax.set_ylabel("normalized flux")
            ax.legend(loc="lower left",ncols=2)
            ax.set_title("Order %d"%(order),color=title_color)
        ax.set_ylim(0,spec_raw[i_obs].max())
        ax=axs[1][0]
        ax.scatter(NEID_JD,NEID_CCFRV,c="grey",s=5,label="all (N=%d)"%len(NEID_JD))
        img = ax.scatter(NEID_JD[SELECT],NEID_CCFRV[SELECT],s=5,label="selected (N=%d)"%len(SELECT))
        ax.legend(loc="upper left")
        ax.set_xlabel("JD")
        ax.set_ylabel("NEID Solar RV [km/s]")
        ax=axs[1][1]
        ax.scatter(timestamp,v_template,c='orange',s=3,
                   label="$v_{template}$ RMS = %.2f m/s"%(v_template.std()))
        ax.scatter(timestamp,CCFRV,c='grey',s=2,label="$v_{CCF,NEID}$ RMS = %.2f m/s"%(CCFRV.std()))

        ax.legend(title="Discrepancy RMS = %.2f m/s"%(v_template-CCFRV).std())
        ax.set_xlabel("JD")
        ax.set_ylabel("RV [m/s]")
        ax=axs[1][2]
        ax.scatter(timestamp,fit_chi,c='orange',s=3)
        ax.set_xlabel("JD")
        ax.set_ylabel("$\chi^2_r$")

        ax.scatter(timestamp,1.5*water_vapor/water_vapor.max(),
                   label="Water Vapor",
                   c='b',s=3)
        ax.set_xlabel("JD")
        #ax.set_ylabel("Water Vapor")
        ax=axs[1][3]
        for i in range(100):
            ax.plot(wave_mean,wave_raw[i]-wave_mean,c='k',lw=1,alpha=0.5)
        ax.plot(wave_mean,wave_std,c='r',lw=2)
        ax.set_xlabel("wavelength")
        ax.set_ylabel("wavelength shifts")
        
        plt.tight_layout()
        plt.savefig(f'./[{datatag}]diagnostic.png',dpi=300)
        plt.clf()
        continue

    v_template = get_timeseries(neid_dict,'v_template_processed',sample_names)
    fit_chi = get_timeseries(neid_dict,'chi_template_processed',sample_names)

    file_batches = neid_dict["info"]["ccf_files"]
    print("file_batches:",file_batches)
    # load generated data
    batch = merge_batch(file_batches)
    ccf_info,spec_raw,weights,ssbrvs,ids = [item.numpy() for item in batch]
    #spec_raw[weights<1]=0
    our_ccf = ccf_info[:,1] - np.mean(ccf_info[:,1])

    fig,axs = plt.subplots(ncols=4,nrows=2,figsize=(15,6),constrained_layout=True)
    plot_sample_selection(axs[1,:], timestamp,v_template,CCFRV,our_ccf,fit_chi,order)
    axs[1,3].plot(timestamp,water_vapor,".",c="skyblue",label="Water Vapor",zorder=-10)
    axs[1,3].set_xlabel("JD");axs[1,3].set_ylabel("Water")

    baseline = neid_dict["info"]["processed_baseline"]
    baseline_w = neid_dict["info"]["baseline_w"]
    wave_obs = input_wave[i]

    print("\n\ngood spectra: %d/%d"%(len(sample_names),len(sel)))
    sn = baseline/baseline_w**(-0.5)
    print("sn:",sn.min(),sn.max(),"mean sn:",sn.mean())
    good = sn>1
    RV_limit = photon_noise(baseline[good],wave_obs[good],sn[good])
    print("Order %d RV_limit: %.2f m/s \n"%(order,RV_limit))


    print_string("$v_{CCF}$",CCFRV,mode="2")
    print_string("Preview: our $v_{CCF}$",our_ccf,mode="2")
    print("\n\n")
    #wave_raw,spec_raw,weights,ssbrvs,ids = [item.numpy() for item in batch]

    ratio = 20
    rank = np.argsort(water_vapor)[::-1]
    i_plots = rank[::2000]
    i_plots = i_plots[:5]

    cmap = get_cmap('plasma')
    cmin,cmax = min(water_vapor),max(water_vapor)
    colors =[cmap((ii-cmin)/(cmax-cmin)) for ii in water_vapor]

    for ax in axs[0, :]:fig.delaxes(ax)
    ax = fig.add_subplot(2, 1, 1)  # Add a new subplot spanning the second row
    for i_obs,obsname in enumerate(sample_names):
        if not i_obs in i_plots:continue
        ccfrv = CCFRV[i_obs]
        yoffset = 0#ccfrv
        date_obs = neid_dict[obsname]['DATE-OBS']
        date = date_obs[5:10]
        time = date_obs[11:16]

        snr = spec_raw[i_obs]/(weights[i_obs]**(-0.5))

        text = "%.2f $v_{CCF}$:%.2f m/s $\chi^2=%.2f$"%(neid_dict[obsname]['timestamp'],ccfrv,fit_chi[i_obs])
        print(text,obsname)
        spec_inflate = ratio*spec_raw[i_obs]+baseline
        spec_inflate[(weights[i_obs]<1)] = 0
        spec_inflate[baseline==0] = 0
        ax.plot(wave_obs,spec_inflate,drawstyle="steps-mid",alpha=1.0,lw=1,
                c=colors[i_obs],label=text)
        ax.set_xlabel("Raw wavelength ($\AA$)")
        ax.set_ylabel("normalized flux")
        ax.legend(loc="lower left",ncols=2)
        ax.set_title("Order %d %s the fire"%(order,suffix),color=title_color)
    plt.tight_layout()
    plt.savefig(f'./[{datatag}]diagnostic.png',dpi=300)
    plt.clf()
    
print("blacklist",blacklist)



