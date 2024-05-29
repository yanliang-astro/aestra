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
from util import moving_median,plot_fft,mem_report,load_batch,merge_batch
from scipy.optimize import curve_fit

dynamic_dir = "/scratch/gpfs/yanliang/neid-dynamic"
datadir = "/scratch/gpfs/yanliang/NEID-SOLAR"
telluric_dir = "/scratch/gpfs/yanliang/NEID-TELLURIC"
device =  torch.device("cpu")

blacklist = ["neidL2_20211216T223047.fits"]
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

def read_multiple_order(filename,order_value,read_keys=['OBSJD','DATE-OBS']):
    hdulist = fits.open(filename)
    header = hdulist[0].header
    ccf_header = hdulist[12].header
    telluric_header = hdulist[10].header
    
    science_wavelength = hdulist[7].data
    science_flux = hdulist[1].data
    science_variance = hdulist[4].data
    science_blaze = hdulist[15].data

    # Close the FITS file
    hdulist.close()
    
    science = [science_wavelength,science_flux,science_variance]
    data = []
    for o in order_value:
        science_order = [item[o] for item in science]
        data.append([science_order,science_blaze[o]])

    SSBRV= get_barycentric_corr_rv(header)
    CCFRV = read_ccf_rv(ccf_header)
    info_dict = {key:header[key] for key in read_keys}
    info_dict.update({key:telluric_header[key] for key in ["ZENITH","WVAPOR"]})
    km_m = 1e3
    for o in order_value:
        info_dict[o] = {"SSBRV":(SSBRV[o]+0.8)*km_m,"CCFRV":CCFRV[o]*km_m}
    info_dict["CCFRVMOD"] = ccf_header["CCFRVMOD"]*km_m
    # time zero point
    info_dict["timestamp"] = np.float32(info_dict["OBSJD"] - 2459350.0) 
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

def calculate_flux_uncertainty(wave_obs, lines_to_mask, width=0.15):
    # Initialize the uncertainty array with zeros
    uncertainty = np.zeros_like(wave_obs)
    # Gaussian width factor (converting FWHM to standard deviation)
    sigma = width / 2.355

    # Loop through the lines to calculate uncertainty
    for line in lines_to_mask:
        line_center = line[0]
        line_depth = max(0.01,0.5*line[1])
        # Calculate Gaussian profile for this line
        gaussian = line_depth*np.exp(-0.5 * ((wave_obs - line_center) / sigma) ** 2)
        # Add this profile to the uncertainty array
        uncertainty = np.maximum(uncertainty, gaussian)
    return uncertainty


def prepare_spectrum(obsname,flag=False):
    known_bad_pixels = {50:[280,297],51:[280,297],52:[280,297]}
    large_number = 1e6
    data,info_dict = read_multiple_order("%s/%s"%(datadir,obsname),order_value=order_value)

    wavelength = np.zeros((len(order_value),N_SPEC))
    spectrum = np.zeros((len(order_value),N_SPEC))
    spectrum_err = np.zeros((len(order_value),N_SPEC))

    for k,o in enumerate(order_value):
        wave_obs = input_wave[k]
        science,blaze = data[k]
        wave_raw,flux,flux_var = science
        #known bad pixels
        if o in known_bad_pixels:
            for i_bad in known_bad_pixels[o]:
                flux_var[i_bad] = large_number**2
        ssbrv = info_dict[o]["SSBRV"]
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

        wavelength[k] = wave_raw
        spectrum[k][~isnan] = normflux[~isnan]
        spectrum_err[k][~isnan] = normflux_err[~isnan]
        spectrum_err[k][isnan] = large_number
    badmask = detect_bad_pixel([wavelength,spectrum,spectrum_err])
    spectrum_err[badmask] = large_number
    return [wavelength,spectrum,spectrum_err],info_dict

def save_batch(wave,specs,w,ssbrv,IDs,filename):
    wave = torch.from_numpy(wave.astype(np.double))
    spec = torch.from_numpy(specs.astype(np.float32))
    weight = torch.from_numpy(w.astype(np.float32))
    ssbrv = torch.from_numpy(ssbrv.astype(np.double))
    ID = torch.from_numpy(IDs.astype(np.float32))

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


def make_batch(sample_names):
    large_number = 1e6
    batch_size = len(sample_names)
    wavemat =  np.zeros((batch_size,n_order,N_SPEC))
    specmat = np.zeros((batch_size,n_order,N_SPEC))
    errmat = np.zeros((batch_size,n_order,N_SPEC))
    good =  np.ones((batch_size),dtype=bool)
    neid_dict = {}
    for i_obs,obsname in enumerate(sample_names):
        data,info_dict = prepare_spectrum(obsname)
        wavelength,spectrum,spectrum_err = data
        # negative flux?
        neg = np.sum(spectrum<0.0,axis=-1)
        if neg.sum()>100:
            good[i_obs] = False
            print("negative!!",obsname,neg)
            print("flux: %.2f, %.2f"%(spectrum.min(),spectrum.max()))
        #if spectrum.min()<0.01:good[i_obs] = False
        if not good[i_obs]: continue
        neid_dict[obsname] = info_dict
        wavemat[i_obs,:,:] = wavelength
        specmat[i_obs,:,:] = spectrum
        errmat[i_obs,:,:] = spectrum_err

    bad = errmat**(-2)<1.0
    print("bad pixels:",(bad.sum()/batch_size))
    print("good:",good.sum())
    specmat[bad] = 0.0
    wavemat=wavemat[good]
    specmat=specmat[good]
    errmat=errmat[good]
    return sample_names[good],wavemat,specmat,errmat,neid_dict

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
    if colname in neid_dict[keys[0]]:
        vector = [neid_dict[key][colname] for key in keys]
        return np.array(vector)
    vector = []
    print("neid_dict[key]:",neid_dict[keys[0]])
    for order in order_value:
        vector.append([neid_dict[key][order][colname] for key in keys])
    vector = np.array(vector)
    print(colname,vector.shape)
    return vector


def velocity_label(velocity,label):
    quantiles=[0.16,0.50,0.84]
    q1,q2,q3 = np.quantile(velocity,quantiles)
    val = "${%.2f}^{+%.2f}_{-%.2f}$"%(q2,q3-q2,q2-q1)
    vlabel = "%s = %s [m/s]"%(label,val)
    return vlabel

def make_batch_worker(batch_id, batch_name, neid_dict):
    batch_id,wavemat,specmat,errmat,sub_dict = make_batch(batch_id)
    ssbrvs = get_timeseries(sub_dict,'SSBRV',batch_id).T
    timestamp = get_timeseries(sub_dict,'timestamp',batch_id)
    save_batch(wavemat,specmat,errmat**(-2),ssbrvs,timestamp,batch_name)
    print("good spectra: %d"%len(batch_id))
    neid_dict.update(sub_dict)
    return 0

def fit_rv_worker(wave,spec,w,wave_baseline,baseline,baseline_w,mdict,obsname,order):
    v_template,base_chi,message = fit_rv(wave,spec,w,wave_baseline,baseline,baseline_w)
    summary = {"v_template":v_template[0],"chi_template":base_chi}
    mdict["%s-%d"%(obsname,order)]=summary
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

def wrap_data(sample_names,datatag,batch_size):
    idx = np.arange(0, len(sample_names), batch_size)
    batches = np.array_split(sample_names, idx[1:])
    file_batches = ["%s/%s_%d.pkl"%(dynamic_dir,datatag,k) for k in range(len(batches))]

    general_info = {"sample_names":sample_names,
                    "files":file_batches,
                    "orders":order_value}

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
                            args=(batch_id, batch_name, mdict))
        process_list.append(work_p)

    for i_start in range(0, len(process_list), num_cores):
        print("Currently running #%i - #%i"%(i_start, i_start+num_cores))
        running_list = process_list[i_start:i_start+num_cores]
        [p.start() for p in running_list]
        [p.join()  for p in running_list]

    neid_dict = {k:v for k,v in mdict.items()}
    batch = merge_batch(file_batches)
    waves,specs,weights,ssbrvs,ids = [item.numpy() for item in batch]
    n_order,n_pix = input_wave.shape
    # interpolate to homogeneous grid - calculate the template spectrum
    print("native grid",waves.shape,"input grid",input_wave.shape)
    n_template = min(100,len(specs))
    
    input_flux = np.zeros((n_template,n_order,n_pix))
    input_weight = np.zeros((n_template,n_order,n_pix))

    for o in range(n_order):
        wave_obs = input_wave[o]
        for i in range(n_template):
            wave_raw = waves[i][o]
            ssbrv = ssbrvs[i][o]
            flux = specs[i][o]
            weight = weights[i][o]
            wave = wave_raw + wave_raw*(ssbrv)/Synthetic.c
            good = (weights[i][o]>1.0)
            inbound = (wave_obs>min(wave[good]))&(wave_obs<max(wave[good]))

            input_flux[i][o][inbound] = interp1d(wave[good], flux[good], kind='linear')(wave_obs[inbound])
            input_weight[i][o][inbound] = interp1d(wave, weight, kind='linear')(wave_obs[inbound])
            input_weight[i][o][~inbound] = 1e-12

            bad = input_weight[i][o]<1.0
            input_flux[i][o][bad] = 0.0

    template = np.median(input_flux,axis=0)
    template_w = np.median(input_weight,axis=0)

    #'''
    dispersion = np.std(input_flux,axis=0,where=(input_flux>0.0))
    dispersion[np.isnan(dispersion)] = 0.0

    fig,axs=plt.subplots(figsize=(16,8),nrows=n_order,constrained_layout=True)
    for o in range(n_order):
        wave_obs = input_wave[o]
        whmax = np.argmax(dispersion[o])
        err = template_w[o]**(-0.5)
        ax=axs[o]
        ax.plot(wave_obs,template[o],"k-",lw=1,drawstyle="steps-mid")
        ax.fill_between(wave_obs,template[o]-err,template[o]+err,
                        color="k",alpha=0.3)
        ax.plot(wave_obs,dispersion[o],"r-",lw=1,drawstyle="steps-mid")
        ax.set_xlim(wave_obs[300]-2,wave_obs[300]+2)
        ax.set_ylim(-0.05,1.1)
    plt.savefig("[%s]template.png"%tag,dpi=200)

    #'''
    template_name="%s/%s.pkl"%(dynamic_dir,datatag+"-template")
    save_batch(input_wave[None,:,:],
               template[None,:,:],template_w[None,:,:],
               np.array([0]),np.array([888]),template_name)
    general_info.update({"baseline":template,"baseline_w":template_w})
    # remove poor-quality spectra
    sample_names = [i for i in sample_names if i in neid_dict]
    print("sample_names:",len(sample_names))

    general_info["sample_names"] = sample_names
    neid_dict.update({"info":general_info})
    with open("%s-param.pkl"%datatag,"wb") as f:
        pickle.dump(neid_dict,f)
    return

def calculate_v_template(sample_names,datatag):
    print("Loading from %s-param.pkl"%datatag)
    with open("%s-param.pkl"%datatag,"rb") as f:
        neid_dict = pickle.load(f)

    # calculate v_template and chi_template
    file_batches = neid_dict["info"]["files"]
    template = neid_dict["info"]["baseline"]
    template_w = neid_dict["info"]["baseline_w"]
    order_value = neid_dict["info"]["orders"]

    batch = merge_batch(file_batches)
    waves,specs,weights,ssbrvs,ids = [item.numpy() for item in batch]
    n_order,n_pix = input_wave.shape

    print("ssbrvs [m/s]:",ssbrvs.min(),ssbrvs.max())
    #process_list = []
    manager = mp.Manager()
    mdict = manager.dict()
    # Use a pool of workers
    pool_size = num_cores  # Number of processes in the pool
    pool = mp.Pool(pool_size)
    tasks = []
    for i_epoch,obsname in enumerate(sample_names):
        for i_order,order in enumerate(order_value):
            # skip existing entries
            if 'v_template' in neid_dict[obsname][order]:continue
            z_ssb = ssbrvs[i_epoch][i_order]/Synthetic.c
            # shift to stellar restframe
            wave_stellar = waves[i_epoch][i_order]*(1.0+z_ssb)
            task_args = (wave_stellar, specs[i_epoch][i_order], weights[i_epoch][i_order], input_wave[i_order],template[i_order],template_w[i_order], obsname, order)
            tasks.append(task_args)

    for i,task in enumerate(tasks):
        pool.apply_async(process_task, args=(task, mdict))
    # Close and join the pool
    pool.close()
    pool.join()

    for k in sample_names:
        for o in order_value:
            key = "%s-%d"%(k,o)
            if key in mdict:neid_dict[k][o].update(mdict[key])
            else:print("%s missing..."%key)

    print("Saving to %s-param.pkl..."%datatag)
    with open("%s-param.pkl"%datatag,"wb") as f:
        pickle.dump(neid_dict,f)
    return 

def get_wavelengths(poly,wave_min,wave_max,n_pix=9216):
    input_pix = np.arange(n_pix+30)
    input_grid = np.zeros((n_pix+30))
    for j in range(len(input_grid)):
        if j==0:input_grid[j] = wave_min;continue
        local_bin = np.polyval(poly,input_grid[j-1])
        input_grid[j] = input_grid[j-1]+local_bin
    if input_grid[-1]<wave_max:
        print("input_grid too short!!",input_grid[-1],wave_max)
        exit()
    return input_grid

def get_order_wavelengths(target_order):
    config_data = np.loadtxt("orders.config").T
    wave_min = config_data[1]
    wave_max = config_data[2]
    wave_poly = config_data[3:6].T
    o = target_order
    wave_obs = get_wavelengths(wave_poly[o],wave_min[o],wave_max[o])
    return wave_obs

def tensor2array(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else: return tensor.detach().numpy()

def load_telluric_model(obsname,reftag="quiet_noplanet_N5000"):
    base = obsname.split(".")[0]
    fname = "%s/%s_%s_telluric.txt"%(telluric_dir,reftag,base)
    if os.path.isfile(fname):
        return True,np.loadtxt(fname)
    print("%s does not exist!"%fname)
    return False,None

def detect_bad_pixel(data,sigma=1.5,snr=3,radius=2):
    wavelength,spectrum,spectrum_err = data
    spec_smooth = gaussian_filter1d(spectrum, sigma)
    ydiff = np.abs(spectrum-spec_smooth)
    badmask = np.zeros(spectrum.shape,dtype=bool)
    n_spec = spectrum.shape[-1]
    for o in range(len(ydiff)):
        candidates = find_deepest_lines(wavelength[o], 1-ydiff[o], num_lines=10,min_separation=0.1, return_ind=True)
        known_bad = np.where(spectrum_err[o]>1)[0]
        valid = []
        for ind,amp in candidates:
            sn = amp/spectrum_err[o,ind]
            if sn<snr:continue
            if ind in known_bad:continue
            # check if coincide with a stellar line
            nearby = spectrum[o,max(0,ind-radius):min(ind+radius,n_spec-1)]
            sorted_ind = np.argsort(nearby)
            fluxes = nearby[sorted_ind]
            if (fluxes[0]<0.8)&((fluxes[1]-fluxes[0])<0.1):
                continue
            valid.extend([ind-1,ind,ind+1])
        valid = [item for item in valid if item>=0 and item<n_spec]
        badmask[o,valid] = True
    #print("badmask:",badmask.sum())
    return badmask

def preview_spectrum(obsname):
    data,info_dict = prepare_spectrum(obsname)
    wavelength,spectrum,spectrum_err = data
    spec_smooth = gaussian_filter1d(spectrum, 1.5)

    ydiff = np.abs(spectrum-spec_smooth)/spectrum_err
    ydiff /= ydiff.max()

    nrows=len(order_value)
    ylim=[0,1.2]
    fig, axs = plt.subplots(figsize=(15,nrows*2.5),nrows=nrows,dpi=200,constrained_layout=True)
    for i,ax in enumerate(axs):

        ax.set_title("Order %d"%order_value[i])
        ax.plot(wavelength[i],spectrum[i],"k-",lw=1,label="data",drawstyle="steps-mid")
        ax.plot(wavelength[i],spec_smooth[i],"b-",lw=1,label="smooth",drawstyle="steps-mid")
        ax.plot(wavelength[i],ydiff[i],"r-",lw=1,drawstyle="steps-mid")
        ax.fill_between(wavelength[i],spectrum[i]-spectrum_err[i],
                        spectrum[i]+spectrum_err[i],step="mid",
                        color="k",alpha=0.3,zorder=-10)
        ax.legend()
        ax.set_ylim(ylim)
        wh = 5500#np.argmax(ydiff[i])
        ax.set_xlim(wavelength[i][wh]-3,wavelength[i][wh]+3)
    plt.savefig("[%s]single-obs.png"%datatag)
    return

def initialize_restframe_model(input_wave,template,weight):
    wave_rest = input_wave
    spec_rest = np.ones_like(template)
    spec_rest[weight>1.0]=template[weight>1.0]
    n_expand = input_wave.shape[0]*input_wave.shape[1]
    x = input_wave.reshape((n_expand))
    y = template.reshape((n_expand))
    bad = (weight<1.0).reshape((n_expand))
    bound = (input_wave>x[~bad].min()) &(input_wave<x[~bad].max())
    fillmask = (weight<1.0)&bound
    f = interp1d(x[~bad],y[~bad],kind = "nearest")
    for o in range(input_wave.shape[0]):
        mask = fillmask[o]
        spec_rest[o][mask] = f(input_wave[o][mask])
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

np.random.seed(0)
torch.manual_seed(0)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your script')

# Define optional arguments with default values
parser.add_argument('-t', '--tag', help='Tag description', default='test')
parser.add_argument('-n', '--samples', type=int, help='Number of samples', default=100)
parser.add_argument('-batch', '--batch_size', type=int, help='Batch size', default=500)
parser.add_argument('-cpu', '--num_cores', type=int, help='Number of CPU cores', default=10)
parser.add_argument('-load', '--load_data', action='store_true', help='Load data')
parser.add_argument('-o','--orders', nargs='+', help='<Required> Orders', required=True)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
tag = args.tag
n_sample = args.samples
batch_size = args.batch_size
num_cores = args.num_cores
load_data = args.load_data

N_SPEC = 9216

order_value = [int(o) for o in args.orders]

input_wave = [get_order_wavelengths(o) for o in order_value]
input_wave = np.array(input_wave)

n_order = len(order_value)
print("input_wave:",input_wave.shape)

datatag = "%s_N%d"%(tag,n_sample)


# reading the CSV file
csvfilename = 'NEID_2021B.csv'
csvFile = pandas.read_csv("%s/%s"%(datadir,csvfilename))
neid_filenames = np.array(csvFile.filename)
neid_jd = np.array(csvFile.ccfjdsum)
neid_ccfrv = csvFile.ccfrvmod
neid_snr =  np.array(csvFile.extsnr)
quality_flag = (np.array(csvFile.flaggedval,dtype=str)=='x')

excluded = neid_jd<2e6
excluded |= ((neid_jd>2459495)&(neid_jd<2459515))
excluded |= quality_flag
existing_files = os.listdir(datadir)
'''
import time
for obs in existing_files:
    path = '%s/%s'%(datadir,obs)
    file_size = os.path.getsize(path)
    if not obs in neid_filenames:continue
    where = np.where(neid_filenames==obs)[0][0]
    if file_size<86155200 and neid_jd[where]==-1.0:
        ti_m = os.path.getmtime(path)
        ti_m = time.ctime(ti_m)
        #os.system("rm %s"%path)
        print(path,file_size,ti_m,"CCFRV:",neid_ccfrv[where])
'''
available = np.array([name in existing_files for name in neid_filenames])

excluded |= ~available

snr_cut = 400
excluded |= neid_snr < snr_cut
sel = np.arange(len(neid_filenames))[(~excluded)]

print("total number:",len(sel))
np.random.shuffle(sel)
sel = sel[:n_sample]
sample_names = list(neid_filenames[sel])
print("order:",order_value)
print("sample_names:",len(sample_names))

idx = np.arange(0, len(sample_names), batch_size)
batches = np.array_split(sample_names, idx[1:])

file_batches = ["%s/%s_%d.pkl"%(dynamic_dir,datatag,k) for k in range(len(batches))]
print("file_batches:",file_batches)

#preview_spectrum("neidL2_20211109T204628.fits")
#preview_spectrum("neidL2_20220416T185324.fits")
#exit()
n_order = input_wave.shape[0]

if not load_data:
    save_auxfile(input_wave,"%s/%s-wavelength.pkl"%(dynamic_dir,datatag))
    wrap_data(sample_names,datatag,batch_size)
    calculate_v_template(sample_names,datatag)

#with open("skymask.pkl","rb") as f:
#    skymask = pickle.load(f)
#    save_auxfile(skymask,"%s/%s-skymask.pkl"%(dynamic_dir,datatag))

print("Loading from %s-param.pkl"%datatag)
with open("%s-param.pkl"%datatag,"rb") as f:
    neid_dict = pickle.load(f)
print("neid_dict:",len(neid_dict))

#calculate_v_template(sample_names,datatag)
sample_names = neid_dict["info"]["sample_names"]
print("good spectra: %d/%d"%(len(sample_names),len(sel)))

baseline = neid_dict["info"]["baseline"]
baseline_w = neid_dict["info"]["baseline_w"]

init_rest = initialize_restframe_model(input_wave,baseline,baseline_w)
save_auxfile(init_rest,"%s/%s-rest.pkl"%(dynamic_dir,datatag))


nrows=len(order_value)
'''
fig, axs = plt.subplots(figsize=(12,nrows*2.5),nrows=nrows,dpi=200,constrained_layout=True)
for i,ax in enumerate(axs):
    ax.set_title("Order %d"%order_value[i])
    ax.plot(input_wave[i],baseline[i],"k-",label="mean")
    ax.legend()
plt.savefig("[%s]single-obs.png"%datatag)
'''
for i_order,o in enumerate(order_value):
    sn = baseline[i_order]/baseline_w[i_order]**(-0.5)
    print("sn:",sn.min(),sn.max(),"mean sn:",sn.mean())
    good = sn>1
    RV_limit = photon_noise(baseline[i_order][good],
                            input_wave[i_order][good],
                            sn[good])
    print("Order %d RV_limit: %.2f m/s"%(o,RV_limit))

timestamp = get_timeseries(neid_dict,'timestamp',sample_names)
jds = get_timeseries(neid_dict,'OBSJD',sample_names)
v_template_order = get_timeseries(neid_dict,'v_template',sample_names)
base_chi_order = get_timeseries(neid_dict,'chi_template',sample_names)
ssbrvs_order = get_timeseries(neid_dict,'SSBRV',sample_names)
ccfrvs_order = get_timeseries(neid_dict,'CCFRV',sample_names)

ccf_norm = (ccfrvs_order-np.median(ccfrvs_order,axis=-1,keepdims=True))
template_ccf_offset = v_template_order-ccf_norm

def print_string(vname,v,mode="1"):
    if mode=="1":
        quantiles = [v.min(),v.max(),v.mean()]
        string = " ".join(["%.2f"%item for item in quantiles])
    if mode=="2":
        string =  "%.3f +/- %.3f m/s"%(v.mean(),v.std())
    print("%s: %s"%(vname,string))
    return

for i in range(n_order):
    print("\nOrder %d:"%order_value[i])
    print_string("base_chi",base_chi_order[i])
    print_string("$v_{CCF}$",ccf_norm[i],mode="2")
    print_string("$v_{template}$",(v_template_order[i]),mode="2")
    print_string("$v_{template}-v_{CCF}$",
                 template_ccf_offset[i],mode="2")

v_template = v_template_order
v_template -= np.median(v_template,axis=-1,keepdims=True)

'''
fig,ax=plt.subplots(figsize=(8,3),constrained_layout=True)
for i in range(n_order):
    label_template = velocity_label(v_template[i],"$v_{template}$")
    ax.plot(timestamp,v_template[i],".",ms=2,color="lightgrey",zorder=-20)
    xgrid,ygrid,delta_y = moving_median(timestamp,v_template[i],n=50)
    ax.errorbar(xgrid,ygrid,yerr=delta_y,fmt=".",capsize=3,ms=2,
                label="order %d RMS=%.2f m/s"%(order_value[i],v_template[i].std()))

ax.set_xlabel("Time [days]");ax.set_ylabel("$v_{template}$ [m/s]")
ax.legend(ncols=2)
plt.savefig("[%s]v_template.png"%datatag,dpi=300)
'''
skymask = load_batch("%s/%s-skymask.pkl"%(dynamic_dir,datatag)).bool()

# load generated data
batch = merge_batch(file_batches)
wave_raw,spec_raw,weights,ssbrvs,ids = [item.numpy() for item in batch]

spec_raw[:,skymask] = 0
weights[:,skymask] = 1e-6

n_epoch,n_order,N_SPEC = spec_raw.shape
wave_mean = np.median(wave_raw,axis=0)
wave_std = np.std(wave_raw,axis=0)

print("Calculate model weight!")
#save_auxfile(input_wave,"%s/%s-wavelength.pkl"%(dynamic_dir,datatag))
model_weight = np.zeros_like(input_wave)
print(wave_raw.shape,input_wave.shape,ssbrvs.shape)
for o in range(n_order):
    wave_obs = input_wave[o]
    for i in range(n_epoch):
        z = ssbrvs[i][o]/Synthetic.c
        wave_shifted = wave_raw[i][o]*z+wave_raw[i][o]
        # calculated interpolated weight
        func = interp1d(wave_shifted,weights[i][o],kind="nearest")
        mask = (wave_obs>wave_shifted.min())&(wave_obs<wave_shifted.max())
        weight_interp = np.zeros_like(wave_obs)
        weight_interp[mask] = func(wave_obs[mask])
        covered = (weight_interp>1.0)
        model_weight[o][covered] = 1
save_auxfile(model_weight,"%s/%s-weight.pkl"%(dynamic_dir,datatag))
for o in range(n_order):
    plt.plot(input_wave[o],model_weight[o],"-",label="order %d"%o)
plt.savefig("test.png",dpi=200)
zero_w = (model_weight<1).sum()
print("zero weight bins: %d (%.4f)"%(zero_w,zero_w/(n_order*N_SPEC)))
exit()
'''
fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
o = 5
mask = (wave_mean[o]>5168.75) & (wave_mean[o]<5169)
chunk = spec_raw[:,o,mask]
print("outlier:",np.where(chunk==chunk.max()))
for i in range(n_epoch):
    ax.plot(wave_raw[i][o][mask],chunk[i],"k-",alpha=0.1,
            drawstyle="steps-mid")
ax.set_xlabel("wavelength")
plt.savefig("test.png",dpi=300)
exit()
'''

fig,ax=plt.subplots(figsize=(8,3),constrained_layout=True)
for o in range(n_order):
    for i in range(100):
        ax.plot(wave_mean[o],np.abs(wave_raw[i][o]-wave_mean[o]),"k-",alpha=0.1,zorder=-20)
    if wave_std[o].max()>0.01:c="r"
    else: c="b"
    ax.plot(wave_mean[o],wave_std[o],label="order %d"%order_value[o],c=c)
ax.legend(ncols=2,loc="upper left")
ax.set_xlabel("wavelength")
ax.set_ylabel("wavelength dispersion")
plt.savefig("[%s]wavelength.png"%datatag,dpi=300)


n_cut = 10
cut_params = ssbrvs.mean(axis=1)
cuts = np.linspace(cut_params.min(),cut_params.max(),n_cut)
print("cuts:",cuts)
dispersion = np.zeros((n_cut-1,n_order,N_SPEC))
for i in range(n_cut-1):
    who = (cut_params>cuts[i])&(cut_params<cuts[i+1])
    dispersion[i] = np.std(spec_raw[who],axis=0)
print("dispersion:",dispersion.shape)
dispersion = np.median(dispersion,axis=0)
dispersion[np.isnan(dispersion)] = 0
rank = np.argsort(base_chi_order[5])[::-1]
#i_plots = [0,1,2,3,4,5]#
#rank = np.argsort(ssbrvs.mean(axis=0))
i_plots = rank[:50]

baseline = np.median(spec_raw,axis=0)
baseline_w = np.median(weights,axis=0)

for o in range(n_order):
    max_dispersion = np.max(dispersion[o])
    whmax = np.argmax(dispersion[o])
    print("order",o,"max_dispersion:",
          max_dispersion,"where:",whmax)

cmap = get_cmap('plasma_r')
tmin,tmax = min(timestamp[i_plots]),max(timestamp[i_plots])
colors =[cmap((t-tmin)/(tmax-tmin)) for t in timestamp[i_plots]]

mask = np.arange(7800,8000)
#mask = np.arange(0,N_SPEC)
ncols = 3
nrows = n_order//ncols
fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,4*nrows),constrained_layout=True)

for i_order,o in enumerate(order_value):
    base_chi = base_chi_order[i_order]
    i_row,i_col = i_order//ncols,i_order%ncols
    i_image = 0
    ax = axs[i_row,i_col]
    for i_obs,obsname in enumerate(sample_names):
        if not i_obs in i_plots:continue
        ccfrv = ccf_norm[i_order][i_obs]
        yoffset = 0#ccfrv
        date_obs = neid_dict[obsname]['DATE-OBS']
        date = date_obs[5:10]
        time = date_obs[11:16]
        
        snr = spec_raw[i_obs][i_order]/(weights[i_obs][i_order]**(-0.5))

        text = "%.2f $v_{CCF}$:%.2f m/s $\chi^2=%.2f$"%(neid_dict[obsname]['timestamp'],ccfrv,base_chi[i_obs])
        print(text,obsname)
        ax.plot(wave_raw[i_obs][i_order][mask], spec_raw[i_obs][i_order][mask],
                drawstyle="steps-mid",alpha=1,
                c=colors[i_image])#,label=text)

        i_image += 1

    ax.plot(wave_mean[i_order][mask], baseline[i_order][mask],drawstyle="steps-mid",lw=1,c="k",label="mean")
    ax.plot(wave_mean[i_order][mask], dispersion[i_order][mask],drawstyle="steps-mid",lw=1,c="r",label="dispersion")
    ax.plot(wave_mean[i_order][mask], wave_std[i_order][mask],drawstyle="steps-mid",lw=1,c="cyan",label="wave dispersion")

    #wh = np.argmax(dispersion[i_order])
    #ax.set_xlim(wave_obs[wh]-2,wave_obs[wh]+2)
    #ax.set_xlim(wave_mean[i_order][0],wave_mean[i_order][0]+5)
    ax.set_ylabel("normalized flux")
    ax.legend(loc="lower left")
    ax.set_title("Order %d"%o)
plt.savefig("[%s]residual-spectrum.png"%(tag),dpi=300)

grey_colors = ["grey"]*10
bright_colors = ["k"]*10

fig,axs = plt.subplots(nrows=3,figsize=(8,10),constrained_layout=True)

ax=axs[0]
ax.scatter(neid_jd[neid_jd>2e6],neid_ccfrv[neid_jd>2e6],c="grey",s=5,label="all (N=%d)"%len(neid_jd))
img = ax.scatter(neid_jd[sel],neid_ccfrv[sel],c=neid_snr[sel],cmap="inferno",s=5,label="selected (N=%d)"%len(sample_names))
ax.set_ylim(-1.2,-0.2)
cbar = plt.colorbar(img)
cbar.set_label("S/N")
ax.legend(loc="upper left")
ax.set_xlabel("JD")
ax.set_ylabel("NEID Solar RV [km/s]")

ax_in = ax.inset_axes([0.58, 0.1, 0.4, 0.3])
ax_in.hist(neid_snr,color="grey",log=True)
ax_in.axvline(snr_cut,ls="--",color="k")
ax_in.set_title("S/N")

ax=axs[1]
for i,o in enumerate(order_value):
    ax.scatter(timestamp,v_template[i],c=bright_colors[i],s=5,
               label="%d $v_{template}$ vs. $v_{CCF}$ RMS = %.2f m/s vs. %.2f m/s"%(o,v_template[i].std(),ccf_norm[i].std()))
    ax.scatter(timestamp,ccf_norm[i],c=grey_colors[i],s=5)

ax.legend(title="Discrepancy RMS = %.2f m/s"%template_ccf_offset.std())
ax.set_xlabel("JD")
ax.set_ylabel("RV [m/s]")

ax=axs[2]
for i,o in enumerate(order_value):
    ax.scatter(timestamp,base_chi_order[i],c=bright_colors[i],s=5,
               label="%d $\chi^2_{template}$, RMS = %.2f m/s"%(o,base_chi_order[i].std()))
ax.set_xlabel("JD")
ax.set_ylabel("$\chi^2_r$")

plt.savefig("[%s]sample-selection.png"%tag,dpi=300)
plt.clf()

