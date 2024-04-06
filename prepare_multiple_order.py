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
from util import moving_mean,plot_fft,mem_report,load_batch,merge_batch
from scipy.optimize import curve_fit

dynamic_dir = "/scratch/gpfs/yanliang/neid-dynamic"
datadir = "/scratch/gpfs/yanliang/NEID-SOLAR"
telluric_dir = "/scratch/gpfs/yanliang/NEID-TELLURIC"
device =  torch.device("cpu")

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
    telluric = hdulist[10].data

    # Close the FITS file
    hdulist.close()
    
    science = [science_wavelength,science_flux,science_variance]
    data = []
    for o in order_value:
        science_order = [item[o] for item in science]
        data.append([science_order,science_blaze[o],telluric[o]])

    SSBRV= get_barycentric_corr_rv(header)
    CCFRV = read_ccf_rv(ccf_header)
    info_dict = {key:header[key] for key in read_keys}
    info_dict.update({key:telluric_header[key] for key in ["ZENITH","WVAPOR"]})
    for o in order_value:
        info_dict[o] = {"SSBRV":SSBRV[o]+0.8,"CCFRV":CCFRV[o]}
    info_dict["CCFRVMOD"] = ccf_header["CCFRVMOD"]
    # time zero point
    info_dict["timestamp"] = np.float32(info_dict["OBSJD"] - 2459350.0) 
    return data,info_dict

def redshift_chi(rv,wave_rest,yrest,wrest,wave_obs,ydata,wdata):
    wave_shifted = wave_rest*(1 + rv/Synthetic.c)
    bad = yrest==0
    model_obs = CubicSpline(wave_shifted[~bad], yrest[~bad])(wave_obs)
    model_w = interp1d(wave_shifted,wrest)(wave_obs)

    wmodel = np.ones_like(wdata)
    wmodel[(wave_obs<min(wave_shifted))|(wave_obs>max(wave_shifted))]=0
    wmodel[model_w<1.0]=0
    loss = np.sum(wmodel*wdata * (ydata - model_obs)**2) / len(ydata)
    return loss

def find_deepest_lines(wave_obs, raw_spectrum, num_lines=30, min_separation=0.10):
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


def prepare_spectrum(obsname):
    large_number = 1e6
    data,info_dict = read_multiple_order("%s/%s"%(datadir,obsname),order_value=order_value)
    
    flag,telluric_model = load_telluric_model(obsname)
    if flag: f_telluric = interp1d(wave_rest, telluric_model, kind='cubic')

    wavelength = np.zeros((len(order_value),N_SPEC))
    spectrum = np.zeros((len(order_value),N_SPEC))
    spectrum_err = np.zeros((len(order_value),N_SPEC))
    telluric_spectrum = np.zeros((len(order_value),N_SPEC))
    for k,o in enumerate(order_value):
        wave_obs = input_wave[k]
        science,blaze,neid_telluric = data[k]
        wave_raw,flux,flux_var = science

        #print("NEID telluric:",neid_telluric.min(),neid_telluric.max())
        #print("AESTRA telluric:",telluric_model.min(),telluric_model.max())
        ssbrv = info_dict[o]["SSBRV"]
        jd = info_dict["OBSJD"]

        isnan = np.isnan(flux) | np.isnan(blaze)| (flux<=0.0)

        normflux = np.zeros_like(flux)
        normflux_err = np.zeros_like(flux_var)

        denom = blaze

        norm = np.quantile(flux[~isnan]/denom[~isnan],0.5)
        normflux[~isnan] = flux[~isnan]/(norm*denom[~isnan])
        normflux_err[~isnan] = flux_var[~isnan]**0.5/(norm*denom[~isnan])
        if normflux.min()<0:
            print("negative flux!",normflux.min(),normflux.max())
        elif blaze.min()<=0.:
            print("blaze nan!",blaze.min(),blaze.max())

        wavelength[k] = wave_raw
        spectrum[k][~isnan] = normflux[~isnan]
        spectrum_err[k][~isnan] = normflux_err[~isnan]
        if flag: telluric_spectrum[k] = f_telluric(wave_raw)
        elif neid_telluric.ndim ==1:
            telluric_spectrum[k] = neid_telluric
        # telluric lines * telluric continuum
        else: telluric_spectrum[k] = neid_telluric[:,0]*neid_telluric[:,1]
        spectrum_err[k][isnan] = large_number
    data = wavelength,spectrum,spectrum_err,telluric_spectrum
    return data,info_dict

def save_batch(wave,specs,w,ssbrv,IDs,telluric,filename):
    wave = torch.from_numpy(wave.astype(np.double))
    spec = torch.from_numpy(specs.astype(np.float32))
    weight = torch.from_numpy(w.astype(np.float32))
    ssbrv = torch.from_numpy(ssbrv.astype(np.double))
    ID = torch.from_numpy(IDs.astype(np.float32))
    telluric = torch.from_numpy(telluric.astype(np.float32))

    batch = [wave,spec,weight,ssbrv,ID,telluric]
    print("wave:",wave.shape,"spec:",spec.shape,"weight:",weight.shape,
          "ssbrv:",ssbrv,"ID",ID.shape,"telluric",telluric.shape)
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
    telluricmat = np.ones((batch_size,n_order,N_SPEC))
    good =  np.ones((batch_size),dtype=bool)
    neid_dict = {}
    for i_obs,obsname in enumerate(sample_names):
        data,info_dict = prepare_spectrum(obsname)
        wavelength,spectrum,spectrum_err,telluric_spectrum = data
        # negative flux?
        neg = np.sum(spectrum<0.0,axis=-1)
        if neg.sum()>100:
            good[i_obs] = False
            print("negative!!",obsname,neg)
            print("  flux: %.2f, %.2f"%(spectrum.min(),spectrum.max()))
        #if spectrum.min()<0.01:good[i_obs] = False
        if not good[i_obs]: continue
        neid_dict[obsname] = info_dict
        wavemat[i_obs,:,:] = wavelength
        specmat[i_obs,:,:] = spectrum
        errmat[i_obs,:,:] = spectrum_err
        telluricmat[i_obs,:,:] = telluric_spectrum

    bad = errmat**(-2)<1.0
    print("bad pixels:",(bad.sum()/batch_size))
    print("good:",good.sum())
    specmat[bad] = 0.0
    wavemat=wavemat[good]
    specmat=specmat[good]
    errmat=errmat[good]
    telluricmat=telluricmat[good]
    return sample_names[good],wavemat,specmat,errmat,telluricmat,neid_dict

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
    batch_id,wavemat,specmat,errmat,telluricmat,sub_dict = make_batch(batch_id)
    ssbrvs = get_timeseries(sub_dict,'SSBRV',batch_id).T
    timestamp = get_timeseries(sub_dict,'timestamp',batch_id)
    save_batch(wavemat,specmat,errmat**(-2),ssbrvs,timestamp,telluricmat,batch_name)
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
    if n_items%500==0: print("mdict:",n_items)
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
    waves,specs,weights,ssbrvs,ids,tellurics = [item.numpy() for item in batch]
    
    # interpolate to homogeneous grid - calculate the template spectrum
    print("native grid",waves.shape,"input grid",input_wave.shape)
    n_template = min(100,len(specs))
    n_order,n_pix = input_wave.shape
    input_flux = np.zeros((n_template,n_order,n_pix))
    input_weight = np.zeros((n_template,n_order,n_pix))
    input_telluric = np.zeros((n_template,n_order,n_pix))
    for o in range(n_order):
        wave_obs = input_wave[o]
        for i in range(n_template):
            wave_raw = waves[i][o]

            ssbrv = 1e3*ssbrvs[i][o] # km/s to m/s
            flux = specs[i][o]
            weight = weights[i][o]
            telluric = tellurics[i][o]

            wave = wave_raw + wave_raw*(ssbrv)/Synthetic.c
            inbound = (wave_obs>min(wave))&(wave_obs<max(wave))
            good = (weights[i][o]>1.0)

            input_flux[i][o][inbound] = interp1d(wave[good], flux[good], kind='nearest')(wave_obs[inbound])
            input_weight[i][o][inbound] = interp1d(wave, weight, kind='nearest')(wave_obs[inbound])
            input_telluric[i][o][inbound] = interp1d(wave, telluric, kind='nearest')(wave_obs[inbound])
            input_weight[i][o][~inbound] = 1e-12

            bad = input_weight[i][o]<1.0
            input_flux[i][o][bad] = 0.0
            print(o,"raw spec min:",specs[i][o][good].min())
            print(o,"spec min:",input_flux[i][o][~bad].min())

    template = np.median(input_flux,axis=0)
    template_w = np.median(input_weight,axis=0)
    template_telluric = np.median(input_telluric,axis=0)

    dispersion = np.std(input_flux,axis=0)
    for o in range(n_order):
        max_dispersion = np.max(dispersion[o])
        while max_dispersion>0.1:
            whmax = np.argmax(dispersion[o])
            start,end = max(whmax-5,0),min(whmax+5,n_pix)
            template_w[o][start:end] = 1e-12
            dispersion[o][start:end] = 0
            max_dispersion = np.max(dispersion[o])
            print(o,max_dispersion,whmax)
    template[template_w<1.0] = 0.0
    dispersion[template_w<1.0] = 0.0

    '''
    fig,axs=plt.subplots(figsize=(10,8),nrows=n_order)
    for o in range(n_order):
        wave_obs = input_wave[o]
        ax=axs[o]
        ax.plot(wave_obs,template[o],"k-",lw=1)
        ax.plot(wave_obs,dispersion[o],"r-",lw=1)
    plt.savefig("test.png",dpi=200)
    '''
    template_name="%s/%s.pkl"%(dynamic_dir,datatag+"-template")
    save_batch(input_wave[None,:,:],
               template[None,:,:],template_w[None,:,:],
               np.array([0]),np.array([888]),
               template_telluric[None,:,:],template_name)
    general_info.update({"baseline":template,"baseline_w":template_w,
                         "telluric_baseline":template_telluric})
    # remove poor-quality spectra
    sample_names = [i for i in sample_names if i in neid_dict]
    print("sample_names:",len(sample_names))

    general_info["sample_names"] = sample_names
    neid_dict.update({"info":general_info})
    with open("%s-param.pkl"%datatag,"wb") as f:
        pickle.dump(neid_dict,f)

    # calculate v_template and chi_template
    #process_list = []
    manager = mp.Manager()
    mdict = manager.dict()
    # Use a pool of workers
    pool_size = num_cores  # Number of processes in the pool
    pool = mp.Pool(pool_size)
    tasks = []
    for i_epoch,obsname in enumerate(sample_names):
        for i_order,order in enumerate(order_value):
            wave_obs = input_wave[i_order]
            task_args = (waves[i_epoch][i_order], specs[i_epoch][i_order], weights[i_epoch][i_order], input_wave[i_order],template[i_order],template_w[i_order], obsname, order)
            tasks.append(task_args)

    for i,task in enumerate(tasks):
        pool.apply_async(process_task, args=(task, mdict))
    # Close and join the pool
    pool.close()
    pool.join()

    #print("mdict:",mdict)#,"neid_dict:",neid_dict.keys())
    for k in sample_names:
        for o in order_value:
            neid_dict[k][o].update(mdict["%s-%d"%(k,o)])

    print("Saving to %s-param.pkl..."%datatag)
    with open("%s-param.pkl"%datatag,"wb") as f:
        pickle.dump(neid_dict,f)

    return 

def get_wavelengths(poly,wave_min,wave_max,n_pix=9216):
    input_pix = np.arange(n_pix+6)
    input_grid = np.zeros((n_pix+6))
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
    o = target_order
    wave_obs = get_wavelengths(wave_poly[o],wave_min[o],wave_max[o])
    return wave_obs

def simulate_planet(t,amp, Period,t0=0.0):
    phase = ((t-t0)/Period)%1
    return phase,amp*np.sin(2*np.pi*phase)

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
            
def preview_spectrum(input_wave,obsname,flag=False):
    data,info_dict = prepare_spectrum(obsname)
    wavelength,spectrum,spectrum_err,telluric_spectrum = data
    nrows=len(order_value)
    fig, axs = plt.subplots(figsize=(12,nrows*2.5),nrows=nrows,dpi=200,constrained_layout=True)
    for i,ax in enumerate(axs):
        ax.set_title("Order %d"%order_value[i])
        ax.plot(wavelength[i],spectrum[i],"k-")
        ax.plot(wavelength[i],telluric_spectrum[i],"r-",lw=1,label="our telluric model")
        ax.set_xlim(input_wave[i][0],input_wave[i][-1])
        ax.legend()
        #ax.set_ylim(0.9,1.01)
    plt.savefig("[%s]single-obs.png"%datatag)
    return

def initialize_restframe_model(input_wave,telluric_corrected,weight):
    # define restframe grids
    min_sep = np.min(input_wave[:,1:]-input_wave[:,:-1])
    wbin = min(min_sep,0.01) # wavelength bin
    wave_rest = np.arange(input_wave.min(),input_wave.max()+wbin,wbin)
    spec_rest = np.ones_like(wave_rest)

    n_expand = input_wave.shape[0]*input_wave.shape[1]
    x = input_wave.reshape((n_expand))
    y = telluric_corrected.reshape((n_expand))
    bad = (weight<1.0).reshape((n_expand))
    f = interp1d(x[~bad],y[~bad],kind = "nearest")
    mask = (wave_rest>x[~bad].min())&(wave_rest<x[~bad].max())
    spec_rest[mask] = f(wave_rest[mask])

    num_lines = 500
    lines_per_group = 50
    dim = 3

    '''
    lines = find_deepest_lines(wave_rest, spec_rest, num_lines=num_lines,min_separation=0.10)
    lines = np.array(lines)
    param_fit = np.array([(line[1], line[0], 0.04) for line in lines])
    print("param_fit:",param_fit.shape)

    for i in range(num_lines//lines_per_group):
        start,end = max(0,i*lines_per_group-5),min(num_lines,(i+1)*lines_per_group+5)
        adjust = np.zeros((num_lines),dtype=bool)
        fixed = np.zeros((num_lines),dtype=bool)
        adjust[start:end] = True
        fixed[start-3:start] = True
        fixed[end:end+3] = True
        wavemin,wavemax = lines[adjust,0].min(),lines[adjust,0].max()
        mask = (wave_rest>wavemin)&(wave_rest<wavemax)

        fixed_params = param_fit[fixed].reshape((fixed.sum()*dim))
        fixed_spec = multi_gaussian(wave_rest[mask], *fixed_params)

        p0 = param_fit[adjust].reshape((1,(adjust).sum()*dim))
        params, covariance = curve_fit(multi_gaussian, wave_rest[mask], spec_rest[mask]/fixed_spec, p0=p0)
        bestfit_model = multi_gaussian(wave_rest[mask], *params)
        param_fit[adjust] = params.reshape(((adjust).sum(),dim))

        chi = ((spec_rest[mask]/fixed_spec-bestfit_model)**2/0.01**2).mean()
        print("lines %d ~ %d chi: %.2f"%(start,end,chi))
        #plt.plot(wave_rest[mask],spec_rest[mask],"k-")
        #plt.plot(wave_rest[mask],fixed_spec,"b-",lw=0.5)
        #plt.plot(wave_rest[mask],bestfit_model,"r-",lw=0.5)
        #plt.savefig("test.png",dpi=300)
        #exit()

    #with open("[lsf]bestfit.pkl","wb") as f:
    #    pickle.dump(param_fit,f)
    '''
    from scipy.signal import convolve
    with open("[lsf]bestfit.pkl","rb") as f:
        param_fit = pickle.load(f)

    lsf_size = 30
    sigma = 5
    x_sigma = np.arange(-lsf_size//2,lsf_size//2+1)/sigma
    kernel = np.exp(-(x_sigma)**2/(2))
    kernel /= kernel.sum()

    width = 0.03 # about the same as stellar intrinsic lsf
    '''
    bitwise = np.ones_like(wave_rest)
    for param in param_fit:
        amp,mu,sig = param
        mask = (wave_rest>(mu-width))&(wave_rest<(mu+width))
        bitwise[mask] *= (1-amp*sig/width)
    stellar = 1-bitwise
    '''
    thin = np.copy(param_fit)
    thin[:,0] *= thin[:,2]/width
    thin[:,2] = 0.03

    param_fit = param_fit.reshape((num_lines*dim))
    # Flatten the initial guess list for curve_fit
    bestfit_model = multi_gaussian(wave_rest, *param_fit)
    stellar = multi_gaussian(wave_rest, *thin.reshape((num_lines*dim)))
    chi = ((spec_rest-bestfit_model)**2/0.01**2).mean()
    print("bestfit_model chi:",chi)

    intrinsic_model = 1-convolve(1-stellar, kernel, mode='same')
    chi = ((spec_rest-intrinsic_model)**2/0.01**2).mean()
    print("intrinsic_model chi:",chi)
    
    plt.plot(wave_rest,spec_rest,"k-",label="data")
    plt.plot(wave_rest,bestfit_model,"r-",lw=0.5,label="Gaussian")
    plt.plot(wave_rest,stellar,"-",lw=0.5,color="springgreen",
             label="stellar intrinsic")
    plt.plot(wave_rest,intrinsic_model,color="b",lw=0.5,
             label="convolved with LSF")
    plt.xlim(4981,4986)
    plt.legend()
    plt.ylim(0,1.1)
    plt.savefig("test.png",dpi=300)

    init_rest = np.array([wave_rest,stellar])
    print("init_rest:",init_rest.shape)
    return init_rest

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

wave_rest = np.loadtxt("%s/quiet_wave_rest.txt"%(telluric_dir))

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


#bad_day = [2459358,2459359,2459360,2459370,2459380,2459381]
#excluded = np.zeros(len(neid_jd),dtype=bool)
#for t0 in bad_day:excluded |= ((neid_jd>t0)&(neid_jd<(t0+0.5)))

#select_date = (neid_jd>2459700)&(neid_jd<2459710)

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

#preview_spectrum(input_wave,"neidL2_20211223T165223.fits")

if not load_data:
    save_auxfile(input_wave,"%s/%s-wavelength.pkl"%(dynamic_dir,datatag))
    wrap_data(sample_names,datatag,batch_size)

print("Loading from %s-param.pkl"%datatag)
with open("%s-param.pkl"%datatag,"rb") as f:
    neid_dict = pickle.load(f)

print("neid_dict:",len(neid_dict))

sample_names = neid_dict["info"]["sample_names"]
print("good spectra: %d/%d"%(len(sample_names),len(sel)))

telluric_baseline = neid_dict["info"]["telluric_baseline"]
baseline = neid_dict["info"]["baseline"]
baseline_w = neid_dict["info"]["baseline_w"]

#init_rest = initialize_restframe_model(input_wave,baseline/telluric_baseline,baseline_w)
#save_auxfile(init_rest,"%s/%s-rest.pkl"%(dynamic_dir,datatag))
#exit()
nrows=len(order_value)
fig, axs = plt.subplots(figsize=(12,nrows*2.5),nrows=nrows,dpi=200,constrained_layout=True)
for i,ax in enumerate(axs):
    ax.set_title("Order %d"%order_value[i])
    ax.plot(input_wave[i],baseline[i],"k-",label="mean")
    ax.plot(input_wave[i],telluric_baseline[i],"b-",label="mean tellurics")
    ax.legend()
plt.savefig("[%s]single-obs.png"%datatag)


for i_order,o in enumerate(order_value):
    sn = baseline[i_order]/baseline_w[i_order]**(-0.5)
    print("sn:",sn.min(),sn.max(),"mean sn:",sn.mean())
    good = sn>1
    RV_limit = photon_noise(baseline[i_order][good],
                            input_wave[i_order][good],
                            sn[good])
    print("Order %d RV_limit: %.2f m/s"%(o,RV_limit))

km_m = 1e3
timestamp = get_timeseries(neid_dict,'timestamp',sample_names)
jds = get_timeseries(neid_dict,'OBSJD',sample_names)
v_template_order = get_timeseries(neid_dict,'v_template',sample_names)
base_chi_order = get_timeseries(neid_dict,'chi_template',sample_names)
ssbrvs_order = get_timeseries(neid_dict,'SSBRV',sample_names)*km_m
ccfrvs_order = get_timeseries(neid_dict,'CCFRV',sample_names)*km_m

bervs_order = -ssbrvs_order

berv_norm = (bervs_order-np.median(bervs_order,axis=-1,keepdims=True))
ccf_norm = (ccfrvs_order-np.median(ccfrvs_order,axis=-1,keepdims=True))

template_ccf_offset = v_template_order-bervs_order-ccf_norm

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
    print_string("$v_{template}-v_{ssb}$",(v_template_order[i]+ssbrvs_order[i]),mode="2")
    print_string("$v_{template}-v_{ssb}-v_{CCF}$",
                 template_ccf_offset[i],mode="2")

v_template = v_template_order-bervs_order
v_template -= np.median(v_template,axis=-1,keepdims=True)
#plot_fft(timestamp,[v_template],datatag,["$v_{template}$"],
#         period=period_planet,fs=14)
Period = 88.3
phase = (timestamp/Period)%1
v_planet = np.zeros_like(phase)
fig,ax=plt.subplots(figsize=(4,4),constrained_layout=True)
for i in range(n_order):
    label_template = velocity_label((v_template[i]-v_planet),"$v_{template}-v_{planet}$")
    ax.plot(phase,v_template[i],".",ms=2,color="lightgrey",
            label=label_template)
    xgrid,ygrid,delta_y = moving_mean(phase,v_template[i],n=15)
    ax.errorbar(xgrid,ygrid,yerr=delta_y,fmt=".",capsize=3,ms=2,
                label="order %d $v_{template}$ re-binned"%order_value[i])
ax.plot(phase,v_planet,"r.",ms=1)
ax.set_xlabel("Phase");ax.set_ylabel("Residual RV [m/s]")
ax.set_ylim(-1.5,1.5)
#ax.set_ylim(-5,5)
ax.legend()
plt.savefig("[%s]v_template-phase-fold.png"%datatag,dpi=300)

# load generated data
batch = merge_batch(file_batches)
waves,spec_raw,weights,ssbrvs,ids,telluric_spec = [item.numpy() for item in batch]
specs = spec_raw/telluric_spec
n_epoch,n_order,N_SPEC = specs.shape

n_cut = 10

cut_params = ssbrvs.mean(axis=1)
cuts = np.linspace(cut_params.min(),cut_params.max(),n_cut)
print("cuts:",cuts)
dispersion = np.zeros((n_cut-1,n_order,N_SPEC))
for i in range(n_cut-1):
    who = (cut_params>cuts[i])&(cut_params<cuts[i+1])
    dispersion[i] = np.std(specs[who],axis=0)
print("dispersion:",dispersion.shape)
dispersion = np.median(dispersion,axis=0)
#rank = np.argsort(base_chi_order.mean(axis=0))[::-1]
#i_plots = [0,1,2,3,4,5]#
rank = np.argsort(ssbrvs.mean(axis=0))
i_plots = rank[:5]

baseline = np.median(specs,axis=0)
baseline_w = np.median(weights,axis=0)
spec_resid = specs - baseline[None,:,:]

for o in range(n_order):
    max_dispersion = np.max(dispersion[o])
    whmax = np.argmax(dispersion[o])
    print("order",o,"max_dispersion:",
          max_dispersion,"where:",whmax)

cmap = get_cmap('plasma_r')
tmin,tmax = min(timestamp[i_plots]),max(timestamp[i_plots])
colors =[cmap((t-tmin)/(tmax-tmin)) for t in timestamp[i_plots]]


#wh = np.argmax(spec_resid[i_obs][i_order].abs())
mask = np.arange(230,350)
#mask = np.arange(0,N_SPEC)
        
for i_order,o in enumerate(order_value):
    wave_obs = input_wave[i_order]
    spec_base = baseline[i_order]
    base_chi = base_chi_order[i_order]

    fig,axs = plt.subplots(nrows=2,figsize=(10,12),
                           constrained_layout=True,sharex=True)
    i_image = 0
    for i_obs,obsname in enumerate(sample_names):
        if not i_obs in i_plots:continue
        ccfrv = ccf_norm[i_order][i_obs]

        yoffset = 0#ccfrv

        date_obs = neid_dict[obsname]['DATE-OBS']
        date = date_obs[5:10]
        time = date_obs[11:16]
        
        resid_flux = spec_resid[i_obs][i_order][mask]+yoffset
        resid_error = weights[i_obs][i_order][mask]**(-0.5)
        resid_error[resid_error>0.2]=0.2
        snr = specs[i_obs][i_order]/(weights[i_obs][i_order]**(-0.5))

        text = "%.2f %s(%s) $v_{CCF}$:%.2f m/s $\chi^2=%.2f$"%(neid_dict[obsname]['timestamp'],time,date,ccfrv,base_chi[i_obs])
        print(text)

        axs[0].plot(wave_obs[mask], specs[i_obs][i_order][mask],
                    drawstyle="steps-mid",alpha=1,
                    c=colors[i_image],label=text)
        axs[1].fill_between(wave_obs[mask],
                            resid_flux-resid_error,
                            resid_flux+resid_error,step="mid",
                            color=colors[i_image],alpha=0.3)

        axs[1].plot(wave_obs[mask],resid_flux,
                    drawstyle="steps-mid",alpha=1,
                    c=colors[i_image],label=text)

        x_text = np.quantile(wave_obs[mask],0.9)
        y_text = resid_flux[-1] + 0.01
        axs[1].text(x_text,y_text,"S/N = %.1f"%snr.mean(),c=colors[i_image],
                   bbox= dict(facecolor='w',ec='w', alpha=0.8))
        i_image += 1

    axs[0].plot(wave_obs[mask], spec_base[mask],drawstyle="steps-mid",lw=1,c="k",label="mean")
    axs[0].plot(wave_obs[mask], dispersion[i_order][mask],drawstyle="steps-mid",lw=1,c="r",label="dispersion")
    #axs[0].plot(wave_obs[mask], dispersion[i_order][mask],drawstyle="steps-mid",lw=1,c="r",label="dispersion")
    axs[0].set_ylabel("normalized flux")
    #axs[1].set_ylabel("$v_{CCF}$ [m/s]")
    #axs[1].set_ylim(0.1,1.05)
    axs[0].legend()
    axs[0].set_title("Order %d"%o)
    plt.savefig("[%s-order%d]residual-spectrum.png"%(tag,o),dpi=300)

grey_colors = ["grey","skyblue","lightgreen"]
bright_colors = ["k","b","m","cyan"]

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
               label="%d $v_{template}$, RMS = %.2f m/s"%(o,v_template[i].std()))
    ax.scatter(timestamp,ccf_norm[i],c=grey_colors[i],s=5,
               label="%d $v_{CCF}$, RMS = %.2f m/s"%(o,ccf_norm[i].std()))

ax.legend(title="Discrepancy RMS = %.2f m/s"%template_ccf_offset.std())
ax.set_xlabel("JD")
ax.set_ylabel("RV [m/s]")
ax.set_ylim(-500,500)


ax=axs[2]
for i,o in enumerate(order_value):
    ax.scatter(timestamp,base_chi_order[i],c=bright_colors[i],s=5,
               label="%d $\chi^2_{template}$, RMS = %.2f m/s"%(o,base_chi_order[i].std()))
ax.set_xlabel("JD")
ax.set_ylabel("$\chi^2_r$")

plt.savefig("[%s]sample-selection.png"%tag,dpi=300)
plt.clf()

