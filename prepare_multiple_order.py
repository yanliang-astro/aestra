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

def read_multiple_order(filename,order_value,read_keys=['OBSJD','DATE-OBS']):
    hdulist = fits.open(filename)

    header = hdulist[0].header
    ccf_header = hdulist[12].header
    
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
    for o in order_value:
        info_dict[o] = {"SSBRV":SSBRV[o]+0.8,"CCFRV":CCFRV[o]}
    info_dict["CCFRVMOD"] = ccf_header["CCFRVMOD"]
    return data,info_dict

def redshift_chi(rv,wave_rest,yrest,wave_obs,ydata,wdata):
    wave_shifted = wave_rest*(1 + rv/Synthetic.c)
    func = CubicSpline(wave_shifted, yrest)
    model_obs = func(wave_obs)
    wmodel = np.ones_like(wdata)
    wmodel[(wave_obs<min(wave_shifted))|(wave_obs>max(wave_shifted))]=0
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
    return sorted(zip(unique_wavelengths, unique_depths), key=lambda x: x[1], reverse=True)

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

def detrend_polynomial(wavelength,intensity,deg=3):
    ysmooth = gaussian_filter1d(intensity, 30)
    ydiff = intensity-ysmooth
    # Calculate the 1D derivative of the spectrum
    derivative = np.gradient(intensity, wavelength)
    threshold = np.quantile(np.abs(derivative),0.5)
    absorption = (np.abs(derivative)>threshold)|(ydiff<np.quantile(ydiff,0.3))
    # Remove the absorption lines from the data
    wavelength_no_absorption = wavelength[~absorption]
    intensity_no_absorption = intensity[~absorption]
    ppoly = np.polyfit(wavelength_no_absorption, intensity_no_absorption, deg)
    fitted_polynomial = np.polyval(ppoly,wavelength)
    return fitted_polynomial

def prepare_spectrum(input_wave,obsname,divide_telluric=False,mask_telluric=False,store_telluric = True, detrend=False, n_micro=100):
    large_number = 1e6
    data,info_dict = read_multiple_order("%s/%s"%(datadir,obsname),order_value=order_value)

    n_spec = input_wave.shape[1]
    spectrum = np.zeros((len(order_value),n_spec))
    spectrum_err = np.zeros((len(order_value),n_spec))
    if store_telluric: telluric_spectrum = np.zeros((len(order_value),n_spec))
    for k,o in enumerate(order_value):
        wave_obs = input_wave[k]
        science,blaze,telluric = data[k]
        wave_raw,flux,flux_var = science

        ssbrv = info_dict[o]["SSBRV"]
        jd = info_dict["OBSJD"]

        ph,planetary_rv = simulate_planet(jd)
        info_dict["v_planet"] = planetary_rv
        total_rv = 1e3*ssbrv + planetary_rv
        # ssbrv: transform to heliocentric frame
        wave = wave_raw + wave_raw*(total_rv)/Synthetic.c
        isnan = np.isnan(flux)

        normflux = np.zeros_like(flux)
        normflux_err = np.zeros_like(flux_var)

        if divide_telluric:
            telluric_poly = detrend_polynomial(wave_raw,telluric)
            telluric /= telluric_poly
            norm = np.quantile(flux[~isnan]/(blaze*telluric)[~isnan],0.5)
            normflux[~isnan] = (flux/(norm*blaze*telluric))[~isnan]
            normflux_err[~isnan] = flux_var[~isnan]**0.5/(norm*blaze)[~isnan]
            micro_tellurics = find_deepest_lines(wave, telluric, num_lines=n_micro)
            telluric_err = calculate_flux_uncertainty(wave_obs, micro_tellurics)
        else:
            norm = np.quantile(flux[~isnan]/blaze[~isnan],0.5)
            normflux[~isnan] = (flux/(norm*blaze))[~isnan]
            normflux_err[~isnan] = flux_var[~isnan]**0.5/(norm*blaze)[~isnan]
            # Use the function to find the top 10 deepest unique lines
            if mask_telluric:
                top_unique_lines = find_deepest_lines(wave, telluric)
                # Mask the top unique deepest lines in the spectrum
                skymask = mask_deepest_lines(wave_obs, top_unique_lines)
                fraction = skymask.sum()/len(wave_obs)
            else: skymask = np.zeros(len(wave_obs),dtype=bool)

        wh_nan = np.where(isnan)[0]
        index = np.where((wh_nan[1:]-wh_nan[:-1])>1)[0]
        edges = [wh_nan[0]] + list(wh_nan[index]) + list(wh_nan[index+1]) + [wh_nan[-1]]
        edges = wave[np.sort(edges)]

        small = 1e-2
        # select bad regions after interpolation
        bad = (wave_obs<min(wave))|(wave_obs>max(wave))
        for i_chunk in range(len(edges)//2):
            start,end = edges[2*i_chunk:2*i_chunk+2]
            bad |= (wave_obs>(start-small))&(wave_obs<(end+small))
        #print("bad:",bad.sum())
        inbound = (wave_obs>min(wave))&(wave_obs<max(wave))
        locmask = inbound&(~bad)

        # Interpolate flux onto wave_obs
        spectrum[k][locmask] = interp1d(wave[~isnan], normflux[~isnan], kind='cubic')(wave_obs[locmask])
        spectrum_err[k][locmask] = interp1d(wave[~isnan], normflux_err[~isnan], kind='cubic')(wave_obs[locmask])
        spectrum_err[k][~locmask] = large_number
        if divide_telluric:
            combined_err = (spectrum_err[k]**2+telluric_err**2)**0.5
            spectrum_err[k][locmask] = combined_err[locmask]
        else:spectrum_err[k][skymask] = large_number
        spectrum[k][~locmask] = 1.0
        if store_telluric:
            telluric/=np.quantile(telluric,0.5)
            telluric_spectrum[k][inbound] = interp1d(wave, telluric, kind='cubic')(wave_obs[inbound])
            telluric_spectrum[k][~inbound] = 1.0
        else: telluric_spectrum = None
        if detrend:
            cmask = locmask & (~reference_skymask[k])
            p = np.polyfit(wave_obs[cmask],(spectrum[k]-template[k])[cmask],deg=3,w=spectrum_err[k][cmask])
            spectrum_trend = np.polyval(p,wave_obs[locmask])
            spectrum[k][locmask] -= spectrum_trend   
    return spectrum,spectrum_err,telluric_spectrum,info_dict

def save_batch(specs,w,ssbrv,IDs,filename):
    spec = torch.from_numpy(specs.astype(np.float32))
    w = torch.from_numpy(w.astype(np.float32))
    ssbrv = torch.from_numpy(ssbrv.astype(np.double))
    ID = torch.from_numpy(IDs.astype(np.double))
    batch = [spec,w,ssbrv,ID]
    print("spec:",spec.shape,"w:",w.shape,"ssbrv:",ssbrv,"ID",ID.shape)
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

def merge_batch(file_batches):
    spectra = [];weights = [];ssbrv = [];specid = []
    for batchname in file_batches:
        print("batchname:",batchname)
        batch = load_batch(batchname)
        spectra.append(batch[0])
        weights.append(batch[1])
        ssbrv.append(batch[2])
        specid.append(batch[3])
    spectra = torch.cat(spectra,axis=0)
    weights = torch.cat(weights,axis=0)
    ssbrv = torch.cat(ssbrv,axis=0)
    specid = torch.cat(specid,axis=0)
    print("spectra:",spectra.shape,"w:",weights.shape,
          "ssbrv:",ssbrv.shape,"specid:",specid.shape)
    return spectra, weights, ssbrv.T, specid

def make_batch(sample_names):
    large_number = 1e6
    batch_size = len(sample_names)
    specmat = np.ones((batch_size,n_order,n_spec))
    errmat = np.zeros((batch_size,n_order,n_spec))
    telluricmat = np.ones((batch_size,n_order,n_spec))
    neid_dict = {}
    for i_obs,obsname in enumerate(sample_names):
        spectrum,spectrum_err,telluric, info_dict = prepare_spectrum(input_wave,obsname)
        neid_dict[obsname] = info_dict
        specmat[i_obs,:,:] = spectrum
        errmat[i_obs,:,:] = spectrum_err
        if telluric is None:continue
        telluricmat[i_obs,:,:] = telluric
    for k in range(n_order):
        avg_err = np.median(errmat[:,k,:],axis=0)
        errmat[:,k,avg_err>1] = large_number
    avg_err = np.median(errmat,axis=0)[None,:,:]
    w_baseline = (avg_err**(-2)).repeat(batch_size,axis=0)
    baseline = np.median(specmat,axis=0)[None,:,:].repeat(batch_size,axis=0)
    bad = (w_baseline<1e-3)|(errmat**(-2)<1e-3)
    print("bad pixels:",(bad.sum()/batch_size))
    specmat[bad] = baseline[bad]
    return specmat,errmat,telluricmat,neid_dict

def photon_noise(spec_rest,wave_rest,sn):
    A0 = spec_rest*(sn**2)
    dAdl = np.gradient(A0,wave_rest)
    W = (wave_rest**2)*(dAdl)**2/A0
    Ne = A0.sum()
    Q = W.sum()/(Ne**0.5)
    print("W:",W.shape,"Ne:",Ne,"Q:",Q)
    RV_rms = Synthetic.c/(W.sum())**0.5
    return RV_rms

def fit_rv(spec,w,rest_model,wave_obs):
    result = scipy.optimize.minimize(redshift_chi,0.0, method='Nelder-Mead',args=(wave_obs,rest_model,wave_obs,spec,w,))
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
    specmat,errmat,telluricmat,sub_dict = make_batch(batch_id)
    ssbrvs = get_timeseries(sub_dict,'SSBRV',batch_id).T
    timestamp = get_timeseries(sub_dict,'OBSJD',batch_id)
    save_batch(specmat,errmat**(-2),ssbrvs,timestamp,batch_name)
    print("telluricmat:",telluricmat.shape)
    auxname = "%s/%s-%s"%(dynamic_dir,"telluric",os.path.basename(batch_name))
    save_auxfile(np.median(telluricmat,axis=0),auxname)
    neid_dict.update(sub_dict)
    return 0

def fit_rv_worker(wave_obs,spec,w,baseline,mdict,obsname,order):
    v_template,base_chi,message = fit_rv(spec,w,baseline,wave_obs)
    summary = {"v_template":v_template[0],"chi_template":base_chi}
    mdict["%s-%d"%(obsname,order)]=summary
    return 0

def process_task(args, mdict):
    # Unpack arguments
    wave_obs, specs, weights, baseline, obsname, order = args
    # Your existing task logic with the managed dictionary
    fit_rv_worker(wave_obs, specs, weights, baseline, mdict, obsname, order)
    print("mdict:",len(mdict))
    return

def wrap_data(sample_names,datatag,batch_size):
    idx = np.arange(0, len(sample_names), batch_size)
    batches = np.array_split(sample_names, idx[1:])
    file_batches = ["%s/%s_%d.pkl"%(dynamic_dir,datatag,k) for k in range(len(batches))]
    planet_info = [amplitude_planet,period_planet,t0_value]
    general_info = {"sample_names":sample_names,
                    "files":file_batches,
                    "planet_param":planet_info}

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
    specs,weights,ssbrvs,ids = [item.numpy() for item in batch]
    baseline = np.median(specs,axis=0)
    avg_err = np.median(weights**(-0.5),axis=0)
    print("baseline:",baseline.shape)
    
    template_name="%s/%s.pkl"%(dynamic_dir,datatag+"-template")
    save_batch(baseline[None,:,:],avg_err[None,:,:]**(-2),
               np.array([0]),np.array([888]),template_name)

    general_info.update({"baseline":baseline,"avg_err":avg_err})
    neid_dict.update({"info":general_info})
    with open("%s-param.pkl"%datatag,"wb") as f:
        pickle.dump(neid_dict,f)
        
    telluric_batches = ["%s/telluric-%s_%d.pkl"%(dynamic_dir,datatag,k) for k in range(len(batches))]
    telluric = []
    for item in telluric_batches:
        telluric.append(load_batch(item)[None,:,:])
    telluric = torch.cat(telluric).mean(axis=0).numpy()
    save_auxfile(telluric,"%s/%s-telluric.pkl"%(dynamic_dir,datatag))
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
            task_args = (wave_obs, specs[i_epoch][i_order], weights[i_epoch][i_order], baseline[i_order], obsname, order)
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

np.random.seed(0)
torch.manual_seed(0)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your script')

# Define optional arguments with default values
parser.add_argument('-t', '--tag', help='Tag description', default='single')
parser.add_argument('-a', '--amplitude', type=float, help='Amplitude of the planet', default=0.0)
parser.add_argument('-p', '--period', type=float, help='Period of the planet', default=0.11)
parser.add_argument('-t0', '--t0_value', type=int, help='t0 value', default=2459300)
parser.add_argument('-n', '--samples', type=int, help='Number of samples', default=100)
parser.add_argument('-batch', '--batch_size', type=int, help='Batch size', default=500)
parser.add_argument('-cpu', '--num_cores', type=int, help='Number of CPU cores', default=10)
parser.add_argument('-load', '--load_data', action='store_true', help='Load data')
parser.add_argument('-o','--orders', nargs='+', help='<Required> Orders', required=True)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
tag = args.tag
#order_value = args.order_value
amplitude_planet = args.amplitude
period_planet = args.period
t0_value = args.t0_value
n_sample = args.samples
batch_size = args.batch_size
num_cores = args.num_cores
load_data = args.load_data

order_value = [int(o) for o in args.orders]

input_wave = [get_order_wavelengths(o) for o in order_value]
input_wave = np.array(input_wave)

n_order,n_spec = input_wave.shape
print("input_wave:",input_wave.shape)

if amplitude_planet>0:
    per_tag = "%.2fday"%period_planet
    amp_tag = "%dcm"%(amplitude_planet*100)
    datatag = "%s_%s_%s_N%d"%(tag,per_tag,amp_tag,n_sample)
else:datatag = "%s_noplanet_N%d"%(tag,n_sample)

save_auxfile(input_wave,"%s/%s-wavelength.pkl"%(dynamic_dir,datatag))

def simulate_planet(t,amp=amplitude_planet, Period=period_planet,t0=2459300):
    phase = ((t-t0)/Period)%1
    return phase,amp*np.sin(2*np.pi*phase)

#n_sample = 6000
# reading the CSV file
csvfilename = 'SolarRVTable_Level.csv'
csvFile = pandas.read_csv("%s/%s"%(datadir,csvfilename))
neid_filenames = csvFile.filename
neid_jd = np.array(csvFile.ccfjdsum)
neid_ccfrv = csvFile.ccfrvmod
neid_snr =  np.array(csvFile.extsnr)

bad_day = [2459358,2459359,2459360,2459370,2459380,2459381]
excluded = np.zeros(len(neid_jd),dtype=bool)
for t0 in bad_day:excluded |= ((neid_jd>t0)&(neid_jd<(t0+0.5)))

snr_cut = 400
excluded |= neid_snr < snr_cut
sel = np.arange(len(neid_filenames))[~excluded]

print("total number:",len(sel))
np.random.shuffle(sel)
sel = sel[:n_sample]
print("sel:",sel[:10])

sample_names = list(neid_filenames[sel])
print("order:",order_value)
print("sample_names:",len(sample_names))

idx = np.arange(0, len(sample_names), batch_size)
batches = np.array_split(sample_names, idx[1:])

file_batches = ["%s/%s_%d.pkl"%(dynamic_dir,datatag,k) for k in range(len(batches))]
print("file_batches:",file_batches)

load_tag = "calib_noplanet_N6000"
save_template = "%s/%s-template.pkl"%(dynamic_dir,load_tag)
template = load_batch(save_template)[0][0].numpy()
reference_skymask = load_batch("%s/%s-skymask.pkl"%(dynamic_dir,load_tag)).numpy().astype(dtype=bool)


if not load_data:
    wrap_data(sample_names,datatag,batch_size)

print("Loading from %s-param.pkl"%datatag)
with open("%s-param.pkl"%datatag,"rb") as f:
    neid_dict = pickle.load(f)

save_telluric = "%s/%s-telluric.pkl"%(dynamic_dir,datatag)
telluric_spec = load_batch(save_telluric).numpy()
print("wave",input_wave.shape,"telluric_spec:",telluric_spec.shape)
skymask = np.zeros((telluric_spec).shape,dtype=bool)
for i_order,o in enumerate(order_value):
    wave_obs = input_wave[i_order]
    top_unique_lines = find_deepest_lines(wave_obs, telluric_spec[i_order])
    # Mask the top unique deepest lines in the spectrum
    skymask[i_order] = mask_deepest_lines(wave_obs, top_unique_lines)
    fraction = skymask[i_order].sum()/len(wave_obs)
    print("Order %d"%o,fraction)
save_auxfile(skymask,"%s/%s-skymask.pkl"%(dynamic_dir,datatag))

# load generated data
batch = merge_batch(file_batches)
specs,weights,ssbrvs,ids = [item.numpy() for item in batch]
n_epoch,n_order,n_spec = specs.shape

baseline = np.median(specs,axis=0)
avg_err = np.median(weights**(-0.5),axis=0)

for i_order,o in enumerate(order_value):
    sn = baseline[i_order]/avg_err[i_order]
    print("sn:",sn.min(),sn.max(),"mean sn:",sn.mean())
    good = sn>1
    RV_limit = photon_noise(baseline[i_order][good],
                            input_wave[i_order][good],
                            sn[good])
    print("Order %d RV_limit: %.2f m/s"%(o,RV_limit))

km_m = 1e3
timestamp = get_timeseries(neid_dict,'OBSJD',sample_names)
v_template_order = get_timeseries(neid_dict,'v_template',sample_names)
base_chi_order = get_timeseries(neid_dict,'chi_template',sample_names)
ssbrvs_order = get_timeseries(neid_dict,'SSBRV',sample_names)*km_m
ccfrvs_order = get_timeseries(neid_dict,'CCFRV',sample_names)*km_m

bervs_order = -ssbrvs_order

berv_norm = (bervs_order-np.mean(bervs_order,axis=-1,keepdims=True))
ccf_norm = (ccfrvs_order-np.mean(ccfrvs_order,axis=-1,keepdims=True))
v_template = v_template_order-np.mean(v_template_order,axis=-1,keepdims=True)

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
    print_string("$v_{template}$",v_template[i],mode="2")
    print_string("$v_{template}-v_{CCF}$",
                 template_ccf_offset[i],mode="2")

phase,v_planet = simulate_planet(timestamp)

#plot_fft(timestamp,[v_template],datatag,["$v_{template}$"],
#         period=period_planet,fs=14)

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

t0 = 2459370
#mask = ((timestamp-t0)>-1.2)&((timestamp-t0)<0.45)
rank = np.argsort(base_chi_order.mean(axis=0))[::-1]
i_plots = [0,1,2,3,4,5]#rank[:5]

cmap = get_cmap('plasma')
tmin,tmax = min(timestamp[i_plots]),max(timestamp[i_plots])
colors =[cmap((t-tmin)/(tmax-tmin)) for t in timestamp[i_plots]]

spec_resid = specs - baseline

#wh = np.argmax(spec_resid[i_obs][i_order].abs())
mask = np.arange(1000,1500)
        
for i_order,o in enumerate(order_value):
    wave_obs = input_wave[i_order]
    spec_base = baseline[i_order]
    base_chi = base_chi_order[i_order]

    fig,axs = plt.subplots(nrows=2,figsize=(10,8),
                           constrained_layout=True,sharex=True)
    i_image = 0
    for i_obs,obsname in enumerate(sample_names):
        if not i_obs in i_plots:continue
        ccfrv = ccf_norm[i_order][i_obs]
        yoffset = ccfrv

        date_obs = neid_dict[obsname]['DATE-OBS']
        date = date_obs[5:10]
        time = date_obs[11:16]
        
        resid_flux = 20*spec_resid[i_obs][i_order][mask]+yoffset
        resid_error = 20*weights[i_obs][i_order][mask]**(-0.5)
        resid_error[resid_error>2]=2
        snr = specs[i_obs][i_order]/(weights[i_obs][i_order]**(-0.5))

        text = "%s(%s) $v_{CCF}$:%.2f m/s $\chi^2=%.2f$"%(time,date,ccfrv,base_chi[i_obs])
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
    axs[0].set_ylabel("normalized flux")
    axs[1].set_ylabel("$v_{CCF}$ [m/s]")
    #axs[1].set_ylim(0.1,1.05)
    axs[0].legend()
    plt.savefig("[%s-order%d]residual-spectrum.png"%(tag,o),dpi=300)

grey_colors = ["grey","skyblue","lightgreen"]
bright_colors = ["k","b","m","cyan"]

fig,axs = plt.subplots(nrows=3,figsize=(8,10),constrained_layout=True)

ax=axs[0]
ax.scatter(neid_jd,neid_ccfrv,c="grey",s=5,label="all (N=%d)"%len(neid_jd))
img = ax.scatter(neid_jd[sel],neid_ccfrv[sel],c=neid_snr[sel],cmap="inferno",s=5,label="selected (N=%d)"%len(sel))
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
    #ax.scatter(timestamp,berv_norm[i],c="grey",s=5,label="$v_{BERV}$")
    ax.scatter(timestamp,berv_norm[i]+ccf_norm[i],c=grey_colors[i],s=5,label="%d $v_{BERV}+v_{CCF}$"%o)
    ax.scatter(timestamp,berv_norm[i]+v_template[i],c=bright_colors[i],s=5,label="%d $v_{BERV}+v_{template}$"%o)

ax.legend()
ax.set_xlabel("JD")
ax.set_ylabel("RV [m/s]")

ax=axs[2]

for i,o in enumerate(order_value):
    ax.scatter(timestamp,v_template[i],c=bright_colors[i],s=5,
               label="%d $v_{template}$, RMS = %.2f m/s"%(o,v_template[i].std()))
    ax.scatter(timestamp,ccf_norm[i],c=grey_colors[i],s=5,
               label="%d $v_{CCF}$, RMS = %.2f m/s"%(o,ccf_norm[i].std()))

ax.legend(title="Discrepancy RMS = %.2f m/s"%template_ccf_offset.std())
ax.set_xlabel("JD")
ax.set_ylabel("RV [m/s]")

plt.savefig("[%s]sample-selection.png"%tag,dpi=300)
plt.clf()

