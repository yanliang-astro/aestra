#!/usr/bin/env python
# coding: utf-8
import io, os, sys, time, random
import numpy as np
import pickle
import pandas 
import torch

import matplotlib.pyplot as plt
import scipy.optimize

from matplotlib.cm import get_cmap
from astropy.io import fits
from scipy.interpolate import interp1d,CubicSpline
from scipy.special import gamma
from synthetic_data import Synthetic
from util import moving_mean,plot_fft

dynamic_dir = "/scratch/gpfs/yanliang/neid-dynamic"
datadir = "/scratch/gpfs/yanliang/NEID-SOLAR"
device =  torch.device("cpu")

colors = ["k",'b','c','m','orange',"gold",'navy',"skyblue"]
n_colors = len(colors)

def bit_mask(wave,lines,amps,width=0.01):
    profile = np.zeros_like(wave)
    for i,l in enumerate(lines): 
        loc = np.abs(wave-l)<1*width
        profile[loc]=amps[i]
    return profile

def cross_correlation(xobs,yobs,lines_dict,size=100,vstep=200.):
    n = len(yobs)
    velocity = np.arange(-size,size+1,dtype="double")*vstep
    ccf = np.zeros(len(velocity))
    for i,rv in enumerate(velocity):
        rest_lines, amps = lines_dict["loc"],lines_dict["amp"]
        obs_lines = rest_lines*(1+rv/Synthetic.c)
        line_mask = bit_mask(xobs,obs_lines,amps)
        ccf[i] = (line_mask*yobs).sum()/(line_mask**2).sum()
    return ccf,velocity

def auto_corr(x):
    result = np.correlate(x, x, mode='same')
    return result

def auto_correlation(x):
    n = len(x)
    result = np.zeros(n//2)
    for k in range(len(x)//2):
        result[k] = (x[:n-k]*x[k:]).sum()
    return result

def acf_ccf(xobs,yobs,lines_dict,size=100,vstep=200.):
    n = len(yobs)
    velocity = np.arange(-size,size+1,dtype="double")*vstep
    ccf = np.zeros(len(velocity))
    for i,rv in enumerate(velocity):
        rest_lines, amps = lines_dict["loc"],lines_dict["amp"]
        obs_lines = rest_lines*(1+rv/Synthetic.c)
        line_mask = bit_mask(xobs,obs_lines,amps)
        ccf[i] = (line_mask*yobs).sum()/(line_mask**2).sum()
    acf = auto_correlation(ccf)
    return acf,velocity

def gauss(x, *p):
    amp, mu, sigma, b = p
    return amp*np.exp(-(x-mu)**2/(2.*sigma**2))+b

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

def read_neid_L2(filename):
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
    
    SSBRV= get_barycentric_corr_rv(header)
    
    data_dict = {"header":header}
    data_dict["science"] = [science_wavelength,science_flux,science_variance]
    data_dict["science_blaze"] = science_blaze
    data_dict["telluric_model"] = telluric_model
    data_dict["SSBRV"] = SSBRV
    data_dict["CCFRV"] = read_ccf_rv(ccf_header)
    data_dict["CCFRVMOD"] = ccf_header["CCFRVMOD"]
    return data_dict

def read_single_order(filename,order=0,read_keys=['OBSJD','DATE-OBS']):
    hdulist = fits.open(filename)

    header = hdulist[0].header
    ccf_header = hdulist[12].header
    
    science_wavelength = hdulist[7].data[order]
    science_flux = hdulist[1].data[order]
    science_variance = hdulist[4].data[order]
    science_blaze = hdulist[15].data[order]
    #telluric_model = hdulist[10].data[order]

    # Close the FITS file
    hdulist.close()
    
    science = [science_wavelength,science_flux,science_variance]

    SSBRV= get_barycentric_corr_rv(header)
    info_dict = {key:header[key] for key in read_keys}
    info_dict["SSBRV"] = SSBRV[order]
    info_dict["CCFRV"] = read_ccf_rv(ccf_header)[order]
    info_dict["CCFRVMOD"] = ccf_header["CCFRVMOD"]
    return [science,science_blaze],info_dict

def redshift_chi(rv,wave_rest,yrest,wave_obs,ydata,wdata):
    wave_shifted = wave_rest*(1 + rv/Synthetic.c)
    func = CubicSpline(wave_shifted, yrest)
    model_obs = func(wave_obs)
    wmodel = np.ones_like(wdata)
    wmodel[(wave_obs<min(wave_shifted))|(wave_obs>max(wave_shifted))]=0
    loss = np.sum(wmodel*wdata * (ydata - model_obs)**2) / len(ydata)
    return loss

def save_batch(specs,w,aux,IDs,savedir,tag):
    spec = torch.from_numpy(specs.astype(np.float32))
    w = torch.from_numpy(w.astype(np.float32))
    aux = torch.from_numpy(aux.astype(np.double))
    ID = torch.from_numpy(IDs.astype(np.double))
    batch = [spec,w,aux,ID]

    print("spec:",spec.shape,"w:",w.shape,"aux:",aux,"ID",ID.shape)
    filename = os.path.join(savedir, "%s.pkl"%tag)
    print("saving to %s..."%filename)
    with open(filename, 'wb') as f:
        pickle.dump(batch, f)
    return

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

def get_timeseries(colname,sample_names):
    return np.array([neid_dict[key][colname] for key in sample_names])

def simulate_planet(t,amp=0.0, Period=0.11,t0=2459300):
    return amp*np.sin(2*np.pi*(t-t0)/Period)

np.random.seed(0)
torch.manual_seed(0)

tag = sys.argv[1]
select_order = int(sys.argv[2])

wave_obs = Synthetic().wave_obs.numpy()
wave_range = [wave_obs.min(),wave_obs.max()]

n_spec = len(wave_obs)
print("wave_obs:",wave_obs,n_spec)

n_sample = 600
# reading the CSV file
csvFile = pandas.read_csv('SolarRVTable_Level-2.csv')
neid_filenames = csvFile.filename
neid_jd = csvFile.ccfjdsum
neid_ccfrv = csvFile.ccfrvmod
neid_snr =  csvFile.extsnr

sel = np.arange(len(neid_filenames))
#sel = np.arange(3000)

snr_cut = 400
sel = [i for i in sel if neid_snr[i] > snr_cut]
print("total number:",len(sel))
sel = sel[:n_sample]
print("sel:",len(sel))

#sorted(np.random.choice(sel,size=n_sample,replace=False))
sample_names = list(neid_filenames[sel][::-1])
print("sample_names:",len(sample_names))


neid_dict = {}

large_number = 1e6
specmat = np.ones((len(sample_names),n_spec))
errmat = np.zeros((len(sample_names),n_spec))

for i_obs,obsname in enumerate(sample_names):
    data,info_dict = read_single_order("%s/%s"%(datadir,obsname),
                                       order=select_order)
    science,blaze = data
    
    neid_dict[obsname] = info_dict

    wave_raw,flux,flux_var = science

    ssbrv = info_dict["SSBRV"]
    jd = info_dict["OBSJD"]
    
    planetary_rv = simulate_planet(jd)
    neid_dict[obsname]["v_planet"] = planetary_rv

    total_rv = 1e3*ssbrv + planetary_rv
    # ssbrv: transform to heliocentric frame
    wave = wave_raw + wave_raw*(total_rv)/Synthetic.c
    isnan = np.isnan(flux)

    normflux = np.zeros_like(flux)
    normflux_err = np.zeros_like(flux_var)
    
    norm = np.quantile(flux[~isnan]/blaze[~isnan],0.99)
    normflux[~isnan] = (flux/(norm*blaze))[~isnan]
    normflux_err[~isnan] = flux_var[~isnan]**0.5/(norm*blaze)[~isnan]
    
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

    locmask = (wave_obs>min(wave))&(wave_obs<max(wave))&(~bad)
    # Interpolate flux onto wave_obs
    specmat[i_obs][locmask] = interp1d(wave[~isnan], normflux[~isnan], kind='cubic')(wave_obs[locmask])
    errmat[i_obs][locmask] = interp1d(wave[~isnan], normflux_err[~isnan], kind='cubic')(wave_obs[locmask])
    errmat[i_obs][~locmask] = large_number
    specmat[i_obs][~locmask] = 1.0
    
    if i_obs%100==0 or i_obs==(len(sample_names)-1):
        print("neid_dict: %d/%d"%(i_obs,len(sample_names)))

baseline = np.median(specmat,axis=0)
avg_err = np.median(errmat,axis=0)
weight = avg_err**(-2)
sn = baseline/avg_err
print("sn:",sn.min(),sn.max(),"mean sn:",sn.mean())
RV_limit = photon_noise(baseline,wave_obs,sn.mean())
print("RV_limit: %.2f m/s"%RV_limit)

print("bad template",(avg_err>1).sum())
specmat[:,avg_err>1] = 1.0
errmat[:,avg_err>1] = large_number

timestamp = get_timeseries('OBSJD',sample_names)
ssbrvs = get_timeseries('SSBRV',sample_names)
ccfrvs = get_timeseries('CCFRV',sample_names)

print("timestamp:",timestamp)

bervs = -ssbrvs
n_epoch,n_spec = specmat.shape


if "save" in sys.argv:
    save_batch(baseline[None,:],avg_err[None,:]**(-2),
               np.array([bervs.mean()]),np.array([888]),
               dynamic_dir,tag+"-template")
    save_batch(specmat,errmat**(-2),bervs,timestamp,dynamic_dir,tag)



i_plots = np.arange(0,12)
cmap = get_cmap('plasma')
tmin,tmax = min(timestamp[i_plots]),max(timestamp[i_plots])
colors =[cmap((t-tmin)/(tmax-tmin)) for t in timestamp[i_plots]]

'''
i_plots = np.arange(0,12)

fig,ax = plt.subplots(figsize=(15,8),constrained_layout=True)
i_image = 0
for i_obs,obsname in enumerate(neid_dict):
    if not i_obs in i_plots:continue
    offset = i_obs*0.1
    yobs = specmat[i_obs] + offset
    ax.plot(wave_obs,yobs,color=colors[i_image],lw=1)
    ax.fill_between(wave_obs,-1,3,where=(errmat[i_obs]>1),
                    color="grey",lw=1,zorder=-10)
    i_image += 1

ax.plot(wave_obs,baseline,color="k",lw=1)
ax.set_ylim(-0.2,1.1+offset)
plt.savefig("[test]input-spectrum.png",dpi=300)
'''
#mask = (wave_obs>5405)&(wave_obs<5418)
mask = wave_obs<5390
print("mask",mask.sum(),"baseline:",baseline[:20])

base_chi = np.zeros(n_epoch)
base_guess = np.zeros(n_epoch)
for i_epoch,obsname in enumerate(neid_dict):
    result = fit_rv(specmat[i_epoch],errmat[i_epoch]**(-2),baseline,wave_obs)

    base_guess[i_epoch] = result[0]
    base_chi[i_epoch] = result[1]
    label = result[2]
    print("spectrum %d: Planet: %.2f  %s"%(i_epoch,neid_dict[obsname]["v_planet"],label))
    neid_dict[obsname]["v_template"] = base_guess[i_epoch]
    neid_dict[obsname]["chi_template"] = base_chi[i_epoch]

km_m = 1e3
berv_norm = (bervs-np.mean(bervs))*km_m
ccf_norm = (ccfrvs-np.mean(ccfrvs))*km_m
base_guess = base_guess-np.mean(base_guess)

print("base_chi: %.2f, %.2f, %.2f"%(base_chi.min(),base_chi.max(),base_chi.mean()))
berv_ccf_offset = base_guess-ccf_norm
print("$v_{CCF}$: %.3f +/- %.3f m/s"%(ccf_norm.mean(),ccf_norm.std()))
print("$v_{template}$: %.3f +/- %.3f m/s"%(base_guess.mean(),base_guess.std()))
print("$v_{template}-v_{CCF}$: %.3f +/- %.3f m/s"%(berv_ccf_offset.mean(),berv_ccf_offset.std()))


if "save" in sys.argv:
    print("Saving to %s-param.pkl..."%tag)
    with open("%s-param.pkl"%tag,"wb") as f:pickle.dump(neid_dict,f)


fig,axs = plt.subplots(nrows=2,figsize=(10,8),
                       constrained_layout=True,sharex=True)
i_image = 0
for i_obs,obsname in enumerate(sample_names):
    if not i_obs in i_plots:continue
    ccfrv = ccf_norm[i_obs]
    yoffset = ccfrv
    
    date_obs = neid_dict[obsname]['DATE-OBS']
    date = date_obs[5:10]
    time = date_obs[11:16]

    resid_flux = 20*(specmat[i_obs]-baseline)[mask]+yoffset
    resid_error = 20*errmat[i_obs][mask]
    resid_error[resid_error>1]=1
    snr = specmat[i_obs]/errmat[i_obs]
    
    text = "%s(%s) $v_{CCF}$:%.2f m/s $\chi^2=%.2f$"%(time,date,ccfrv,base_chi[i_obs])
    print(text)

    
    axs[0].plot(wave_obs[mask], specmat[i_obs][mask],
                drawstyle="steps-mid",alpha=1,
                c=colors[i_image],label=text)
    axs[1].fill_between(wave_obs[mask],
                        resid_flux-resid_error,
                        resid_flux+resid_error,
                        color=colors[i_image],alpha=0.3)
    
    axs[1].plot(wave_obs[mask],resid_flux,
                drawstyle="steps-mid",alpha=1,
                c=colors[i_image],label=text)
    
    x_text = np.quantile(wave_obs[mask],0.9)
    y_text = resid_flux[-1] + 0.01
    axs[1].text(x_text,y_text,"S/N = %.1f"%snr.mean(),c=colors[i_image],
               bbox= dict(facecolor='w',ec='w', alpha=0.8))
    i_image += 1

axs[0].plot(wave_obs[mask], baseline[mask],drawstyle="steps-mid",lw=1,c="k",label="mean")
axs[0].set_ylabel("normalized flux")
axs[1].set_ylabel("$v_{CCF}$ [m/s]")
#axs[0].set_ylim(0.1,1.05)
axs[0].legend()
plt.savefig("[test]residual-spectrum.png",dpi=300)


fig,axs = plt.subplots(nrows=3,figsize=(5,8),constrained_layout=True)

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
ax.scatter(timestamp,berv_norm,c="grey",s=5,label="$v_{BERV}$")
ax.scatter(timestamp,berv_norm+ccf_norm,c="orange",s=5,label="$v_{BERV}+v_{CCF}$")
ax.scatter(timestamp,berv_norm+base_guess,c="k",s=5,label="$v_{BERV}+v_{template}$")

ax.legend()
ax.set_xlabel("JD")
ax.set_ylabel("Order %d RV [m/s]"%select_order)

ax=axs[2]
ax.scatter(timestamp,base_guess,c="grey",s=5,label="$v_{template}$, RMS = %.2f m/s"%(base_guess.std()))
ax.scatter(timestamp,ccf_norm,c="orange",s=5,label="$v_{CCF}$, RMS = %.2f m/s"%(ccf_norm.std()))

ax.legend(title="Discrepancy RMS = %.2f m/s"%berv_ccf_offset.std())
ax.set_xlabel("JD")
ax.set_ylabel("Order %d RV [m/s]"%select_order)

plt.savefig("[test]sample-selection.png",dpi=300)
plt.clf()

