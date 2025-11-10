#!/usr/bin/env python
# coding: utf-8
import io, os, sys, time, random, re
sys.path.insert(1, './')
import numpy as np
import pickle
import torch
#from torchinterp1d import Interp1d
from synthetic_data import Synthetic
from util import load_batch,interpolate_to_input_grid,normalize_residual,divide_sky_model
from util import load_model,simulate_planet
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.optimize
from scipy.interpolate import interp1d
from scipy.special import digamma
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

data_dir = "/scratch/gpfs/yanliang"

if "cpu" in sys.argv:device =  torch.device("cpu")
else:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device('cuda:2')

def tensor2array(tensor):
    if not torch.is_tensor(tensor):return tensor
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else: return tensor.detach().numpy()

def get_timeseries(timetable,timestamp,colname):
    timeseries = [timetable[t][colname] for t in timestamp]
    return np.array(timeseries)

def get_colors(cdata,cmap_name):
    cmap = plt.get_cmap(cmap_name)
    tmin,tmax = min(cdata),max(cdata)
    colors =[cmap((t-tmin)/(tmax-tmin)) for t in cdata]
    return colors

def plot_wavelength(wave_raw,colors,highlight=[]):
    n_epoch,n_order,N_SPEC = wave_raw.shape
    wave_mean = np.median(wave_raw,axis=0)
    wave_std = np.std(wave_raw,axis=0)

    fig,ax=plt.subplots(figsize=(8,3),constrained_layout=True)
    for o in range(n_order):
        for i in range(n_epoch):
            if i in highlight:zorder=0
            else:zorder=-20
            ax.plot(wave_mean[o],np.abs(wave_raw[i][o]-wave_mean[o]),"-",c=colors[i],zorder=zorder)
        if wave_std[o].max()>0.01:c="r"
        else: c="b"
        ax.plot(wave_mean[o],wave_std[o],label="order %d"%o,c=c)
    ax.legend(ncols=2,loc="upper left")
    ax.set_xlabel("wavelength")
    ax.set_ylabel("wavelength dispersion")
    plt.savefig("diagnostic.png",dpi=300)
    '''
    wave_mean = np.mean(wave_raw,axis=0,keepdims=True)
    wave_std = (wave_raw-wave_mean).std(axis=-1).T
    print("wave_std:",wave_std.shape,wave_std)

    for i in range(3):
        plt.hist(wave_std[i],label="order %d"%i,log=True)
        whmax = np.argmax(wave_std[i])
        print("index: %.3f  wave std: %.4f"%(whmax,wave_std[i][whmax]))
    plt.legend()
    plt.savefig("diagnostic.png",dpi=200)
    '''
    return

def find_deepest_lines(wave_obs, raw_spectrum, num_lines=100, min_separation=0.30,return_ind=False):
    spectrum = raw_spectrum/np.quantile(raw_spectrum,0.99)
    depth = 1 - spectrum
    sorted_indices = np.argsort(depth)[::-1]  # Indices of depths sorted from largest to smallest

    # Start with the deepest line
    unique_indices = [sorted_indices[0]]

    # Iterate over the sorted indices and dataset_tag peaks with the required minimum separation
    for index in sorted_indices[1:]:
        # Check if this index is sufficiently far from all previously dataset_taged peaks
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


def visualize_spectrum(spec,spec_resid,obsname_star,template,vel):
    n_epoch,n_order,N_SPEC = spec.shape
    dispersion = spec_resid.std(axis=0)
    mask = np.arange(6700,7000)
    deeplines = [find_deepest_lines(wave_obs[i_order] [mask],template[i_order][mask], num_lines=5) for i_order in range(n_order)]
    v_template = vel["v_template"]
    v_trad = vel["v_trad"]
    
    rank = np.argsort(v_template.mean(axis=-1))
    index_samples = np.hstack((rank[:3],rank[-3:]))
    print("index_samples:",index_samples)

    fig,axs = plt.subplots(ncols=3,nrows=3,figsize=(22,10),constrained_layout=True)
    for k in range(3):
        for i_order,o in enumerate(orders):
            if k==2:colors=get_colors(v_template[:,i_order],"rainbow") 
            else: colors=get_colors(water_vapor,"plasma")

            ax = axs[i_order][k]
            for i_obs in index_samples:
                obsname = obsname_star[i_obs]
                date_obs = neid_dict[obsname]['DATE-OBS']
                date = date_obs[5:10]
                time = date_obs[11:16]
                text = "%.2f $v_{template}$:%.2f m/s $\chi^2=%.2f$"%(neid_dict[obsname]['timestamp'],v_template[i_obs][i_order],chi_template[i_obs][i_order])
                #print(text,obsname)
                if k==0:y_smooth=spec[i_obs][i_order][mask]
                else:y_smooth = gaussian_filter1d(spec_resid[i_obs][i_order][mask],1)
                ax.plot(wave_obs[i_order][mask], y_smooth,drawstyle="steps-mid",alpha=1,c=colors[i_obs],label=text)
            ax.plot(wave_obs[i_order][mask], dispersion[i_order][mask],drawstyle="steps-mid",lw=1,c="k",label="dispersion")
            for line in deeplines[i_order]:ax.axvline(line[0],ls="--",color="k")
            #whmax = np.argmax(dispersion[i_order])
            #ax.set_xlim(wave_obs[i_order][whmax]-5,wave_obs[i_order][whmax]+5)
            ax.set_ylabel("normalized flux")
            ax.set_title("Order %d"%o)
            if k==0:ax.legend()
    plt.savefig("[%s]residual-spectrum.png"%(dataset_tag),dpi=300)
    return

def redshift_chi(rv,wave_model,yrest,weight_rest,wave_data,ydata,wdata):
    wave_shifted = wave_model*(1 + rv/Synthetic.c)
    bad = yrest==0
    mask = (wave_data>min(wave_shifted[~bad]))&(wave_data<max(wave_shifted[~bad]))
    model_obs = interp1d(wave_shifted[~bad], yrest[~bad])(wave_data[mask])
    loss = wdata[mask]* (ydata[mask] - model_obs)**2
    loss = np.sum(loss)/len(ydata[mask])
    return loss

def fit_rv(args,mdict):
    i,wave,spec,w,wave_rest,rest_model,weight_model = args
    result = scipy.optimize.minimize(redshift_chi,0.1, method='Nelder-Mead',args=(wave_rest,rest_model,weight_model,wave,spec,w,))
    label = "RV_fit=%.4f $\chi^2$:%.4f"%(result.x,result.fun)
    print(label)
    mdict[i] = {"v_bestfit":result.x,"chi_bestfit":result.fun}
    return 0

import multiprocessing as mp
def measure_velocities(wave_data,spec_data,w,wave_rest,rest_model):
    num_cores = 10
    batch_size = spec_data.shape[0]
    v_bestfit = np.zeros((batch_size))
    chi_bestfit = np.zeros((batch_size))
 
    manager = mp.Manager()
    mdict = manager.dict()
    # Use a pool of workers
    pool_size = num_cores  # Number of processes in the pool
    pool = mp.Pool(pool_size)
    tasks = []
    for i in range(batch_size):
        task_args = (i,wave_data,spec_data[i],w[i],wave_rest,rest_model[i],None)
        tasks.append(task_args)

    for i,task in enumerate(tasks):
        pool.apply_async(fit_rv, args=(task, mdict))
    # Close and join the pool
    pool.close()
    pool.join()

    for i in range(batch_size):
        if i in mdict:
            v_bestfit[i] = mdict[i]["v_bestfit"]
            chi_bestfit[i] = mdict[i]["chi_bestfit"]
    return v_bestfit,chi_bestfit

def process_spectra(model,instrument,batch,template=None,skymask=None,full=False,v_planet=0):

    #v_trad,spec_raw,w,ssbrv,jd = batch
    ccf_info,spec_raw,w,ssbrv,jd = batch
    v_trad = ccf_info[:,[0]]

    # inject planet
    _,v_planet = simulate_planet(jd,amp=planet_amp,
                                 period=planet_period,
                                 phase_t0=t0)


    template = template_data[1]
    if ssbrv.ndim==1:ssbrv = ssbrv.unsqueeze(1)
    if jd.ndim==1:jd = jd.unsqueeze(1)

    z_loss = 0
    rv = v_trad + v_planet 

    # replace rv estimator with v_apparent
    indices = torch.searchsorted(aux_data[0].contiguous(), jd)
    v_apparent = aux_data[1][indices]
    v_offset = aux_data[2][indices]
    quality_mask = aux_data[3][indices][:,0].bool()

    #rv = v_apparent
    print("v_apparent:",v_apparent.shape,"v_trad:",v_trad.shape)
    print(f"[Pretrained RV] NN vs. Trad Difference: {(v_apparent-v_trad-v_planet).std():.3f} m/s")

    print("rv:",rv.shape,rv.min().item(),rv.max().item())

    # testing new idea: shift spectra back to zero rv
    spec_zero_rv, w, _ = interpolate_to_input_grid(batch,instrument,template,extra_rv=(-v_apparent+v_planet))

    s = model.encode(spec_zero_rv)
    y_act = model.decode(s)
    spectrum_observed = model.decoder.spec_rest+y_act

    v_act = model.activity_estimator(s)
    v_doppler = model.estimate_doppler_rv(jd)

    v_doppler = v_doppler.T
    # normalize residual model
    model_resid = spectrum_observed-template
    model_resid[w<1.0] = 0


    # compare residual model
    fid_loss = model._loss(spec_zero_rv, w, model_resid, individual=True)
    
    #v_offset = torch.zeros_like(rv)
    v_doppler_sum = v_doppler.sum(dim=1)[:,None]
    v_resid = v_apparent-v_offset-v_act-v_doppler_sum

    print("fid_loss:",fid_loss.min().item(),fid_loss.max().item())
    print(f"Initial RMS:{v_apparent[quality_mask].std().item():.3f} m/s")
    print(f"Residual RMS:{v_resid[quality_mask].std().item():.3f} m/s")

    #v_activity = model.estimate_v_act(s)
    v_ssb = (ssbrv).mean(dim=-1)
    #if s_sky is None: spec_sky = torch.ones_like(jd)

    info = {"ids":jd,"ssbrv":v_ssb,"s":s,#"s_aug":s_aug,
            'v_template': ccf_info[:,[0]],
            'v_ccf': ccf_info[:,[1]],'features':ccf_info[:,2:],
            "v_apparent":v_apparent,"v_act":v_act,
            "v_doppler":v_doppler,"v_offset":v_offset,
            "loss":fid_loss,"v_planet":v_planet}

    if full:
        output = {"spec":spec_zero_rv+template,
                  "w":w,"model":model_resid+template}
    else: output = {}

    output.update(info)
    output = {key:tensor2array(item[quality_mask]) for key,item in output.items()}
    return output


def calculate_v_apparent(model,instrument,batch,template=None,v_planet=0):

    ccf_info,spec_raw,w,ssbrv,jd = batch
    v_trad = ccf_info[:,[0]]

    template = template_data[1]
    if ssbrv.ndim==1:ssbrv = ssbrv.unsqueeze(1)
    if jd.ndim==1:jd = jd.unsqueeze(1)

    #rv = v_trad + v_planet 

    spec_input, w, _ = interpolate_to_input_grid(batch,instrument,template,extra_rv=v_planet)
    rv,rv_err = model.estimate_rv(spec_input)
    print(f"\nrv:  [{rv.min():.2f},{rv.max():.2f}] RMS={rv.std():.3f}m/s")
    print(f"rv_trad: RMS={(v_trad+v_planet).std():.3f}m/s")
    print(f"NN vs. Trad Difference: {(rv-v_trad-v_planet).std():.3f} m/s")
 
    print("rv:",rv.shape,rv.min().item(),rv.max().item())


    info = {"ids":jd,'v_template': ccf_info[:,[0]],
            'v_ccf': ccf_info[:,[1]],'features':ccf_info[:,2:],
            "v_encode":rv,"v_err":rv_err,
            "v_planet":v_planet}

    output = {key:tensor2array(item) for key,item in info.items()}
    return output

def get_info(timetable,timestamp):
    info = {}
    colnames = [#'v_template','chi_template',#'CCFRV',#
                'obsname','WVAPOR','ZENITH']
    for name in colnames:
        info[name] = get_timeseries(timetable,timestamp,name)
    return info

def k_means_clustering(S, num_clusters = 5):
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(S)
    # Get the cluster labels
    labels = kmeans.labels_
    # Get the cluster centers
    centers = kmeans.cluster_centers_
    return labels

def cluster_mean_spectra(labels,ydata):
    y_avg_norm = ydata.mean(axis=0)
    uniq_labels = sorted(np.unique(labels))
    y_eigen = {}
    for label in uniq_labels:
        wh = labels==label
        y_mean = ydata[wh].mean(axis=0)-y_avg_norm
        y_eigen[label] = y_mean
    return y_eigen

def correct_velocity_offset(aux_file, threshold=800):
    t, v, v_offset = torch.load(aux_file)
    quality = t>0
    return torch.stack([t, v, v_offset, quality])

def save_latents(data,fname):
    time = data['ids']
    s = data['s']
    s_data = np.hstack((time,s))
    print("s_data:",s_data.shape)

    sort_ind = np.argsort(s_data[:,0])
    s_data = s_data[sort_ind]
    s_data = torch.from_numpy(s_data.astype(np.float32)).T
    latents_path = f"{dynamic_dir}/aux/{fname}_latents.pkl"
    print("saving to:",latents_path)
    torch.save(s_data,latents_path)
    return

#-------------------------------------------------------
import matplotlib
dynamic_dir = "/scratch/gpfs/yanliang/neid-production"


model_file = sys.argv[1]
output_dir = "summary_file"

stellar_pattern = r'period(\d+\.\d+)d_K(\d+\.\d+)m_phase(\d+\.\d+)_(\w+).pt'

basename = os.path.basename(model_file)
if not "purez" in model_file:
    trim = "_".join(basename.split("_")[:-1])
    aux_data = correct_velocity_offset(f"{dynamic_dir}/aux/{trim}_v_apparent_full.pkl")
    mode = "fid"
else: 
    aux_data = torch.zeros((1,4))
    mode = "purez"
out_file = os.path.join(output_dir,re.sub(r'\.pt$', f'_{mode}_sum.pkl', basename))
print("\n-- ",out_file," --\n")

if re.search(os.path.basename(stellar_pattern), model_file): 
    # Searc4h for the pattern in the file path
    match = re.search(stellar_pattern, model_file)
    # Extract the numbers as a list of integers
    planet_period = float(match.group(1))
    planet_amp = float(match.group(2)) # m/s
    t0 = float(match.group(3))

else:
    print("Pattern not found")
    # Extract the numbers as a list of integers
    planet_period = 3.162#80.1
    planet_amp = 0.5
    t0 = 0.0

#dataset_tag = f"{tag}_order{order_value}"
#dataset_tag = f"safe_full"
dataset_tag = f"newprod_full"

print("planet_period,planet_amp,t0:",planet_period,planet_amp,t0 )

template_data = load_batch("%s/merge/%s-template.pkl"%(dynamic_dir,dataset_tag))


#template_data = load_batch("%s/%s-template.pkl"%(dynamic_dir,dataset_tag))
wave_obs = template_data[0]
template = template_data[1]
wave_obs = wave_obs.to(device=device)
instrument = Synthetic(wave_obs)
aux_data = aux_data.to(device=device)
model, _, _ = load_model("%s"%(model_file),instrument, device)

template_data = [item.to(device=device) for item in template_data]
print("template_data:",template_data[1].min(),template_data[1].max())

files = ["%s/ccf_info/%s"%(dynamic_dir,ii) for ii in os.listdir(f"{dynamic_dir}/ccf_info") if ii.startswith(dataset_tag) and bool(re.search(r'_\d+.pkl$', ii))]


param_name = f"{dynamic_dir}/params/{dataset_tag}-param.pkl"
with open(param_name,"rb") as f:
    neid_dict = pickle.load(f)
sample_names = neid_dict["info"]["sample_names"]
print("neid_dict[info]:",neid_dict[sample_names[0]].keys())


for item in sample_names:neid_dict[item]["obsname"]=item
# Create a time-to-parameters mapping
timetable = {np.float32(neid_dict[item]["timestamp"]): neid_dict[item] for item in sample_names}

try:
    skymask = load_batch("%s/%s-skymask.pkl"%(dynamic_dir,dataset_tag)).bool()
    if skymask.ndim==1:skymask=skymask.unsqueeze(0)
except:skymask=None
#'''

full=False
summary = {'info':{}}


tlin = torch.linspace(200,1500,2000,device=instrument.wave_obs.device)[:,None]
quasi_terms = model.doppler_model.get_quasi_periodic_terms(tlin)


batch_size = 20
for batch_name in files:
    print("batch_name:",batch_name)
    batch = load_batch(batch_name)
    batch = [item.to(device=device) for item in batch]
    N = len(batch[0])
    sections = np.arange(batch_size,N,batch_size)
    print(sections)
    idxs = np.split(np.arange(N),sections)
    for idx in idxs:
        print("%d~%d"%(idx[0],idx[-1]))
        batch_i = [item[idx] for item in batch]
        timestamp = tensor2array(batch_i[4][:,0])

        # inject planet
        phase,v_planet = simulate_planet(batch_i[4],amp=planet_amp,period=planet_period,phase_t0=t0)

        info_dict = get_info(timetable,timestamp)
        
        if mode=="fid":
            view_dict = process_spectra(model,instrument,batch_i,template=template,full=full,v_planet=v_planet)
        elif mode=="purez":
            view_dict = calculate_v_apparent(model,instrument,batch_i,template=template,v_planet=v_planet)

        if not "data" in summary:
            summary["info"] = info_dict
            summary["data"] = view_dict
            continue
        for key in summary["data"]:
            #if key in ["spec_sky","spec_input"]:continue
            summary["data"][key] = np.concatenate((summary["data"][key],view_dict[key]),axis=0)
        for key in summary["info"]:
            summary["info"][key] = np.concatenate((summary["info"][key],info_dict[key]),axis=0)

summary['info']['planet_params'] = planet_period,planet_amp,t0
summary['info']['wave_obs']= tensor2array(wave_obs)
summary['info']['template']= tensor2array(template_data[1])

summary['info']['planet_solution'] = tensor2array(model.doppler_model.planet_params)
summary['info']['period_solution'] = tensor2array(model.doppler_model.get_periods())

summary['info']['quasi_terms']= tensor2array(quasi_terms)

if mode =='fid':
    save_latents(summary['data'],trim)


#cluster_labels = k_means_clustering(summary["data"]["s"])
#summary["data"]["cluster_labels"] = cluster_labels


with open(out_file,"wb") as f:
    pickle.dump(summary,f)

print("\n-- ",out_file," --\n")