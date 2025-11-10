#!/usr/bin/env python
# coding: utf-8
import torch,os,sys,pickle,re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KDTree
from scipy.optimize import minimize

from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from util import load_batch,moving_mean
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def get_colormap(cdata,cmap,vmin=0,vmax=1):
    cdata[cdata<vmin] = vmin
    cdata[cdata>vmax] = vmax
    mycmap = plt.get_cmap(cmap)
    normalized_cdata = (cdata-cdata.min())/(cdata.max()-cdata.min())
    colors = [mycmap(ii) for ii in normalized_cdata]
    return colors

def smooth_kernel(points,target,sigma=0.5,fraction=0.1,n_radius=30,
                  self_weight=0.0):
    tree = KDTree(points)
    target_smooth = np.zeros_like(target)
    target_dispersion = np.zeros_like(target)

    k_neighbor = max(int(fraction*len(target)),n_radius)
    distance,neighbors = tree.query(points,k=k_neighbor)
    weight_neighbor =  np.zeros_like(target)
    num_neighbor = np.zeros_like(target)

    dsquare = np.sum((points[neighbors] - points[:,None,:])**2,axis=-1)
    sigma *= points.std()
    #print("sigma:",sigma,"default:",np.median(dsquare[:,:n_radius]**0.5))
    weights = np.exp(-0.5*dsquare/sigma**2)
    weights[dsquare==0.]=self_weight
    for i,n in enumerate(neighbors):
        close = weights[i]>0.001
        if close.sum()==0:continue
        target_nb = target[n][close]
        target_smooth[i] = np.average(target[n,0],weights=weights[i])
        target_dispersion[i] = np.std(target_nb)
        weight_neighbor[i] = weights[i].sum()
    return target_smooth,target_dispersion,weight_neighbor

def simulate_planet_np(t,amp=1,period=0.11,phase_t0=0,t0=800):
    #phase = ((t/period)-phase_t0)%1
    phase = (((t-t0)/period)+phase_t0)%1
    v_planet = amp*np.sin(2*np.pi*phase)
    return phase,v_planet

def planet_param_chi(param,time,v_obs,v_err,p0=None,full=False):
    amp,period,phase = param[:3]
    _,v_planet = simulate_planet_np(time,amp=amp,period=period,phase_t0=phase)
    loss = np.mean((v_obs-v_planet)**2/v_err**2)
    if full: 
        plin,cov = np.polyfit(v_planet,v_obs,deg=1,cov=True)
        slope,b = plin
        label=f"slope = {slope:.2f}+/- {cov[0][0]**0.5:.2f}"
        print(label)
        return v_planet,loss
    if p0 is not None:
        delta = 0.1
        p_guess = p0[1]
        loss += (period/p_guess-1)**2/delta**2
    return loss

def initial_guess(period,time,v_ccf,v_err):
    phase_grid = np.arange(0,1,0.05)
    amp_grid = np.arange(0.1,0.8,0.1)
    chi_best = np.inf

    for ph in phase_grid:
        for a in amp_grid:
            p_guess = [a,period,ph]
            chi = planet_param_chi(p_guess,time,v_ccf,v_err)
            if chi<chi_best:
                best_guess = p_guess
                chi_best = chi
                #print(p_guess,"chi:",chi)
    return best_guess,chi_best

def print_params(params):
    param_names = ["K [m/s]  ","Period [d]",  "Phase    "]
    print_p = np.copy(params)
    string = ""
    for i in range(len(params)):
        string += f"{param_names[i]}: {print_p[i]:.4f}\n"
    return string

def tensor2array(tensor):
    if not torch.is_tensor(tensor):return tensor
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else: return tensor.detach().numpy()

def calculate_harmonics(P):
    harmonics = []
    for num in range(1, 3):  # Numerators: 1, 2, 3
        for denom in range(1, 3):  # Denominators: 1, 2, 3
            value = (num / denom) * P
            if denom == 1:
                fraction_str = f"{num}P" 
            elif denom==num:continue
            else:
                fraction_str = f"{num}/{denom}P"
            harmonics.append((value, fraction_str, P))

    # Sort by numerical value of the harmonics
    harmonics.sort()
    return harmonics

def detect_ls_peaks(per,power,n_peaks = 25,frac=0.1,delta=0.1):
    rank = np.argsort(power)[::-1]
    top_n_periods = np.zeros((n_peaks))
    top_n_power = np.zeros((n_peaks))

    i = 0
    for rk in rank:
        per_max = per[rk]
        if np.abs(per_max - 1)<0.02:continue
        if np.abs(per_max - 0.5)<0.02:continue
        nonzero = top_n_periods>0
        if nonzero.sum()>0:
            if (np.abs(top_n_periods[nonzero]-per_max)/per_max).min()<frac:continue
            if np.abs(top_n_periods[nonzero]-per_max).min()<delta:continue
        top_n_periods[i] = per_max
        top_n_power[i] = power[rk]
        i+=1
        #harmoniscs.extend(calculate_harmonics(per_max))
        if (top_n_periods>0).sum()>=n_peaks:
            print("break!")
            break
    mask = (top_n_periods>0)&(top_n_periods<350)
    top_n_periods = top_n_periods[mask]
    top_n_power = top_n_power[mask]
    return top_n_periods, top_n_power

def remove_close_periods(periods,powers,threshold=0.05):
    # Sort the power in ascending order
    rank = np.argsort(powers)[::-1]
    periods = periods[rank]

    # Initialize the filtered list with the first period
    filtered_periods = [periods[0]]

    for p in periods[1:]:
        # Check if the new period is sufficiently different from all previously kept ones
        if all(abs(p - fp) > threshold * fp for fp in filtered_periods):
            filtered_periods.append(p)

    return filtered_periods

def check_alignment_harmonics(f_aestra,P_activity,periods,tol=0.03):
    harmonics = []
    for p in P_activity:
        harmonics.extend(calculate_harmonics(p))

    comments = {}
    good_periods =  periods>0
    for p in P_activity:
        if (np.abs(periods-p)/p).min()>tol:continue
        # check main period
        wh = np.argmin(np.abs(periods-p)/p)
        main_power = f_aestra(periods[wh])
        print(periods[wh],"main_power:",main_power)
        good_periods[wh] = False
        comment =  f"1P harmonic with {p:.2f}d"
        comments[periods[wh]] = comment
        print(f"tol:{tol} {periods[wh]:.2f}d {comment}")
        # if aligns with main period, check for secondaries
        harmonics = calculate_harmonics(p)
        print("harmonics:",harmonics)
        for item in harmonics:
            if (np.abs(periods-item[0])/item[0]).min()>tol:continue
            wh = np.argmin(np.abs(periods-item[0])/item[0])
            sub_power = f_aestra(periods[wh])
            print("checking period",item[0])
            # check if the ratio is reasonable
            if sub_power>0.8*main_power:
                print("secondary peak too strong, chance alignment!")
                continue
            print(periods[wh],"sub_power:",sub_power)
            good_periods[wh] = False
            comment =  f"{item[1]} harmonic with {item[2]:.2f}d"
            comments[periods[wh]] =comment
            print(f"tol:{tol} {periods[wh]:.2f}d {comment}")
    return comments,good_periods

def get_timeseries(neid_dict,colname,keys):
    vector = np.array([neid_dict[key][colname] for key in keys])
    print(colname,vector.shape)
    return vector

def weighted_average_time_series(df):
    # Compute weights = 1 / (sigma^2)
    df['weight'] = 1.0 / (df['velocity_error'] ** 2)

    # Group by time and compute weighted average and its error
    grouped = df.groupby('time').apply(
        lambda g: pd.Series({
            'velocity': np.average(g['velocity'], weights=g['weight']),
            'velocity_error': np.sqrt(1.0 / g['weight'].sum())
        })
    ).reset_index()
    return grouped

def sinusoidality(t, y, yerr, P, K=5):
    # phase fold
    phi = (t % P) / P
    w = 1.0 / yerr**2

    # design matrix: cols = [1, cos1, sin1, cos2, sin2, ...]
    cols = [np.ones_like(phi)]
    for k in range(1, K+1):
        cols.append(np.cos(2*np.pi*k*phi))
        cols.append(np.sin(2*np.pi*k*phi))
    X = np.vstack(cols).T

    # weighted linear least squares
    W = np.diag(w)
    XtW = X.T * w  # broadcast
    beta = np.linalg.solve(XtW @ X, XtW @ y)

    C = beta[0]
    ak = beta[1::2][:K]  # cos terms
    bk = beta[2::2][:K]  # sin terms

    A = np.sqrt(ak**2 + bk**2)
    S2 = A[0]**2 / np.sum(A**2)
    return S2#, A, C, ak, bk


def polynomial_detrend(time, vel, features, tgrid = [0,800,1600], degs=[2,1]):
    v_detrend = np.copy(vel)
    for i in range(len(tgrid)-1):
        mask = (time>tgrid[i])&(time<=tgrid[i+1])
        vlin = linear_detrend(vel[mask],input_features[mask])
        #v_detrend[mask] -= vlin
        poly = np.polyfit(time[mask],v_detrend[mask],deg=degs[i])
        v_detrend[mask] -= np.polyval(poly,time[mask])
    return v_detrend

def linear_detrend(v_power,input_features):
    # Example inputs
    rv = v_power[:,None]
    X = input_features 
    # Add intercept (constant term)
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])

    # Solve for coefficients via least squares
    coeffs, *_ = np.linalg.lstsq(X_design, rv, rcond=None)
    # Predicted trend
    rv_trend = X_design @ coeffs
    return rv_trend[:,0]

def save_to_auxfile(aux_data, filepath, datadir="/scratch/gpfs/yanliang/neid-production"):

    print("aux_data:",aux_data.shape)
    sort_ind = np.argsort(aux_data[:,0])
    aux_data = aux_data[sort_ind]
    aux_data = torch.from_numpy(aux_data.astype(np.float32)).T
    print("aux_data:",aux_data.shape)
    
    torch.save(aux_data,f"{datadir}/{filepath}")
    return

def get_info(timetable,timestamp):
    info = {}
    colnames = ['v_template','chi_template',
                'obsname','WVAPOR','ZENITH']
    for name in colnames:
        info[name] = get_timeseries(timetable,timestamp,name)
    return info

def pack_ccf_info(tag="safe"):
    sum_file = f"/scratch/gpfs/yanliang/neid-production/params/{tag}_full-param.pkl"
    with open(sum_file,"rb") as f:
        neid_dict = pickle.load(f)
    samples = neid_dict['info']['sample_names']
    files = neid_dict['info']['ccf_files']
    ccf_info = []
    time = []
    print(f"{len(files)} files:",files[:5])
    for i,batch_name in enumerate(files):
        batch = load_batch(batch_name)
        ccf_info.append(batch[0])
        time.append(batch[4])
        if (i%10)==0:print(f"{i} files...")
    ccf_info = tensor2array(torch.cat(ccf_info))
    time = tensor2array(torch.cat(time))
    v_template = ccf_info[:,[0]]
    v_ccf = ccf_info[:,[1]]
    features = ccf_info[:,2:]
    with open(f"{tag}_ccf_summary.pkl", "wb") as f:
        pickle.dump([time,v_template,v_ccf,features],f)
    print("features:",features.shape,"v_template",v_template.shape)
    return

def read_ccf_info(tag="safe"):
    with open(f"{tag}_ccf_summary.pkl", "rb") as f:
        info = pickle.load(f)
    return info
    
def find_bestfit_planet(P, time, v, v_err=1):
    string = "\n\n"
    p_guess,chi_guess = initial_guess(P,time,v,v_err)
    args = (time,v,v_err,p_guess)
    res = minimize(planet_param_chi, p_guess, method='Nelder-Mead', 
                   tol=1e-6, args=args)
    pfit = res.x
    bestchi = res.fun

    string += f"Period: {pfit[1]:.3f}d chi: {bestchi:.4f}\n"
    string += print_params(pfit)
    print(string)
    return string
   

def filter_unique_periods(all_periods,frac=0.05):
    uniq_periods = np.zeros_like(all_periods)
    for i,P in enumerate(all_periods):
        if np.abs(uniq_periods/P-1).min()<frac:continue
        else:uniq_periods[i]=P
    print("uniq_periods:",uniq_periods)
    return uniq_periods>0

'''
planet_pattern = r'period(\d+\.\d+)d_K(\d+\.\d+)m_phase(\d+\.\d+)'
if re.search(planet_pattern, basename): 
    # Search for the pattern in the file path
    match = re.search(planet_pattern, basename)
    # Extract the numbers as a list of integers
    planet_period = float(match.group(1))
    planet_amp = float(match.group(2)) # m/s
    t0 = float(match.group(3))
    #suffix = match.group(5)

else:
    print("Pattern not found")
    # Extract the numbers as a list of integers
    planet_period = 80.1
    planet_amp = 0.0
    t0 = 0.0
'''



'''
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors[2] = 'k'
truth = [0.0,16.228,0.6]
max_f = 1/1.1

colors = ["r","skyblue","b"]
ls = ["-","--","-.",":","-","--","-"]
lw = [0.5,0.5,0.5,0.5,1.5,1.5,4]
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
for k,tag in enumerate(["safe","safeblue","extremblue"]):
    time,v_template,v_ccf,features = read_ccf_info(tag)
    time = time[:,0]
    v_template = v_template[:,0]

    ph,v_planet = simulate_planet_np(time,*truth)
    v_inject = v_template + v_planet
    t_seg = [0,500,800,1200,1600]
    masks = [(time>t_seg[k])&(time<t_seg[k+1]) for k in range(len(t_seg)-1)]

    
    #masks = [~m for m in masks]
    #masks = [(m1 | m2) for m1, m2 in itertools.combinations(masks, 2)]
    for mask in masks:
        vlin = linear_detrend(v_inject[mask],features[mask])
        poly = np.polyfit(time[mask],v_inject[mask],deg=1)
        v_poly = np.polyval(poly,time[mask])
        v_inject[mask] -= v_poly

    labels = [f"{tag} n={m.sum()}" for m in masks]
    all_powers = []
    all_periods = []
    for mask in masks:
        #frequency = 1.0/per
        base=time[mask].max()-time[mask].min()
        min_f = 2/base
        max_f = 30/base
        freq,power = LombScargle(time[mask], v_inject[mask]).autopower(minimum_frequency=min_f,maximum_frequency=max_f,samples_per_peak=10)
        
        find_bestfit_planet(truth[1], time[mask], v_inject[mask])
        
        all_powers.append([1/freq,power])
        top_n_periods,top_n_power = detect_ls_peaks(1/freq,power)
        print("top_n_periods:",top_n_periods)
        all_periods.append(top_n_periods)

    for j,power in enumerate(all_powers):
        ax.semilogx(power[0],power[1],alpha=1.0,lw=lw[j],c=colors[k],
                    label=labels[j],ls=ls[j])
        for period in all_periods[j]:
            ax.axvline(period,lw=1,color='k',ls="-",alpha=0.3,zorder=-10)

ax.axvline(truth[1],lw=8,color="gold",alpha=0.5,zorder=-20,
           label=f"Truth P={truth[1]:.2f}d")
ax.legend(loc="upper left")
ax.set_ylim(0,0.22)
ax.set_xlabel("Period [days]");ax.set_ylabel("Power")
plt.savefig(f"test.png",dpi=200)
plt.clf()
'''
#basename="period3.162d_K0.1m_phase0.0"
#v_long_term = v_inject-v_power
#aux_data = np.vstack((time,v_inject,v_long_term)).T
#filepath = f"{datadir}/aux/{basename}_v_apparent.pkl"
#save_to_auxfile(aux_data, filepath)

#pack_ccf_info('newprod')
#exit()
n_planet = 15
deg = 3
np.random.seed(0)
planet_amp = np.ones(n_planet)*0.0
inject_t0 = np.round(np.random.uniform(0,1,size=(n_planet)),1)
inject_periods = np.round(np.logspace(0.5,2.5,n_planet),3)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i_count,planet_period in enumerate(inject_periods):
    #if not planet_period>200:continue
    #if not planet_period<90:continue
    #if not planet_period==61.054:continue
    truth = [planet_amp[i_count],planet_period,inject_t0[i_count]]
    basename = f"period{truth[1]:.3f}d_K{truth[0]:.1f}m_phase{truth[2]:.1f}"
    outdir = "initial_guess"
    output_txt = f"{outdir}/{basename}_debug.txt"
    #output_txt = f"{outdir}/{basename}_fine.txt"
    
    truth_text = f"Truth:   P={truth[1]:.2f}d   K={truth[0]:.2f}m/s   ph={truth[2]:.1f}"
    print("planet:",truth)
    times = []
    v_powers =[]
    all_powers = []
    all_periods = []
    peak_quality = []


    #labels = ['before','after']
    #colors = ['b','r']
    for k,tag in enumerate(["newprod"]):#["safe","extremblue"]):
        time,v_template,v_ccf,features = read_ccf_info(tag)
        time = time[:,0]
        v_template = v_template[:,0]
        v_ccf = v_ccf[:,0]

        ph,v_planet = simulate_planet_np(time,*truth)
        #v_inject = v_template + v_planet
        v_inject = v_ccf + v_planet
        v_power = np.zeros_like(v_inject)
        v_err = np.ones_like(v_template)

        t_seg = [0,500,800,1200,1600]
        #t_seg  = [0,800,1600]
        dmasks = [(time>t_seg[k])&(time<t_seg[k+1]) for k in range(len(t_seg)-1)]
        
        for j,mask in enumerate(dmasks):
            t = time[mask]
            v = v_inject[mask]
            input_features = features[mask]
            print(f"v_inject RMS={(v).std():.2f}m/s")
            vlin = linear_detrend(v,input_features)
            #poly = np.polyfit(t,v-vlin,deg=3)
            #v_poly = np.polyval(poly,t)
            v_power[mask] = v-vlin
            print(f"v_power RMS={v_power[mask].std():.2f}m/s")
        
        v_long_term = v_inject-v_power
        aux_data = np.vstack((time,v_inject,v_long_term)).T
        filepath = f"aux/{basename}_v_apparent_full.pkl"
        print(f"Saving to {filepath}...")
        save_to_auxfile(aux_data, filepath)
        exit()
        break
        
        t_seg  = [0,800,1600]
        masks = [(time>t_seg[k])&(time<t_seg[k+1]) for k in range(len(t_seg)-1)]
        per_range = [[1.5,340] for m in masks]
        n_peaks = [20 for m in masks]

        masks += [time<1600]
        per_range += [[1.5,340]]
        n_peaks += [30]
        
        print(f"v_power RMS={(v_power).std():.2f}m/s")
        for j,mask in enumerate(masks):
            t = time[mask]
            v = v_power[mask]
            min_p,max_p = per_range[j]
            print("min_p,max_p",min_p,max_p)
            freq,power = LombScargle(t, v).autopower(minimum_frequency=1/max_p,maximum_frequency=1/min_p,samples_per_peak=10)

            top_n_periods,top_n_power = detect_ls_peaks(1/freq,power,n_peaks=n_peaks[j])
            print("top_n_periods:",top_n_periods)
            all_periods.append(top_n_periods)
            all_powers.append([1/freq,power,per_range[j]])
            
            peak_quality.append(top_n_power)
            times.append(t)
            v_powers.append(v)            
    continue
    #------------------ plotting ------------------
    labels = [int(ii+1) for ii in range(len(all_powers))]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    n_inst = len(times)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for j,power in enumerate(all_powers):
        lw = min(power[0].max()/15,2)
        ax.semilogx(power[0],power[1],'-',alpha=1.0,lw=lw,label=labels[j])
        for p in all_periods[j]: ax.axvline(p,ls="-",color="lightsteelblue",alpha=0.3,zorder=-10)
    #ax.legend()
    ax.axvline(planet_period,lw=8,color="gold",alpha=0.5,
               zorder=-20,label="Truth")
    ax.set_xlabel("Period [days]");ax.set_ylabel("Power")
    plt.title(truth_text,weight="bold")
    plt.savefig(f"{outdir}/guess_{basename}.png",dpi=200)
    plt.clf()
    #exit()

    all_periods = np.concatenate(all_periods)
    peak_quality = np.concatenate(peak_quality)
    rank = np.argsort(peak_quality)[::-1]
    all_periods = all_periods[rank]
    uniq = filter_unique_periods(all_periods,frac=0.05)
    top_n_periods = all_periods[uniq]

    
    n_periods = len(top_n_periods)
    v_err = 1
    n_lin = 100
    t_sin = np.linspace(0,1,n_lin)

    v_subtract = [np.copy(item) for item in v_powers]
    rejected = np.full((n_periods,n_inst),False)
    sins = np.zeros((n_inst,n_lin))
    top_n_Ks = np.zeros(n_inst)
    top_n_params = np.zeros((n_inst,3))
    for i,P in enumerate(top_n_periods):
        string = "\n\n"
        for j in range(n_inst):
            min_p,max_p = all_powers[j][2]
            if P<min_p or P>max_p:
                rejected[i,j] = True
                continue
            p_guess,chi_guess = initial_guess(P,times[j],v_subtract[j],v_err)
            args = (times[j],v_subtract[j],v_err,p_guess)
            res = minimize(planet_param_chi, p_guess, method='Nelder-Mead', 
                           tol=1e-6, args=args)
            pfit = res.x
            bestchi = res.fun

            string += f"Period: {pfit[1]:.3f}d Inst: {j} chi: {bestchi:.4f}\n"
            if pfit[0]<0:
                pfit[0] = np.abs(pfit[0])
                pfit[2] += 0.5
            top_n_Ks[j] = pfit[0]
            top_n_params[j] = pfit
            string += print_params(pfit)
            #print(string)
            sins[j] = simulate_planet_np(t_sin*P,*pfit)[1]

        K_ref = np.median(top_n_Ks[~rejected[i]])
        K_std = np.std(top_n_Ks[~rejected[i]])

        G = sins @ sins.T 
        d = np.diag(G)
        chi2 = d[:, None] + d[None, :] - 2*G
        RMS = np.sqrt(chi2 / sins.shape[1])
        if np.abs(P-planet_period)<1 or K_std>0.2 or np.abs(P-181)<100:
            print(string)
            print("top_n_Ks",K_ref,K_std,top_n_Ks[~rejected[i]])
            #print("RMS",RMS[good])

        good = ~rejected[i]
        # only two instruments
        if good.sum()==2:
            good_K = top_n_Ks[good]
            K_diff = good_K.max()/good_K.min()-1
            if  K_diff>0.5:
                print(f"\nReject all instr P={P:.2f}d K_diff {K_diff*100:.1f}%\n\n")
                rejected[i,:] = True
                continue

        # purpose: reject strong activity
        for j in range(n_inst):
            if rejected[i,j]:continue
            rms = RMS[j]
            f = rms[good]

            enemy = np.sum(f>max(0.3,0.05+top_n_Ks[j]))
            friend = np.sum((f>0) & (f<(0.05+0.5*top_n_Ks[j])))
            majority = good.sum()/3

            if good.sum()==1:continue
            if (enemy<=majority) and (friend>=majority):continue
            if (enemy==0) and (friend==0):continue

            print(f"K={top_n_Ks[j]:.2f}, rms:",f)
            print(f"Friend {friend}/{good.sum()}! enemy: {enemy} K={top_n_Ks[j]:.2f}, rms/K:",f)
            
            rejected[i,j] = True
            
            _,rej_act = simulate_planet_np(times[j],*top_n_params[j])


            print(f" Reject instr {j} P={P:.2f}d rms {v_subtract[j].std():.2f} m/s -> {(v_subtract[j]-rej_act).std():.2f}m/s\n")
            if np.abs(pfit[1]-planet_period)<0.05*planet_period:
                print("Truth rejected??")
            if top_n_Ks[j]<0.2:continue
            v_subtract[j] -= rej_act


        #if rejected[i].sum()>0:print(string)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for j in range(n_inst):
        t = times[j]

        min_p,max_p = all_powers[j][2]
        freq,power = LombScargle(t, v_subtract[j]).autopower(minimum_frequency=1/max_p,maximum_frequency=1/min_p,samples_per_peak=10)
        lw = min(max_p/15,2)
        ax.semilogx(1/freq,power,'-',alpha=1.0,lw=lw,label=labels[j])
    for p in all_periods: ax.axvline(p,ls="-",color="lightsteelblue",alpha=0.3,zorder=-10)

    #ax.legend()
    ax.axvline(planet_period,lw=8,color="gold",alpha=0.5,
               zorder=-20,label="Truth")
    ax.set_xlabel("Period [days]");ax.set_ylabel("Power")
    plt.title(truth_text,weight="bold")
    plt.savefig(f"{outdir}/resid_{basename}.png",dpi=200)
    plt.clf()
    
    #exit()

    top_n_params = np.zeros((len(top_n_periods),3))
    top_n_Ks = np.zeros((n_periods))

    for i,P in enumerate(top_n_periods):
        available = (~rejected[i]).sum()
        if available<1:
            print("Period:",P, "No signal left! continue...")
            continue
        t_all = np.concatenate([ii for j,ii in enumerate(times) if not rejected[i,j]])
        v_all = np.concatenate([ii for j,ii in enumerate(v_subtract) if not rejected[i,j]])
        v_err = np.ones_like(t_all)
        
        p_guess,chi_guess = initial_guess(P,t_all,v_all,v_err)
        print("Period:",P,"instruments:",available,
              "len:",len(t_all))
        args = (t_all,v_all,v_err,p_guess)
        res = minimize(planet_param_chi, p_guess, method='Nelder-Mead', 
                       tol=1e-6, args=args)
        pfit = res.x
        bestchi = res.fun

        print(f"Joint Period:{pfit[1]:.3f}d chi: {bestchi:.4f}")       
        top_n_Ks[i] = pfit[0]
        top_n_params[i] = pfit
        print(print_params(pfit))

    planet_rank = np.argsort(top_n_Ks)[::-1]
    top_n_params = top_n_params[planet_rank]
    top_n_Ks = top_n_Ks[planet_rank]
    uniq = filter_unique_periods(top_n_params[:,1])
    np.savetxt(output_txt,top_n_params[uniq])
    print(top_n_params[uniq])
    continue
    n_rows = 8
    n_cols = 5
    fig,axs=plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(15,10),constrained_layout=True)

    t_loc = (time/planet_period)%2
    sort = np.argsort(t_loc)
    xgrid,y,delta_y = moving_mean(t_loc[sort],v_power[sort],n=15)
    _,y_truth,_ = moving_mean(t_loc[sort],v_planet[sort],n=15)
    chi = np.sum((y-y_truth)**2)

    ax=axs[0][0]
    ax.plot(t_loc,v_power,".",color="lightgrey")
    ax.plot(t_loc[sort],v_planet[sort],"r-",label=f"P={planet_period:.3f}d K={truth[0]:.2f}m/s")
    ax.legend(loc="lower left")
    ax.errorbar(xgrid,y,yerr=delta_y,fmt=".",color="k",capsize=5,ms=10)
    ax.set_ylim(-2*truth[0],1.5*truth[0])
    ax.set_xticks([]);ax.set_yticks([])
    for i,P in enumerate(top_n_periods):
        i_row = (i+1)//n_cols
        i_col = (i+1)%n_cols
        if i>=(n_rows*n_cols-1):break
        ax=axs[i_row][i_col]

        p_guess,chi_guess = initial_guess(P,time,v_subtract,v_err)
        #top_n_params[i] = best_guess

        print("Period:",P,"p_guess:",p_guess)
        chi = planet_param_chi(p_guess,time,v_subtract,v_err)

        res = minimize(planet_param_chi, p_guess, method='Nelder-Mead', 
                       tol=1e-6, args=(time,v_subtract,v_err,))
        pfit = res.x
        bestchi = res.fun

        ph,v_show = simulate_planet_np(time,*pfit)

        print(f"Period:{pfit[1]:.3f}d chi: {chi:.4f}")
        top_n_Ks[i] = pfit[0]
        top_n_params[i] = pfit
        top_n_chi[i] = bestchi
        #top_n_v_act[i] = v_act_nn
        #top_n_v_doppler[i] = v_doppler
        print_params(pfit)
        t_loc = (time/pfit[1])%2
        sort = np.argsort(t_loc)
        xgrid,y,delta_y = moving_mean(t_loc[sort],v_power[sort],n=15)
        chi = np.sum((y-y_truth)**2)
        #s_raw = sinusoidality(xgrid, y, delta_y, planet_period)

        ax.plot(t_loc,v_power,".",color="lightgrey")
        ax.plot(t_loc[sort],v_show[sort],"r-",alpha=0.5,label=f"P={pfit[1]:.3f}d  K={pfit[0]:.2f}m/s")
        ax.legend(loc="lower left")

        ax.errorbar(xgrid,y,yerr=delta_y,fmt=".",color="k",capsize=5,ms=10)
        ylim = max(-2*pfit[0],1.3*pfit[0])
        ax.set_ylim(-ylim,ylim)
        ax.set_xticks([]);ax.set_yticks([])

    plt.savefig(f"{outdir}/grid_{basename}.png",dpi=200)

    planet_rank = np.argsort(top_n_Ks)[::-1]
    top_n_params = top_n_params[planet_rank]
    

    np.savetxt(output_txt,top_n_params)
    print("top_n_params:",top_n_params)
