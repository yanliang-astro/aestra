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
from util import load_batch,moving_mean


def noise_level(per,power,min_noise=0.008,quantile=0.9):
    logp = np.log10(per)
    noise = np.zeros_like(per)
    x = np.linspace(logp.min(),logp.max(),3)
    xmid = 0.5*x[:-1]+0.5*x[1:]
    ymid = np.zeros_like(xmid)
    for i in range(len(x)-1):
        mask = (logp>x[i])&(logp<=x[i+1])
        ymid[i] = max(min_noise,np.quantile(power[mask],0.9))
    xmid[0] = logp.min();xmid[-1] = logp.max()
    noise = interp1d(xmid,ymid,kind="linear")(logp)
    return noise

def simulate_planet_np(t,amp=1,period=0.11,phase_t0=0,t0=800):
    #phase = ((t/period)-phase_t0)%1
    phase = (((t-t0)/period)+phase_t0)%1
    v_planet = amp*np.sin(2*np.pi*phase)
    return phase,v_planet

def planet_param_chi(param,time,v_obs,full=False):
    amp,period,phase = param[:3]
    _,v_planet = simulate_planet_np(time,amp=amp,period=period,phase_t0=phase)
    loss = np.mean((v_obs-v_planet)**2)
    return loss


n_planet = 15
deg = 3
np.random.seed(0)
planet_amp = np.ones(n_planet)*0.2
inject_t0 = np.round(np.random.uniform(0,1,size=(n_planet)),1)
inject_periods = np.round(np.logspace(0.5,2.5,n_planet),3)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

output_dir = "detection_pipeline"

for i_count,planet_period in enumerate(inject_periods):
    #if planet_period<80:continue
    #if not planet_period==84.834:continue
    truth = [planet_amp[i_count],planet_period,inject_t0[i_count]]
    basename = f"period{truth[1]:.3f}d_K{truth[0]:.1f}m_phase{truth[2]:.1f}"
    summary_file = f"summary_file/{basename}_12e_fid_sum.pkl"
    if not os.path.isfile(summary_file):
        summary_file = f"summary_file/{basename}_12d_fid_sum.pkl"
        if not os.path.isfile(summary_file):continue


    output_txt = f"{output_dir}/{basename}_recover.txt"
    
    truth_text = f"Truth:   P={truth[1]:.2f}d   K={truth[0]:.2f}m/s   ph={truth[2]:.1f}"
    print(truth_text)

    with open(summary_file,"rb") as f:
        summary_dict=pickle.load(f)
    data = summary_dict['data']
    time = data['ids'][:,0]

    #planet_truth =  summary_dict['info']['planet_params']
    planet_sol = summary_dict['info']['planet_solution']
    periods = summary_dict['info']['period_solution']
    v_planet = data['v_planet'][:,0]
    v_apparent = data['v_apparent'][:,0]
    #v_err = data['v_err'][:,0]
    v_template = data['v_template'][:,0]
    v_ccf = data['v_ccf'][:,0]
    s = data['s']
    v_act = data['v_act'][:,0]
    v_offset = data['v_offset'][:,0]
    v_doppler = data['v_doppler']
    v_aestra = v_apparent-v_act-v_offset
    print("v_aestra:",v_aestra.shape)

    #t_seg = [0,500,800,1200,1600]
    #masks = [(time>t_seg[k])&(time<t_seg[k+1]) for k in range(len(t_seg)-1)]
    masks = [time<800,time>800]
    powers = []
    for m in masks:
        fmin = 1/400
        fmax = 1/1.5
        freq,pact = LombScargle(time[m], v_act[m]).autopower(minimum_frequency=fmin,samples_per_peak=20,
                                                             maximum_frequency=fmax)
        powers.append([1/freq,pact])

    good_period = periods<400
    plt.figure(figsize=(8,4),dpi=200,constrained_layout=True)
    plt.axvline(truth[1],lw=8,label="truth",color="gold",alpha=0.5,zorder=-10)

    planet_solutions = []
    noises = [0.025,0.03]
    for i,m in enumerate(masks):
        per,act_power = powers[i]

        #noise_floor = noise_level(per,act_power,quantile=0.1,min_noise=np.quantile(act_power,0.99))

        
        noise_floor = np.zeros_like(per) + noises[i]
        local_good = periods<390
        for j,p in enumerate(periods):
            #if not good_period[j]:continue
            mask = (per>0.98*p)&(per<1.02*p)
            if mask.sum()==0:continue
            max_p = act_power[mask].max()
            whmax = np.argmax(act_power[mask])
            if max_p<noise_floor[mask][whmax]:continue
            # missing the actual peak
            snr = max_p/noise_floor[mask][whmax]
            if (whmax == 0) or (whmax == (mask.sum()-1)):
                print(i,f"P={p:.2f}d s/n:{snr:.2f}, missing the peak!")
                #print(act_power[mask])
                if snr<1:continue
            print(i,"activity peak detected!",p)
            plt.fill_between(per[mask],0,1,alpha=0.5,color="lightgrey")
            local_good[j] = False
        good_period &= local_good
        plt.semilogx(per,act_power,'-',alpha=1.0,lw=1, label="$v_{act}$")
        plt.fill_between(per,np.zeros_like(per),noise_floor,alpha=0.1)
    for p in periods:plt.axvline(p,lw=1,color="grey",ls='--')
    for p in periods[good_period]:plt.axvline(p,lw=1,color="r",ls='--')
    plt.legend(loc=2)
    plt.xlim(2,380)
    plt.ylim(0,0.08)
    plt.xlabel("Period [days]")
    plt.ylabel("Normalized Power")
    plt.savefig(f"{output_dir}/{basename}_diag.png")
    plt.clf()

    print("\n\n")
    K = planet_sol[:,0]
    K_diff = np.zeros_like(K)
    phase = planet_sol[:,1]
    for i in np.argsort(K)[::-1]:
        P = periods[i]
        tfold = time%P

        yfold = np.array([moving_mean(tfold[m],v_act[m],n=12)[1] for m in masks])
        K_diff[i] = np.std(yfold[1]-yfold[0])/K[i]
        corr = [np.corrcoef(v_act[m],v_doppler[m,i])[0][1] for m in masks]
        corr = np.array(corr)
        text = f"P:{periods[i]:.4f}, K:{K[i]:.3f}m/s K_diff: {K_diff[i]:.3f} Corr: {np.round(corr,3)}"
        period_factor = max(100,P)/100

        if np.abs(corr).max()/period_factor>0.1 or (K_diff[i])>(0.8*period_factor):
            #good_period[i] = False
            print(f"{text} Reject?")
        elif not good_period[i]:print(f"{text} Bad")
        else: print(text)

    print(f"Saving to {output_txt}...")

    good_period &= np.abs(K)>0.1

    planets = np.vstack((K,periods,phase)).T
    planets = planets[good_period]
    rank = np.argsort(planets[:,0])[::-1]
    planets = planets[rank]
    print("planets:",planets)
    np.savetxt(output_txt,planets)

    if good_period.sum()==0:
        print("No planet detected! Continue...")
        continue
    wh = K==K[good_period].max()
    #wh = K_diff == K_diff[good_period].min()

    t_all = []
    v_all = []

    v_bg = v_doppler[:,~wh].sum(axis=1)
    v_without_act = v_aestra - v_bg
    v_without_act -= v_without_act.mean()
    t_all = np.hstack((t_all,time))
    v_all = np.hstack((v_all,v_without_act))
    
    p0 = [K[wh],periods[wh],phase[wh]]
    p0 = [item[0] for item in p0]
    print("p0",p0)
    # Fit the sinusoidal model
    result = minimize(planet_param_chi,p0, args=(t_all,v_all,),method='Nelder-Mead')
    best_planet_params = result.x
    best_chi = result.fun
    Kbest,Pbest,t0best = best_planet_params
    
    label_best = f"P = {Pbest:.3f}d, K = {Kbest:.3f} m/s"
    if np.abs(Pbest-truth[1])/truth[1] < 0.1:
        print("True signal detected!!")
    print(f"\nBestfit {label_best}")

    tmin = 200
    tmax = min(1400,200+50*Pbest)

    x_axis = t_all
    t_truth = np.linspace(x_axis.min(),x_axis.max(),10000)

    _,v_truth = simulate_planet_np(t_truth,*truth)
    _,v_fit = simulate_planet_np(t_truth,*(Kbest,Pbest,t0best))


    xspan = x_axis.max()-x_axis.min()
    nbins = int(60*xspan/(tmax-tmin))

    fig,axs=plt.subplots(figsize=(8,3.5),nrows=2,constrained_layout=True)


    ax=axs[0]
    ax.plot(x_axis,v_apparent-v_offset,".",ms=3,color="lightgrey",label=f"$v_{{traditional}}$: RMS={(v_apparent-v_offset).std():.2f} m/s")
    if truth[1]>50:
        xgrid,y,delta_y = moving_mean(x_axis,v_apparent-v_offset,n=nbins)
        ax.errorbar(xgrid,y,yerr=delta_y,fmt=".",color="k",capsize=5,ms=10)

    #plt.plot(x_axis,v_aestra-v_aestra.mean(),".", ms=3,color="b",label=f"Before rejection: RMS={(v_aestra-v_planet).std():.2f} m/s")
    ax=axs[1]
    ax.plot(x_axis,v_all,".",ms=3,color="lightgrey",label=f"$v_{{aestra}}$: RMS={(v_all-v_planet).std():.2f} m/s")
    if truth[1]>50:
        xgrid,y,delta_y = moving_mean(x_axis,v_all,n=nbins)
        ax.errorbar(xgrid,y,yerr=delta_y,fmt=".",color="k",capsize=5,ms=10)
    ax.plot(t_truth,v_fit,lw=2,color='b',label=label_best)
    

    for ax in axs:
        ax.plot(t_truth,v_truth,"-",lw=2,color='r',label=f"Truth")
        ax.legend(loc=4)

        ax.set_xlim(tmin,tmax)
        ax.set_ylim(-1.4,1.4)
        ax.set_ylabel("RV [m/s]")
        #plt.ylim(2*Kbest,-2*Kbest)
    ax.set_xlabel("Time [days]");
    ax.text(tmin+20,-1.,f"Truth: K={truth[0]:.2f}m/s  P={truth[1]:.3f}d",
             fontsize=15,weight='bold',color="k",alpha=1.0)
    plt.savefig(f"{output_dir}/ts_{basename}.png",dpi=200)
    plt.clf()
    print("Continue.........................\n\n")