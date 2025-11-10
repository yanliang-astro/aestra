#!/usr/bin/env python

import time, argparse, os
import numpy as np
import functools
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator
# allows one to run fp16_train.py from home directory
import sys;sys.path.insert(1, './')
import subprocess
from spender_model import SpectrumAutoencoder,NullRVEstimator
from synthetic_data import Synthetic
from util import mem_report,gaussian_kernel_1d
from functools import partial
from util import load_batch,interpolate_to_input_grid,normalize_residual,divide_sky_model
from util import simulate_planet
from torch.utils.data import DataLoader,Dataset
from torchinterp1d import Interp1d
from line_profiler import LineProfiler
from scipy.special import digamma

import torch.nn.functional as F

# Check the value of CUDA_VISIBLE_DEVICES
cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')

# Print the visible CUDA devices
node_name = os.getenv('SLURMD_NODENAME')

# Print the node name
print(f"Running on node: {node_name} CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")

def pearson_corrcoef_batch(x,Y):
    """
    Computes the Pearson correlation coefficient between a 1D tensor x and multiple 1D tensors stacked in Y.
    
    Args:
    - x: Tensor of shape (N,)
    - Y: Tensor of shape (M, N) where each row is a different 1D tensor (e.g., y1, y2, y3)

    Returns:
    - A tensor of shape (M,) containing correlation coefficients for each row in Y.
    """
    x_mean = torch.mean(x)
    Y_mean = torch.mean(Y, dim=1, keepdim=True)

    x_diff = x - x_mean
    Y_diff = Y - Y_mean

    numerator = torch.sum(x_diff * Y_diff, dim=1)
    denominator = torch.sqrt(torch.sum(x_diff ** 2)) * torch.sqrt(torch.sum(Y_diff ** 2, dim=1))

    return numerator / denominator

def corrcoef(tensor, rowvar=True, bias=False):
    """Estimate a corrcoef matrix (np.corrcoef)
    https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    covmat = factor * tensor @ tensor.transpose(-1, -2).conj()
    std = torch.diag(covmat)**0.5
    covmat /= std[:, None]
    covmat /= std[None, :]
    return covmat

def avgdigamma(dist,radius):
    num_points = torch.count_nonzero(dist<(radius-1e-15),dim=0).double()
    return (torch.digamma(num_points)).mean()

def mutual_information(x0, y0, k=50, base=2):
    x = (x0-x0.mean(dim=0))/x0.std(dim=0)
    y = (y0-y0.mean())/y0.std()
    """Mutual information of x and y 
    """
    assert x.shape[0] == y.shape[0], "Arrays should have same length"
    assert y.shape[1] == 1,     "Single value function"
    assert k <= x.shape[0] - 1, "Set k smaller than num. samples - 1"
    # Find nearest neighbors in joint space, 
    points = torch.cat((x, y),dim=1)

    # Find nearest neighbors in joint space, p=inf means max-norm
    distmat = torch.abs(points[:,None]-points[None,:])
    dvec = torch.kthvalue(torch.amax(distmat,2),k+1,dim=1)[0]
    
    a, b, c, d = (
        avgdigamma(torch.amax(distmat[:,:,:-1],2),dvec),
        avgdigamma(distmat[:,:,-1],dvec),
        digamma(k),
        digamma(x.shape[0]),
    )
    return (-a - b + c + d) / np.log(base)

def prepare_train(seq,niter=100000):
    for d in seq:
        if not "iteration" in d:d["iteration"]=niter
        if not "encoder" in d:d.update({"encoder":d["data"]})
    return seq

def build_ladder(train_sequence):
    n_iter = sum([item['iteration'] for item in train_sequence])

    ladder = np.zeros(n_iter,dtype='int')
    n_start = 0
    for i,mode in enumerate(train_sequence):
        n_end = n_start+mode['iteration']
        ladder[n_start:n_end]= i
        n_start = n_end
    return ladder

def get_all_parameters(models,instruments):
    model_params = []
    model = models[0]

    model_params += model.rv_estimator.parameters()
    # telluric model training
    if model.telluric is not None:
        model_params += model.telluric.parameters()
    if model.fringe is not None:
        model_params += model.fringe.parameters()
    if model.encoder is not None:
        model_params += model.encoder.parameters()
        model_params += model.decoder.parameters()
        model_params += model.activity_estimator.parameters()
        model_params += model.doppler_model.parameters()

    dicts = [{'params':model_params}]
    n_parameters = sum([p.numel() for p in model_params if p.requires_grad])

    return dicts,n_parameters

def consistency_loss(s, s_aug, individual=False, sigma_s=1.0):
    batch_size, s_size = s.shape
    ds = torch.sum((s_aug - s)**2/(sigma_s)**2,dim=1)/(s_size)
    cons_loss = torch.sigmoid(ds)-0.5# zero = perfect alignment
    if individual:
        return cons_loss
    return cons_loss.sum()

def flexibility_loss(spectrum,weight,sigma=0.01):
    spec_size = spectrum.shape[-1]
    flex_loss = torch.sum(weight*spectrum**2/sigma**2,dim=-1)/spec_size
    return flex_loss.sum()

def z_offset_loss(z_off, z_off_true, sigma_z=3.33e-9,individual=False):
    z_loss = ((z_off - z_off_true)/sigma_z)**2
    if individual:return z_loss
    return z_loss.sum()

def restframe_weight(model,sn=300):
    w = model.decoder.weight_rest*(sn**2)
    return w

def similarity_loss(x,slope=1.0,wid=5.0):
    sim_loss = torch.sigmoid(slope*x-wid/2)+torch.sigmoid(-slope*x-wid/2)
    sim_loss -= 1.0/(1+np.exp(wid/2))*2 # adjust zero point to 0
    return sim_loss

def similarity_restframe(model, spec, s, slope=1.0, sigma_s=1.0,
                         individual=False, sn=300):
    _, s_size = s.shape
    device = s.device

    batch_size, n_order, spec_size = spec.shape
    # randomly permute spectra
    rand = torch.randperm(batch_size)

    W = restframe_weight(model,sn=sn)

    # dissimilarity of latents
    s_sim = ((s - s[rand,:])**2/sigma_s**2).sum(-1) / s_size
    spec_sim = torch.zeros_like(s_sim)

    for o in range(n_order):
        S = (spec[:,o,:]-spec[rand,o,:])**2
        # pairwise dissimilarity of spectra
        #S = (spec[None,:,o,:] - spec[:,None,o,:])**2
        # dissimilarity of spectra
        # of order unity, larger for spectrum pairs with more comparable bins
        spec_sim += (W[o] * S).sum(dim=(-1)) 
    spec_sim /= W.count_nonzero()

    x = s_sim-spec_sim
    sim_loss = similarity_loss(x,slope=slope)
    #diag_mask = torch.diag(torch.ones(batch_size,device=device,dtype=bool))
    #sim_loss[diag_mask] = 0
    if individual:
        return s_sim,spec_sim,sim_loss
    # total loss: sum over N terms,
    # needs to have amplitude of N terms to compare to fidelity loss
    return sim_loss.sum()# / batch_size

def tensor2array(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else: return tensor.detach().numpy()

def leaky_relu(x, alpha):
    return torch.where(x > 0, x, alpha * x)

def v_continuity_reg_loss(delta_t,delta_v,k=0.2,alpha=0.0,tmax=3):
    mask = (delta_t>0) & (delta_t<=tmax)
    v_reg_loss = leaky_relu(delta_v[mask]-(k*delta_t[mask]),alpha)
    v_reg_loss += alpha*(k*tmax)
    return 0.1*len(delta_t)*v_reg_loss.mean()

def plot_similarity(s_sim,spec_sim,sim_loss,slope=1.0,sigma_s=1.0):
    import matplotlib.pyplot as plt
    spec_sim = tensor2array(spec_sim)
    s_sim = tensor2array(s_sim)
    sim_loss = tensor2array(sim_loss)
    print("sim_loss:",sim_loss.shape)
    print("spec_sim:",spec_sim.min(),spec_sim.mean(),
          spec_sim.max(),spec_sim.shape)
    print("s_sim:",s_sim.min(),s_sim.mean(),s_sim.max(),s_sim.shape)

    s_chi = s_sim.max()*torch.rand(size=(1000,1))
    spec_chi = spec_sim.max()*torch.rand(size=(1000,1))

    x = s_chi - spec_chi
    loss_bg = similarity_loss(x,slope=slope)
    vmax = 0.5
    fig,ax=plt.subplots(nrows=1,constrained_layout=True,
                        figsize=(5,5))
    img = ax.scatter(spec_sim,s_sim,c=sim_loss,ec="k",cmap="plasma",vmax=vmax)
    ax.scatter(spec_chi,s_chi,c=loss_bg,cmap="plasma",zorder=-10,vmax=vmax)
    ax.set_xlabel("spectral $\chi^2$")
    ax.set_ylabel("latent $\chi^2$")
    cbar = fig.colorbar(img, ax = ax, shrink=0.5)
    cbar.set_label("similarity loss")
    ax.set_title("slope %.2f, sig_s=%.2f"%(slope,sigma_s))
    plt.savefig("[similarity].png",dpi=300)
    #exit()
    return

def sinusoidality_harmonic(t, y, yerr, P, K=5):
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

def plot_diagnostic(diags,model,instrument,ratio=50,n_window=2):
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from astropy.timeseries import LombScargle
    from util import moving_mean
    
    tlin = torch.linspace(200,1500,2000,device=instrument.wave_obs.device)
    #v_spline = model.doppler_model(tlin[:,None],ph_shift=0.25)
    v_spline = model.estimate_doppler_rv(tlin[:,None])
    periods = model.doppler_model.get_periods()

    quasi_terms = model.doppler_model.get_quasi_periodic_terms(tlin[:,None])
    quasi_terms = tensor2array(quasi_terms)
    print("quasi_terms:",quasi_terms.shape)

    tlin = tensor2array(tlin)
    periods = tensor2array(periods)
    v_spline = tensor2array(v_spline)

    fig,ax=plt.subplots(figsize=(12,0.5*len(periods)+2),constrained_layout=True)
    ampl = v_spline.std(axis=1)
    rank = np.argsort(periods)

    for i_rank in range(len(periods)):
        i = rank[i_rank]
        p = periods[i]
        v_spline[i]-=np.mean(v_spline[i])
        s2 = sinusoidality_harmonic(tlin, v_spline[i], np.ones_like(v_spline[i]), p)
        var1,var2 = quasi_terms[i].max(axis=0)-quasi_terms[i].min(axis=0)
        phase_avg = quasi_terms[i,:,1].mean()
        label = f"P={p:.2f}d  var=[{var1:.2f},{var2:.2f}]  S={s2:.2f}"
        #print(label,"phi_coeff:",model.doppler_model.phi_coef[i])
        #print("amp_coeff:",model.doppler_model.amp_coef[i])

        y_off = (i_rank+1)*0.5
        line,=ax.plot(tlin,v_spline[i]+y_off,"-",label=label)
        #line,=ax.plot(tlin,quasi_terms[i,:,1]+y_off,"-",label=label)
        ax.text(-200,y_off,label,color = line.get_color())
        ax.set_yticks([])
    #ax.legend()
    ax.set_xlim(tlin.min(),tlin.max())
    plt.savefig("vspline.png",dpi=200)
    if not "input_data" in diags:
        time,v_trad,v_encode,v_planet = [tensor2array(item[:,0]) for item in diags["rv"]]
        print(f"v_encode: {v_encode.std():.3f} m/s")
        plin = np.polyfit(v_planet,v_encode,deg=1)
        print(f"slope: {plin[0]:.2f}")
        fig,ax=plt.subplots(figsize=(5,3),constrained_layout=True)
        ax.plot(v_planet,v_encode,"k.")
        x = sorted(v_planet)
        ax.plot(x,np.polyval(plin,x),"r--",label=f"slope={plin[0]:.3f}")
        ax.legend()
        plt.savefig("test.png",dpi=200)
        exit()

    if "rv" in diags:
        time,v_trad,v_encode,v_planet,v_act,v_doppler = [tensor2array(item[:,0]) for item in diags["rv"]]

        planet_amp,planet_period,phase=diags["planet"]
        v_aestra = v_encode-v_act
        v_aestra -= v_aestra.mean()
        print("v_doppler:",v_doppler.shape)
        
        ind = np.argsort(time)
        frequency = 1.0/np.logspace(0.2, 2.6, 10000)
        power = LombScargle(time[ind], v_encode[ind]).power(frequency)
        print(len(v_encode[ind]))

        n_peaks = 10
        per = 1/frequency
        rank = np.argsort(power)[::-1]
        top_n_periods = np.zeros((n_peaks))

        i = 0
        for rk in rank:
            per_max = per[rk]
            nonzero = top_n_periods>0
            if per_max>360:continue
            if nonzero.sum()>0:
                if (np.abs(top_n_periods[nonzero]-per_max)/per_max).min()<0.2:continue
                if np.abs(top_n_periods[nonzero]-per_max).min()<2:continue
            top_n_periods[i] = per_max;i+=1
            nonzero = top_n_periods>0
            if nonzero.sum()>=n_peaks:break
        print("top_n_periods:",top_n_periods)
        
        fig,axs=plt.subplots(figsize=(6,2.5),ncols=2,constrained_layout=True)

        ax=axs[0]
        ax.semilogx(1/frequency, power,"-",lw=4,
                    color="lightgrey",label="$v_{encode}$")
        power = LombScargle(time, v_aestra).power(frequency)
        ax.semilogx(1/frequency, power,"k-",label="$v_{aestra}$")
        ax.axvline(planet_period,lw=5,color="gold",
                   alpha=0.5,zorder=-20,label="truth")
        ax.legend()
        ax.set_xlabel("Period [days]")
        ax=axs[1]
        t_loc = ((time/planet_period)-phase)%1
        sort = np.argsort(t_loc)
        ax.plot(t_loc,v_aestra,".",color="lightgrey",label="$v_{aestra}$")
        ax.plot(t_loc[sort],v_planet[sort],"r",label="Truth")
        xgrid,y,delta_y = moving_mean(t_loc,v_aestra,n=15)
        ax.errorbar(xgrid,y,yerr=delta_y,
                    fmt=".",color="k",capsize=5,ms=10)
        ax.legend()
        ax.set_ylim(-2*planet_amp,2*planet_amp)
        ax.set_xlabel("Phase");ax.set_ylabel("$v_{aestra} [m/s]$")
        plt.savefig("test.png",dpi=200)
        #exit()

    if "model" in diags:model_resid = tensor2array(diags["model"].squeeze(1))

    if "y_act" in diags: y_act = tensor2array(diags["y_act"].squeeze(1))

    template_data = [tensor2array(item[0]) for item in diags["template"]]
    wave_obs,template,w_template,_,_ = template_data
    spec_zero_rv,w = [tensor2array(item.squeeze(1)) for item in diags["input_data"]]

    #print("err:",np.quantile(w**(-0.5),[0.01,0.5,0.99]))

    n_batch,n_spec = spec_zero_rv.shape
    temp_err = w_template**(-0.5)

    loss = w*(spec_zero_rv-model_resid)**2
    loss_ind = np.sum(loss, axis=-1) / np.sum(w>1,axis=-1)
    
    plt.clf()
    plt.hist(loss_ind,bins=20)
    plt.axvline(loss_ind.mean(axis=0),color="k",ls="--")
    plt.savefig("loss.png",dpi=200)
    plt.clf()

    loss_avg = gaussian_filter1d(loss.mean(axis=0),2)
    loss_avg -= np.quantile(loss_avg,0.3)
    print("loss_ind:",loss_ind.shape)

    print("masked:",(w<=1).sum()/(n_batch*n_spec))
    print("loss:",loss_ind.shape,loss_ind.mean())
    

    diag = np.copy(loss_ind)
    #diag_full = loss
    #smooth_diag = gaussian_filter1d(diag_full[i,o_max],2)
    #diag = np.abs(model_resid.sum(axis=-1))
    #diag_full = np.abs(spec_obs)

    print("loss:",loss.shape)
    i,wh = np.argwhere(loss==loss.max())[0]

    #i = 92
    print("i:",i,"wh:",wh)

    drawstyle = "steps-mid"
    stepstyle = "mid"
    ncols = 1
    
    spec_rest = tensor2array(diags["spec_rest"][0])

    c_order = "k"
    err_c = "lightgrey"
    zorder = 0
    window = 2.5
    #colors = ["k","b","darkgreen"]*n_order
    #c_err = ["lightgrey","lavender","palegreen"]*n_order
    fig,axs=plt.subplots(figsize=(12,7),ncols=ncols,nrows=2,
                         gridspec_kw={'height_ratios': [2.5, 1]},
                         constrained_layout=True)
    for k in range(ncols):
        center = wave_obs[wh]
        ax = axs[0]#[k]
        spec_inflate = ratio*spec_zero_rv[i]+template
        #aug_inflate = ratio*spec_aug[i]+template
        model_inflate = ratio*model_resid[i]+template
        #spec[w[i]<1] = 0
        #ax.plot(wave_obs,template,"-",color='navy',drawstyle=drawstyle,label="template",zorder=zorder)
        ax.plot(wave_obs,template,"-",color='b',drawstyle=drawstyle,label="template",zorder=zorder)
        ax.plot(wave_obs,spec_inflate,"k-",drawstyle=drawstyle,label=f"data ({ratio}x inflated activity)",zorder=zorder)
        #ax.plot(wave_obs,aug_inflate,"b-",lw=0.5,drawstyle=drawstyle,label="augment",zorder=zorder)
        ax.plot(wave_obs,model_inflate,"-",color='r',lw=1,drawstyle=drawstyle,label=f"model ({ratio}x inflated activity)",zorder=zorder)
    
        err = (w[i]**(-0.5))
        ax.fill_between(wave_obs,spec_inflate-ratio*err,spec_inflate+ratio*err,
                        color="k",alpha=0.3,lw=0,step=stepstyle,zorder=-20)

        loss_ind = loss[i]
        loss_io = loss_ind.sum()/(w[i]>1).sum()
        print("loss_ind:",loss_io,sorted(loss_ind,reverse=True)[:10])

        ax.plot(wave_obs,loss_ind/loss_ind.max(),"-",color="grey",lw=1.0,drawstyle="steps-mid",label="loss = %.2f"%(loss_io))
        ax.plot(wave_obs,loss_avg,"-",color="orange",lw=1.0,drawstyle="steps-mid",label="mean loss")
        ax.fill_between(wave_obs,0,1/loss_ind.max(),color="lightgrey",zorder=-20)
        ax.set_ylim(spec_inflate.min(),spec_inflate.max())

        #ax.set_ylim(-0.002,0.002)
        ax = axs[1]#[k]
        if "y_act" in diags: 
            y_show,yname = y_act,"y_act"
        else:y_show,yname = spec_zero_rv,"resid"

        disp = y_show.std(axis=0)
        for i_spec in range(min(50,n_batch)):
            ax.plot(wave_obs,y_show[i_spec],c="grey",lw=1,alpha=0.5,drawstyle="steps-mid")
        ax.plot(wave_obs,-disp,lw=1,c="cyan",
                drawstyle="steps-mid",label="dispersion")
        #ax.fill_between(wave_obs,y_show[i]-err[i],y_show[i]+err[i],color="k",alpha=0.3,lw=0,step=stepstyle,zorder=-20)
        #ax.plot(wave_obs,spec_zero_rv[i],c="k",lw=1,drawstyle="steps-mid",label="data")
        ax.plot(wave_obs,y_show[i],c="k",lw=1,drawstyle="steps-mid",label="%s"%(yname))
        #ax.set_ylim(-0.01,0.01)
        #ylim = np.quantile(y_show,[0.01,0.99])
        #ax.set_ylim(ylim)
        for i_row in range(2):
            ax = axs[i_row]#[k]
            #xlim = [wave_obs[wh]-window,wave_obs[wh]+window]
            #xlim = [wave_obs[-1]-3,wave_obs[-1]+0.1]
            xlim = [4957,4961]
            ax.set_xlim(xlim);
            ax.legend()

    plt.savefig("reconstruction.png",dpi=300)
    exit()
    return

def print_planet_solutions(p0,p,current_Ks,periods,
                           n_string=10,corr=None):
    rank = torch.argsort(current_Ks,descending=True)
    for i in rank:
        K0,P0 = p0[i,:2]
        #K,P = p[i,:2]
        P = periods[i]
        K = current_Ks[i]
        str1 = f"{P0:.4f}d"
        str1 += " "*(n_string-len(str1))
        str1 += f"K={K0:.3f}m/s"
        str2 = f"{P:.4f}d"
        str2 += " "*(n_string-len(str2))
        str2 += f"K={K:.3f}m/s"
        if corr is None: str3 = ""
        else: str3 = f"  corr: {corr[i]:.3f}"
        print(str1," --> ",str2,str3)
    return

def hinge_loss_penalty(x, x0=14.35, dx=0.15):
    return torch.maximum(torch.zeros_like(x), dx - torch.abs(x - x0))/dx

def doppler_correlation(jd,vel,v_doppler,t_seg=800):
    m1 = jd.squeeze(1)<t_seg
    m2 = jd.squeeze(1)>t_seg
    corr1 = pearson_corrcoef_batch(vel[m1].squeeze(1),v_doppler[:,m1])
    corr2 = pearson_corrcoef_batch(vel[m2].squeeze(1),v_doppler[:,m2])
    return corr1,corr2

def get_losses(model,
               instrument,
               batch,
               template,
               aux_data=None,
               planet_param=None,
               skymask=None,
               aug_fct=None,
               similarity=True,
               consistency=True,
               flexibility=True,
               regularize_v=True,
               slope=0,
               sigma_s=0.5,
               stellar_activity=True,
               skipz=False,
               telluric=True,
               ):

    wave_obs = instrument.wave_obs
    print("wave_obs:",wave_obs.shape)
    print("template:",template.shape)

    v_reg_loss = 0
    fid_loss = sim_loss = flex_loss = cons_loss = 0

    ccf_info,spec_raw,w,ssbrv,jd = batch
    #v_trad = ccf_info[:,[0]]
    v_trad = ccf_info[:,[1]]

    print("v_trad:",v_trad.shape)

    batch_size = jd.shape[0]

    # inject planet
    planet_amp,planet_period,phase=planet_param
    print("planet_param:",planet_param)
    _,v_planet = simulate_planet(jd,amp=planet_amp,
                                 period=planet_period,
                                 phase_t0=phase)

    rv = v_trad + v_planet # just replace our RV Estimator


    if ssbrv.ndim==1:ssbrv = ssbrv.unsqueeze(1)
    if jd.ndim==1:jd = jd.unsqueeze(1)

    if args.debug:slope=1.0
    if skymask is not None:
        print("Sky Mask Fraction: %.2f"%(skymask.sum()/torch.numel(skymask)))

    z_loss = 0
    
    if not skipz:
        spec_input, w, _ = interpolate_to_input_grid(batch,instrument,template,extra_rv=v_planet)
        rv,rv_err = model.estimate_rv(spec_input)
        print("spec_input:",spec_input.min(),spec_input.max())
        print(f"\n[RV Estimator]rv:  [{rv.min():.2f},{rv.max():.2f}] RMS={rv.std():.3f}m/s")
        print(f"[RV Estimator]rv_err:  [{rv_err.min():.2f},{rv_err.max():.2f}] Mean={rv_err.mean():.3f}m/s")
        print(f"rv_trad: RMS={(v_trad+v_planet).std():.3f}m/s")
        print(f"[NN RV Training] NN vs. Trad Difference: {(rv-v_trad-v_planet).std():.3f} m/s")

        #  generate augment spectra
        spec_input, w, z_off_true = interpolate_to_input_grid(batch,instrument,template,aug=True,extra_rv=v_planet)

        rv_aug,_ = model.estimate_rv(spec_input)
        z_loss = (rv_aug-rv-z_off_true*instrument.c)**2/rv_err**2
        z_loss = z_loss.sum()
        flex_loss = torch.log(rv_err**2).sum()

    else:
        # replace rv estimator with pre-trained values
        indices = torch.searchsorted(aux_data[0].contiguous(), jd)
        v_apparent = aux_data[1][indices]
        v_offset = aux_data[2][indices]
        quality_mask = aux_data[3][indices][:,0].bool()
        # replace auto encoder with pre-trained values
        start,end = 5,len(aux_data)
        s_load = [aux_data[ii][indices] for ii in range(start,end)]
        s_load = torch.cat(s_load,dim=1)
        #s = s_load
        #stellar_activity = False
        print("quality_mask:",quality_mask.sum())
        #rv = v_nn
        print(f"[Pretrained RV] Apparent vs. Trad Difference: {(v_apparent-v_trad-v_planet).std():.3f} m/s")

    if stellar_activity:

        # testing new idea: shift spectra back to zero apparent rv
        spec_zero_rv, w, _ = interpolate_to_input_grid(batch,instrument,template,extra_rv= -v_apparent+v_planet)

        s = model.encode(spec_zero_rv)
        y_act = model.decode(s)
        spectrum_observed = model.decoder.spec_rest+y_act

        # normalize residual model
        model_resid = spectrum_observed-template
        model_resid[w<1.0] = 0

        print("y_act:",y_act.std(dim=0).mean())

        # compare residual model
        loss = model._loss(spec_zero_rv, w, model_resid, individual=True)
        
        fid_loss = 2*loss[quality_mask].sum()
        print("loss:",loss.shape,fid_loss.item()/batch_size)
        if fid_loss.item()/batch_size < 0.7:
            print("\n\nfidelity_loss<0.7!",fid_loss.item()/batch_size)
            print("loss:",loss.min().item(),loss.max().item())
            print("quality_mask:",quality_mask.sum(),
                   quality_mask.shape)
            args.debug=True
    else: pass

    if flexibility:
        flex_loss += slope*10*(y_act**2).sum()

    if similarity:
        sim_loss = similarity_restframe(model, y_act, s, slope=slope,sigma_s=sigma_s)

    #if regularize_v:
    if (consistency and (fid_loss/batch_size < 0.96)) or (stellar_activity and args.debug):
        #    s_aug = model.encode(spec_input_aug)
        #    cons_loss = consistency_loss(s, s_aug
        v_act = model.activity_estimator(s)
        v_doppler = model.estimate_doppler_rv(jd)
        #v_doppler = model.activity_estimator.doppler_rv(jd)

        v_doppler_sum = v_doppler.sum(dim=0)[:,None]

        #quality_mask = jd>500
        v_resid = v_apparent-v_act-v_doppler_sum-v_offset
        v_reg_loss = 0.05*torch.sum((v_resid[quality_mask])**2)
        print("v_reg_loss:",v_reg_loss.item()/batch_size)

        v_shifted = model.doppler_model(jd,ph_shift=0.25)

        model_params = model.doppler_model.planet_params
        current_Ks = model_params[:,0]
        periods = model.doppler_model.get_periods()
        
        corr = doppler_correlation(jd,v_act,v_doppler)
        anti_corr = doppler_correlation(jd,v_act,v_shifted)
        
        doppler_corr = (corr[1]-corr[0])**2
        # weaker constraints for long periods
        doppler_corr[periods>200] *= 0.0
        #doppler_corr += (corr1-corr2)**2
        print("periods:",periods)
        print("doppler_corr:",doppler_corr.shape)

        cons_loss = batch_size*0.05*torch.sum(doppler_corr)
        print("cons_loss:",cons_loss.item()/batch_size)

        if v_resid.std().item()>0.4:cons_loss=0
        if v_resid.std().item()<0.1:cons_loss=0
        #K_reg_loss = slope*torch.exp(-current_Ks[(current_Ks>0.1)])
        #cons_loss += K_reg_loss.sum()
        #print("K_reg_loss:",K_reg_loss.sum()/batch_size)

        print(f"v_act RMS:{v_act.std():.5f} m/s")
        print(f"v_doppler RMS:{v_doppler_sum.std():.3f} m/s")
        print(f"v_offset RMS:{v_offset.std():.3f} m/s")
        print(f"Initial RMS:{rv[quality_mask].std().item():.3f} m/s")
        print(f"Residual RMS:{(v_resid[quality_mask]).std().item():.3f} m/s")

        s_corr = pearson_corrcoef_batch(rv.T,s.T)
        print("s_i correlation with v_apparent:",s_corr.detach())


        #0.5*(v_doppler.max(dim=-1)[0]-v_doppler.min(dim=-1)[0])
        print_planet_solutions(planet_params,
                               model_params,
                               current_Ks,
                               model.doppler_model.get_periods(),
                               corr=doppler_corr)
        #print(f"planet_params:{model.activity_estimator.planet_params[rank]}")
        #print(f"planet_params:{model.activity_estimator.planet_params}")
        
        #delta_v = torch.abs(rv-rv.T)
        #delta_t = torch.abs(jd-jd.T)
        #v_reg_loss = v_continuity_reg_loss(delta_t,delta_v)
    else:
        v_act = torch.zeros_like(v_apparent)
        v_doppler = torch.zeros_like(v_apparent)

    if args.debug:
        diags = {"template":template_data,
                 "spec_rest":model.decoder.spec_rest,
                 'planet':planet_param}

        if stellar_activity:
            diags['rv'] = [jd,v_trad,rv,v_planet,v_act,v_doppler]
            diags["input_data"]=[spec_zero_rv,w]
            diags["model"] = model_resid
            diags["y_act"] = y_act
        else:
            diags['rv'] = [jd,v_trad,v_apparent,v_planet]

        if similarity:
            slope = 1.0
            s_sim,spec_sim,sim_loss = similarity_restframe(model, y_act, s, 
                                                           slope=slope,sigma_s=sigma_s,individual=True)
            plot_similarity(s_sim,spec_sim,sim_loss,sigma_s=sigma_s,slope=slope)
        plot_diagnostic(diags,model,instrument)
        exit()
    return fid_loss, sim_loss, z_loss, cons_loss, v_reg_loss, flex_loss


def checkpoint(accelerator, args, optimizer, scheduler, n_inst, outfile, losses):
    unwrapped = [accelerator.unwrap_model(args_i).state_dict() for args_i in args]

    accelerator.save({
        "model": unwrapped,
        "losses": losses,
    }, outfile)
    return

def update_rv_estimator(rvfile, models, instruments, name='rv_estimator'):
    device = instruments[0].wave_obs.device
    rv_model = torch.load(rvfile, map_location=device)["model"][0]
    short = lambda key: key.split(name+".")[1]
    rv_model = {short(key):val for key,val in rv_model.items() if name in key}
    for model in models: model.rv_estimator.load_state_dict(rv_model)
    return models

def update_telluric(updatefile, models, instruments, name='telluric'):
    if updatefile == "null":
        print("No telluric model...")
        for model in models: model.telluric = None
        return models
    device = instruments[0].wave_obs.device
    update_model = torch.load(updatefile, map_location=device)["model"][0]
    short = lambda key: key.split(name+".")[1]
    update_model = {short(key):val for key,val in update_model.items() if name in key}
    for model in models: model.telluric.load_state_dict(update_model)
    return models

def load_model(mainfile, models, instruments, new_act = False):
    device = instruments[0].wave_obs.device
    model_struct = torch.load(mainfile, map_location=device)

    for i, model in enumerate(models):
        init_dict = model.state_dict()
        model_dict = model_struct['model'][i]

        filtered = {}
        for k, v in model_dict.items():
            if not "activity_estimator" in k and not "doppler_model" in k:
                filtered[k] = v
                continue

            if  k in init_dict:
                shape1 = model_dict[k].shape
                shape2 = init_dict[k].shape
                if shape1==shape2:
                    filtered[k] = v
                    continue
                else:
                    # shape mismatch -- load to the first entry
                    filtered[k] = init_dict[k]
                    #filtered[k][0] = v
                    print(k,v.shape,init_dict[k].shape)
                    new_act = True
            else:new_act = True

        model.load_state_dict(filtered, strict=False)
    if new_act:
        print("New activity_estimator!")
    losses = model_struct['losses']
    return models, losses

def correct_velocity_offset_old(aux_file, threshold=800):
    t, v, quality = torch.load(aux_file)
    quality = quality.bool()
    v_offset = torch.where(t < threshold,
                           torch.median(v[(t < threshold) & quality]),
                           torch.median(v[(t >= threshold) & quality]))
    return torch.stack([t, v-v_offset, v_offset, quality])

def correct_velocity_offset(aux_file, threshold=800):
    t, v, v_offset = torch.load(aux_file)
    quality = t>0
    return torch.stack([t, v, v_offset, quality])

def log_cuda_info(device,cuda_log_path="cuda_info.log", smi_log_path="nvidia_smi.log"):
    """
    Logs basic CUDA device info and optional nvidia-smi output.
    """
    lines = [f"Default device: {device}\n"]

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        lines.append("CUDA is available.")
        lines.append(f"Current device ID: {device_id}")
        lines.append(f"Device name: {device_name}")
    else:
        lines.append("CUDA is NOT available.")

    # Print and save CUDA info
    for line in lines:
        print(line)
    with open(cuda_log_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    # Try to log nvidia-smi output
    try:
        smi_output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        with open(smi_log_path, "w") as f:
            f.write(smi_output)
    except Exception as e:
        print(f"Failed to run nvidia-smi: {e}")
    return

def filter_unique_periods(all_periods,basefrac=0.02):
    uniq_periods = np.zeros_like(all_periods)
    for i,P in enumerate(all_periods):
        if P>100:frac = 0.3
        else:frac = basefrac 

        if np.abs(uniq_periods/P-1).min()<frac:continue
        else:uniq_periods[i]=P
    print("uniq_periods:",uniq_periods)
    return uniq_periods>0

def train(models,
          instruments,
          trainloaders,
          templates,
          aux_data=None,
          planet_param=None,
          skymask=None,
          n_epoch=200,
          outfile=None,
          losses=None,
          verbose=False,
          lr=1e-4,
          n_batch=50,
          aug_fcts=None,
          similarity=True,
          consistency=True,
          flexibility=True,
          stellar_activity=True,
          telluric=True,
          skipz=False
          ):

    n_inst = len(models)
    model_parameters, n_parameters = get_all_parameters(models,instruments)

    if verbose:
        print("model parameters:", n_parameters)
        mem_report()

    ladder = build_ladder(train_sequence)
    optimizer = optim.Adam(model_parameters, lr=lr, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                              total_steps=n_epoch)

    accelerator = Accelerator(mixed_precision='fp16')
    #accelerator = Accelerator()
    models = [accelerator.prepare(model) for model in models]
    instruments = [accelerator.prepare(instrument) for instrument in instruments]
    trainloaders = [accelerator.prepare(loader) for loader in trainloaders]

    optimizer = accelerator.prepare(optimizer)
    device = instruments[0].wave_obs.device
    log_cuda_info(device)
    templates = [item.to(device) for item in templates]
    aux_data = aux_data.to(device)
    print("templates:",templates)
    torch.autograd.set_detect_anomaly(True)
    # define losses to track
    n_loss = 6
    epoch = 0
    if losses is None:
        detailed_loss = np.zeros((2, n_inst, n_epoch, n_loss))
    else:
        try:
            non_zero = np.sum(losses[0][0],axis=1)!=0
            losses = losses[:,:,non_zero,:]

            epoch = len(losses[0][0])

            n_epoch += epoch
            detailed_loss = np.zeros((2, n_inst, n_epoch, n_loss))
            detailed_loss[:, :, :epoch, :] = losses

            if verbose:
                losses = tuple(detailed_loss[0, :, epoch-1, :])
                vlosses = tuple(detailed_loss[1, :, epoch-1, :])
                print(f'====> Epoch: {epoch-1}')
                print('TRAINING Losses:', losses)
                print('VALIDATION Losses:', vlosses)
        except: # OK if losses are empty
            print("loss empty...")
            pass

    if outfile is None:
        outfile = "checkpoint.pt"

    for epoch_ in range(epoch, n_epoch):

        mode = train_sequence[ladder[epoch_ - epoch]]

        # turn on/off model decoder
        for p in models[0].decoder.parameters():
            p.requires_grad = mode['decoder']
        models[0].decoder.spec_rest.requires_grad = mode['spec_rest']

        slope = ANNEAL_SCHEDULE[(epoch_ - epoch)%len(ANNEAL_SCHEDULE)]
        if n_epoch-epoch_<=10: slope=0 # turn off similarity
        
        if verbose and similarity:
            print("similarity info:",slope)

        for which in range(n_inst):
            # turn on/off encoder
            print("Encoder:",mode['encoder'][which])
            if models[which].encoder is not None:
                for p in models[which].encoder.parameters():
                    p.requires_grad = mode['encoder'][which]
            # turn on/off rv_estimator
            print("RV estimator:",mode['rv'][which])
            for p in models[which].rv_estimator.parameters():
                p.requires_grad = mode['rv'][which]
            if models[which].telluric is not None:
                for p in models[which].telluric.parameters():
                    p.requires_grad = mode['telluric']
            if not mode['fringe']:models[which].fringe = None
            if models[which].fringe is not None:
                for p in models[which].fringe.parameters():
                    p.requires_grad = mode['fringe']
            print("Telluric:",mode['telluric'])
            print("Fringe:",mode['fringe'])

            # optional: training on single dataset
            if not mode['data'][which]:
                continue

            models[which].train()
            instruments[which].train()

            n_sample = 0
            for k, batch in enumerate(trainloaders[which]):
                batch_size = len(batch[0])
                losses = get_losses(
                    models[which],
                    instruments[which],
                    batch,
                    templates[which],
                    aux_data=aux_data,
                    planet_param=planet_param,
                    skymask=skymask,
                    aug_fct=aug_fcts[which],
                    similarity=similarity,
                    consistency=consistency,
                    flexibility=flexibility,
                    slope=slope,
                    stellar_activity=stellar_activity,
                    telluric=telluric,
                    skipz=skipz
                )
                # sum up all losses
                loss = functools.reduce(lambda a, b: a+b , losses)
                accelerator.backward(loss)
                # clip gradients: stabilizes training with similarity
                accelerator.clip_grad_norm_(model_parameters[0]['params'], 1.0)
                # once per batch
                optimizer.step()
                optimizer.zero_grad()

                # logging: training
                detailed_loss[0][which][epoch_] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                n_sample += batch_size

                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break
            detailed_loss[0][which][epoch_] /= n_sample

        scheduler.step()

        if verbose:
            #mem_report()
            losses = tuple(detailed_loss[0, :, epoch_, :])
            vlosses = tuple(detailed_loss[1, :, epoch_, :])
            print('====> Epoch: %i'%(epoch_))
            print('TRAINING Losses:', losses)
            print('VALIDATION Losses:', vlosses)

        if epoch_ % 100 == 0 or epoch_ == n_epoch - 1:
            args = models
            checkpoint(accelerator, args, optimizer, scheduler, n_inst, outfile, detailed_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="dataset name")
    parser.add_argument("outfile", help="output file name")
    parser.add_argument("-dir","--dir", help="data file directory", default="/scratch/gpfs/JNWINN/yanliang/neid-production/")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=512)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-a", "--amp", help="planet signal amplitude", type=float, default=0.)
    parser.add_argument("-per", "--period", help="planet signal period", type=float, default=100.1)
    parser.add_argument("-phase", "--phase", help="initial phase (0~1)", type=float, default=0.)
    parser.add_argument("-z", "--rv_file", help="rv estimator", type=str, default="None")
    parser.add_argument("-sky", "--sky_file", help="telluric model", type=str, default="None")
    parser.add_argument("-it", "--iteration", help="number of interation", type=int, default=100000)
    parser.add_argument("-s", "--similarity", help="add similarity loss", action="store_true")
    parser.add_argument("-star_act", "--star_act", help="enable stellar activity", action="store_true")
    parser.add_argument("-skipz", "--skipz", help="skip rv loss", action="store_true",default=False)
    parser.add_argument("-c", "--consistency", help="add consistency loss", action="store_true")
    parser.add_argument("-d", "--double", help="double precision", action="store_true",default=False)
    parser.add_argument("-f", "--flexibility", help="constrian model flexibility", action="store_true")
    parser.add_argument("-seed", "--planet_seed", help="specify planet seed file",type=str, default="ccf")
    parser.add_argument("-new_seed", "--new_planet_seed", help="Override planet solution", action="store_true")
    parser.add_argument("-telluric", "--telluric", help="train telluric model", action="store_true")
    parser.add_argument("-fringe", "--fringe", help="train fringe model", action="store_true")
    parser.add_argument("-C", "--clobber", help="continue training of existing model", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    parser.add_argument("-debug", "--debug", help="show diagnostic plots", action="store_true")
    parser.add_argument("-shuffle", "--shuffle", help="shuffle inputs", action="store_true")
    args = parser.parse_args()
    
    if args.debug:torch.cuda.set_device('cuda:3')
    print("args.outfile:",args.outfile)
    basename = os.path.basename(args.outfile)
    basename = "_".join(basename.split("_")[:-1])
    if args.skipz:
        seed_file = f"initial_guess/{basename}_{args.planet_seed}.txt"
        print(f"Loading from {seed_file}...")
        planet_params = np.loadtxt(seed_file)
        #print("planet_params:",planet_params)
        min_K = 0.1 # m/s
        #planet_params[:,0] *= 0.8
        # reject repetitive entries
        mask = planet_params[:,0]>min_K
        mask &= filter_unique_periods(planet_params[:,1], basefrac=0.05)#baselinefrac=0.15
        planet_params = planet_params[mask]
        
        #rank = np.argsort(planet_params[:,1])
        #planet_params = planet_params[rank]
        #planet_params = planet_params[1::2]
        #mask &= planet_params[:,1]<300
        #mask &= np.abs(planet_params[:,1]-3.16)<0.1
        #planet_params[:,0] = 0.3
        if not args.consistency:
            planet_params = np.zeros((2,3))
        
    else:
        planet_params = np.zeros((2,3))

    print("data:",args.data)
    templates = []
    trainloaders = []
    instruments = []
    models = []

    if args.skipz:
        aux_data = correct_velocity_offset(f"{args.dir}/aux/{basename}_v_apparent_full.pkl")
        latents = torch.load(f"{args.dir}/aux/trad_latents.pkl")
        aux_data = torch.cat([aux_data,latents])

    else: aux_data = torch.zeros((1,4))
    
    template_data = load_batch("%s/merge/%s-template.pkl"%(args.dir,args.data))
    wave_obs = wave_rest = template_data[0]
    init_restframe = template_data[1].float().detach().clone()

    instrument = Synthetic(wave_obs)


    model = SpectrumAutoencoder(instrument,
                                wave_rest,
                                spec_rest=init_restframe,
                                weight_rest=None,
                                planet_params=planet_params,
                                n_latent=args.latents,
                                normalize=False,
                                skip_encoding=not args.star_act)

    data_loader = instrument.get_data_loader(f"{args.dir}/ccf_info", select=f"{args.data}", which="train",batch_size=args.batch_size, shuffle=args.shuffle)

    templates.append(template_data[1])
    trainloaders.append(data_loader)
    instruments.append(instrument)
    models.append(model)

    # number of different instruments
    n_inst = len(instruments)

    if args.double:
        template_data = [item.double() for item in template_data]
        if args.init: init_restframe = init_restframe.double()

    # get augmentation function
    aug_fcts = [ inst.augment_spectra  for inst in instruments]
    planet_param = [args.amp,args.period,args.phase]

    # define training sequence
    FULL = {"data":[True],"encoder":[True],"rv":[True],
            "decoder":True,"spec_rest":False,
            "telluric":args.telluric, "fringe":args.fringe}
    train_sequence = prepare_train([FULL],niter=args.iteration)

    annealing_step = 100
    ANNEAL_SCHEDULE = np.linspace(1.0,1.0,annealing_step)
    ANNEAL_SCHEDULE = np.hstack((ANNEAL_SCHEDULE,np.zeros(4*annealing_step)))
    if args.verbose and args.similarity:
        print("similarity_slope:",len(ANNEAL_SCHEDULE),ANNEAL_SCHEDULE)

    #print("Encoder: n_latent=%d"%models[0].encoder.n_latent)
    #print("Decoder: n_latent=%d"%models[0].decoder.n_latent)
    #print("Telluric: n_latent=%d"%models[0].telluric.n_latent)

    # always use the same doppler model
    for i in range(n_inst):
        if i==0:continue
        models[i].doppler_model = models[0].doppler_model

    if args.double:[model.double() for model in models]
    n_epoch = sum([item['iteration'] for item in train_sequence])
    init_t = time.time()
    if args.verbose:
        print("torch.cuda.device_count():",torch.cuda.device_count())
        if torch.cuda.device_count()==0:
            print("No visible cuda device! MIG?")
            exit()
        print (f"--- Model {args.outfile} ---")

    if args.skipz:
        print("Skipping... RV Estimator...")
        models[0].rv_estimator = NullRVEstimator()
    # check if outfile already exists, continue only of -c is set
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit("\nOutfile exists! Set option -C to continue training.")
    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print("\nLoading file %s"%args.outfile)
        models, losses = load_model(args.outfile, models, instruments,new_act=args.new_planet_seed)

    if os.path.isfile(args.rv_file):
        if args.verbose:
            print("\nUpdating RV estimator based on file %s"%args.rv_file)
        models = update_rv_estimator(args.rv_file, models, instruments)
    if os.path.isfile(args.sky_file) or args.sky_file == "null":
        print("\nUpdating telluric model based on file %s"%args.sky_file)
        models = update_telluric(args.sky_file, models, instruments)

    profiler = LineProfiler()
    profiler.add_function(get_losses)
    lpWrapper = profiler(train)
    lpWrapper(models, instruments, trainloaders, templates, aux_data=aux_data, skymask=None, planet_param=planet_param,n_epoch=n_epoch,
          n_batch=args.batch_number, lr=args.rate, aug_fcts=aug_fcts, similarity=args.similarity, consistency=args.consistency, flexibility=args.flexibility, stellar_activity=args.star_act,skipz=args.skipz,telluric=args.telluric,outfile=args.outfile, losses=losses, verbose=args.verbose)
    
    profiler.print_stats()

    if args.verbose:
        print("--- %s seconds ---" % (time.time()-init_t))
