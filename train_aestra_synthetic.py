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
from spender_model import SpectrumAutoencoder,NullRVEstimator
from synthetic_data import Synthetic
from util import mem_report
from functools import partial
from util import load_batch,interpolate_to_input_grid,normalize_residual,divide_sky_model
from util import simulate_planet
from torch.utils.data import DataLoader,Dataset
from torchinterp1d import Interp1d
from line_profiler import LineProfiler
from scipy.special import digamma

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

    # telluric model training
    if model.telluric is not None:
        model_params += model.telluric.parameters()
    if model.fringe is not None:
        model_params += model.fringe.parameters()
    if model.encoder is not None:
        model_params += model.encoder.parameters()
        model_params += model.decoder.parameters()
        model_params += model.activity_estimator.parameters()
        model_params += model.rv_estimator.parameters()

    dicts = [{'params':model_params}]
    n_parameters = sum([p.numel() for p in model_params if p.requires_grad])

    return dicts,n_parameters

def consistency_loss(s, s_aug, individual=False, sigma_s=0.5):
    batch_size, s_size = s.shape
    ds = torch.sum((s_aug - s)**2/(sigma_s)**2,dim=1)/(s_size)
    cons_loss = torch.sigmoid(ds)-0.5# zero = perfect alignment
    if individual:
        return cons_loss
    return cons_loss.sum()

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

def plot_diagnostic(diags,instrument,n_window=2):
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    raw_data = [tensor2array(item) for item in diags["raw_data"]]
    if "rv" in diags:rv,rv_aug,v_offset,v_planet = [tensor2array(item[:,0]) for item in diags["rv"]]
    if "model" in diags:spec_obs = tensor2array(diags["model"])
    if "telluric" in diags: 
        spec_telluric = tensor2array(diags["telluric"])
        fringe = tensor2array(diags["fringe"])
    else: spec_telluric=None
    if "y_act" in diags: y_act = tensor2array(diags["y_act"])

    template_data = [tensor2array(item[0]) for item in diags["template"]]
    spec_input,spec_aug,w = [tensor2array(item) for item in diags["input_data"]]

    if "rv" in diags:
        poly,cov = np.polyfit(v_offset,rv_aug-rv,deg=1,cov=True)
        slope_uncertainty = cov[0][0]**0.5 
        vlabel = "slope = %.3f+/-%.3f"%(poly[0],slope_uncertainty)
        print("\npredicted offset vs. true offset:",vlabel)
        fig,axs=plt.subplots(figsize=(5,5),nrows=2,constrained_layout=True)
        ax=axs[0]
        ax.plot(v_offset,rv_aug-rv,"k.",ms=3)
        ax.plot(v_offset,np.polyval(poly,v_offset), "r--",label=vlabel)
        ax.legend()
        ax=axs[1]
        poly,cov = np.polyfit(v_planet,rv,deg=1,cov=True)
        slope_uncertainty = cov[0][0]**0.5 
        vlabel = "slope = %.3f+/-%.3f"%(poly[0],slope_uncertainty)
        print("Truth vs. Encoded RV:",vlabel)
        print("RMS: %.2f m/s"%(rv).std())
        ax.plot(v_planet,rv,"k.",label=vlabel)
        ax.set_xlabel("true planetary Doppler shift")
        ax.set_ylabel("v_encode")
        ax.legend()
        plt.savefig("[v_encode]test.png",dpi=200)
        plt.clf()

    if "telluric" in diags:
        x = spec_input.mean(axis=-1)[:,0]
        fig,ax=plt.subplots(figsize=(5,3),constrained_layout=True)
        ax.plot(x,spec_obs.mean(axis=-1)[:,0],"k.",ms=3,label="spec model")
        ax.plot(x,fringe.mean(axis=-1)[:,0],".",c="cyan",label="fringe")
        ax.legend()
        plt.savefig("[yoffset]test.png",dpi=200)
        plt.clf()

    if not "model" in diags:exit()
    wave_raw,spec_raw,w_raw,ssbrv,jd = raw_data
    wave_obs,template,w_template,_,_ = template_data

    n_batch,n_order,n_spec = spec_raw.shape
    temp_err = w_template**(-0.5)

    loss = w*(spec_input-spec_obs)**2
    loss_ind = np.sum(loss, axis=-1) / np.sum(w>1,axis=-1)
    loss_avg = gaussian_filter1d(loss.mean(axis=0),2)
    loss_avg -= np.quantile(loss_avg,0.3)
    print("loss_avg:",loss_avg.shape)

    print("masked:",(w<=1).sum()/(n_batch*n_order*n_spec))
    print("loss:",loss_ind.shape,loss_ind.mean())
    sky_z = ssbrv/instrument.c

    diag = np.copy(loss_ind)
    diag_full = loss

    #diag = np.abs(spec_obs.sum(axis=-1))
    #diag_full = np.abs(spec_obs)

    i,o_max = np.where(diag==diag.max())
    #i,o_max,i_bin = np.where(diag==diag.max())
    i=i[0]
    show_orders = [o_max[0]]
    show_bins = [np.argmax(diag_full[i,o_max])]

    for k in range(n_window-1):
        diag[i,o_max]=0
        o_max = np.argmax(diag[i])
        smooth_diag = gaussian_filter1d(diag_full[i,o_max],2)
        show_orders.append(o_max)
        show_bins.append(np.argmax(smooth_diag))

    print("show_orders:",show_orders,"show_bins:",show_bins)
    drawstyle = "steps-mid"
    stepstyle = "mid"

    window = [1,10]
    ncols = len(show_orders)
    colors = ["k","b","darkgreen"]*n_order
    c_err = ["lightgrey","lavender","palegreen"]*n_order
    fig,axs=plt.subplots(figsize=(12,8),ncols=ncols,nrows=3,
                         constrained_layout=True)
    for k in range(ncols):
        wh = show_bins[k]
        center = wave_raw[i][show_orders[k]][wh]
        for o in range(n_order):
            if wave_obs[o].min()>center or wave_obs[o].max()<center:continue
            if o==show_orders[k]:
                c_order = "orange"
                err_c = "wheat"
                zorder = 0
            else:
                c_order = colors[o]
                err_c = c_err[o]
                zorder = None

            ax = axs[0][k]
            spec_err = w_raw[i][o]**(-0.5)
            ax.fill_between(wave_raw[i][o]*(1+sky_z[i][o]),spec_raw[i][o]-spec_err,spec_raw[i][o]+spec_err,color=err_c,step=stepstyle,zorder=-10)
            ax.plot(wave_raw[i][o]*(1+sky_z[i][o]),spec_raw[i][o],"-",color=c_order,drawstyle=drawstyle,label="order %d data"%o,zorder=zorder)
            if spec_telluric is not None:
                ax.plot(wave_obs[o]*(1+sky_z[i][o]),spec_telluric[i][o],"-",lw=0.5,drawstyle=drawstyle,color="b",label="telluric")

        # order specific info
        o = show_orders[k]
        xlim = [wave_raw[i][o][wh]-window[k],wave_raw[i][o][wh]+window[k]]
        #if k==1: xlim = (5440,5455)
        loss_ind = loss[i][o]
        print("i,o:",i,o)
        loss_io = loss_ind.sum()/(w[i][o]>1).sum()
        print("loss_ind:",loss_io,sorted(loss_ind,reverse=True)[:10])

        ax.plot(wave_obs[o],loss_ind/loss_ind.max(),"-",color="grey",lw=1.0,drawstyle="steps-mid",label="order %d loss = %.2f"%(o,loss_io))
        ax.plot(wave_obs[o],loss_avg[o],"-",color="r",lw=1.0,drawstyle="steps-mid",label="mean loss")
        ax.fill_between(wave_raw[i][o],0,1/loss_ind.max(),color="lightgrey",zorder=-20)
        ax.set_ylim(0,1.2)

        ax = axs[1][k]
        ax.plot(wave_obs[o],spec_input[i][o],c="k",lw=1,drawstyle="steps-mid",label="order %d resid"%(o))
        #ax.plot(wave_obs[o,mask],spec_aug[i][o][mask],c="b",lw=1,drawstyle="steps-mid",label="order %d aug v=%.2f m/s"%(o,v_offset[i]))
        ax.plot(wave_obs[o],spec_obs[i][o],c="r",lw=1,drawstyle="steps-mid",label="order %d loss = %.2f"%(o,loss_io))
        if spec_telluric is not None:
            ax.plot(wave_obs[o],fringe[i][o],"-",lw=1.0,drawstyle=drawstyle,color="cyan",label="fringe")
        err = w[i][o]**(-0.5)
        ax.fill_between(wave_obs[o],spec_input[i][o]-err,spec_input[i][o]+err,color="k",alpha=0.3,step=stepstyle,zorder=-20)
        ax.set_ylim(-0.01,0.01)
        ax = axs[2][k]
        if "y_act" in diags: y_show,yname = y_act,"activity"
        else:y_show,yname = spec_input,"resid"

        for i_spec in range(n_batch):
            ax.plot(wave_obs[o],y_show[i_spec][o],c="grey",lw=1,alpha=0.5,drawstyle="steps-mid")
        ax.plot(wave_obs[o],y_show[i][o],c="k",lw=1,drawstyle="steps-mid",label="order %d %s"%(o,yname))
        for i_row in range(3):
            ax = axs[i_row][k]
            ax.set_xlim(xlim);
            #if k==1:ax.set_xlim(5434,5438);
            ax.legend()

    plt.savefig("test.png",dpi=300)
    exit()
    return

def get_losses(model,
               instrument,
               batch,
               template_data,
               skymask=None,
               aug_fct=None,
               similarity=True,
               consistency=True,
               flexibility=True,
               smoothness=True,
               slope=0,
               sigma_s=0.5,
               stellar_activity=True,
               skipz=False
               ):

    v_reg_loss = 0
    fid_loss = sim_loss = flex_loss = cons_loss = 0
    # Raw spectra are in the Earth frame
    wave_raw,spec_raw,w_raw,ssbrv,jd = batch
    template = template_data[1]

    z_null = torch.zeros((wave_raw.shape[0],1),device=wave_raw.device)

    if args.debug:slope=1.0
    if skymask is not None:
        print("Sky Mask Fraction: %.2f"%(skymask.sum()/torch.numel(skymask)))

    # evaluate telluric & fringe models
    if model.telluric is not None:
        # interpolate raw spectra: telluric and fringe patterns
        spec,w,_ = interpolate_to_input_grid(batch,instrument,template_data)
        # normalize residual model
        spec_input = normalize_residual(spec,w,template)
        y_fringe = model.fringe(spec_input)
        fringe_spec = model.fringe.cubic_interpolation(y_fringe, z_null)
        print("fringe_spec:",fringe_spec.std().item())
        s_sky = model.telluric.encode(spec_input)
        # telluric lines are defined in the Earth frame -- shift to stellar frame
        spec_sky = model.telluric(s_sky,ssbrv/instrument.c,
                                  instrument.wave_obs,None)
    else:
        fringe_spec = 0
        spec_sky = 1

    # telluric & fringe pre-training, no rv model, no activity
    if not stellar_activity:
        # intrinsic stellar model - no variability
        spectrum_restframe = model.decoder.spec_rest.repeat(z_null.shape[0],1,1)
        spectrum_observed = model.decoder.transform(spectrum_restframe, z_null, instrument.wave_obs)
        # full model = intrinsic stellar model * telluric model
        spectrum_observed = spectrum_observed*spec_sky
        # normalize residual model
        model_resid = spectrum_observed-template
        model_resid += fringe_spec
        model_resid[w<1.0] = 0

        # compare residual model
        fid_loss = model._loss(spec_input, w, model_resid)
        # constrain telluric flexibility
        continuum = spec_sky>torch.quantile(spec_sky,0.05)
        flex_loss = 50*slope*((1-spec_sky[continuum])**2).sum()

    if skipz:
        z = z_null
        z_loss = 0
        spec_input_aug = spec_input
    else:
        # inject planet
        _,v_planet = simulate_planet(jd,amp=0.5,period=100.1,t0=0.0)
        z_sky = (v_planet+ssbrv)/instrument.c

        spec,w,_ = interpolate_to_input_grid(batch,instrument,template_data,planetary_rv=v_planet)
        spec_sky = model.telluric(s_sky,z_sky,instrument.wave_obs,skymask)
        spec_input = divide_sky_model(spec,w,spec_sky,fringe_spec,template)

        #  generate augment spectra
        spec_aug,w_aug,z_off_true = interpolate_to_input_grid(batch,instrument,template_data,aug=True,planetary_rv=v_planet)
        spec_sky_aug = model.telluric(s_sky,z_sky+z_off_true,instrument.wave_obs,skymask)
        spec_input_aug = divide_sky_model(spec_aug,w_aug,spec_sky_aug,fringe_spec,template)

        rv =  model.estimate_rv(spec_input)
        z = (rv)/instrument.c

        rv_aug = model.estimate_rv(spec_input_aug)

        z_off = (rv_aug - rv)/instrument.c
        z_loss = z_offset_loss(z_off, z_off_true)
        print("z_loss:",z_loss.item(),
              "RV: %.2f, %.2f"%(rv.min().item(),rv.max().item()),
              "RV_aug: %.2f, %.2f"%(rv_aug.min().item(),rv_aug.max().item()))

    # stellar acitivity training
    if stellar_activity:
        s = model.encode(spec_input)
        y_act, spectrum_restframe, spectrum_observed = model._forward(s, z)
        # normalize residual model
        model_resid = spectrum_observed-template
        model_resid[w<1.0] = 0

        print("s:",s.std(dim=0))
        print("y_act:",y_act.std(dim=0).mean())

        # compare residual model
        fid_loss = model._loss(spec_input, w, model_resid)

    else: s = 0.0

    if smoothness:
        sigma_v = 5 # m/s
        #v_activity = model.estimate_v_act(s)
        #v_doppler = rv-v_activity
        v_reg_loss = (rv**2/sigma_v**2).sum()
        print("v_reg_loss:",v_reg_loss)

    if stellar_activity:
        if consistency:
            s_aug = model.encode(spec_input_aug)
            cons_loss = consistency_loss(s, s_aug)
        if flexibility:
            #flex_loss = (w*spec_input**2).mean(dim=-1).sum()
            flex_loss += slope*(y_act**2).sum()
            #print("flex_loss:",flex_loss)

    if similarity:
        sim_loss = similarity_restframe(model, y_act, s, slope=slope,sigma_s=sigma_s)

    if args.debug:
        diags = {"raw_data":[wave_raw,spec_raw,w_raw,ssbrv,jd],
                 "input_data":[spec_input,spec_input_aug,w],
                 "template":template_data}

        if model.telluric is not None:
            diags["model"] = model_resid
            diags["telluric"] = spec_sky
            diags["fringe"] = fringe_spec
        if stellar_activity:
            diags["model"] = model_resid
            diags["y_act"] = y_act

        if not skipz:diags["rv"]=[rv,rv_aug,z_off_true*instrument.c,v_planet]

        if similarity:
            slope = 1.0
            s_sim,spec_sim,sim_loss = similarity_restframe(model, y_act, s, 
                                                           slope=slope,sigma_s=sigma_s,individual=True)
            plot_similarity(s_sim,spec_sim,sim_loss,sigma_s=sigma_s,slope=slope)
        plot_diagnostic(diags,instrument)
    return fid_loss, sim_loss, z_loss, cons_loss, v_reg_loss, flex_loss


def checkpoint(accelerator, args, optimizer, scheduler, n_encoder, outfile, losses):
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

def load_model(mainfile, models, instruments):
    device = instruments[0].wave_obs.device
    model_struct = torch.load(mainfile, map_location=device)

    for i, model in enumerate(models):
        model_dict = model_struct['model'][i]
        model.load_state_dict(model_dict, strict=False)

    losses = model_struct['losses']
    return models, losses

def train(models,
          instruments,
          trainloaders,
          template_data,
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
          skipz=False
          ):

    n_encoder = len(models)
    model_parameters, n_parameters = get_all_parameters(models,instruments)

    if verbose:
        print("model parameters:", n_parameters)
        mem_report()

    ladder = build_ladder(train_sequence)
    optimizer = optim.Adam(model_parameters, lr=lr, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                              total_steps=n_epoch)

    #accelerator = Accelerator(mixed_precision='fp16')
    accelerator = Accelerator()
    models = [accelerator.prepare(model) for model in models]
    instruments = [accelerator.prepare(instrument) for instrument in instruments]
    trainloaders = [accelerator.prepare(loader) for loader in trainloaders]
    template_data = [accelerator.prepare(item) for item in template_data]
    optimizer = accelerator.prepare(optimizer)

    device = instruments[0].wave_obs.device
    template_data = [item.to(device) for item in template_data]
    torch.autograd.set_detect_anomaly(True)
    # define losses to track
    n_loss = 6
    epoch = 0
    if losses is None:
        detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))
    else:
        try:
            non_zero = np.sum(losses[0][0],axis=1)>0
            losses = losses[:,:,non_zero,:]

            epoch = len(losses[0][0])

            n_epoch += epoch
            detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))
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

        for which in range(n_encoder):
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
                for p in models[which].fringe.parameters():
                    p.requires_grad = mode['telluric']
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
                    template_data,
                    skymask=skymask,
                    aug_fct=aug_fcts[which],
                    similarity=similarity,
                    consistency=consistency,
                    flexibility=flexibility,
                    slope=slope,
                    stellar_activity=stellar_activity,
                    skipz=skipz
                )
                # sum up all losses
                loss = functools.reduce(lambda a, b: a+b , losses)
                accelerator.backward(loss)
                # clip gradients: stabilizes training with similarity
                accelerator.clip_grad_norm_(model_parameters[0]['params'], 1.0)
                # once per batch
                optimizer.step()

                if models[which].decoder.lsf is not None:
                    print("update lsf!")
                    # lsf weights are non-negative & sum to 1
                    non_negative_weights = torch.clamp(models[which].decoder.lsf.weight.data, min=0)
                    norm_weights = non_negative_weights / non_negative_weights.sum(dim=-1)[:,:,None]
                    models[which].decoder.lsf.weight.data = norm_weights

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

        if epoch_ % 30 == 0 or epoch_ == n_epoch - 1:
            args = models
            checkpoint(accelerator, args, optimizer, scheduler, n_encoder, outfile, detailed_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="dataset name")
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("outfile", help="output file name")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=512)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-z", "--rv_file", help="rv estimator", type=str, default="None")
    parser.add_argument("-sky", "--sky_file", help="telluric model", type=str, default="None")
    parser.add_argument("-it", "--iteration", help="number of interation", type=int, default=100000)
    parser.add_argument("-s", "--similarity", help="add similarity loss", action="store_true")
    parser.add_argument("-star_act", "--star_act", help="enable stellar activity", action="store_true")
    parser.add_argument("-skipz", "--skipz", help="skip rv loss", action="store_true",default=False)
    parser.add_argument("-c", "--consistency", help="add consistency loss", action="store_true")
    parser.add_argument("-d", "--double", help="double precision", action="store_true",default=False)
    parser.add_argument("-f", "--flexibility", help="constrian model flexibility", action="store_true")
    parser.add_argument("-telluric", "--telluric", help="train telluric model", action="store_true")
    parser.add_argument("-fringe", "--fringe", help="train fringe model", action="store_true")
    parser.add_argument("-C", "--clobber", help="continue training of existing model", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    parser.add_argument("-debug", "--debug", help="show diagnostic plots", action="store_true")
    args = parser.parse_args()

    init_rest = load_batch("%s%s-rest.pkl"%(args.dir,args.data))
    wave_obs = wave_rest = init_rest[0]
    init_restframe = init_rest[1].float()

    try:skymask = load_batch("%s%s-skymask.pkl"%(args.dir,args.data)).bool()
    except: skymask = None

    # define instruments
    instruments = [ Synthetic(wave_obs) ]
    n_encoder = len(instruments)

    # data loaders
    trainloaders = [ inst.get_data_loader(args.dir, select=args.data, which="train",
                     batch_size=args.batch_size) for inst in instruments ]

    template_data = load_batch("%s%s-template.pkl"%(args.dir,args.data))

    if args.double:
        template_data = [item.double() for item in template_data]
        if args.init: init_restframe = init_restframe.double()

    # get augmentation function
    aug_fcts = [ inst.augment_spectra  for inst in instruments]

    # define training sequence
    FULL = {"data":[True],"encoder":[True],"rv":[True],
            "decoder":True,"spec_rest":True,
            "telluric":args.telluric, "fringe":args.fringe}
    train_sequence = prepare_train([FULL],niter=args.iteration)

    annealing_step = 100
    ANNEAL_SCHEDULE = np.linspace(0.0,1.0,annealing_step)
    if args.verbose and args.similarity:
        print("similarity_slope:",len(ANNEAL_SCHEDULE),ANNEAL_SCHEDULE)

    # define and train the model
    n_hidden = (64, 256, 1024)
    models = [ SpectrumAutoencoder(instrument,
                                   wave_rest,
                                   spec_rest=init_restframe,
                                   weight_rest=None,
                                   n_latent=args.latents,
                                   n_hidden=n_hidden,
                                   normalize=False,
                                   skip_encoding=not args.star_act)
              for instrument in instruments ]

    #print("Encoder: n_latent=%d"%models[0].encoder.n_latent)
    #print("Decoder: n_latent=%d"%models[0].decoder.n_latent)
    print("Telluric: n_latent=%d"%models[0].telluric.n_latent)

    # use same decoder
    if n_encoder==2:models[1].decoder = models[0].decoder
    if args.double:[model.double() for model in models]
    n_epoch = sum([item['iteration'] for item in train_sequence])
    init_t = time.time()
    if args.verbose:
        print("torch.cuda.device_count():",torch.cuda.device_count())
        print (f"--- Model {args.outfile} ---")

    # check if outfile already exists, continue only of -c is set
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit("\nOutfile exists! Set option -C to continue training.")
    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print("\nLoading file %s"%args.outfile)
        models, losses = load_model(args.outfile, models, instruments)

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
    lpWrapper(models, instruments, trainloaders, template_data, skymask=skymask, n_epoch=n_epoch,
          n_batch=args.batch_number, lr=args.rate, aug_fcts=aug_fcts, similarity=args.similarity, consistency=args.consistency, flexibility=args.flexibility, stellar_activity=args.star_act,skipz=args.skipz,outfile=args.outfile, losses=losses, verbose=args.verbose)
    
    profiler.print_stats()

    if args.verbose:
        print("--- %s seconds ---" % (time.time()-init_t))
