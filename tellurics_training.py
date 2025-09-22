#!/usr/bin/env python

import time, argparse, os
import numpy as np
import functools
import torch
import subprocess
from torch import nn
from torch import optim
from accelerate import Accelerator
# allows one to run fp16_train.py from home directory
import sys;sys.path.insert(1, './')
from spender_model import TelluricModel,gaussian_kernel_1d
from synthetic_data import Synthetic
from util import mem_report
from functools import partial
from util import load_batch,interpolate_to_input_grid_raw
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

def get_all_parameters(models,instruments,lr):
    model_params = []
    model = models[0]
    accelerate_params = []
    for p in model.parameters():
        if p.shape[0] == 4:accelerate_params.append(p)
        else:model_params.append(p)

    print("accelerate_params:",accelerate_params)
    dicts = [{'params':model_params,"lr":lr},
             {'params':accelerate_params,"lr":50*lr}]
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

def plot_diagnostic(diags,model,instrument,ratio=1,n_window=2):
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from astropy.timeseries import LombScargle
    from util import moving_mean
    if "model" in diags:
        model = tensor2array(diags["model"])
        lines = tensor2array(diags["lines"].squeeze(1))
        continuum = tensor2array(diags["continuum"].squeeze(1))
        y_act = tensor2array(diags["y_act"])
        correlation = tensor2array(diags["correlation"])
        lsf = tensor2array(diags["lsf"].squeeze(0))
        polyb = tensor2array(diags["polyb"])
        print("lines range:",np.quantile(lines,[0,0.5,1.0]))
    wave_obs = tensor2array(instrument.wave_obs[0])
    template = tensor2array(diags["template"][0])
    print("template:",len(template))
    spec = tensor2array(diags["spec"])
    w = tensor2array(diags["w"])

    n_batch,n_spec = spec.shape
    loss = w*(spec-model)**2
    loss_ind = np.sum(loss, axis=-1) / np.sum(w>1,axis=-1)

    loss_avg = gaussian_filter1d(loss.mean(axis=0),2)
    loss_avg -= np.quantile(loss_avg,0.3)
    print("loss_ind:",loss_ind.shape)

    print("w:",w.shape,w.mean())
    print("masked:",(w<=1).sum()/(n_batch*n_spec))
    print("loss:",loss_ind.shape,loss_ind.mean())

    plt.figure(figsize=(5,4),dpi=200)
    lsf_init = gaussian_kernel_1d(sigma=3,kernel_size=41)[0][0]
    plt.plot(lsf_init,'k--',alpha=0.5,label="initialization")
    plt.plot(lsf[0],'k-',label="trained")
    plt.title(f"Order {args.orders[0]}")
    plt.legend()
    plt.savefig("lsf.png")


    plt.figure(figsize=(5,4),dpi=200)
    for i in range(10):
        plt.plot(polyb[i],'k-',alpha=0.2)
    plt.title(f"Order {args.orders[0]}")
    plt.savefig("polyb.png")

    diag = np.copy(loss_ind)

    print("loss:",loss.shape)
    i,wh = np.argwhere(loss==loss.max())[0]
    #i,wh = np.argwhere(spec==spec.max())[0]

    print("i:",i,"wh:",wh)

    drawstyle = "steps-mid"
    stepstyle = "mid"
    ncols = 1

    c_order = "k"
    err_c = "lightgrey"
    zorder = 0
    window = 3
    #colors = ["k","b","darkgreen"]*n_order
    #c_err = ["lightgrey","lavender","palegreen"]*n_order
    fig,axs=plt.subplots(figsize=(12,7),ncols=ncols,nrows=3,
                         gridspec_kw={'height_ratios': [2.5, 1,1]},
                         constrained_layout=True)
    for k in range(ncols):
        center = wave_obs[wh]
        ax = axs[0]#[k]
        spec_inflate = ratio*(spec[i]-template)+template
        #aug_inflate = ratio*spec_aug[i]+template
        model_inflate = ratio*(model[i]-template)+template
        ax.plot(wave_obs,template,"-",color='lightgrey',drawstyle=drawstyle,label="template",zorder=zorder)
        ax.plot(wave_obs,spec_inflate,"k-",drawstyle=drawstyle,label=f"data ({ratio}x inflated activity)",zorder=zorder)

        ax.plot(wave_obs,1-lines[i],"-",color='cyan',lw=1,drawstyle=drawstyle,label=f"lines")
        ax.plot(wave_obs,1+10*continuum[i],"-",color='b',lw=1,drawstyle=drawstyle,label=f"10xcontinuum")
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
        #ax.set_ylim(spec_inflate.min(),spec_inflate.max())
        ax.set_ylim(0.8,1.1)
        #ax.set_ylim(-0.002,0.002)
        ax = axs[1]#[k]
        #if "y_act" in diags: 
        #    y_show,yname = y_act,"y_act"
        #else:
        y_show,yname = continuum-1,"continuum"


        disp = y_show.std(axis=0)
        for i_spec in range(min(50,n_batch)):
            ax.plot(wave_obs,y_show[i_spec],c="grey",lw=1,alpha=0.5,drawstyle="steps-mid")
        ax.plot(wave_obs,y_show[i],c="k",lw=1,drawstyle="steps-mid",label="%s"%(yname))
        #ax.set_ylim(-0.01,0.01)
        #ylim = np.quantile(y_show,[0.01,0.99])
        #ax.set_ylim(ylim)
        ax = axs[2]
        ax.plot(wave_obs,correlation,"-",color="k",lw=1.0,drawstyle="steps-mid",label="correlation")
        #ax.set_ylim(-1,1)
    
        for i_row in range(3):
            ax = axs[i_row]#[k]
            xlim = [wave_obs[wh]-window,wave_obs[wh]+window]
            #xlim = [wave_obs[-1]-3,wave_obs[-1]+0.1]
            ax.set_xlim(xlim);
            ax.legend()

    plt.savefig("reconstruction.png",dpi=300)
    exit()
    return

def print_planet_solutions(p0,p,periods,n_string=10,corr=None):
    current_Ks = 0.5*(p.max(dim=-1)[0]-p.min(dim=-1)[0])
    rank = torch.argsort(current_Ks,descending=True)
    for i in rank:
        K0,P0 = p0[i,:2]
        #K,P = p[i,:2]
        P = torch.exp(periods[i])
        K = current_Ks[i]
        str1 = f"{P0:.2f}d"
        str1 += " "*(n_string-len(str1))
        str1 += f"K={K0:.2f}m/s"
        str2 = f"{P:.2f}d"
        str2 += " "*(n_string-len(str2))
        str2 += f"K={K.abs():.2f}m/s"
        if corr is None: str3 = ""
        else: str3 = f"  corr: {corr[i]:.3f}"
        print(str1," --> ",str2,str3)
    return

def hinge_loss_penalty(x, x0=14.35, dx=0.15):
    return torch.maximum(torch.zeros_like(x), dx - torch.abs(x - x0))/dx

def colwise_corr(a, b, eps = 1e-12):
    # a, b: shape (N, L) -> returns shape (L,)
    a = a.float()
    b = b.float()
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    num = (a_c * b_c).sum(dim=0)
    den = a_c.norm(dim=0) * b_c.norm(dim=0)
    return num / den.clamp_min(eps)

def get_losses(model,
               instrument,
               batch,
               template,
               planet_param=None,
               skymask=None,
               aug_fct=None,
               similarity=True,
               consistency=True,
               flexibility=True,
               regularize_v=True,
               slope=0,
               n_epoch=0,
               sigma_s=0.5,
               stellar_activity=True,
               skipz=False,
               telluric=True,
               ):

    wave_obs = instrument.wave_obs
    print("wave_obs:",wave_obs.shape)
    print("template:",template.shape)

    v_reg_loss = 0
    fid_loss = z_loss = sim_loss = flex_loss = cons_loss = 0

    #telluric_offset = model.evaluate_telluric_rv_offset(batch[4])
    #print(f"\nAdditional Telluric Offset:{telluric_offset.min().item():.3f} m/s, {telluric_offset.max().item():.3f} m/s")
    polyb = model.evaluate_wavelength_polynomial(batch[0],batch[4])
    print(f"\nWavelength shift:{polyb.std(dim=1).max()*instrument.c:.3f} m/s \n")

    # shift to the (apparent) stellar restframe - frame of wave_obs
    spectrum, w, ssbrv,jd = interpolate_to_input_grid_raw(batch,instrument,template,polyb=polyb)
    spec_input = spectrum - template
    spec_input[w<1] = 0
    
    batch_size = jd.shape[0]

    print("spec_input:",spec_input.max(),spec_input.min())
    cons_loss = 1e2*torch.sum((spec_input.std(dim=0)).pow(2))*batch_size
    if consistency:return fid_loss, sim_loss, z_loss, cons_loss, v_reg_loss, flex_loss

    wave_raw,spec_raw,w_raw,ssbrv,jd = batch
    #z_sky = (ssbrv+telluric_offset)/instrument.c
    z_sky = (ssbrv)/instrument.c
    s_sky = model.encode(spec_input)

    lines,continuum,y_act,spec_model =  model._forward(s_sky,z_sky,wave_obs,skymask=skymask)

    #print("spec_input:",spec_input.min(),spec_input.max())
    print(f"\nlines: {lines.std().item():.4f}",
          f"\ncontinuum:  {continuum.std().item():.4f}",
          f"\ny_act:  {y_act.std().item():.4f}")

    #corr_1 = colwise_corr(lines.squeeze(1),y_act)
    #corr_1 = colwise_corr(continuum.squeeze(1),lines.squeeze(1))
    #corr_3 = colwise_corr(continuum.squeeze(1),y_act)
    continuum_std = continuum.std(dim=0).squeeze(0)
 
    c_level = 0.005 # 0.002

    reg_term = torch.zeros_like(continuum_std)
    high_std = continuum_std>c_level
    reg_term[high_std] = 1e3*(continuum_std[high_std]-c_level)
    corr = reg_term
    print("corr:",corr.shape)
    print("y_act:",y_act.shape,lines.shape,continuum.shape)

    cons_loss = batch_size*torch.mean(reg_term)
    #print("correlation:",corr.min(),corr.max())

    loss_ind = w * (spectrum - spec_model).pow(2)
    loss_ind = torch.sum(loss_ind, dim=-1) / torch.sum(w>1,dim=-1)
    fid_loss = torch.sum(loss_ind)

    if args.debug:slope = 1
    mask = (y_act.std()<0.002)&(spectrum/(1-lines.squeeze(1))>0.9)
    flex_loss = 10*torch.sum((y_act[mask]).pow(2))
    flex_loss += batch_size*torch.sum((continuum.mean(dim=0)).pow(2))
    flex_loss += 2e-2*torch.sum((lines[lines<0.05].pow(2)))
    
    #mask = dispersion<2e-4

    #lines = lines.squeeze(1)
    #telluric_proxy = lines[:,torch.argmax(lines.std(dim=0))]
    #print("telluric_proxy:",telluric_proxy.shape)
    #correlation = pearson_corrcoef_batch(telluric_proxy,lines.T)
    #mask = correlation<0.5
    #print("mask:",mask.sum())
    #cons_loss = slope*torch.sum((lines[:,mask]).pow(2))

    print("fid_loss:",fid_loss/batch_size)
    print("flex_loss:",flex_loss/batch_size)
    print("cons_loss:",cons_loss/batch_size)

    print("\nlsf:",model.lsf_kernel.max().item())

    #if regularize_v:
    if args.debug or torch.isnan(fid_loss):
        diags = {"template":template,"spec":spectrum,
                 "lines":lines,"continuum":continuum,
                 "correlation":corr,'polyb':polyb,
                 'lsf':model.lines_act(model.lsf_kernel),
                 "y_act":y_act,"w":w,"model":spec_model}
        plot_diagnostic(diags,model,instrument)
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

def load_model(mainfile, models, instruments):
    device = instruments[0].wave_obs.device
    model_struct = torch.load(mainfile, map_location=device)

    for i, model in enumerate(models):
        init_dict = model.state_dict()
        model_dict = model_struct['model'][i]
        model.load_state_dict(model_dict, strict=False)
    losses = model_struct['losses']
    return models, losses

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

def train(models,
          instruments,
          trainloaders,
          templates,
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
    model_parameters, n_parameters = get_all_parameters(models,instruments,lr)

    if verbose:
        print("model parameters:", n_parameters)
        mem_report()

    ladder = build_ladder(train_sequence)
    optimizer = optim.Adam(model_parameters, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                              total_steps=n_epoch)

    #accelerator = Accelerator(mixed_precision='fp16')
    accelerator = Accelerator()
    models = [accelerator.prepare(model) for model in models]
    instruments = [accelerator.prepare(instrument) for instrument in instruments]
    trainloaders = [accelerator.prepare(loader) for loader in trainloaders]

    optimizer = accelerator.prepare(optimizer)

    device = instruments[0].wave_obs.device
    log_cuda_info(device)

    templates = [item.to(device) for item in templates]
    print("templates:",templates)
    
    torch.autograd.set_detect_anomaly(True)
    # define losses to track
    n_loss = 6
    epoch = 0
    if losses is None:
        detailed_loss = np.zeros((2, n_inst, n_epoch, n_loss))
    else:
        try:
            non_zero = np.sum(losses[0][0],axis=1)>0
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

        slope = ANNEAL_SCHEDULE[(epoch_ - epoch)%len(ANNEAL_SCHEDULE)]
        if n_epoch-epoch_<=10: slope=0 # turn off similarity
        
        if verbose and similarity:
            print("similarity info:",slope)

        for which in range(n_inst):
            for p in models[which].parameters():
                p.requires_grad = True

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
                    planet_param=planet_param,
                    skymask=skymask,
                    aug_fct=aug_fcts[which],
                    similarity=similarity,
                    consistency=consistency,
                    flexibility=flexibility,
                    slope=slope,
                    n_epoch=epoch_,
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
    parser.add_argument("-dir","--dir", help="data file directory", default="/scratch/gpfs/yanliang/neid-production/")
    parser.add_argument("-when","--when", help="before or after the fire", default="after")
    parser.add_argument('-o','--orders', nargs='+', help='<Required> Orders', required=True)
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=5)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=512)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-it", "--iteration", help="number of interation", type=int, default=100000)
    parser.add_argument("-c", "--consistency", help="add consistency loss", action="store_true")
    parser.add_argument("-d", "--double", help="double precision", action="store_true",default=False)
    parser.add_argument("-f", "--flexibility", help="constrian model flexibility", action="store_true")
    parser.add_argument("-C", "--clobber", help="continue training of existing model", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    parser.add_argument("-debug", "--debug", help="show diagnostic plots", action="store_true")
    parser.add_argument("-shuffle", "--shuffle", help="shuffle inputs", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda')
    if args.debug:torch.cuda.set_device('cuda:2')
    #torch.cuda.set_device('cuda:2')

    assert torch.cuda.is_available(), "CUDA is not available!"
    assert device.type == 'cuda', "Device is not CUDA!"

    print("args.outfile:",args.outfile)
    basename = os.path.basename(args.outfile)
    basename = "_".join(basename.split("_")[:-1])

    print("args.orders",args.orders,"data:",args.data)
    templates = []
    trainloaders = []
    instruments = []
    models = []

    for order in args.orders:
        template_data = load_batch("%s%s%s_%s-template.pkl"%(args.dir,args.data,order,args.when))
        wave_obs = wave_rest = template_data[0]

        if int(order)<=50:skymask = wave_obs<0
        else: skymask = wave_obs>0
            
        init_restframe = template_data[1].float().detach().clone()
        instrument = Synthetic(wave_obs)

        model = TelluricModel(wave_rest,init_restframe,
                              instrument,
                              n_latent=args.latents)

        select_tag = f"{args.data}{order}_{args.when}"
        data_loader = instrument.get_data_loader(f"{args.dir}", select=select_tag, which="train",batch_size=args.batch_size, shuffle=args.shuffle)

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
    planet_param = None
    # define training sequence
    FULL = {"data":[True,True],"encoder":[True,True],"rv":[True,True],
            "decoder":True,"spec_rest":True}
    train_sequence = prepare_train([FULL],niter=args.iteration)

    annealing_step = 3000
    ANNEAL_SCHEDULE = np.linspace(0.0,1.0,annealing_step)
    ANNEAL_SCHEDULE = np.hstack((ANNEAL_SCHEDULE))
    #print("Encoder: n_latent=%d"%models[0].encoder.n_latent)
    #print("Decoder: n_latent=%d"%models[0].decoder.n_latent)
    #print("Telluric: n_latent=%d"%models[0].telluric.n_latent)

    if args.double:[model.double() for model in models]
    n_epoch = sum([item['iteration'] for item in train_sequence])
    init_t = time.time()
    if args.verbose:
        print("torch.cuda.device_count():",torch.cuda.device_count())
        if torch.cuda.device_count()==0:
            print("No visible cuda device! MIG?")
            exit()
        print (f"--- Model {args.outfile} ---")

    # check if outfile already exists, continue only of -c is set
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit("\nOutfile exists! Set option -C to continue training.")
    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print("\nLoading file %s"%args.outfile)
        models, losses = load_model(args.outfile, models, instruments)


    profiler = LineProfiler()
    profiler.add_function(get_losses)
    lpWrapper = profiler(train)
    lpWrapper(models, instruments, trainloaders, templates, skymask=skymask, planet_param=planet_param,n_epoch=n_epoch,
          n_batch=args.batch_number, lr=args.rate, aug_fcts=aug_fcts, consistency=args.consistency, flexibility=args.flexibility,outfile=args.outfile, losses=losses, verbose=args.verbose)
    
    profiler.print_stats()

    if args.verbose:
        print("--- %s seconds ---" % (time.time()-init_t))
