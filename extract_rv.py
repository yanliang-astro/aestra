#!/usr/bin/env python
# coding: utf-8
import torch
import sys,os,time,pickle
import numpy as np
import emcee,corner
import matplotlib.pyplot as plt
from torch import nn,optim
from scipy.special import digamma
from torchinterp1d import Interp1d
from sklearn.neighbors import KDTree
from scipy.interpolate import interp1d
from scipy.optimize import minimize,curve_fit
from util import moving_mean,plot_fft,visualize_encoding

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device="cpu"

class MLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden=(3,2),
                 act=(nn.LeakyReLU(),nn.LeakyReLU(),nn.Identity()),
                 dropout=0):
        super(MLP, self).__init__()

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_)-1):
                layer.append(nn.Linear(n_[i], n_[i+1]))
                layer.append(act[i])
                layer.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layer)

    def forward(self, x):
        return self.mlp(x)

def avgdigamma(dist,radius,slope=100):
    #num_points = torch.count_nonzero(dist<(radius-1e-15),dim=0)#.double()
    num_points = torch.sum(torch.sigmoid(-slope*(dist-radius)),dim=0)
    return (torch.digamma(num_points)).mean()

def mutual_information(x0, y0, k=100, base=2):
    x = (x0-x0.mean(dim=0))/x0.std(dim=0)
    y = (y0-y0.mean())/y0.std()
    if k > x.shape[0]-1: k=x.shape[0] - 1
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
    #print("a, b, c, d:",a, b, c, d)
    return (-a - b + c + d) / np.log(base)


def mi_loss(latent,delta_rv,amp=1e3):
    return amp*torch.abs(mutual_information(latent, delta_rv))


def smooth_kernel_old(tree,points,target,sigma=0.1,self_weight=0.):
    assert sigma.shape[0] == points.shape[1]
    RV_smooth = np.zeros_like(target)
    neighbors = tree.query_radius(points, 3*sigma.max())
    #neighbors = tree.query(points, k=4)[1]
    num_neighbor =  np.zeros_like(target)
    weight_neighbor =  np.zeros_like(target)
    for i,n in enumerate(neighbors):
        x = points[i]
        dsquare = np.sum(((points[n] - x)/sigma)**2,axis=1)
        weights = np.exp(-0.5*dsquare)
        #print("n:",n,"dsquare:",dsquare,"weights:",weights)
        weights[dsquare==0.]=self_weight
        if weights.sum()>self_weight:RV_smooth[i] = np.average(target[n,0],weights=weights)
        else:
            # no neighbor...expand range and increase self weight
            distance,nearest = tree.query([x],k=int(0.05*len(target)))
            near_weights = np.exp(-0.5*(distance/(3*sigma.mean()))**2)
            RV_smooth[i] = np.average(target[nearest[0]].T,weights=near_weights)
        num_neighbor[i] = len(weights)
        weight_neighbor[i] = weights.sum()
    return torch.tensor(RV_smooth),num_neighbor,weight_neighbor

def smooth_kernel(tree,points,target,fraction=0.5,n_radius=20,
                  self_weight=0.):
    RV_smooth = np.zeros_like(target)
    k_neighbor = max(int(fraction*len(target)),n_radius)
    distance,neighbors = tree.query(points,k=k_neighbor)
    weight_neighbor =  np.zeros_like(target)
    num_neighbor = np.zeros_like(target)
    dsquare = np.sum((points[neighbors] - points[:,None,:])**2,axis=-1)
    sigma = np.mean(dsquare[:,:n_radius]**0.5)
    print(dsquare[:,:n_radius].shape,"sigma:",sigma)
    weights = np.exp(-0.5*dsquare/sigma**2)
    weights[dsquare==0.]=self_weight
    w_norm = weights/weights.sum(axis=1)[:,None]
    v0_error = np.sqrt(np.sum(w_norm**2,axis=1))
    for i,n in enumerate(neighbors):
        RV_smooth[i] = np.average(target[n,0],weights=weights[i])
        #print("RV_smooth[i]:",RV_smooth[i]);exit();
        weight_neighbor[i] = weights[i].sum()
        num_neighbor[i] = (weights[i]>=self_weight).sum()
    return RV_smooth,v0_error,num_neighbor,weight_neighbor,sigma

def plot_encoding(latent,colors,cnames,cmap,fname, mask=None,
                  center=None, radius=None, titles=[]):
    
    if center is None: center=latent.mean(axis=0)
    if mask is None: mask = np.ones(len(latent),dtype=bool)

    if titles == []: titles = [None]*len(colors)
    nrows = len(colors) // 2
    ncols = 2
    fig,axs = plt.subplots(nrows=nrows,ncols=ncols,
                           figsize=(12,4*nrows+1),
                           constrained_layout=True)
    if nrows==2:axs = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
    for i,ccode in enumerate(colors):
        ax = axs[i]
        if cnames[i]=="$\chi^2$":vmin,vmax=0,20
        else: vmin,vmax=ccode.min(),ccode.max()
        
        img=ax.scatter(latent[mask,0],latent[mask,1],s=1,
                       cmap=cmap[i],c=ccode[mask],
                       vmin=vmin,vmax=vmax)
        ax.set_title(titles[i])
        if not radius is None:
            ax.set_xlim(center[0]-r,center[0]+r)
            ax.set_ylim(center[1]-r,center[1]+r)

        cbar=fig.colorbar(img,ax=ax)
        cbar.set_label(cnames[i])
    plt.savefig(fname,dpi=200)
    return

def plot_mutual_info(input_params,title,name,ms=2,plot=True):
    act_rv,rv_weight,data, v_encode,RV_ccf = input_params
    print("act_rv:",act_rv.shape,"rv_weight:",rv_weight.shape)
    mi_init = mutual_information(data, v_encode).item()
    mi_after = mutual_information(data,v_encode-act_rv).item()
    
    print("mi_init: %.3f"%mi_init,"mi_after: %.3f"%mi_after)

    act_rv = act_rv.detach().cpu()
    v_encode = v_encode.detach().cpu()
    delta_rv = v_encode-act_rv
    delta_rv = delta_rv[:,0].detach().cpu()

    RV_ccf = RV_ccf.detach().cpu()
    latent = data.detach().cpu()
    
    diff = RV_ccf - delta_rv
    diff -= diff.mean()

    rms = torch.std(diff).item()
    outlier = np.abs(diff)>3*rms

    avg_weight = rv_weight.mean()
    outlier_weight = rv_weight[outlier].mean()
    print("mean weight: %.1f outlier weight: %.1f"%(avg_weight,outlier_weight))
    print("\n\nv_CCF - v_correct RMS: %.3f m/s"%rms)
    print("\n\nv_correct RMS: %.3f m/s"%delta_rv.std())

    if RV_ccf.max()-RV_ccf.min() == 0:
        xlin = sorted(delta_rv);plin = [0,0]
        linear_fit = np.polyval(plin,xlin)
    else:
        plin,cov = np.polyfit(RV_ccf,delta_rv,1,
                              cov=True)#,w=rv_weight[:,0])
        p_err = [cov[i][i]**0.5 for i in range(len(cov))]
        label_lin = "slope=%.2f+/-%.2f"%(plin[0],p_err[0])
        print(label_lin)
        xlin = sorted(RV_ccf)
        linear_fit = np.polyval(plin,xlin)

    if not plot: return

    fig,axs=plt.subplots(nrows=2,ncols=3,figsize=(12,6),
                         constrained_layout=True,
                         dpi=200)
    ax = axs[0][0]
    ax.set_title("mutual information = %.3f"%(mi_init))
    img=ax.scatter(latent[:,0],latent[:,1],c=v_encode,
                   label="N=%d"%len(v_encode),s=ms)
    cbar=plt.colorbar(img,ax=ax)
    cbar.set_label("v_encode [m/s]")
    ax.legend()
    ax = axs[0][1]
    ax.set_title(title)
    img=ax.scatter(latent[:,0],latent[:,1],c=act_rv,s=ms)
    cbar=plt.colorbar(img,ax=ax)
    cbar.set_label("v_encode [m/s]")

    ax = axs[0][2]
    ax.set_title("mutual information = %.3f"%(mi_after))
    img=ax.scatter(latent[:,0],latent[:,1],c=delta_rv,s=ms)
    cbar=plt.colorbar(img,ax=ax)
    cbar.set_label("residual RV [m/s]")

    ax = axs[1][0]
    ax.plot(RV_ccf,delta_rv,"k.",ms=ms,
            label="prefit RMS=%.3f m/s"%torch.std(v_encode))
    ax.plot(xlin,linear_fit,"-",c="orange",label=label_lin)
    ax.set_xlabel("$v_{ccf}$ [m/s]")
    ax.set_ylabel("$v_{correct}$ [m/s]")
    ax.legend(loc=2)
    ax = axs[1][1]
    ax.plot(RV_ccf[~outlier],diff[~outlier],"k.",ms=ms,
            label="RMS=%.3f m/s w=%.1f"%(rms,avg_weight))
    ax.plot(RV_ccf[outlier],diff[outlier],"r.",
            label="outlier w=%.1f"%(outlier_weight))
    ax.set_xlabel("$v_{ccf}$ [m/s]")
    ax.set_ylabel("$v_{ccf}$ [m/s] - $v_{correct}$ [m/s]")
    ax.legend(loc=2)
    ax = axs[1][2]
    ax.plot(latent[:,0],latent[:,1],'k.',ms=ms,label="data")
    ax.plot(latent[outlier,0],latent[outlier,1],'r.',label="outlier")
    ax.legend(loc=2)
    plt.savefig("[%s]mutual-info.png"%name,dpi=200)
    return

def phase_fold(phase,RV_data,n=20):
    xgrid,ygrid,delta_y = moving_mean(phase,RV_data,n=n)
    if not full: return sig_chi,ref_chi
    print("sig chi^2: %.2f (delta: %.2f)"%(sig_chi,ref_chi-sig_chi))
    return [xgrid,ygrid,delta_y],[sig_chi,ref_chi]

def double_y_axis(axleft,data,labels,colors=["k","orange"],legends=[None,None]):
    x,y1,y2 = data
    xlabel,y1label,y2label = labels
    c1,c2 = colors
    l1,l2 = legends

    axright = axleft.twinx()
    axleft.plot(x, y1, ".",color=c1, ms=3,label=l1)
    if "dynamic" in sys.argv:
        axright.plot(x, y2, color=c2, lw=2, label=l2)
    else: axright.plot(x, y2, ".",color=c2, ms=2, label=l2)

    axleft.set_xlabel(xlabel)
    axleft.set_ylabel(y1label, color=c1)
    axleft.tick_params(axis="y", labelcolor=c1)
    axright.set_ylabel(y2label,color=c2)
    axright.tick_params(axis="y", labelcolor=c2)
    axright.set_ylim(y2.min(),1.2*y2.max())
    return

def density_weights(feature,target,radius=0.17):
    feature = feature.cpu().numpy()
    tree = KDTree(feature)
    target = v_encode.cpu().numpy()
    print("[density_weights]: radius = %.2f"%radius)
    sigma = np.array([radius]*3)
    act_rv,v0_error,num,weights = smooth_kernel(tree,feature,target,sigma=sigma)
    return weights

def load_param(select):
    with open("%s-param.pkl"%select,"rb") as f:param_dict=pickle.load(f)
    return param_dict

def plot_latent_space(select,model_file,v_target,v_name,cmap):
    save_latent = "[%s]%s.pkl"%(model_file,select)
    with open(save_latent,"rb") as f:
        latent = pickle.load(f)
        latent_aug = pickle.load(f)
        fitted_RV = pickle.load(f)
        encoded_RV = pickle.load(f)
        RV_ccf = pickle.load(f)
        fit_chi = pickle.load(f)
    visualize_encoding(latent,latent_aug,v_target,v_name,
                       tag=model_file,cmap=cmap)
    return

def load_model_latent(select,model_file,device):
    save_latent = "[%s]%s.pkl"%(model_file,select)
    with open(save_latent,"rb") as f:
        latent = pickle.load(f)
        latent_aug = pickle.load(f)
        fitted_RV = pickle.load(f)
        encoded_RV = pickle.load(f)
        RV_ccf = pickle.load(f)
        fit_chi = pickle.load(f)
    print("fitted_RV:",fitted_RV.min(),fitted_RV.max())
    print("encoded_RV:",encoded_RV.min(),encoded_RV.max())

    latent = torch.tensor(latent.T,device=device)
    v_encode = encoded_RV
    v_bestfit = fitted_RV
    RV_ccf = torch.tensor(RV_ccf,device=device)
    fit_chi = torch.tensor(fit_chi,device=device)[:,None]
    return latent,v_encode,v_bestfit,RV_ccf,fit_chi

def log_prior(param):
    bounds = mcmc_dict["bounds"]
    for i,b in enumerate(bounds):
        if param[i]<b[0] or param[i]>b[1]:
            return -np.inf
    return 0

def log_prob(param):
    prior = log_prior(param)
    if not np.isfinite(prior): return prior
    period, amplitude, phi = param
    timestamp = mcmc_dict["timestamp"]
    RV_data = mcmc_dict["RV_data"]
    noise = mcmc_dict["RV_noise"]
    phi *= np.pi/180.0
    #phase = (timestamp%period)/period
    RV_model = np.sin(2*np.pi*timestamp/period+phi)*amplitude
    chi_model = (((RV_data-RV_model)/noise)**2).sum()
    return mcmc_dict["chi_0"]-chi_model

def run_mcmc(pool,p_mean,p_radius,
             mcmc_tag="mcmc",mcmc_dir="MCMCbackend",
             nwalkers=32,n_epoch=20000):
    ndim = len(p_mean)
    p0 = p_mean+np.random.uniform(-1,1,size=(nwalkers, ndim))*p_radius
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_prob,pool=pool)
    sampler.run_mcmc(p0, n_epoch, progress=True)
    samples = sampler.get_chain(flat=False)
    logP = sampler.get_log_prob(flat=False)
    filename = "%s/%s.pkl"%(mcmc_dir,mcmc_tag)
    print("saving to %s..."%filename)
    with open(filename,"wb") as f:
        pickle.dump(samples,f)
        pickle.dump(logP,f)
    return samples,logP

def mcmc_stats(mcmc_samples,quantiles=[0.16,0.50,0.84]):
    stats = []
    for i in range(mcmc_samples.shape[1]):
        low,mean,high = np.quantile(mcmc_samples[:,i], quantiles)
        ubar = high-mean
        lbar = mean-low
        digit = -int(np.floor(np.log10(abs(min(ubar,lbar))))) + 1
        digit = min(digit,2)
        mean,ubar,lbar = [str(np.round(ii,digit)) for ii in [mean,ubar,lbar]]
        text = "${%s}^{+%s}_{-%s}$"%(mean,ubar,lbar)
        stats.append(text)
    return stats

def plot_mcmc(mcmc_tag="mcmc",mcmc_dir="MCMCbackend",burnin=0.2,
              stats_alone=False,truth=[]):
    filename = "%s/%s.pkl"%(mcmc_dir,mcmc_tag)
    with open(filename,"rb") as f:
        samples = pickle.load(f)
        logP = pickle.load(f)

    n_burnin = int(burnin*len(logP))
    samples = samples[n_burnin:]
    logP = logP[n_burnin:]
    

    n_epoch,n_walker,n_dim = samples.shape
    samples=samples.reshape((n_epoch*n_walker,n_dim))
    logP=logP.reshape((n_epoch*n_walker))
    stats = mcmc_stats(samples)
    
    Ksample = np.quantile(samples[:,1],[0.021,0.16,0.50])
    print("Ksample:",Ksample)
    
    if stats_alone: return stats
    labels = mcmc_dict["labels"]
    best = logP==max(logP)
    print("samples:",samples.shape,"logP:",logP.shape)
    print("max log_prob:",max(logP),"param:",samples[best])
    print("stats:",stats)
    c_sigma = np.array([1,2,3])
    levels= 1 - np.exp(-0.5*c_sigma**2)
    print("levels:",levels)
    
    fig = plt.figure(figsize=(7,7))
    corner.corner(samples,labels=labels,fig=fig,
                  quantiles=[0.16,0.50,0.84],#range=[0.99]*n_dim,
                  plot_contours=True,fill_contours=True,
                  smooth=0.2,levels=(levels),alpha=0.5,color='k',
                  show_titles=True)
    if len(truth)>0:
        corner.overplot_lines(fig, truth, color="b")
        corner.overplot_points(fig, truth[None], marker="s",
                               color="b")
    #fig.tight_layout()
    plt.savefig("[%s]cornerplot.png"%mcmc_tag,dpi=300)
    return

def velocity_label(velocity,label):
    quantiles=[0.16,0.50,0.84]
    q1,q2,q3 = np.quantile(velocity,quantiles)
    val = "${%.2f}^{+%.2f}_{-%.2f}$"%(q2,q3-q2,q2-q1)
    vlabel = "%s = %s [m/s]"%(label,val)
    return vlabel


# ----------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)


select = sys.argv[1]
model_file = sys.argv[2]

tag = select
latent,v_encode,v_bestfit,RV_ccf,fit_chi  = load_model_latent(select,model_file,device)

print("latent std:",latent.std(dim=0))
print("fit_chi: %.3f, %.3f, %.3f"%(fit_chi.min(),fit_chi.max(),fit_chi.mean()))
print("where max",np.argmax(fit_chi))

latent -= latent.mean(dim=0)
latent /= latent.std(dim=0)

n_data, n_latent = latent.shape

neid_dict = load_param(select)
select_order=60
timestamp = np.array([neid_dict[key]['OBSJD'] for key in neid_dict])
ssbrv = np.array([neid_dict[key]["SSBRV"]for key in neid_dict])
v_template = np.array([neid_dict[key]["v_template"] for key in neid_dict])
v_planet = np.array([neid_dict[key]["v_planet"] for key in neid_dict])

print("ssbrv:",ssbrv)

param_dict = {"timestamp":timestamp,
              "v_template":v_template,
              "v_ccf":RV_ccf.numpy(),
              "v_encode":v_encode,
              "v_bestfit":v_bestfit,
              "v_planet":v_planet,
              "ssbrv":ssbrv
             }

print("v_encode:",v_encode.shape,"latent:",latent.shape,
  "RV_ccf",RV_ccf.shape)

for vtag in ["v_ccf","v_encode","v_bestfit","ssbrv"]:
    vtensor =  torch.tensor(param_dict[vtag])[:,None]
    mi = mutual_information(latent,vtensor).item()
    print("%s vs. latents mutual information: %.3f"%(vtag,mi))

#mi_encode = mutual_information(latent, torch.tensor(v_encode)[:,None]).item()
#cdata,clabel,ctitle,cmap = v_encode,"$v_{encode}$ [m/s]","mi = %.3f"%mi_encode,"viridis"
print("v_ccf > 10:",np.where(RV_ccf.numpy()>10))

def visualize_latents(latent,cdata,clabel,cmap,tag):
    indices = [[0,1],[1,2],[0,2]]
    fig,axs=plt.subplots(figsize=(7,2),ncols=len(indices),
                         constrained_layout=True)
    for i,wh in enumerate(indices):
        ax=axs[i]
        img=ax.scatter(latent[:,wh[0]],latent[:,wh[1]],c=cdata,cmap=cmap,
                       label="N=%d"%len(cdata),s=5)
        ax.set_xlabel("$s_%d$"%(wh[0]+1));ax.set_ylabel("$s_%d$"%(wh[1]+1))
    cbar=plt.colorbar(img,ax=ax)
    cbar.set_label(clabel)
    plt.savefig("[%s]latent-%s.png"%(model_file,tag),dpi=300)
    return

cdata,clabel,cmap = timestamp-timestamp[0],"Time [days]","plasma"
visualize_latents(latent,cdata,clabel,cmap,"time")

cdata,clabel,cmap = ssbrv,"barycentric RV","rainbow"
visualize_latents(latent,cdata,clabel,cmap,"ssbrv")

cdata,clabel,cmap = fit_chi,"$\chi^2$","inferno"
cdata = np.minimum(cdata,1.0)
visualize_latents(latent,cdata,clabel,cmap,"chi")
# initialization
data = latent.to(device).float()


if len(data) > 100: self_weight = 0.0 # if mean weight > 10.0
else: self_weight = 1.0 # if mean weight < 10.0

feature = latent.numpy()
tree = KDTree(feature)
#target = v_encode[:,None];v_name = "encode"
target = v_bestfit[:,None];v_name = "bestfit"
#target = param_dict["v_ccf"][:,None];v_name = "ccf"

print("target:",target.shape)
figname = "%s-%s"%(tag,model_file[:-3])

if "3d" in sys.argv:
    #cdata,cname,cmap = v_bestfit,"bestfit","viridis"
    cdata,clabel,cmap = fit_chi,"$\chi^2$","inferno"
    plot_latent_space(select,model_file,cdata,clabel,cmap=cmap)

self_weight = 0.0
act_rv,v0_error,num,weights,d_measure = smooth_kernel(tree,feature,target,self_weight=self_weight)

print("weights:",weights.min(),weights.max(),
      weights.mean())
print("latent Dij: %.3f"%d_measure)

print("v_activity: %.3f m/s"%act_rv.std())
#act_rv = act_rv.mean() + torch.zeros_like(act_rv)
v_correct = (target-act_rv).T[0]

print("v_encode RMS: %.3f m/s"%v_encode.std())
print("v_correct RMS: %.3f m/s"%v_correct.std())
print("v_CCF RMS: %.3f m/s"%RV_ccf.std())

labels = [r"$v_{encode}$",r"$v_{best-fit}$",
          r"$v_{correct}$",r"$v_{ccf}$",r"$v_{template}$"]
v_matrix = np.array([v_encode,v_bestfit,v_correct,RV_ccf.numpy(),v_template]).T
print("v_matrix:",v_matrix.shape)
n_sample,n_dim = v_matrix.shape



with open("[%s-%s]vmatrix.pkl"%(model_file,select),"wb") as f:
    pickle.dump(v_matrix,f)
    pickle.dump(labels,f)

#fig = plt.figure(figsize=(7,7))
if "vmat" in sys.argv:
    corner.corner(v_matrix,labels=labels,range=[0.99]*n_dim,
                  quantiles=[0.16,0.50,0.84],fontsize=15,
                  color='k',show_titles=True)
    plt.savefig("[%s]v_matrix.png"%model_file,dpi=300)

    
if "planet" in sys.argv:
    
    period=0.11
    phase = ((timestamp-timestamp[0])/period)%1
    
    fig,ax=plt.subplots(figsize=(4,4),constrained_layout=True)

    label_template = velocity_label((v_template-v_planet),"$v_{template}-v_{planet}$")
    label_correct = velocity_label((v_correct-v_planet),"$v_{correct}-v_{planet}$")
    #ax.plot(phase,v_template,".",ms=2,color="lightgrey",label=label_template)
    ax.plot(phase,v_correct,".",ms=2,color="grey",label=label_correct)
    
    xgrid,ygrid,delta_y = moving_mean(phase,v_correct,n=15)
    ax.errorbar(xgrid,ygrid,yerr=delta_y,fmt="k.",capsize=3,ms=2,
                label="$v_{correct}$ re-binned")
    
    #ax.plot(phase,v_bestfit,".",color="orange",label="v_bestfit")
    ax.plot(phase,v_planet,"r.",ms=1)
    ax.set_xlabel("Phase");ax.set_ylabel("Residual RV [m/s]")
    #ax.set_ylim(-1.5,1.5)
    ax.set_ylim(-5,5)
    ax.legend()
    plt.savefig("[%s]phase-fold.png"%model_file,dpi=300)

    nrows,ncols = 2,3
    fig,axs=plt.subplots(figsize=(8,5),dpi=200,nrows=nrows,ncols=ncols,constrained_layout=True)

    for i in range(len(labels)):
        i_col = i%ncols
        i_row = i//ncols

        vel = v_matrix[:,i]-v_matrix[:,i].mean()
        poly,cov = np.polyfit(v_planet,vel,deg=1,cov=True)
        slope_uncertainty = cov[0][0]**0.5 
        label = velocity_label((vel-v_planet),labels[i])
        text = "slope = %.2f+/-%.2f"%(poly[0],slope_uncertainty)
        if i in [1,2,4]:print(label,text)
        ax=axs[i_row][i_col]
        ax.plot(v_planet,vel,"k.",ms=1)
        ax.plot(v_planet,np.polyval(poly,v_planet),"r.",ms=1,
                label=text)
        ax.legend()
        ax.set_ylim(-10,10)
        ax.set_title(label)
    plt.savefig("[%s]planet.png"%model_file,dpi=300)
    exit()

def plot_time_series(latent,v_matrix,timestamp):
    velocities = v_matrix - v_matrix.mean(axis=0)
    v_encode,v_bestfit,v_correct,v_ccf,v_template = velocities.T
    outlier = np.abs(v_ccf)>100
    
    ccflabel = velocity_label(v_ccf,"CCF")
    predlabel=velocity_label(v_correct,"correct")
    print(ccflabel,predlabel)
    
    nbins = 20
    fig,ax=plt.subplots(figsize=(3,3),constrained_layout=True)
    ax.hist(v_ccf,color="k",alpha=0.5,bins=nbins,
            log=True,label=ccflabel)
    ax.hist(v_correct,color="b",alpha=0.5,bins=nbins,
            log=True)
    ax.set_title(predlabel)
    ax.legend()
    ax.set_xlabel("radial velocity [m/s]")
    ax.set_ylabel("$N$")
    ax.set_xlim(-60,30)
    plt.savefig("[%s]v-summary.png"%model_file,dpi=300)
    
    xlim=ylim=(-50,20)
    fig,axs=plt.subplots(figsize=(8,5),nrows=2,ncols=2,
                         width_ratios=[3, 1],
                         constrained_layout=True)
    ax=axs[0][0]
    colors = ["skyblue","r","b"]
    for i in range(n_latent):
        ax.plot(timestamp,latent[:,i],".",c=colors[i],ms=2,label="$s_%d$"%(i+1))
    ax.set_xlabel("JD")
    ax.set_ylabel("Latent variables")
    ax.legend()

    ax=axs[0][1]
    ax.plot(v_ccf,v_bestfit,"k.",ms=2)

    ax.set_xlabel("$v_{CCF}$")
    ax.set_ylabel("$v_{bestfit}$")
    ax.set_xlim(xlim);ax.set_ylim(ylim)
    ax.set_title("Correlation = %.2f"%(np.corrcoef(v_ccf[~outlier], v_bestfit[~outlier])[0,1]))

    ax=axs[1][1]
    ax.plot(v_ccf,v_correct,"k.",ms=2)
    ax.set_xlabel("$v_{CCF}$")
    ax.set_ylabel("$v_{correct}$")
    ax.set_xlim(xlim);ax.set_ylim(ylim)
    ax.set_title("Correlation = %.2f"%np.corrcoef(v_ccf[~outlier], v_correct[~outlier])[0,1])

    ax=axs[1][0]

    
    ax.plot(timestamp,v_ccf,".",c="k",ms=2,lw=1,label=ccflabel)
    #ax.plot(timestamp,v_encode,".",c="b",ms=2,
    #        label=velocity_label(v_encode,"encode"))
    #ax.plot(timestamp,v_bestfit,".",c="r",ms=2,
    #        label=velocity_label(v_bestfit,"bestfit"))

    ax.plot(timestamp,v_correct,".",c="orange",ms=2,lw=1,label=predlabel)
    ax.axhspan(-2,2,color="lightgrey",zorder=-10)
    ax.legend(ncol=2)
    ax.set_ylim(ylim)
    ax.set_xlabel("JD")
    ax.set_ylabel("Order %d RV [m/s]"%select_order)
    plt.savefig("[%s]compare-timeseries.png"%model_file,dpi=300)

    fig,ax=plt.subplots(figsize=(3,3),#nrows=2,ncols=2,
                         constrained_layout=True)
    ax.plot(v_encode,v_bestfit,"k.",ms=2)
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_xlabel("$v_{encode}$")
    ax.set_ylabel("$v_{bestfit}$")
    #ax.set_xlim(xlim);ax.set_ylim(ylim)
    ax.set_title("Correlation = %.2f"%np.corrcoef(v_encode[~outlier], v_bestfit[~outlier])[0,1])
    plt.savefig("[%s]test.png"%model_file,dpi=300)
    return

mask = (weights>0.0)[:,0]
#mask = (timestamp-timestamp[0])>2.5
#mask &= (timestamp-timestamp[0])<3.5
plot_time_series(latent[mask],v_matrix[mask],timestamp[mask])




#labels = [r"$v_{doppler}$",r"$v_{apparent}$",r"$v_{encode}$",r"$v_{correct}$",r"$v_{ref}$"]
#signals = [param_dict["rv"],param_dict["v_template"],
#           target.T[0],v_correct,param_dict["rv_ref"]]
#plot_fft(timestamp,signals,figname,labels)
