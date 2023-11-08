#!/usr/bin/env python
# coding: utf-8
import io, os, sys, time, random
sys.path.insert(1, './')
import numpy as np
import pickle
import torch
#from torchinterp1d import Interp1d
from spender_model import SpectrumAutoencoder
from synthetic_data import Synthetic
from util import mem_report,load_batch
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.interpolate import CubicSpline
import scipy.optimize
from scipy.interpolate import interp1d
from scipy.special import digamma
from util import evaluate_lines,visualize_encoding
from sklearn.neighbors import KDTree

data_dir = "/scratch/gpfs/yanliang"
dynamic_dir = "/scratch/gpfs/yanliang/neid-dynamic"
#device =  torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def avgdigamma(dist,radius):
    num_points = torch.count_nonzero(dist<(radius-1e-15),dim=0).double()
    return (torch.digamma(num_points)).mean()

def mutual_information(x, y, k=100, base=2):
    """Mutual information of x and y 
    """
    assert x.shape[0] == y.shape[0], "Arrays should have same length"
    assert y.shape[1] == 1,     "Single value function"
    assert k <= x.shape[0] - 1, "Set k smaller than num. samples - 1"

    # Find nearest neighbors in joint space, 
    points = torch.cat((x, y),dim=1)
    print("x:",x.min(),x.max(),"y:",y.min(),y.max())
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

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]

def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)

def _avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    #print("num_points:",num_points)
    return np.mean(digamma(num_points))

def build_tree(points):
    return KDTree(points, metric="chebyshev")

def ee_mutual_information(x, y, k=100, base=2):
    if k > len(x) - 1: k = len(x) - 1
    """Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    # Find nearest neighbors in joint space, 
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]

    points = np.hstack(points)
    #print("points:",points.shape)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    #print("k:",k,"dvec:",dvec)
    
    a, b, c, d = (
        _avgdigamma(x, dvec),
        _avgdigamma(y, dvec),
        digamma(k),
        digamma(len(x)),
    )
    #print("a:",a,"b:",b,"c:",c,"d:",d)
    return (-a - b + c + d) / np.log(base)

def model_encode(model,spec,template):
    return model.encode(spec-template)

def permute_indices(length,n_redundant=1):
    wrap_indices = torch.arange(length).repeat(n_redundant)
    rand_permut = wrap_indices[torch.randperm(length*n_redundant)]
    return rand_permut

def load_model(path, instruments, normalize=False):
    mdict = torch.load(path, map_location=device)
    models = []
    wave_rest = mdict['model'][0]['decoder.wave_rest']
    spec_rest = mdict['model'][0]['decoder.spec_rest']
    n_latent = len(mdict['model'][0]['encoder.mlp.mlp.9.bias'])

    for j in range(len(instruments)):
        model = SpectrumAutoencoder(instruments[j],
                                    wave_rest,
                                    spec_rest=spec_rest,
                                    n_latent=n_latent,
                                    n_aux=0,
                                    normalize=normalize)
        model.load_state_dict(mdict["model"][j],strict=False)
        model.to(device)
        #except: print("\n\nloading error...\n\n")
        models.append(model)
        #print("decoder wave_rest:",model.decoder.wave_rest.device,"\n\n")
    return models,mdict["losses"],n_latent

def add_emission(ax,z=0, ymax=0, xlim=(3000,7000),alpha=0.3,
                 color="grey",ytext=0.5,zorder=5):
    
    lines = [i for i in emissionlines]
    
    if not ymax:ymax = ax.get_ylim()[1]
    
    ymin = -10
    shifted = np.array(lines)*(1+z)
    ax.vlines(x=shifted,ymin=ymin,ymax=1.1*ymax,
              color=color,lw=1,alpha=alpha,zorder=zorder)
    
    for line in emissionlines:
        
        name = emissionlines[line]
        shifted = np.array(line)*(1+z)
        
        if shifted<xlim[0] or shifted>xlim[1]:continue
            
        ax.text(shifted,ytext,name,fontsize=20,
                color=color,rotation=90,zorder=zorder)
                
    return

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
    
def plot_loss(loss, ax=None,xlim=None,ylim=None,fs=12):
    latest = loss[:,-1]
    ep = range(len(loss[0]))
    
    if not ax:fig,ax=plt.subplots(dpi=200)
    labels = ["fidelity","similarity","z_offset","consistency",
              "zero_point","flexibility"]
    colors = ['k','r','b','orange','c','gold']
    order = [10,15,0,-5,-10,20]
    ax.axhline(1.0,ls="-",color="grey")
    print("loss:",loss.shape,loss.sum(axis=0).shape)
    minimum = np.min(loss.sum(axis=0))

    for i in range(len(loss)):
        if sum(loss[i])==0:continue
        ax.semilogy(ep,loss[i],lw=1,
                    label="%s(loss=%.2f)"%(labels[i],latest[i]),
                    zorder = order[i], color=colors[i])
    low = min(loss[loss>0])
    low = 0.1
    high = min(1e6,loss.max())
    ax.axhline(1.0,ls="--",color="grey")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xlim(ep[-1]-10000,ep[-1])
    ax.set_ylim(max(low,1e-3),high)
    ax.legend(loc="best",fontsize=fs).set_zorder(30)
    
    if sum(latest)==minimum:
        print("\n\n%.2f is the minimum!\n\n"%minimum)
    return

def merge_batch(file_batches):
    spectra = [];weights = [];z = [];specid = [];norm = []
    for batchname in file_batches:
        print("batchname:",batchname)
        batch = load_batch(batchname)
        spectra.append(batch[0])
        weights.append(batch[1])
        z.append(batch[2])
        specid.append(batch[3])
        norm.extend(batch[4])
        #print("spec:",batch[0].shape)
    spectra = torch.cat(spectra,axis=0)
    weights = torch.cat(weights,axis=0)
    z = torch.cat(z,axis=0)
    specid = torch.cat(specid,axis=0)
    print("spectra:",spectra.shape,"w:",weights.shape,
          "z:",z.shape)
    return spectra, weights, z, specid, norm

def redshift_chi(rv,wave_rest,yrest,wave_obs,ydata,wdata):
    func = CubicSpline(wave_rest*(1 + rv/Synthetic.c), yrest)
    model_obs = func(wave_obs)
    loss = np.sum(wdata * (ydata - model_obs)**2) / len(ydata)
    return loss

def fit_rv(input_data,p0):
    #for item in input_data:print(item.shape)
    wrest,yrest,wave_obs,y_obs,w = input_data
    rv_fit = torch.zeros((len(yrest),1))
    chi = np.zeros(len(yrest))
    for i in range(len(yrest)):
        result = scipy.optimize.minimize(redshift_chi,p0[i], \
        method='Nelder-Mead',args=(wrest,yrest[i],wave_obs,y_obs[i],w[i],))
        rv_fit[i] = result.x[0]
        chi[i]=result.fun
    return rv_fit,chi



def spectra_1D(spec,w,berv,ID,instrument,template,label="",aux=None,tag="test"):
    colors = ['mediumseagreen',"skyblue",'salmon','orange','gold']
    title = "%s JD:%.5f"%(select,ID.item())

    wave_obs = tensor2array(instrument.wave_obs)
    wrest = model.decoder.wave_rest

    s = model_encode(model,spec,template)
    yrest = model.decoder.spec_rest+model.decode(s)
    #print("y_rest:",yrest.shape,"wave_obs:",wave_obs.shape)
    rv_fit = model.estimate_rv(spec-template)

    z_fit = rv_fit/instrument.c
    print("v_encode: %.3f m/s"%rv_fit.item())
    y_recon = model.decoder.transform(yrest, z_fit,instrument=instrument)

    loss = model._loss(spec, w, y_recon)
    if not aux is None:loss_ref =  model._loss(spec, w, aux)
    #_,yrest,y_recon = model._forward(spec, w, s, z_fit)

    y_act = tensor2array(y_recon-template)[0]
    true_act = tensor2array(spec-template)[0]

    spec_new,w_new,_,z_offset=instrument.augment_spectra([spec,w,0,ID])
    s_new = model_encode(model,spec_new,template)
    rv_new = model.estimate_rv(spec_new-template)
    
    augloss = model.loss(spec_new, w_new, s_new,rv_new/instrument.c)

    print("v_offset: %.3f m/s"%(z_offset.item()*instrument.c))
    print("loss:",loss.item(),"augment_loss: %.5f"%augloss.item())
    print(label)
    
    z_new = (z_fit+z_offset)[0][0]
    z_new = tensor2array(z_new)

    print("fitted v_offset:",rv_new-rv_fit)

    y_recon = tensor2array(y_recon[0])
    yrest = tensor2array(yrest[0])
    spec_new = tensor2array(spec_new[0])
    
    if w.dim()==1:
        scalar_w = w[0].item()**(-0.5)
        err=np.ones_like(y_recon)*scalar_w
    else:
        err=tensor2array(w[0]**(-0.5))

    flux=tensor2array(spec[0])
    print("flux[j]: %.2f +/- %.2f"%(flux.mean(),flux.std()))

    s = tensor2array(s[0])
    s_new = tensor2array(s_new[0])
    
    view_dict = {"wave_obs":wave_obs,
                 "spec":flux,"spec_err":err, "model":y_recon,
                 "loss":loss.item(),"loss_aug":augloss.item(),
                 "latent":s,"latent_aug":s_new}
    
    drawstyle = 'steps-mid'

    fig = plt.figure(figsize=(12,5),dpi=300)
    gs = fig.add_gridspec(2, 2) #width_ratios=[1,1],)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    #axs = [ax1,ax2]
    ax1.set_title(title)

    
    window = [5414,5418]
    window = [5407.9,5421.1]
    mask = (wave_obs>window[0]) & (wave_obs<window[1])
    

    local_act = true_act[mask]
    ymin,ymax = local_act.min(),local_act.max()
    yextra = 0.2*(ymax-ymin)
    
    ax = ax1
    ax.fill_between(wave_obs,flux-err,flux+err,
                    color="lightgrey",step="mid",
                    zorder=-10)
    ax.plot(wave_obs,flux,"k-",drawstyle=drawstyle,lw=1.0,label="data")

    ax.plot(wave_obs,y_recon,drawstyle=drawstyle, c='r',lw=0.5,
            label="model ($\chi^2_r$: %.2f)"%(loss))
    
    ax.set_xticks([])
    ax.set_ylim(0.8*flux.min(),1.1)
    ax.set_ylabel("flux")
    ax = ax2
    ax.fill_between(wave_obs,true_act-err,true_act+err,
                    color="lightgrey",step="mid",
                    zorder=-10)
    ax.plot(wave_obs,true_act,drawstyle=drawstyle, c='k',
            label="data",lw=1.0)
    ax.plot(wave_obs,y_act,drawstyle=drawstyle, c='r',
            label="model ($\chi_r^2$=%.2f)"%loss,lw=0.5)

    text = "$s_1=%.3f,s_2=%.3f,s_3=%.3f$"%(s[0],s[1],s[2])
    print("latent:",text)
    ax.text(window[0]+0.5,ymin-0.5*yextra,text,
            fontsize=12,color="grey")
    ax.set_xlim(window)
    ax.set_ylim(ymin-yextra,ymax+yextra)
    ax.legend(loc="best",ncol=3,frameon=True)
    ax.set_xlabel("observed $\lambda (\AA)$")
    ax.set_ylabel("flux - baseline")
    
    ax = ax3

    ax.fill_between(wave_obs[mask],
                    (flux-err)[mask],(flux+err)[mask],
                    color="lightgrey",step="mid",zorder=-10)
    ax.plot(wave_obs[mask],flux[mask],
            drawstyle=drawstyle, c='k',
            label="data",lw=1.0)
    ax.plot(wave_obs[mask],y_recon[mask],
            drawstyle=drawstyle, c='r',
            label="model ($\chi_r^2$=%.2f)"%loss,lw=0.5)
    ax.set_ylim(0.9,1.0)
    ax.legend(loc="best",ncol=3,frameon=True)
    ax.set_xlabel("observed $\lambda (\AA)$")
    ax.set_ylabel("flux")
    
    
    fig.align_ylabels([ax1,ax2,ax3])
    plt.subplots_adjust(left=0.08, bottom=0.13, right=0.98, top=0.95,hspace=0.0)
    plt.savefig("pdf/%s-spectrum-%d.png"%(tag,wh[0]),dpi=300)
    return view_dict


def restframe_weight(model,instrument,xrange=[5372.5470,5476.0942],sn=1000):
    x = model.decoder.wave_rest
    w = torch.zeros_like(x).float()
    w[(x>xrange[0])*(x<xrange[1])]=sn**2
    return w

def similarity_loss(x,slope=1.0,wid=5.0):
    sim_loss = torch.sigmoid(slope*x-wid/2)+torch.sigmoid(-slope*x-wid/2)
    return sim_loss

def similarity_restframe(instrument, model, s=None, slope=1.0, sig_s=1.0,
                         individual=False, sn=200):
    _, s_size = s.shape
    device = s.device

    spec = model.decode(s)
    batch_size, spec_size = spec.shape

    # pairwise dissimilarity of spectra
    S = (spec[None,:,:] - spec[:,None,:])**2
    # dissimilarity of spectra
    # of order unity, larger for spectrum pairs with more comparable bins
    W = restframe_weight(model,instrument,sn=sn)
    print("W:",W.min(),W.max())
    print("sn:",sn)
    spec_sim = (W * S).sum(-1) / W.count_nonzero()
    # dissimilarity of latents
    s_sim = ((s[None,:,:] - s[:,None,:])**2/sig_s**2).sum(-1) / s_size

    x = s_sim-spec_sim
    sim_loss = similarity_loss(x,slope=slope)
    diag_mask = torch.diag(torch.ones(batch_size,device=device,dtype=bool))
    sim_loss[diag_mask] = 0
    if individual:
        return s_sim,spec_sim,sim_loss
    # total loss: sum over N^2 terms,
    # needs to have amplitude of N terms to compare to fidelity loss
    return sim_loss.sum() / batch_size

def plot_latent(embed, true_z, axs, titles=[], locid=[]):
    if titles==[]:titles=[""]*n_encoder
    if locid==[]:locid = ["z=%.2f"%n for n in newz] + ["true z=%.2f"%true_z[j]]
    # plot latent variables
    for j in range(n_encoder):
        ax = axs[j]
        s = embed[j].detach().numpy()
        # visualize latent space
        ax.matshow(s,vmax=vmax,vmin=vmin)#,cmap='winter')
        
        title="%s"%(titles[j])
        
        for i in range(len(locid)):
            ax.text(-0.1,i+0.2,locid[i],c="w", weight='bold',fontsize=18)
        ax.set_xlabel("latent variables");ax.set_ylabel("copies")
        ax.set_title(title,weight='bold',fontsize=20)
        ax.set_xticks([]);ax.set_yticks([])
    return


def plot_model(spec, model, axs=[],fs=12):

    if axs==[]:
        fig,axs = plt.subplots(1,2,figsize=(8,4),dpi=200,
                               constrained_layout=True,
                               gridspec_kw={'width_ratios': [1.5,1]})
    #losses = np.zeros((2,loss.shape[2]))
    losses = loss[0][0]
    #losses[1] = loss[1][0].sum(axis=1)
    
    non_zero = np.sum(losses,axis=1)>0
    losses = losses[non_zero].T
    
    #print("mask:",non_zero.shape)
    print("\n\n",model_file,"losses:",losses.shape)
    #exit()
    plot_loss(losses,ax=axs[0],fs=fs)
    
    np.random.seed(23)
    rand = np.random.randint(len(spec), size=10)

    ax=axs[1]

    inds = rand
    s = model_encode(model,spec[inds],template)

    s = tensor2array(s)
    s -= s.mean(axis=0)
    batch_size,s_size = s.shape
    locid = ["sample %d"%(ii) for ii in rand] 

    vmin,vmax=s.min(),s.max()
    print("smin,smax:",vmin,vmax)
    ax.matshow(s,vmax=vmax,vmin=vmin)#,cmap='winter')
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel("latent variables");
    ax.set_ylabel("spectra")

    for i in range(len(locid)):
        ax.text(s_size,i+0.2,locid[i],c="k", weight='bold', 
                fontsize=12,alpha=1)
    axs[0].set_title("(%s) best loss = %.2f"%(model_file,min(losses[0])),fontsize=fs)
    return

def normalization(x, m, w):
    # apply constant factor c that minimizes (c*m - x)^2
    mw = m*w
    c = (mw * x).sum() / (mw * m).sum()
    return c

def polynomial(model_obs,ydata, wdata, wave_obs,wavemid=6000.0):
    ratio = ydata/model_obs
    x = wave_obs/wavemid
    p = np.polyfit(x,ratio,3,w=wdata)
    ypoly = np.polyval(p,x)
    print("p:",p)
    return model_obs*ypoly, p

def ccf_1d(y1,y2,n=10):
    ccf = np.zeros(n)
    for k in range(n):
        ccf[k] = (y1[k:]*y2[:len(y2)-k]).mean()
    return ccf

def kl_divergence(x,n=10):
    bins = np.linspace(x.min(),x.max(),n)
    digitized = np.digitize(x, bins)
    counts = np.array([(digitized == i).sum() for i in range(1, n)])
    P = counts/counts.sum()
    print("bins:",bins,"counts:",counts)
    Q = np.ones_like(P)/len(P)
    #kl = (P * (P / Q).log()).sum()
    kl = (P * np.log(P / Q)).sum()
    return kl

def moving_median(x,y,n=20):
    xgrid = np.linspace(x.min(),x.max(),n)
    ygrid = np.zeros_like(xgrid)
    delta_y = np.zeros_like(xgrid)
    dx = xgrid[1]-xgrid[0]
    print("dx:",dx)
    
    for i,xmid in enumerate(xgrid):
        mask = x>(xmid-0.5*dx)
        mask *= x<(xmid+0.5*dx)
        
        ygrid[i] = np.mean(y[mask])
        delta_y[i] = np.std(y[mask])/np.sqrt(mask.sum())
    return xgrid,ygrid,delta_y

def show_restframe(ids,model,spec,w,template=0,acf=None):

    instrument = model.encoder.instrument
    WAVE_REST = tensor2array(model.decoder.wave_rest)
    WAVE_OBS = tensor2array(instrument.wave_obs)
    WAVE_REST=np.array(WAVE_REST,dtype="double")
    WAVE_OBS=np.array(WAVE_OBS,dtype="double")

    s = model_encode(model,spec,template)
    fitted_RV = model.estimate_rv(spec-template)
    z = (fitted_RV)/instrument.c
    
    yrest = tensor2array(model.decode(s))

    loss = model.loss(spec, w, s, z, individual=True)
    loss = tensor2array(loss)

    fitted_RV = tensor2array(fitted_RV)
    
    #yrest -= yrest[0]#.max(axis=0)
    sel = np.arange(len(spec))#[0,2,13,22,24,27] 
    #sel = np.arange(20,30)
    colors = ["k",'b','c','m','orange','navy',"gold","skyblue"]
    
    #color = ["k",'b','orange','springgreen','navy']
    xlim = [None]
    ylim=(-0.01,0.01)
    ylabel = ["relative flux", "derivative","chi^2"]
    nrows=len(ylabel)

    fig,ax=plt.subplots(nrows=1,constrained_layout=True,
                         figsize=(5,5))
    for i_color,i in enumerate(sel[:8]):
        yoffset = i_color*0.005
        ax.plot(WAVE_REST,yoffset+yrest[i],"-",
                c=colors[i_color%(len(colors))],
                lw=0.5,drawstyle="steps-mid",
                label="%d RV_fit=%.2f m/s loss=%.2f"%(ids[i],fitted_RV[i],loss[i]))

    baseline = tensor2array(model.decoder.spec_rest)
    #ax.plot(WAVE_REST,baseline,"r-",label="baseline")
    ax.legend()
    ax.set_ylabel("activity spectra")
    ax.set_xlim(5407,5423)
    ax.set_ylim(-0.01,yoffset+0.005)
    plt.savefig("[restframe]%s.png"%model_file[:-3],dpi=300)

    slope = 2.0
    sig_s = 4.0
    s_sim,spec_sim,sim_loss = similarity_restframe(instrument, model, s=s, individual=True,slope=slope,sn=300,sig_s=sig_s)

    spec_sim = tensor2array(spec_sim)
    s_sim = tensor2array(s_sim)
    sim_loss = tensor2array(sim_loss)
    print("spec_sim:",spec_sim.min(),spec_sim.mean(),
          spec_sim.max())
    print("s_sim:",s_sim.min(),s_sim.mean(),s_sim.max())

    s_chi = s_sim.max()*torch.rand(size=(1000,1))
    spec_chi = spec_sim.max()*torch.rand(size=(1000,1))

    x = s_chi - spec_chi
    loss_bg = similarity_loss(x,slope=slope)
    vmax = 0.3
    fig,ax=plt.subplots(nrows=1,constrained_layout=True,
                        figsize=(5,5))
    img = ax.scatter(spec_sim,s_sim,c=sim_loss,ec="k",cmap="plasma",vmax=vmax)
    ax.scatter(spec_chi,s_chi,c=loss_bg,cmap="plasma",zorder=-10,vmax=vmax)
    ax.set_xlabel("spectral $\chi^2$")
    ax.set_ylabel("latent $\chi^2$")
    cbar = fig.colorbar(img, ax = ax, shrink=0.5)
    cbar.set_label("similarity loss")
    ax.set_title("slope %.2f, sig_s=%.2f"%(slope,sig_s))
    plt.savefig("[similarity]%s.png"%model_file[:-3],dpi=300)
    return

def z_offset_loss(z_off, z_off_true, sigma_z=1e-9,individual=False):
    z_loss = ((z_off - z_off_true)/sigma_z)**2
    if individual:return z_loss
    return z_loss.sum()

def get_timeseries(colname,sample_names):
    return np.array([neid_dict[key][colname] for key in sample_names])

def tensor2array(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else: return tensor.detach().numpy()
#-------------------------------------------------------
import matplotlib

model_file = sys.argv[2]
select = sys.argv[1]


instrument = Synthetic()

n_encoder = 1

models, loss, n_latent = load_model("%s"%(model_file),[instrument])
model = models[0]
model.eval()
print("loss:",loss.shape)

epoch = max(np.nonzero(loss[0][0])[0])+1
print("loss:",loss.shape,"epoch:",epoch)
print("Train :", loss[0][0][epoch-3:epoch])
print("Valid :", loss[1][0][epoch-3:epoch])

torch.manual_seed(0)
random.seed(0)
batchname = os.path.join(dynamic_dir,"%s.pkl"%select)
wave_obs = instrument.wave_obs
spec_star,w_star,berv_star,ids_star = load_batch(batchname)
spec_star = spec_star.to(device=device)
w_star = w_star.to(device=device)
print("spec_star:",spec_star.shape,spec_star.is_cuda)

template = load_batch("%s/%s-template.pkl"%(dynamic_dir,select))[0]
template = template.to(device=device)

with open("%s-param.pkl"%select,"rb") as f:
    neid_dict=pickle.load(f)
    select_order = 60
print(neid_dict["neidL2_20210617T175957.fits"].keys())

sample_names = list(neid_dict.keys())

timestamp = get_timeseries('OBSJD',sample_names)
ssbrvs = get_timeseries('SSBRV',sample_names)
ccfrvs = get_timeseries('CCFRV',sample_names)
chi_template = np.array([neid_dict[key]["chi_template"] for key in sample_names])

km_m = 1e3
#rv_star,depth,skewness,broaden = param.T
rv_star = (ccfrvs-np.mean(ccfrvs))*km_m
#z_doppler = rv_star/Synthetic.c
print("chi_template: %.2f, %.2f, %.2f"%tuple(np.quantile(chi_template,[0.0,0.5,1.0])))

if "model" in sys.argv:
    plot_model(spec_star, model, axs=[])
    plt.savefig("check_model.png",dpi=300)
    exit()

if "sampler" in sys.argv:
    #which_star = np.arange(0,len(spec_star),100)#[70,302]
    #summary_tag = "ordinary"
    which_star = [3952, 5555, 2282, 2868, 3007, 1392, 2903, 2897, 3006, 2876]
    summary_tag = "outlier"
    summary_dict = {}
    for count,where in enumerate(which_star):
        wh = [where]
        obsname = sample_names[where]
        print("\n\n v_ccf: %.3f m/s"% (rv_star[where]))
        label = "chi_template: %.5f"%chi_template[where]
        view_dict = spectra_1D(spec_star[wh],w_star[wh],berv_star[wh],
                               ids_star[wh],instrument,template,label=label,tag=summary_tag)
        view_dict.update(neid_dict[obsname])
        summary_dict[where] = view_dict
        if (count%10 == 0) or count==(len(which_star)-1):
            print("summary_dict:",len(summary_dict))
            with open("[%s]summary_dict.pkl"%summary_tag,"wb") as f:
                pickle.dump(summary_dict,f)
    exit()

if "rest" in sys.argv:
    #wh = np.arange(0,50)
    #wh = [4,70,101,109,302,499]
    wh = np.arange(0,len(spec_star))[::20]
    #wh = np.arange(0,100)[::5]
    spec = spec_star[wh]
    w = w_star[wh]
    show_restframe(wh,model,spec,w,template=template)
    exit()
    
if "plot-rv" in sys.argv:
    wh = np.arange(0,100)
    rv_fit = model.estimate_rv(spec_star[wh]-template)
    timestamp = ids_star[wh].detach().numpy()
    berv_star = berv_star[wh].detach().numpy()
    fitted_rv = rv_fit.detach().numpy()
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    ax.plot(timestamp,berv_star-berv_star.mean(),
             ".-",c="grey",label="Barycentric RV")
    ax.plot(timestamp,fitted_rv-fitted_rv.mean(),
             ".-",c="r",label="$v_{encode}$")
    ax.set_xlabel("JD")
    ax.set_ylabel("Radial Velocity (m/s)")
    ax.legend()
    plt.savefig("test.png",dpi=200)
    exit()
    
if "loss" in sys.argv:
    print("spec_star:",spec_star.shape)
    s = model_encode(model,spec_star,template)
    rv = model.estimate_rv(spec_star-template)
    z = (rv)/instrument.c
    y_act, _, spectrum_observed = model._forward(spec_star, w_star, s, z)
    fid_loss = model._loss(spec_star, w_star, spectrum_observed,individual=True)
    
    print("fid_loss:",fid_loss.min(),fid_loss.max(),
          "where",torch.argmax(fid_loss))
    exit()
    flex_loss = (y_act**2/0.01).sum(dim=1)

    batch_copy = instrument.augment_spectra([spec_star,w_star,_,ids_star],z)
    z_aug = model.estimate_rv(batch_copy[0]-template)/instrument.c
    z_off = z_aug - z
    z_off_true = batch_copy[3]
    z_loss = z_offset_loss(z_off, z_off_true,individual=True)
    losses = [fid_loss,z_loss,flex_loss]
    sloss =["%.3f"%loss.mean().item() for loss in losses]
    print("fid_loss & z_loss & flex_loss ")
    print(" & ".join(sloss))
    exit()



N = len(ids_star)
batch_size = 200
wh = np.arange(N)

spec,w,ID = spec_star[wh],w_star[wh],ids_star[wh]
rv = rv_star[wh]
timestamp = ID

spec_array = tensor2array(spec)
w_array = tensor2array(w)

wave_rest = model.decoder.wave_rest

waverest_array = tensor2array(wave_rest)
wave_obs_array = tensor2array(wave_obs)

save_latent = "[%s]%s.pkl"%(model_file,select)
if "load" in sys.argv:
    with open(save_latent,"rb") as f:
        latent = pickle.load(f)
        latent_aug = pickle.load(f)
        fitted_RV = pickle.load(f)
        encoded_RV = pickle.load(f)
        rv = pickle.load(f)
        fit_chi = pickle.load(f)
else:
    latent = np.zeros((N,n_latent))
    encoded_RV = np.zeros((N))
    fitted_RV = np.zeros((N))
    latent_aug = np.zeros_like(latent)
    fit_chi = np.zeros((N))
    
    sections = np.arange(batch_size,N,batch_size)
    print(sections)
    idxs = np.split(np.arange(N),sections)
    
    for idx in idxs:
        s = model_encode(model,spec[idx],template)
        rv_encode = model.estimate_rv(spec[idx]-template)
        #z_fit = rv_fit/instrument.c

        latent[idx] = tensor2array(s)
        # try chi^2 fitting???
        yrest = model.decoder.spec_rest+model.decode(s)
        rv_fit,chi = fit_rv([waverest_array,tensor2array(yrest),
                             wave_obs_array,spec_array[idx],w_array[idx]],
                             p0=tensor2array(rv_encode))
        z_fit = rv_fit/instrument.c

        encoded_RV[idx] = tensor2array(rv_encode).T[0]
        fitted_RV[idx] = tensor2array(rv_fit).T[0]
        fit_chi[idx] = chi
        #if "aug" in sys.argv:
        spec_aug,_,_,_ = Synthetic.augment_spectra([spec[idx],w[idx],None,ID[idx]])
        s_aug = model_encode(model,spec_aug,template)
        latent_aug[idx] = tensor2array(s_aug)
        print("s_aug:",s_aug.shape)

    latent = latent.T
    latent_aug = latent_aug.T

    with open(save_latent,"wb") as f:
        pickle.dump(latent,f)
        pickle.dump(latent_aug,f)
        pickle.dump(fitted_RV,f)
        pickle.dump(encoded_RV,f)
        pickle.dump(rv,f)
        pickle.dump(fit_chi,f)


var = np.array([item/item.std() for item in latent])
#target = param[0]/param[0].std()
'''
target = encoded_RV/encoded_RV.std()
mi_init = ee_mutual_information(var.T,target)
print("encoded_RV RV mi_init:",mi_init)
'''
target = fitted_RV/fitted_RV.std()
mi_init = ee_mutual_information(var.T,target)
print("fitted_RV RV mi_init:",mi_init)
