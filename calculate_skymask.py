#!/usr/bin/env python
# coding: utf-8
import io, os, sys, time, random, re
import numpy as np
import pickle
import torch
from synthetic_data import Synthetic
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d
from spender_model import TelluricModel
from util import load_batch,interpolate_to_input_grid_raw

def plot_latents(latent,cdata,clabel,cmap="viridis",vrange=[]):
    mask = []
    if latent.shape[1]==3:indices = [[0,1],[1,2],[0,2]]
    elif latent.shape[1]==5:indices = [[0,1],[1,2],[0,2],[3,4]]

    if vrange==[]:vmin,vmax = np.quantile(cdata,[0.01,0.99])
    else: vmin,vmax = vrange
    if cdata.ndim==2:nrows = cdata.shape[1]
    else: nrows = 1
    if nrows == 1: cdata=cdata[:,None]
    if mask != []:
        cdata = cdata[mask,:]
        latent = latent[mask,:]
    ncols=len(indices)
        
    fig,axs=plt.subplots(figsize=(1+2*ncols,2*nrows),nrows=nrows,dpi=200,ncols=ncols,
                         constrained_layout=True)
    for j in range(nrows):
        for i,wh in enumerate(indices):
            if nrows==1: ax=axs[i]
            else: ax=axs[j,i]
            img=ax.scatter(latent[:,wh[0]],latent[:,wh[1]],c=cdata[:,j],cmap=cmap,vmin=vmin,vmax=vmax,
                           label="N=%d"%len(cdata[:,j]),s=10)
            ax.set_xlabel("$s_%d$"%(wh[0]+1));ax.set_ylabel("$s_%d$"%(wh[1]+1))
        cbar=plt.colorbar(img,ax=ax)
        cbar.set_label(clabel)
    plt.savefig("[2D-latents]test.png",dpi=200)
    return axs

def load_model(mainfile, instrument, device):
    mdict = torch.load(mainfile, map_location=device)
    wave_rest =  mdict['model'][0]['wave_rest']
    spec_rest =  mdict['model'][0]['spec_rest']
    bias = mdict['model'][0]['encoder.mlp.mlp.9.bias']

    model = TelluricModel(wave_rest,spec_rest,instrument)
    model.load_state_dict(mdict['model'][0], strict=False)
    model.to(device)
    losses = mdict['losses']
    return model, losses

def save_auxfile(input_data,filename):
    input_data = torch.from_numpy(input_data.astype(np.double))
    print("input_data",input_data.shape)
    print("saving to %s..."%filename)
    with open(filename, 'wb') as f:
        pickle.dump(input_data, f)
    return

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
    return sorted(zip(unique_wavelengths, unique_depths), key=lambda x: x[0])

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


def visualize_telluric(line,wave_raw,spec_raw,telluric_absorption,unique_lines,i_image=0,tag="telluric"):
    dispersion = telluric_absorption.std(axis=0)
    #cdatas = [np.copy(z),np.copy(water_vapor),np.copy(zenith)]
    #mycmaps = ['gist_rainbow','plasma','inferno_r']
    #clabels = ["Barycentric Motion","Water Vapor","Sun Angle"]
    cdata,mycmap,clabel=np.copy(water_vapor),plt.get_cmap('plasma'),"Water Vapor"
    normalized_cdata = (cdata-cdata.min())/(cdata.max()-cdata.min())
    colors = [mycmap(ii) for ii in normalized_cdata]

    print(line)
    o = line["order"]
    ind = line["ind"]
    snr = line["snr"]
    noise_level = line["noise"]
    wave,depth = line["wave"],line["depth"]
    corr = line["correlation_water"]
    line_depths = 1-telluric_absorption[:,o,ind]
    xlim = [wave-0.5,wave+0.5]
    mask = (wave_obs[o]>xlim[0])&(wave_obs[o]<xlim[1])

    star_line_depth=line["stellar_depth"]
    stellar_wave=line["stellar_wave"]
    star_distance = wave-stellar_wave

    starlabel = "Stellar Line (Depth=%.2f,$\Delta \lambda/0.1\AA=%.1f$)"%(star_line_depth,np.abs(star_distance/0.1))
    print(starlabel)
    #if stellar or telluric:continue
    fig,axs = plt.subplots(figsize=(12,6),ncols=2,constrained_layout=True)
    ax=axs[0]
    for i,key in enumerate(targets):
        ax.plot(wave_raw[i][o],spec_raw[i][o],"k-",alpha=0.2,lw=1)
        ax.plot(wave_obs[o][mask],telluric_absorption[i][o][mask],"-",color=colors[i],lw=1)
    cbaxes = inset_axes(ax, width="30%", height="3%", loc=2)
    sm = plt.cm.ScalarMappable(cmap=mycmap)
    cbar = plt.colorbar(sm,cax=cbaxes,orientation='horizontal',ticks=[])
    cbar.set_label(clabel)
    for item in unique_lines:ax.axvline(item[0],ls="--",color="grey",zorder=0)
    ax.axvline(wave,ls="--",color="r",label="Target Line (Depth=%.3f, S/N=%.2f)"%(depth,snr))
    #ax.axvline(stellar_wave,ls="--",color="b",alpha=0.7,label=starlabel)
    ax.fill_between(wavemean[o],0,1.2,where=raw_mask[o],color="lightgrey",alpha=0.5,
                 zorder=-10)
    #ax.plot(wave_obs[o],1+dispersion[o],"r-",lw=1,label="Element-wise Dispersion")
    #ax.fill_between(wave_obs[o],1,1+noise_level,color="r",alpha=0.3,label="Dispersion Noise Level",lw=0)
    ax.set_xlim(xlim)
    ax.set_ylim(0.9,1.05)
    ax.legend(loc=2)
    ax=axs[1]
    ax.plot(line_depths,water_vapor,"k.",label="Correlation: %.3f"%corr)
    ax.legend()
    ax.set_xlabel("Line Depth")
    ax.set_ylabel("Water Vapor")
    plt.savefig("runtime/%s-%d.png"%(tag,i_image),dpi=200)
    return

def correlation_coefficients(x,y):
    print("x:",x.shape,"y:",y.shape)
    # Standardize x
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_standardized = (x - x_mean) / x_std

    # Standardize y
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True)
    y_standardized = (y - y_mean) / y_std

    # Calculate correlation coefficients
    corr_coeff = np.dot(y_standardized, x_standardized) / len(x)
    return corr_coeff
# ----------------------------------------

data_dir = "/scratch/gpfs/yanliang"

if "cpu" in sys.argv:device =  torch.device("cpu")
else:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device('cuda:2')
    
def tensor2array(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else: return tensor.detach().numpy()

def get_timeseries(timetable,timestamp,colname):
    timeseries = [timetable[t][colname] for t in timestamp]
    return np.array(timeseries)

def evaluate_sky_model(model,instrument,batch,template=None):  
    telluric_offset = model.evaluate_telluric_rv_offset(batch[4])
    #print(f"\n Additional Telluric Offset:{telluric_offset.min().item():.3f} m/s, {telluric_offset.max().item():.3f} m/s")
    polyb = model.evaluate_wavelength_polynomial(batch[0],batch[4])
    print(f"\nWavelength shift:{polyb.std(dim=1).max()*instrument.c:.3f} m/s \n")

    # shift to the stellar restframe - frame of wave_obs
    spectrum, w, ssbrv,jd = interpolate_to_input_grid_raw(batch,instrument,template,polyb=polyb)#,extra_rv=extra_rv)
    spec_input = spectrum - template
    spec_input[w<1] = 0

    wave_raw,spec_raw,w_raw,ssbrv,jd = batch
    z_sky = (ssbrv)/instrument.c
    s_sky = model.encode(spec_input)

    lines,continuum,y_act,spec_model =  model._forward(s_sky,z_sky,wave_obs)

    loss_ind = w * (spectrum - spec_model).pow(2)
    loss_ind = torch.sum(loss_ind, dim=-1) / torch.sum(w>1,dim=-1)
    fid_loss = torch.sum(loss_ind)
    print("fid_loss:",fid_loss/spec_raw.shape[0])

    #lines = model.transform(lines,z_sky,wave_obs)
    
    model_resid = spec_model - template
    model_resid[w<1] = 0
    output = {"time":jd,"ssbrv":ssbrv,
              "wave_raw":wave_raw,"spec_raw":spec_raw,
              "spectrum":spectrum,"model":spec_model,"w":w,
              "s_sky":s_sky,"y_sky":lines.squeeze(1),
              "y_act":y_act.squeeze(1),
              "continuum":continuum.squeeze(1)}
    output = {key:tensor2array(item) for key,item in output.items()}
    return output

def get_info(timetable,timestamp):
    info = {}
    colnames = ['CCFRV','obsname','WVAPOR','ZENITH']
    for name in colnames:
        info[name] = get_timeseries(timetable,timestamp,name)
    return info

def is_single_peaked(kernel, rel_height_thresh=0.1):
    """
    rel_height_thresh: fraction of the global max required 
                       for a peak to be considered significant.
    
    Returns True if only one *significant* peak is found.
    """
    k = np.ravel(kernel)
    peaks = np.where((k[1:-1] > k[:-2]) & (k[1:-1] > k[2:]))[0] + 1
    if len(peaks) == 0:
        return False
    
    max_height = k[peaks].max()
    significant = [p for p in peaks if k[p] >= rel_height_thresh * max_height]
    return len(significant) == 1

#-------------------------------------------------------
dynamic_dir = "/scratch/gpfs/yanliang/neid-production"

model_file = sys.argv[1]
#telluric_pattern = r'puresky_order(\d+)_(\w+)\.pt'
telluric_pattern = r'\w+_order(\d+)_(\w+)\.pt'
match = re.search(telluric_pattern, model_file)
order_value = int(match.group(1))
suffix = match.group(2)

suffix = 'full'
dataset_tag = f"newprod_order{order_value}_{suffix}"

template_data = load_batch(f"{dynamic_dir}/{dataset_tag}-template.pkl")
wave_obs = template_data[0]

wave_obs = wave_obs.to(device=device)
template_data = [item.to(device=device) for item in template_data]

template = template_data[1]
template_w = template_data[2]
instrument = Synthetic(wave_obs)


files = ["%s/%s"%(dynamic_dir,ii) for ii in os.listdir(dynamic_dir) if ii.startswith(dataset_tag) and bool(re.search(r'_\d+.pkl$', ii))]

model,losses = load_model("%s"%(model_file),instrument, device)
lsf = tensor2array(model.lines_act(model.lsf_kernel)[0][0])
lsf_single_peaked = is_single_peaked(lsf)


print("files",len(files))

with open(f"{dynamic_dir}/params/{dataset_tag}-param.pkl","rb") as f:
    neid_dict=pickle.load(f)
sample_names = neid_dict["info"]["sample_names"]
for item in sample_names:neid_dict[item]["obsname"]=item
# Create a time-to-parameters mapping
timetable = {np.float32(neid_dict[item]["timestamp"]): neid_dict[item] for item in sample_names}


summary = {}

for i,batch_name in enumerate(files[:10]):
    print("batch_name:",batch_name)
    batch = load_batch(batch_name)
    batch = [item.to(device=device) for item in batch]
    timestamp = tensor2array(batch[4][:,0])
    info_dict = get_info(timetable,timestamp)
    loc_dict = evaluate_sky_model(model,instrument,batch,template)

    if i==0:
        summary['data']=loc_dict
        summary['info']=info_dict
        continue

    for key in loc_dict:
        summary['data'][key] = np.concatenate((summary['data'][key],loc_dict[key]))
    for key in info_dict:
        summary['info'][key] = np.concatenate((summary['info'][key],info_dict[key]))


summary['info']['wave_obs']= tensor2array(wave_obs)
summary['info']['spec_rest']= tensor2array(model.spec_rest)

star_latent = summary["data"]["s_sky"][:,2:5]

timestamp = summary["data"]["time"]

water_vapor = summary["info"]["WVAPOR"]
print("water_vapor:",water_vapor.shape,
      water_vapor.min(),water_vapor.max())
mask = water_vapor>=0
zenith = summary["info"]["ZENITH"]

wave_raw = summary["data"]["wave_raw"]
spec_raw = summary["data"]["spec_raw"]
y_sky = summary["data"]["y_sky"]
y_act = summary["data"]["y_act"]
y_continuum = 1+summary["data"]["continuum"]
spec_data = summary["data"]["spectrum"]
model = summary["data"]["model"]
w = summary["data"]["w"]
ssbrv = summary["data"]["ssbrv"]
wave_obs = summary['info']['wave_obs'][0]

cont_avg = y_continuum.mean(axis=0)

sky_latent = summary["data"]["s_sky"][:,:2]
star_latent = summary["data"]["s_sky"][:,2:5]


cdata,clabel,cmap = water_vapor,"water vapor","inferno"
#cdata,clabel,cmap = timestamp,"Time","plasma"
plot_latents(star_latent,cdata,clabel,cmap=cmap)

#'''

spec_avg = np.median(spec_data,axis=0)
spec_resid = spec_data-spec_avg
model_resid = model-spec_avg

wh = np.argmax(y_continuum.std(axis=1))
print("wh:",wh)
good = (w[wh]>10)&(spec_avg>0)
plt.figure(figsize=(10,4))
plt.plot(wave_obs[good],spec_resid[wh][good],"k-",label="data")
plt.plot(wave_obs[good],model_resid[wh][good],"r-",label="model")
plt.plot(wave_obs,y_continuum[wh]-cont_avg,c="cyan",label="continuum")
#plt.plot(wave_obs,y_continuum[:20].T)
#plt.plot(wave_obs,cont_avg,'k-')
#plt.plot(wave_obs,continuum[wh].T,'c-')
#plt.ylim(-0.1,0.05)
plt.legend()
plt.savefig("test.png",dpi=200)

#exit()

print(summary['info'].keys())

#print("water_vapor:",water_vapor.shape)
#plt.hist(water_vapor)
#plt.savefig("test.png")

cdatas = [np.copy(water_vapor),np.copy(zenith)]
mycmaps = ['plasma','inferno_r']
clabels = ["Water Vapor","Sun Angle"]


batch_size,n_spec = y_sky.shape
np.random.seed(0)
targets = np.random.choice(np.arange(len(water_vapor)), size=500)
#np.arange(0,min(500,batch_size))
dispersion = np.std(y_sky,axis=0)
y_sky = 1-y_sky

print("dispersion:",dispersion.shape)

#'''
for n_it in range(5):
    dispersion = np.std(y_sky,axis=0)
    stdcut = np.quantile(dispersion,0.5)
    print("80% quantile dispersion:",stdcut)
    continuums = []

    continuum = dispersion<stdcut
    telluric_continuum = y_sky[:,continuum]

    norm_continuum = np.median(telluric_continuum,axis=1,keepdims=True)

    print("telluric_continuum:",telluric_continuum.shape)
    print("norm_continuum:",norm_continuum.shape)
    print((telluric_continuum/norm_continuum).shape)
    diag = (telluric_continuum/norm_continuum).std(axis=0).mean()
    print("continuum: %d dispersion: %.4f"%(continuum.sum(),diag))
    continuums.append(continuum)

    y_sky /= norm_continuum
    #continue
    fig,ax = plt.subplots(figsize=(12,2),constrained_layout=True)
    for i,key in enumerate(targets[:50]):
        ax.plot(wave_obs,y_sky[i],"b-",alpha=0.2,lw=1)
    ax.plot(wave_obs,1-dispersion,"r-",lw=1)
    ymin,ymax= y_sky.min(),y_sky.max()
    ax.fill_between(wave_obs,ymin,ymax,lw=0,
                    where=~continuum,color="lightgrey")
    ax.set_ylim(ymin,ymax)
    plt.savefig("runtime/[iter-%d]renormalize.png"%n_it,dpi=300)
#'''
#maximum = y_sky.max(axis=0,keepdims=True)
#telluric_absorption = 1+y_sky-maximum
telluric_absorption = y_sky
print("y_sky:",y_sky.shape)
print("water_vapor:",water_vapor.shape)

# check if deepest line is deeper than 0.1% flux
print("dispersion max:",dispersion.max())

if lsf.max()<0.135 or not lsf_single_peaked:
    message = f"Proxy: NEID Water Vapor  (Max LSF={lsf.max():.3f}, Single Peak: {lsf_single_peaked})"
    telluric_proxy = water_vapor
else:
    message = f"Proxy: Empirical  (Max LSF={lsf.max():.3f}, Single Peak: {lsf_single_peaked})"
    telluric_proxy = 1-y_sky[:,np.argmax(dispersion)]
print("message:",message)
#save_dict = {'wave_obs':wave_obs,'water_vapor':water_vapor,
#             'spec_data':spec_data,'w':w,'model':model}
#with open(f"{dataset_tag}_diag.pkl","wb") as f:
#    pickle.dump(save_dict,f)

dispersion = np.std(y_sky,axis=0)
corr_coeff = np.zeros((n_spec))

y = 1-telluric_absorption.T
corr_coeff = correlation_coefficients(telluric_proxy,y)
print("corr_coeff:",corr_coeff.shape)

corr_coeff = gaussian_filter1d(corr_coeff,1)
corr_coeff[corr_coeff<0] = 0


n_spec = wave_obs.shape[0]
# identify deepest lines
all_lines = []
max_lines = 200

lines = find_deepest_lines(wave_obs, 1-corr_coeff, num_lines=max_lines, min_separation=0.2)
lines = [item for item in lines if item[1]>0.90]
print("Solid lines:",len(lines))
all_lines.extend(lines)

skymask = mask_deepest_lines(wave_obs,all_lines,mask_width=0.2)
#raw_mask[o] = mask_deepest_lines(wavemean[o],all_lines,mask_width=0.10)
summary['info']['skymask'] = skymask

print("mask fraction: %.2f"%(skymask.sum()/n_spec))
save_auxfile(skymask,"/scratch/gpfs/yanliang/neid-production/skymask/%s-skymask.pkl"%(dataset_tag))

dispersion = np.std(y_act,axis=0)
dispersion[:20] = 0
dispersion[-20:] = 0

sample_spec = tensor2array(template)
sample_spec[sample_spec==0] = 1
max_dispersion = np.argmin(sample_spec)
#max_dispersion = np.argmax(dispersion)
print("max_dispersion:",max_dispersion)
activity_level = y_act[:,max_dispersion]
print("activity_level:",activity_level.shape)
print("telluric_absorption:",telluric_absorption.shape)
print("y_continuum:",y_continuum.shape)

def spectral_correlation(a,b):
    coeff = np.zeros((a.shape[1]))
    for i in range(a.shape[1]):
        coeff[i] = np.corrcoef(a[:,i],b[:,i])[0][1]
        #coeff[i] = np.dot(a[:,i],b[:,i])
    #print("a:",a.min(),a.max())
    #print("b:",b.min(),b.max())
    #coeff = 1e3*np.abs(np.mean(a*b,axis=0))
    coeff[:4] = 0
    coeff[-4:] = 0
    print(f"Corr: min:{coeff.min():.2f} max:{coeff.max():.2f} mean:{coeff.mean():.2f}")
    return coeff

print("telluric-activity correlation:")
corr = spectral_correlation(telluric_absorption,y_act)
print("telluric-continuum correlation:")
sky_cont_corr = spectral_correlation(telluric_absorption,y_continuum-1)
print("activity-continuum correlation:")
act_cont_corr = spectral_correlation(y_act,y_continuum)
if np.abs(act_cont_corr).max()>np.abs(sky_cont_corr).max():
    continuum_clabel = "Activity Level"
else: continuum_clabel = "Telluric Level"

continuum_clabel = "Activity Level"

style = {"Activity Level":[np.copy(activity_level),'inferno'],
         "Telluric Level":[np.copy(telluric_proxy),'plasma'],
         "SSBRV":[np.copy(ssbrv),'gist_rainbow'],
         "Time":[np.copy(timestamp),'gist_rainbow']}
specs = [telluric_absorption,1+y_act,y_continuum]
clabels = ["Telluric Level","Activity Level",continuum_clabel]
ylabels = ["Telluric Spec","Activity Spec","Continuum Spec"]

#clabels = ['Time']*3
#clabels =["Telluric Level"]*3

#skymask = corr_coeff>0.7
#print("skymask:",skymask.sum()/(n_order*n_spec))

print("wave_raw:",wave_raw.shape)
fig,axs = plt.subplots(figsize=(20,8),nrows=5,constrained_layout=True)
ax=axs[0]
for i in targets:
    ax.plot(wave_raw[i],spec_raw[i],"k-",alpha=0.1,lw=1)
    ax.plot(wave_obs,telluric_absorption[i],"b-",alpha=0.2,lw=1)
    
ax.set_title(f"Order {order_value}")

ax.set_ylim(0.8,1.05)
#ax.set_ylim(-0.1,0.1)
ylims = [(0.97,1.01),(0.996,1.003),(0.996,1.003)]
for i_row in range(3):
    ax=axs[i_row+1]
    if ylims[i_row] is None: ax.set_ylim(specs[i_row].min(),specs[i_row].max())
    else:ax.set_ylim(ylims[i_row])
    clabel = clabels[i_row]
    cdata,cmap = style[clabel]
    mycmap = plt.get_cmap(cmap)
    cmin,cmax = np.quantile(cdata,[0.01,0.99])
    print("clabel:",clabel,cmin,cmax)
    normalized_cdata = (cdata-cmin)/(cmax-cmin)
    colors = [mycmap(ii) for ii in normalized_cdata]
    for i in targets:
        ax.plot(wave_obs,specs[i_row][i],"-",c=colors[i],lw=1,zorder=normalized_cdata[i]+20)
        #ax.plot(wave_raw[i],spec_raw[i],"-",c=colors[i],lw=1)
    cbaxes = inset_axes(ax, width="20%", height="3%", loc=2)
    sm = plt.cm.ScalarMappable(cmap=mycmap)
    cbar = plt.colorbar(sm,cax=cbaxes,orientation='horizontal',ticks=[])
    cbar.set_label(clabel)
    #ax.plot(wave_obs,1-dispersion,"r-",lw=1,zorder=100)
    #ax.set_yticks([])
    #ax.legend(loc="lower left")
    ax.set_ylabel(ylabels[i_row])
    if clabel=="Telluric Level":
        ax.text(wave_obs[50],0.975,message,alpha=0.5,fontsize=16,
                weight="bold")


ax = axs[-1]
ax.plot(wave_obs,corr,"k-",lw=1,zorder=100)
ax.set_ylim(-1.0,1.0)
ax.set_ylabel("Correlation Coefficient")

for i,ax in enumerate(axs):
    ylim = ax.get_ylim()
    ax.fill_between(wave_obs,ylim[0],ylim[1],where=skymask,zorder=-10,
                    color="lightgrey",label="Telluric Mask")
    for line in all_lines:ax.axvline(line[0],ls="--",color="grey",zorder=0)
ax.set_xlabel("Wavelength ($\AA$)")

for ax in axs:
    #ax.set_xlim(5030,5050)
    ax.set_xlim(wave_obs[0],wave_obs[-1])
    print("wave_obs[max_dispersion]:",wave_obs[max_dispersion])
plt.savefig("runtime/[telluric]merge-%s.png"%dataset_tag,dpi=200)


