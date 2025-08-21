#!/usr/bin/env python
# coding: utf-8
import torch,os,sys,pickle,re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KDTree
from scipy.optimize import minimize

from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from util import load_batch,moving_mean,simulate_planet
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#### Simple MLP ####
class MLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden=(16, 16, 16),
                 act=(nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()),
                 dropout=0,
                 bias=True):
        super(MLP, self).__init__()

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_)-1):
                layer.append(nn.Linear(n_[i], n_[i+1],bias=bias))
                layer.append(act[i])
                layer.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layer)

    def forward(self, x):
        return self.mlp(x)

class ActivityEstimator(nn.Module):
    def __init__(self,
                 n_in,
                 n_out=1,
                 initial_guess=None,
                 n_channel=1,
                 n_hidden=(2,),
                 act=(nn.PReLU(), nn.Identity()),
                 dropout=0):
        super(ActivityEstimator, self).__init__()
        self.mlp = MLP(n_in,n_out,n_hidden=n_hidden,act=act,dropout=dropout)
        # Initialize additional trainable parameters
        if initial_guess is not None:
            K,P,ph = initial_guess
            self.planet_params = nn.Parameter(torch.tensor(initial_guess,device=device))
            self.P_init = P
        else: 
            self.planet_params = torch.zeros((3),device=device)
            self.P_init = 0

    def forward(self, x):
        x = self.mlp(x)
        return x

    def get_period(self):
        #delta_P = self.planet_params[1]
        #print(f"P{self.P_init:.3f} + {delta_P:.6f}")
        return self.planet_params[1]
    
    def doppler_rv(self,t):
        K,_,ph = self.planet_params
        P = self.get_period()
        _, v_doppler = simulate_planet(t,*(K,P,ph))
        return v_doppler

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

def simulate_planet_np(t,amp=1,period=0.11,phase_t0=0):
    phase = ((t/period)-phase_t0)%1
    v_planet = amp*np.sin(2*np.pi*phase)
    return phase,v_planet

def planet_param_chi(param,time,v_obs,v_err,full=False):
    amp,period,phase = param[:3]
    _,v_planet = simulate_planet_np(time,amp=amp,period=period,phase_t0=phase)
    loss = np.mean((v_obs-v_planet)**2/v_err**2)
    if full: 
        plin,cov = np.polyfit(v_planet,v_obs,deg=1,cov=True)
        slope,b = plin
        label=f"slope = {slope:.2f}+/- {cov[0][0]**0.5:.2f}"
        print(label)
        return v_planet,loss
    return loss

def initial_guess(period,time,v_ccf,v_err):
    phase_grid = np.arange(0,1,0.05)
    chi_best = np.inf
    for i,ph in enumerate(phase_grid):
        p_guess = [v_ccf.std(),period,ph]
        chi = planet_param_chi(p_guess,time,v_ccf,v_err)
        if chi<chi_best:
            best_guess = p_guess
            chi_best = chi
    amp_grid = np.arange(0.1,1,0.05)
    chi_best = np.inf
    for i,a in enumerate(amp_grid):
        p_guess = [a] + best_guess[1:]
        chi = planet_param_chi(p_guess,time,v_ccf,v_err)
        if chi<chi_best:
            best_guess = p_guess
            chi_best = chi
    return best_guess,chi_best

def print_params(params,param_names):
    print_p = np.copy(params)
    for i in range(len(params)):
        print(f"{param_names[i]}: {print_p[i]:.4f}")
    return

def compute_ccf(spectra, wavelengths, velocity_grid=np.linspace(-15, 15, 201)):    
    """
    Compute and normalize the Cross-Correlation Function (CCF) for a set of spectra against a template.
    
    Parameters:
        spectra (numpy.ndarray): Array of shape (N_spectra, N_pixels) containing spectral data.
        wavelengths (numpy.ndarray): 1D array of shape (N_pixels,) containing wavelength values.
        velocity_grid (numpy.ndarray): 1D array of velocity shifts in km/s.
        
    Returns:
        ccf_matrix (numpy.ndarray): Normalized array of shape (N_spectra, len(velocity_grid)) containing CCF values.
    """
    N_spectra, N_pixels = spectra.shape
    
    # Compute the template using absorption profile (1 - mean spectrum)
    template = 1 - np.mean(spectra, axis=0)
    
    # Initialize CCF matrix
    ccf_matrix = np.zeros((N_spectra, len(velocity_grid)))

    # Speed of light in km/s
    c = 299792.458

    for i, velocity in enumerate(velocity_grid):
        # Doppler shift the template wavelengths
        shifted_wavelengths = wavelengths * np.sqrt((1 + velocity / c) / (1 - velocity / c))

        # Interpolate the template to match the original wavelength grid
        interp_func = interp1d(shifted_wavelengths, template, kind='cubic', bounds_error=False, fill_value=np.nan)
        shifted_template = interp_func(wavelengths)

        # Compute cross-correlation for each spectrum
        for j in range(N_spectra):
            valid = ~np.isnan(shifted_template)
            ccf_matrix[j, i] = np.sum(spectra[j, valid] * shifted_template[valid])

    # Normalize CCF to range [0, 1]
    ccf_matrix = (ccf_matrix) / (
        np.max(ccf_matrix, axis=1, keepdims=True)
    )
    
    return ccf_matrix, velocity_grid

# Define a Gaussian function
def gaussian(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

def fit_gaussian_ccf(velocity_grid, ccf):
    """
    Fit a Gaussian to the CCF to determine the velocity offset.
    
    Parameters:
        velocity_grid (numpy.ndarray): 1D array of velocity values (km/s).
        ccf (numpy.ndarray): 1D array of normalized CCF values.
    
    Returns:
        popt (tuple): Best-fit parameters (amplitude, mean, sigma, offset).
        velocity_offset (float): The best-fit Gaussian center (km/s).
    """
    # Initial guesses
    a_init = np.min(ccf)  # Amplitude (min of CCF)
    mu_init = velocity_grid[np.argmin(ccf)]  # Initial guess for center
    sigma_init = 5  # Rough estimate of width
    c_init = np.max(ccf)  # Baseline

    p0 = [a_init, mu_init, sigma_init, c_init]

    # Fit Gaussian to CCF
    popt, _ = curve_fit(gaussian, velocity_grid, ccf, p0=p0)

    # Extract velocity offset (Gaussian mean)
    velocity_offset = popt[1]
    
    return popt, velocity_offset


def compute_bisspan(velocity_grid,ccf_matrix):
    v_ccf = np.zeros((ccf_matrix.shape[0]))
    params =  np.zeros((ccf_matrix.shape[0],4))
    for i,ccf in enumerate(ccf_matrix):
        popt, velocity_offset = fit_gaussian_ccf(velocity_grid, ccf)
        v_ccf[i] = velocity_offset*1e3
        params[i] = popt
    v_ccf -= v_ccf.mean()

    pos = velocity_grid>0
    depth = ccf_matrix.max()-ccf_matrix.min()
    depths_top = 1-np.linspace(0.1, 0.4, 20)*depth
    depths_bottom = 1-np.linspace(0.6, 0.9, 20)*depth
    bisspan = np.zeros_like(v_ccf)

    for i,ccf in enumerate(ccf_matrix):
        left_half = interp1d(ccf[~pos],velocity_grid[~pos],kind="linear")
        right_half =  interp1d(ccf[pos],velocity_grid[pos],kind="linear")

        v_top = np.array([np.mean([left_half(x),right_half(x)]) for x in depths_top])
        v_bottom = np.array([np.mean([left_half(x),right_half(x)]) for x in depths_bottom])
        bisspan[i] = np.mean(v_top)-np.mean(v_bottom)
    return params,v_ccf,bisspan

def tensor2array(tensor):
    if not torch.is_tensor(tensor):return tensor
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else: return tensor.detach().numpy()


def train_nn(input_data, planet_guess=None, n_epochs = 500, batch_size = 2000):
    input_tensors = []
    train_loaders = []
    models = []
    if planet_guess is None:doppler=False
    else:doppler=True

    for data_i in input_data:
        time,v_ccf,v_err,input_features = data_i

        features = np.copy(input_features)
        print(f"feature shape: {features.shape}")
        features -= np.mean(features,axis=0)
        features /= np.std(features,axis=0)
        features = torch.tensor(features,device=device).float()
        time_tensor = torch.tensor(time[:, None],device=device)
        v_ccf_tensor = torch.tensor((v_ccf)[:, None],device=device)
        verr_tensor = torch.tensor((v_err)[:, None],device=device)
        input_tensor = (features,v_ccf_tensor,verr_tensor,time_tensor)

        train_dataset = TensorDataset(*input_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = ActivityEstimator(features.shape[1], 1, planet_guess)
        model.to(device)

        input_tensors.append(input_tensor)
        train_loaders.append(train_loader)
        models.append(model)

    for j in range(len(models)):
        if j==0:continue
        models[j].planet_params = models[0].planet_params

    all_params = []
    for model in models: all_params.extend(model.mlp.parameters())
    all_params.append(models[0].planet_params)
    
    optimizer = torch.optim.Adam(all_params, lr=0.002)
    

    for epoch in range(n_epochs):
        for j in range(len(input_data)):
            model = models[j]
            model.train()
            optimizer.zero_grad()
            loss = 0
            for batch in train_loaders[j]:
                latent_i,v_app_i,sigma_v_i,times_i = batch
                
                v_act = model(latent_i)

                if doppler:v_doppler = model.doppler_rv(times_i)
                else: v_doppler = 0

                loss_ = (v_app_i - v_act - v_doppler)**2 / sigma_v_i**2
                # allow 1% outliers
                #print("loss_:",loss_.shape,loss_)
                #exit()
                loss += loss_.sum()# + loss_smooth

                break
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f'Instrument {j} Epoch {epoch + 1}/{n_epochs}, Resid RMS:{(v_app_i - v_act - v_doppler).std():.2f} m/s')

            #if epoch % 500 == 0:
            #    print("Saving model to %s..." % filepath)
            #    torch.save(v_activity_estimator.state_dict(), filepath)
    print("Training complete.")
    v_activity = []
    inst_rms = []

    for j,model in enumerate(models):
        model.eval()

        features,v_ccf,_,time = input_tensors[j]
        v_act_nn = model(features)
        if doppler:
            v_doppler = model.doppler_rv(time)
            v_doppler = tensor2array(v_doppler)
        else: v_doppler = np.zeros((len(time),1))
        v_act_nn = tensor2array(v_act_nn)
        rms = (tensor2array(v_ccf) - v_act_nn - v_doppler).std()
        print(f"Instrument {j} Residual RMS:{rms:.3f}m/s")
        period = model.get_period()
        planet_bestfit = tensor2array(model.planet_params)
        planet_bestfit[1] = period
        v_activity.append(v_act_nn)
        inst_rms.append(rms)
    return v_activity,v_doppler,planet_bestfit,inst_rms

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

def detect_ls_peaks(per,power,n_peaks = 30,frac=0.1,delta=0.1):
    rank = np.argsort(power)[::-1]
    top_n_periods = np.zeros((n_peaks))
    top_n_power = np.zeros((n_peaks))

    i = 0
    for rk in rank:
        per_max = per[rk]
        if per_max>390:continue
        if per_max<1:continue
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
    mask = top_n_periods>0
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

basename = sys.argv[1]
outdir = "initial_guess"
#output_txt = f"{outdir}/{basename}_ccf.txt"
output_txt = f"{outdir}/{basename}_test.txt"


per = np.logspace(-1.2, 2.6, 5000)
frequency = 1.0/per
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

planet_pattern = r'period(\d+\.\d+)d_K(\d+\.\d+)m_phase(\d+\.\d+)'

tag = "prod"

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


truth = [planet_amp,planet_period,t0]
print("planet:",truth)

sum_file = f"summary_file/{basename}_purez_sum.pkl"
with open(sum_file,"rb") as f:
    info_dict = pickle.load(f)
    print("info_dict:",info_dict['data'].keys())

ids = info_dict['data']['ids'][:,0]
v_encode = info_dict['data']['v_encode'][:,0]
v_err = info_dict['data']['rv_err'][:,0]
print("v_encode:",v_encode.shape)

datadir = "/scratch/gpfs/yanliang/neid-production"
#dataset_tag = "prep_order50_before"
param_names = ["K [m/s]  ","Period [d]",  "Phase    ", "Sigma    "]

power_orders = []
all_periods = []
all_periods_power = []
all_times = []
all_v = []
all_verr = []
all_data = []

dataset_tag = f"{tag}_full"
print("dataset_tag:",dataset_tag)

files = ["%s/ccf_info/%s"%(datadir,ii) for ii in os.listdir(f"{datadir}/ccf_info") if ii.startswith(dataset_tag) and bool(re.search(r'_\d+.pkl$', ii))]

ccf_info = []
time = []
for file in files:
    batch = load_batch(file)
    ccf_info.append(batch[0])
    time.append(batch[4])
print("ccf_info:",len(ccf_info))
ccf_info = torch.cat(ccf_info,dim=0)
time = torch.cat(time,dim=0)


time = tensor2array(time[:,0])
v_trad = np.copy(tensor2array(ccf_info[:,0]))


#v_trad = np.copy(v_ccf)

# remove outliers
tgrid = np.linspace(time.min(),time.max(),50)
good = (time<800)|(time>920)
print("good:",good.sum())
#'''
for i in range(len(tgrid)-1):
    mask = (time>tgrid[i])&(time<=tgrid[i+1])
    if mask.sum()<10:continue
    v_ref, v_std = np.median(v_trad[mask]),v_trad[mask].std()
    outlier = np.abs(v_trad-v_ref)>3*v_std
    good[mask&outlier] = False

print("bad:",(~good).sum())
print(f"v_trad: {v_trad.std():.2f} m/s v_trad[good]: { v_trad[good].std():.2f} m/s")

'''
ph,v_planet = simulate_planet_np(time,*truth)
#basename="period3.162d_K0.1m_phase0.0"
aux_data = np.vstack((ids,v_encode,good)).T
print("aux_data:",aux_data.shape)
sort_ind = np.argsort(aux_data[:,0])
aux_data = aux_data[sort_ind]
aux_data = torch.from_numpy(aux_data.astype(np.float32)).T
print("aux_data:",aux_data.shape)
filepath = f"{datadir}/aux/{basename}_v_encode.pkl"
torch.save(aux_data,filepath)
exit()
'''
#v_encode_tensor = torch.from_numpy(v_encode.astype(np.float32))

#'''

time = time[good]
v_trad = v_trad[good]
v_encode = v_encode[good]
v_err = v_err[good]
v_template,v_ccf,depth,bisspan,sigma,c = tensor2array(ccf_info[good]).T

print(f"v_ccf RMS={(v_ccf).std():.2f}m/s")
print(f"v_template RMS={(v_template).std():.2f}m/s")
print(f"Resid RMS={(v_ccf - v_template).std():.2f}m/s")

ph,v_planet = simulate_planet_np(time,*truth)
# add true planet signal to v_ccf
v_trad += v_planet

# remove offset
select = [time<800,time>800]
offset = []
for sel in select:
    print("v_offset:",np.median(v_encode[sel]))
    v_trad[sel] -= np.median(v_trad[sel])
    v_encode[sel] -= np.median(v_encode[sel])


print(f"v_trad RMS={(v_trad).std():.2f}m/s")
print(f"v_encode RMS={(v_encode).std():.2f}m/s")
print(f"Difference RMS={(v_encode-v_trad).std():.2f}m/s")

print(f"min:{v_trad.min():.2f} m/s, max:{v_trad.max():.2f} m/s")
input_features = np.vstack((depth,bisspan,sigma,c)).T
#input_data = [[time,v_trad,v_err,input_features]]

input_data = []

#v_input = v_trad
v_input = v_encode

#v_err = np.ones_like(v_input)*v_input.std()
full_data = [time,v_input,v_err,input_features]

for mask in select:input_data.append([item[mask] for item in full_data])

v_activity,_,_,rms=train_nn(input_data)
v_detrend = np.zeros_like(v_input)

for i,mask in enumerate(select):
    v_detrend[mask] = v_input[mask] - v_activity[i][:,0]

print(f"v_detrend RMS={(v_detrend).std():.2f}m/s")
power = LombScargle(time, v_detrend).power(frequency)
top_n_periods,top_n_power = detect_ls_peaks(per,power)

print("time:",time.shape) 
print("v_detrend:",v_detrend.shape)
# Example usage

print("top_n_periods:",top_n_periods)
#top_n_periods[0] = 3.162
#------------------ plotting ------------------
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
ax.semilogx(per,power,'-',alpha=1.0,lw=1.5,label=f"Power")
#ax.semilogx(per,power_avg,'k-',alpha=1.0,lw=1,label=f"Combined")
#ax.semilogx(period_grid,guess_params[:,0],'-',alpha=1.0,label=f"Guess")
ax.axvline(planet_period,lw=8,color="gold",alpha=0.5,zorder=-20,label="Truth")
ax.set_xlabel("Period [days]");ax.set_ylabel("Power")
for period in top_n_periods: ax.axvline(period,lw=1,color="lightgrey",ls="-",zorder=-10)
ax.legend()
plt.savefig(f"{outdir}/guess_{basename}.png",dpi=200)
plt.clf()
 
top_n_params = np.zeros((len(top_n_periods),3))
top_n_Ks = np.zeros_like(top_n_periods)
top_n_chi = np.zeros_like(top_n_periods)
top_n_v_doppler = np.zeros((len(top_n_periods),len(time)))

for i,try_period in enumerate(top_n_periods):
    best_guess,chi_guess = initial_guess(try_period,time,v_detrend,v_err)
    top_n_params[i] = best_guess
    print("Period:",try_period,"best_guess:",best_guess)

    v_act_nn,v_doppler,pfit,rms = train_nn(input_data,best_guess)
    v_doppler = v_doppler[:,0]
    
    pfit[0] = np.abs(pfit[0])
    chi = np.mean(rms)
    print(f"Period:{pfit[1]:.3f}d chi: {chi:.4f}")
    top_n_Ks[i] = pfit[0]
    top_n_params[i] = pfit
    top_n_chi[i] = chi
    #top_n_v_act[i] = v_act_nn
    #top_n_v_doppler[i] = v_doppler
    print_params(pfit,param_names)

planet_rank = np.argsort(top_n_Ks)[::-1]
top_n_params = top_n_params[planet_rank]
np.savetxt(output_txt,top_n_params)
print("top_n_params:",top_n_params)

exit()
doppler_sum = top_n_v_doppler.sum(axis=0)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sort_t = np.linspace(time.min(),time.max(),1000)#np.argsort(time)

planet_rank = np.argsort(top_n_Ks)[::-1]
identified_planets = top_n_params[planet_rank]

fig,axs=plt.subplots(nrows=2,ncols=4,figsize=(15,8),
                     constrained_layout=True,gridspec_kw = {'height_ratios':[2,1]})
gs = axs[0,0].get_gridspec()
ax_sub = [ii for ii in axs[1,:]]
for ax in axs[0,:]:ax.remove()
ax_time = fig.add_subplot(gs[0,:])
ax_time.plot(time,v_detrend,"k.",ms=1,label=f"Residual RMS:{(v_detrend-doppler_sum).std():.2f} m/s")
ax_time.set_ylim(-3,3)
ax=ax_sub[0]
power_i = LombScargle(time, v_detrend).power(frequency)
ax.semilogx(per,power_i,'k-',alpha=1.0,lw=2,label="$v_{detrend}$")
ax.axvline(planet_period,lw=8,color="gold",alpha=0.5,zorder=-20,label="Truth")
ax.set_xlabel("Period [days]");ax.set_ylabel("Power")

v_bg = np.zeros_like(sort_t)
# plot all phasefolded planets
for i,r in enumerate(planet_rank):
    ax=ax_sub[i+1]
    planet_show = identified_planets[i]
    if np.abs(planet_show[1]-planet_period)/planet_period<0.1:
        print("True signal!")
        title = f"Truth: P={planet_show[1]:.3f}d"
        planet_show = [planet_amp,planet_period,t0]
        model_c = 'r';lw=3
        
    else: 
        title = f"P={planet_show[1]:.3f}d K={planet_show[0]:.2f} m/s"
        model_c = colors[i]; lw=1
        ph,v_show = simulate_planet_np(sort_t,*planet_show)
        v_bg += v_show
        ax_sub[0].axvline(planet_show[1],c=colors[i],ls="--")

    ph,v_show = simulate_planet_np(sort_t,*planet_show)
    ax_time.plot(sort_t,v_show,c=model_c,label=title)
    v_signal = v_detrend.copy()
    v_signal -= v_signal.mean() 

    t_loc = ((time/planet_show[1])-planet_show[2])%1
    sort = np.argsort(ph)
    ax.plot(t_loc,v_signal,".",color="lightgrey",label=title)
    ax.plot(ph[sort],v_show[sort],model_c,lw=lw)
    ax.legend(loc="lower left")
    xgrid,y,delta_y = moving_mean(t_loc,v_signal,n=15)
    ax.errorbar(xgrid,y,yerr=delta_y,fmt=".",color="k",capsize=5,ms=10)
    ylim = max(2*planet_amp,1.3*planet_show[0])
    ax.set_ylim(-ylim,ylim)
    ax.set_xlabel("Phase")
    if i==len(ax_sub)-2:break

#ax_time.plot(sort_t,v_bg,"-",color="r",lw=1)
ax_time.legend()
ax_sub[0].legend()
ax_time.set_ylabel("$v_{ccf} [m/s]$")
ax_time.set_xlabel("Time [days]")

title = f"Truth: K={planet_amp:.2f} m/s P={planet_period:.3f}d  phase={t0:.2f}"

best = top_n_params[planet_rank[0]]
title += f"\nRecovered: K={best[0]:.2f} m/s P={best[1]:.3f}d phase={best[2]:.2f}"

fig.suptitle(title,fontsize=20)
fig.tight_layout()
plt.savefig(f"{outdir}/trad_{basename}.png",dpi=300)
plt.clf()

