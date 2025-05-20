#!/usr/bin/env python
# coding: utf-8
import torch,os,sys,pickle,re
import numpy as np
import matplotlib.pyplot as plt
from util import load_batch,moving_mean,simulate_planet

from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KDTree
from scipy.optimize import minimize

from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
            self.planet_params = nn.Parameter(torch.tensor(initial_guess,device=device))
        else: self.planet_params = torch.zeros((3),device=device)

    def forward(self, x):
        x = self.mlp(x)
        return x

    def doppler_rv(self,t):
        _, v_doppler = simulate_planet(t,*self.planet_params)
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

def initial_guess(period,time,v_ccf):
    phase_grid = np.arange(0,1,0.05)
    chi_best = np.inf
    for i,ph in enumerate(phase_grid):
        p_guess = [v_ccf.std(),period,ph]
        chi = planet_param_chi(p_guess,time,v_ccf,v_err)
        if chi<chi_best:
            best_guess = p_guess
            chi_best = chi
    amp_grid = np.arange(0.1,2,0.1)
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

def train_nn(time,v_ccf,input_features,planet_guess=None, n_epochs = 200):
    features = np.copy(input_features)
    print(f"feature shape: {features.shape}")
    features -= np.mean(features,axis=0)
    features /= np.std(features,axis=0)
    features = torch.tensor(features,device=device).float()
    time_tensor = torch.tensor(time[:, None],device=device)

    input_tensors = (features,torch.tensor((v_ccf)[:, None],device=device), 
                     time_tensor)
    batch_size = min(len(v_ccf), 2000)
    train_dataset = TensorDataset(*input_tensors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    if planet_guess is None:doppler=False
    else:doppler=True

    v_activity_estimator = ActivityEstimator(features.shape[1], 1, planet_guess)

    v_activity_estimator.to(device)
    optimizer = torch.optim.Adam(v_activity_estimator.parameters(), lr=0.005)
    v_activity_estimator.train()

    sigma_v = 1
    for epoch in range(n_epochs):
        for batch in train_loader:
            latent_i,v_app_i,times_i = batch
            optimizer.zero_grad()
            v_act = v_activity_estimator(latent_i)

            if doppler:v_doppler = v_activity_estimator.doppler_rv(times_i)
            else: v_doppler = 0
            #loss_smooth = compute_velocity_gradient(latent_i, v_act).sum()
            loss = (v_app_i - v_act - v_doppler)**2 / sigma_v**2
            loss = loss.sum()# + loss_smooth
            loss.backward()
            optimizer.step()
            break
        if epoch % 50 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, Resid RMS:{(v_app_i - v_act - v_doppler).std():.2f} m/s')
        #if epoch % 500 == 0:
        #    print("Saving model to %s..." % filepath)
        #    torch.save(v_activity_estimator.state_dict(), filepath)
    print("Training complete.")
    v_activity_estimator.eval()
    v_act_nn = v_activity_estimator(features)
    if doppler:
        v_doppler = v_activity_estimator.doppler_rv(time_tensor)
        v_doppler = tensor2array(v_doppler[:, 0])
    else: v_doppler = np.zeros_like(v_ccf)
    v_act_nn = tensor2array(v_act_nn[:, 0])
    rms = (v_ccf - v_act_nn - v_doppler).std()
    print(f"Residual RMS:{rms:.3f}m/s")
    planet_bestfit = tensor2array(v_activity_estimator.planet_params)
    return v_act_nn,v_doppler,planet_bestfit,rms


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

def detect_ls_peaks(per,power,n_peaks = 15,frac=0.2,delta=2):
    rank = np.argsort(power)[::-1]
    top_n_periods = np.zeros((n_peaks))

    i = 0
    for rk in rank:
        per_max = per[rk]
        if per_max>360:continue
        nonzero = top_n_periods>0
        if nonzero.sum()>0:
            if (np.abs(top_n_periods[nonzero]-per_max)/per_max).min()<frac:continue
            if np.abs(top_n_periods[nonzero]-per_max).min()<delta:continue
        top_n_periods[i] = per_max;i+=1
        #harmoniscs.extend(calculate_harmonics(per_max))
        if (top_n_periods>0).sum()>=n_peaks:
            print("break!")
            break
    top_n_periods = top_n_periods[top_n_periods>0]
    return top_n_periods

def remove_close_periods(periods, threshold=0.05):
    if not periods:
        return []

    # Sort the periods in ascending order
    periods = sorted(periods)

    # Initialize the filtered list with the first period
    filtered_periods = [periods[0]]

    for p in periods[1:]:
        # Check if the new period is sufficiently different from all previously kept ones
        if all(abs(p - fp) > threshold * fp for fp in filtered_periods):
            filtered_periods.append(p)

    return filtered_periods

def prepare_ccf_information(summary_dict,CCF_file):
    wave_obs = summary_dict['info']["wave_obs"][0]
    template = summary_dict['info']["template"][0]
    print("planet_period:",planet_period)
    data = summary_dict['data']
    time = data['ids']
    for key in data:
        if type(data[key]) == dict:print(key, len(data[key]))
        else:print(key, data[key].shape)

    if data['spec_input'].ndim==3:
        spectra = data['spec_input'][:,0,:]+template
    else:spectra = data['spec_input']+template
    ccf_matrix, velocity_grid = compute_ccf(spectra, wave_obs)
    params,v_ccf,bisspan = compute_bisspan(velocity_grid,ccf_matrix)

    ccf_dict = {"velocity_grid":velocity_grid,
                   "time":time,"v_ccf":v_ccf,
                   "ccf_matrix":ccf_matrix,
                   "params":params,"bisspan":bisspan}
    with open(CCF_file,"wb") as f:
        pickle.dump(ccf_dict,f)
    print(f"Saving to {CCF_file}...")
    return

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

basename = sys.argv[1]
outdir = "initial_guess"
output_txt = f"{outdir}/{basename}_ccf.txt"
CCF_file = f"{outdir}/CCF/CCF_{basename}.pkl"
fallback_CCF_file = f"{outdir}/CCF/CCF_modelfree.pkl"


per = np.logspace(0.2, 2.6, 10000)
frequency = 1.0/per
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

stellar_pattern = r'period(\d+\.\d+)d_K(\d+\.\d+)m_phase(\d+\.\d+)_(\d+)_(\w+)'
debug_pattern = r'\w+_(\d+)_(\w+)\.pt'

if re.search(stellar_pattern, basename): 
    # Search for the pattern in the file path
    match = re.search(stellar_pattern, basename)
    # Extract the numbers as a list of integers
    planet_period = float(match.group(1))
    planet_amp = float(match.group(2)) # m/s
    t0 = float(match.group(3))
    order_value = int(match.group(4))
    suffix = match.group(5)

else:
    print("Pattern not found")
    match = re.search(debug_pattern, basename)
    # Extract the numbers as a list of integers
    planet_period = 80.1
    planet_amp = 0.0
    t0 = 0.0
    order_value = int(match.group(1))
    suffix = match.group(2)

truth = [planet_amp,planet_period,t0]
print("planet:",truth)

if os.path.isfile(sys.argv[1]):
    with open(sys.argv[1],"rb") as f:
        summary_dict=pickle.load(f)
    planet_period,planet_amp,t0 = summary_dict['info']['planet_params']
    basename = os.path.basename(sys.argv[1])
    basename = "_".join(basename.split("_")[:-2])
    output_txt = f"{outdir}/{basename}_ccf.txt"
    update_txt = f"{outdir}/{basename}_finetune.txt"
    v_planet = summary_dict['data']['v_planet'][:,0]

    imgname = f"{outdir}/harmonics_{basename}.png"
    time =  summary_dict['data']["ids"][:,0]
    star_latent = summary_dict["data"]["s"]
    v_encode =  summary_dict['data']['v_encode'][:,0]
    v_err =  summary_dict['data']['v_encode_err'][:,0]
    v_activity = summary_dict['data']['v_act'][:,0]
    
    planets = summary_dict['info']['planet_solution']
    non_zero = (planets[:,1]>per.min())&(planets[:,1]<per.max())
    planets = planets[non_zero]

    v_doppler_ind = summary_dict['data']['v_doppler'][:,non_zero]

    reject_tols = [0.03,0.01]
    v_doppler_sum = v_doppler_ind.sum(axis=-1)
 
    v_aestra = v_encode-v_activity
    v_aestra -= v_aestra.mean()
    power_aestra = LombScargle(time, v_aestra, v_err).power(frequency)
    noise_level = np.quantile(power_aestra,0.9)
    print("10% level:",noise_level)

    periods = planets[:,1]
    f_aestra = interp1d(per,power_aestra)
    print("periods,",periods,per)
    promising = f_aestra(periods)>noise_level

    print("planets:",planets[:4,1],f_aestra(planets[:4,1]))
    
    s_corr = np.corrcoef(v_activity,star_latent.T)[0][1:]
    max_corr = np.argmax(np.abs(s_corr))
    print("s_corr:",s_corr, max_corr)

    power_act_label = f"$s_{max_corr+1}$"#"$v_{act}$"
    power_act = LombScargle(time, star_latent[:,max_corr]).power(frequency)
    P_activity = detect_ls_peaks(per,power_act,n_peaks=3,frac=0.05,delta=0.5)

    for r in range(len(planets)):
        if planets[r][1]<=0:continue
        corr = np.corrcoef(v_doppler_ind[:,r].T,star_latent.T)[0][1:]
        print(f"Period {planets[r][1]:.3f}d corrcoef with s: {corr}")

    #P_activity = remove_close_periods(P_activity)
    print("P_activity:",P_activity)

    promising = f_aestra(periods)>noise_level
    
    i = 0
    comments,good = check_alignment_harmonics(f_aestra,P_activity,periods,tol=reject_tols[i])
    while (promising&good).sum()==0:
        i+=1
        print("All promising candiates excluded!!")
        comments,good = check_alignment_harmonics(f_aestra,P_activity,periods,tol=reject_tols[i])

    print("promising&good candidates",planets[promising&good])
    keep = np.where(promising&good)[0]
    print("keep:",keep,planets[keep])
    new_planets = np.zeros((len(keep),3))
    for i,index in enumerate(keep):
        p0 = planets[index]
        print("p0",p0)
        # Fit the sinusoidal model
        result = minimize(planet_param_chi,p0, args=(time, v_aestra,v_err,),method='Nelder-Mead')
        bestfit_params = result.x
        bestK,bestP,bestph = bestfit_params
        print(f"Recovered: {bestP:.3f}d  {bestK:.2f} m/s")
        if np.abs(bestP - new_planets[:,1]).min()/bestP<0.05:
            print("Duplicate periods!");continue
        new_planets[i] = bestfit_params
    new_planets = new_planets[new_planets[:,0]>0]
    print("Updated planet solution!",update_txt)
    np.savetxt(update_txt,new_planets)
    
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)

    scale = power_aestra.max()/power_act.max()
    ax.semilogx(per,scale*power_act,'-',c="darkgrey", label=power_act_label)

    #for i in range(star_latent.shape[1]):
    #    power_s = LombScargle(time, star_latent[:,i]).power(frequency)
    #    scale = power_aestra.max()/power_s.max()
    #    ax.semilogx(per,scale*power_s,'-',alpha=1.0, label=f"$s_{i+1}$")

    ax.semilogx(per,power_aestra,'k-',alpha=1.0, label="$v_{aestra,init}$")
    ax.axhspan(0,noise_level,color="lightgrey",lw=0,
               zorder=-20,label="noise")
    for p in periods[keep]:ax.axvline(p,color="darkgrey",lw=1,zorder=-10)

    for i,p in enumerate(comments.keys()):
        ax.axvline(p,c=colors[i],ls="--",label=f"{p:.2f}d")
        ax.text(1.05*p,power_aestra.max(),
                comments[p],verticalalignment='top',
                color=colors[i],rotation=90)
    ax.axvline(planet_period,lw=8,color="gold",alpha=0.5,zorder=-20,label="Truth")
    title = f"Truth: K={planet_amp:.2f} m/s P={planet_period:.3f}d  phase={t0:.2f}"
    ax.set_title(title)
    ax.set_xlabel("Period [days]");ax.set_ylabel("Power")
    ax.legend()
    plt.savefig(imgname,dpi=200)
    plt.clf()
    '''
    with open(output_txt,"r") as f:
        content = f.readlines()
    ampls = np.array([float(line.split()[0]) for line in content])
    periods = np.array([float(line.split()[1]) for line in content])
    
    promising = f_aestra(periods)>noise_level # 1% peaks
    good = periods<0
    i = 0
    comments,good = check_alignment_harmonics(harmonics,planets[:,1],tol=reject_tols[i])
    while (promising&good).sum()==0:
        i+=1
        print("All promising candiates excluded!!")
        reject_tol = reject_tols[i]
        comments,good = check_alignment_harmonics(harmonics,planets[:,1],tol=reject_tols[i])
    print("promising&good candidates",(promising&good).sum())

    with open(update_txt,"w") as f:
        for i,line in enumerate(content):
            if i>=len(good):break
            if good[i]:newline = line
            else: newline ="#"+line.strip()+"\n"
            f.writelines(newline)
    '''
    exit()

# calculate_ccf_file
print(CCF_file)
if not os.path.isfile(CCF_file):
    if not os.path.isfile(fallback_CCF_file):
        filename = "/scratch/gpfs/yanliang/neid-dynamic/runtime/modelfree_50_after_sum.pkl"
        with open(filename,"rb") as f:
            summary_dict=pickle.load(f)
        prepare_ccf_information(summary_dict,fallback_CCF_file)
    print("Fallback CCF file: ",fallback_CCF_file)
    CCF_file = fallback_CCF_file
    fallback = True
else: fallback = False

print(f"loading from {CCF_file}...")
with open(CCF_file,"rb") as f: ccf_dict = pickle.load(f)
time =  ccf_dict['time'][:,0]
v_ccf = ccf_dict['v_ccf']
v_err = np.ones_like(v_ccf)
params = ccf_dict['params']
bisspan = ccf_dict['bisspan']

ph,v_planet = simulate_planet_np(time,*truth)
# add true planet signal to v_ccf
if fallback: v_ccf += v_planet

for key in ccf_dict: print(key,ccf_dict[key].shape)

depth,_,sigma,c = params.T
#s1,s2,s3 =  summary_dict['data']['s'].T
input_features = np.vstack((depth,bisspan,sigma,c)).T
lowest_rms = np.inf

for i in range(3):
    v_act,_,_,rms=train_nn(time,v_ccf,input_features)
    if rms<lowest_rms:
        lowest_rms = rms
        v_activity = v_act

param_names = ["K [m/s]  ","Period [d]",  "Phase    ", "Sigma    "]

v_detrend = v_ccf-v_activity

print(f"v_detrend RMS={(v_detrend).std():.2f}m/s")

power = LombScargle(time, v_detrend).power(frequency)
print(v_detrend.shape,"time",time.shape)

per = 1/frequency

top_n_periods = detect_ls_peaks(per,power)

top_n_params = np.zeros((len(top_n_periods),3))
top_n_Ks = np.zeros_like(top_n_periods)
top_n_chi = np.zeros_like(top_n_periods)
top_n_v_act = np.zeros((len(top_n_periods),len(time)))
top_n_v_doppler = np.zeros((len(top_n_periods),len(time)))

print("top_n_periods:",top_n_periods)

#------------------ plotting ------------------
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
power_i = LombScargle(time, v_detrend).power(frequency)
ax.semilogx(per,power_i,'k-',alpha=1.0,lw=2,label="$v_{detrend}$")
ax.axvline(planet_period,lw=8,color="gold",alpha=0.5,zorder=-20,label="Truth")
ax.set_xlabel("Period [days]");ax.set_ylabel("Power")
for period in top_n_periods: ax.axvline(period,lw=2,color="r",ls="--")
ax.legend()
plt.savefig(f"{outdir}/guess_{basename}.png",dpi=200)
plt.clf()

v_exist = np.zeros_like(v_detrend)
for i,try_period in enumerate(top_n_periods):
    best_guess,chi_guess = initial_guess(try_period,time,v_detrend)
    top_n_params[i] = best_guess
    print("Period:",try_period,"best_guess:",best_guess)

    v_act_nn,v_doppler,pfit,rms = train_nn(time,v_ccf,input_features,best_guess)
    pfit[0] = np.abs(pfit[0])
    chi = np.mean((v_ccf-v_act_nn-v_doppler)**2)
    print(f"Period:{pfit[1]:.3f}d chi: {chi:.4f}")
    top_n_Ks[i] = pfit[0]
    top_n_params[i] = pfit
    top_n_chi[i] = chi
    top_n_v_act[i] = v_act_nn
    top_n_v_doppler[i] = v_doppler
    print_params(pfit,param_names)
    v_exist += v_doppler

np.savetxt(output_txt,top_n_params)
print("top_n_params:",top_n_params)

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

