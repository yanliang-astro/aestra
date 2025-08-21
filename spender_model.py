import numpy as np
import torch
from torch import nn
from torchinterp1d import Interp1d
from torchcubicspline import natural_cubic_spline_coeffs
from periodic_spline_model import PeriodicSplineRV
import torch.nn.functional as F

# Define a 1D Gaussian kernel with standard deviation sigma
def gaussian_kernel_1d(sigma = 20,kernel_size=51, device=None):
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()  # normalize
    kernel = kernel.view(1, 1, -1)  # shape: (out_channels, in_channels, kernel_size)
    return kernel

def cubic_evaluate(coeffs, tnew):
    t = coeffs[0]
    a,b,c,d = [item.squeeze(-1) for item in coeffs[1:]]
    maxlen = b.size(-1) - 1
    index = torch.bucketize(tnew, t) - 1
    index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
    # will never access the last element of self._t; this is correct behaviour
    fractional_part = tnew - t[index]

    batch_size, spec_size = tnew.shape
    batch_ind = torch.arange(batch_size,device=tnew.device)
    batch_ind = batch_ind.repeat((spec_size,1)).T

    inner = c[batch_ind, index] + d[batch_ind, index] * fractional_part
    inner = b[batch_ind, index] + inner * fractional_part
    return a[batch_ind, index] + inner * fractional_part

def cubic_transform(xrest, yrest, wave_shifted):
    coeffs = natural_cubic_spline_coeffs(xrest, yrest.unsqueeze(-1))
    out = cubic_evaluate(coeffs, wave_shifted)
    return out

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

#### MLP with one input channel and multiple output channels####
class MultipleMLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 n_channel=1,
                 n_hidden=(16, 16, 16),
                 act=(nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()),
                 dropout=0):
        super(MultipleMLP, self).__init__()
        self.mlp = nn.ModuleList([MLP(n_in,n_out,n_hidden=n_hidden,act=act,dropout=dropout) for i in range(n_channel)])

    def forward(self, x):
        x = [mlp(x)[:,None,:] for mlp in self.mlp]
        x = torch.cat(x,dim=1)
        return x

#### MLP with multiple input and output channels####
class ParallelMLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 n_channel=1,
                 n_hidden=(16, 16, 16),
                 act=(nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()),
                 dropout=0,
                 bias=True):
        super(ParallelMLP, self).__init__()
        self.n_channel=n_channel
        self.mlp = nn.ModuleList([MLP(n_in,n_out,n_hidden=n_hidden,act=act,dropout=dropout,bias=bias) for i in range(n_channel)])

    def forward(self, x):
        x = [mlp(x[:,i,:])[:,None,:] for i,mlp in enumerate(self.mlp)]
        x = torch.cat(x,dim=1)
        return x

def simulate_planet(t,amp=1,period=0.11,phase_t0=0):
    if period==0: return None,torch.zeros_like(t)
    phase = ((t/period)-phase_t0)%1
    v_planet = amp*torch.sin(2*np.pi*phase)
    return phase,v_planet

#### MLP which infers activity RV from latent vectors ####
class ActivityEstimator(nn.Module):
    def __init__(self,
                 n_in,
                 n_out=1,
                 n_channel=1,
                 n_hidden=(2,),
                 act=(nn.PReLU(), nn.Identity()),
                 dropout=0):
        super(ActivityEstimator, self).__init__()

        self.mlp = MLP(n_in,n_out,n_hidden=n_hidden,act=act,dropout=dropout)

    def forward(self, x):
        x = self.mlp(x)
        return x


class SpeculatorActivation(nn.Module):
    """Activation function from the Speculator paper
    .. math:
        a(\mathbf{x}) = \left[\boldsymbol{\gamma} + (1+e^{-\boldsymbol\beta\odot\mathbf{x}})^{-1}(1-\boldsymbol{\gamma})\right]\odot\mathbf{x}
    Paper: Alsing et al., 2020, ApJS, 249, 5
    Parameters
    ----------
    n_parameter: int
        Number of parameters for the activation function to act on
    plus_one: bool
        Whether to add 1 to the output
    """

    def __init__(self, n_parameter, plus_one=False):
        super().__init__()
        self.plus_one = plus_one
        self.beta = nn.Parameter(torch.randn(n_parameter), requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(n_parameter), requires_grad=True)

    def forward(self, x):
        """Forward method
        Parameters
        ----------
        x: `torch.tensor`
        Returns
        -------
        x': `torch.tensor`, same shape as `x`
        """
        # eq 8 in Alsing+2020
        x = (self.gamma + (1 - self.gamma) * torch.sigmoid(self.beta * x)) * x
        if self.plus_one:
            return x + 1
        return x


class RVEstimator(nn.Module):
    def __init__(self,
                 input_shape,
                 sizes = [5,10],
                 n_hidden=(128, 64, 32),
                 act=(nn.PReLU(128),nn.PReLU(64),nn.PReLU(32), nn.PReLU()),
                 dropout=0):
        super(RVEstimator, self).__init__()

        if len(input_shape)==2:
            n_order,n_in = input_shape
        else:
            n_order=1
            n_in = input_shape[0]

        filters = [n_order,128,64]
        self.conv1,self.conv2 = self._conv_blocks(filters, sizes, dropout=dropout)
        self.n_feature = filters[-1] * ((n_in //sizes[0])//sizes[1])

        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s) for s in sizes[:2])
        print("self.n_feature:",self.n_feature)
        self.mlp = MLP(self.n_feature, 2, n_hidden=n_hidden, act=act, dropout=dropout)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=-1)

    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for i in range(1,len(filters)):
            f_in = filters[i-1]
            f = filters[i]
            s = sizes[i-1]
            p = s // 2
            conv = nn.Conv1d(in_channels=f_in,
                             out_channels=f,
                             kernel_size=s,
                             padding=p,
                            )
            norm = nn.InstanceNorm1d(f)
            act = nn.PReLU(num_parameters=f)
            drop = nn.Dropout(p=dropout)
            convs.append(nn.Sequential(conv, norm, act, drop))
        return tuple(convs)

    def forward(self, x):
        if x.ndim==2:x = x.unsqueeze(1)
        # compression
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.softmax(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x

class NullRVEstimator(nn.Module):
    def __init__(self):
        super(NullRVEstimator, self).__init__()

    def forward(self, x):
        return torch.zeros((x.shape[0],1),device=x.device)

#### Spectrum encoder    ####
#### based on Serra 2018 ####
#### with robust feature combination from Geisler 2020 ####
class SpectrumEncoder(nn.Module):
    def __init__(self,
                 instrument,
                 n_latent,
                 n_hidden=(128, 64, 32),
                 act=(nn.PReLU(128), nn.PReLU(64), nn.PReLU(32), nn.Identity()),
                 n_aux=0,
                 dropout=0):

        super(SpectrumEncoder, self).__init__()
        self.instrument = instrument
        self.n_latent = n_latent
        self.n_aux = n_aux
        if instrument.wave_obs.ndim==2:
            self.n_order = instrument.wave_obs.shape[0]
        else: self.n_order = 1

        #filters = [self.n_order, 128, 256, 512]
        filters = [self.n_order, 64, 128, 256]
        sizes = [5, 11, 21]
        self.conv1, self.conv2, self.conv3 = self._conv_blocks(filters, sizes, dropout=dropout)
        self.n_feature = filters[-1] // 2

        # pools and softmax work for spectra and weights
        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s, padding=s//2) for s in sizes[:2])
        self.softmax = nn.Softmax(dim=-1)

        # small MLP to go from CNN features to latents
        self.mlp = MLP(self.n_feature + n_aux, self.n_latent, n_hidden=n_hidden, act=act, dropout=dropout)

    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for i in range(1,len(filters)):
            f_in = filters[i-1]
            f = filters[i]
            s = sizes[i-1]
            p = s // 2
            conv = nn.Conv1d(in_channels=f_in,
                             out_channels=f,
                             kernel_size=s,
                             padding=p,
                            )
            norm = nn.InstanceNorm1d(f)
            act = nn.PReLU(num_parameters=f)
            drop = nn.Dropout(p=dropout)
            convs.append(nn.Sequential(conv, norm, act, drop))
        return tuple(convs)

    def _downsample(self, x):
        # compression
        if x.ndim==2:x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        C = x.shape[1] // 2
        # split half channels into attention value and key
        h, a = torch.split(x, [C, C], dim=1)
        return h, a

    def forward(self, x, aux=None):
        # run through CNNs
        h, a = self._downsample(x)
        # softmax attention
        a = self.softmax(a)
        # apply attention
        x = torch.sum(h * a, dim=2)
        # redshift depending feature combination to final latents
        if aux is not None and aux is not False:
            x = torch.cat((x, aux), dim=-1)
        x = self.mlp(x)
        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class StablePolynomialBackground(nn.Module):
    def __init__(self, n_points=7):
        super().__init__()
        coeffs = torch.empty(n_points).uniform_(-1, 1)  # small uniform coefficients
        coeffs -= coeffs.mean()
        # Use small initial values for stability
        self.raw_coeffs = nn.Parameter(coeffs)  # start from 0
        self.coeff_scale = 1e-7  # scale factor to keep coefficients small

    def spline(self, x, values):
        """
        Cubic Catmull-Rom spline interpolation
        x: shape [T], in [0, 1)
        values: shape [N]
        """
        N = values.shape[0]
        x_scaled = x * N  # [0, N)
        i = torch.floor(x_scaled).long() % N
        f = x_scaled - i.float()

        def get(idx):
            j = i + idx
            j = torch.clamp(j, 0, N - 1)
            return values[j]

        y0 = get(-1)
        y1 = get(0)
        y2 = get(1)
        y3 = get(2)

        # Catmull-Rom spline coefficients
        a = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3
        b = y0 - 2.5*y1 + 2*y2 - 0.5*y3
        c = -0.5*y0 + 0.5*y2
        d = y1

        return ((a * f + b) * f + c) * f + d

    def forward(self, x):
        # Normalize x to [-1, 1] along the last dimension
        x_norm = torch.linspace(0,1,x.shape[0]+1,device=x.device)
        # Coefficients (shared across batch): shape (D+1,)
        coeffs = self.coeff_scale * self.raw_coeffs
        result = self.spline(x_norm[:-1], coeffs)
        return result  # shape (B, N)


class TelluricModel(nn.Module):
    def __init__(self,
                 wave_rest,
                 spec_rest,
                 instrument,
                 n_latent=1,
                ):


        super(TelluricModel, self).__init__()
        n_orders,n_spec = instrument.wave_obs.shape
        n_sky = n_latent
        n_star = 3
        n_continuum = 5

        self.n_latent = n_sky+n_star+n_continuum+1
        self.n_sky = n_sky
        self.n_star = n_star
        self.instrument = instrument
        self.encoder = SpectrumEncoder(instrument, self.n_latent)
        self.sky_decoder = MLP(n_sky,n_spec,n_hidden=(),act=(nn.LeakyReLU(),))
        self.star_decoder = MLP(n_star,n_spec,n_hidden=(),act=(nn.Identity(),))
        self.continuum_decoder = MLP(n_continuum,n_spec,n_hidden=(),act=(nn.Identity(),))

        self.lines_act = nn.LeakyReLU()
        self.instrument_poly = StablePolynomialBackground()

        self.register_buffer('wave_rest', wave_rest)
        self.register_buffer('broad_kernel', gaussian_kernel_1d(sigma=25,kernel_size=201))

        lsf_kernel = gaussian_kernel_1d(sigma=3,kernel_size=41)
        self.lsf_kernel = nn.Parameter(lsf_kernel)
        #self.register_buffer('lsf_kernel', )
        # internal stellar model!!!
        self.spec_rest= torch.nn.Parameter(spec_rest.float())

        self.overall_rv_offset = torch.nn.Parameter(torch.rand(1).float()-0.5)
        self.telluric_rv_offset = torch.nn.Parameter(torch.rand(1).float()-0.5)

        # initialize weights to avoid large fluctuation
        for p in self.star_decoder.parameters():torch.nn.init.normal_(p,std=1e-3)
        for p in self.sky_decoder.parameters():torch.nn.init.normal_(p,std=1e-3)
        for p in self.continuum_decoder.parameters():torch.nn.init.normal_(p,std=1e-3)

    def evaluate_wavelength_polynomial(self,wave_raw,jd):
        wavelength_shift = torch.zeros_like(wave_raw)
        spline = self.instrument_poly(wave_raw.mean(dim=0).float())
        print("spline:",spline.shape)
        mask = jd.squeeze(1)<800
        wavelength_shift[mask] += spline
        return wavelength_shift

    def evaluate_telluric_rv_offset(self,jd):
        # threshold: jd=800
        extra_rv = torch.zeros_like(jd)
        # order of 10 meters per second
        extra_rv[jd<800] = self.telluric_rv_offset
        return extra_rv

    def convolve_kernel(self,x,kernel):
        padding = kernel.shape[-1]//2
        x = F.conv1d(x, kernel, padding=padding)
        return x

    # rectify solution through high-pass and low-pass filter
    def rectify(self, x):
        # high-pass filter
        y_act = x[:,[0],:]
        y_slow = self.convolve_kernel(y_act,self.broad_kernel)
        continuum = self.convolve_kernel(x[:,[1],:],self.broad_kernel)
        return y_act-y_slow,continuum
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        x = x.reshape((x.shape[0],self.decoder.n_channel,self.n_star))
        x = self.decoder(x)
        x = self.rectify(x)
        return x

    def forward(self, s, z, wave, skymask):
        if s is None: return 1.0
        x = self.decode(s)
        if skymask is not None: x[:,~skymask] = 0
        x = 1.0 - self.transform(x,z,wave)
        return x

    def _forward(self, s, z_sky, wave, skymask=None):
        aux = s[:, -1:]

        # Telluric lines
        lines = self.sky_decoder(s[:, :self.n_sky]).unsqueeze(1)
        if skymask is not None: lines[:,:,~skymask] = 0
        lsf = self.lines_act(self.lsf_kernel)
        lines = self.convolve_kernel(lines, lsf / lsf.sum())

        # Stellar component
        y_star = self.star_decoder(s[:, self.n_sky:self.n_sky+self.n_star]).unsqueeze(1)
        y_star = y_star - self.convolve_kernel(y_star, self.broad_kernel)

        # Continuum
        continuum = self.continuum_decoder(s[:, self.n_sky+self.n_star:-1]).unsqueeze(1)
        continuum = self.convolve_kernel(continuum, self.broad_kernel)

        # Construct spectrum
        x = (1 - lines) * (1 + continuum)
        x = self.transform(x, z_sky, wave)
        x *= (self.spec_rest + y_star).squeeze(1)
        x += 1e-3 * aux

        return lines,continuum,y_star.squeeze(1),x

    def transform(self, spectrum_restframe, z, wave):
        n_batch = spectrum_restframe.shape[0]
        #n_order,n_spec = wave.shape
        if wave.ndim==2:n_order,n_spec = wave.shape
        else:
            n_order = 1
            n_spec = wave.shape[0]
        if z.ndim==1:z = z.unsqueeze(1)
        xx = self.wave_rest.repeat(n_batch,1,1)
        spectrum = torch.zeros((n_batch,n_order,n_spec),device=wave.device)
        for i in range(n_order):
            wave_redshifted = - wave[i] * z[:,[i]] + wave[i]
            spectrum[:,i,:] = Interp1d()(xx[:,i,:], spectrum_restframe[:,i,:], wave_redshifted)
        return spectrum.squeeze(1)

#### Spectrum decoder ####
#### Simple MLP but with explicit redshift and instrument path ####
class SpectrumDecoder(MultipleMLP):
    def __init__(self,
                 wave_rest,
                 spec_rest,
                 weight_rest=None,
                 n_latent=5,
                 n_order=1,
                 n_hidden=(64, 256, 1024),
                 act=None,
                 dropout=0,
                 datatag="mockdata",
                ):
        print("wave_rest:",wave_rest.shape,wave_rest.dim())
        if wave_rest.dim() == 1:
            n_channel,n_spec = 1,wave_rest.shape[0]
        else: n_channel,n_spec = wave_rest.shape

        if act==None: 
            act = [nn.LeakyReLU() for i in range(len(n_hidden))]
            # Last layer should allow negative outputs
            act.append(nn.PReLU())
            #act.append(nn.Identity())

        super(SpectrumDecoder, self).__init__(
            n_latent,
            n_spec,
            n_channel=n_channel,
            n_hidden=n_hidden,
            act=act,
            dropout=dropout,
            )

        self.n_latent = n_latent
        self.lsf = None

        if spec_rest is None:
            self.spec_rest= torch.nn.Parameter(torch.randn(wave_rest.shape))
        else: self.spec_rest= torch.nn.Parameter(spec_rest.float())
        self.register_buffer('wave_rest', wave_rest)
        self.register_buffer('weight_rest', weight_rest)


    def decode(self, s):
        x = 1e-2*super().forward(s)
        if x.shape[1]==1:x = x.squeeze(1)
        return x

    def forward(self, s):
        return self.decode(s)

    def transform(self, spectrum_restframe, z, wave):
        if wave.ndim==1:
            n_batch = spectrum_restframe.shape[0]
            n_spec = wave.shape[0]
            xx = self.wave_rest.repeat(n_batch,1)
            wave_redshifted = - wave * z + wave
            spectrum = Interp1d()(xx, spectrum_restframe[:,0,:], wave_redshifted)
            spectrum = spectrum.unsqueeze(1)

        elif wave.ndim==2:
            n_batch = spectrum_restframe.shape[0]
            n_order,n_spec = wave.shape
            xx = self.wave_rest.repeat(n_batch,1,1)
            spectrum = torch.zeros((n_batch,n_order,n_spec),device=wave.device)
            for i in range(n_order):
                wave_redshifted = - wave[i] * z + wave[i]
                spectrum[:,i,:] = Interp1d()(xx[:,i,:], spectrum_restframe[:,i,:], wave_redshifted)

        elif wave.ndim==3:
            n_batch,n_order,n_spec = wave.shape
            xx = self.wave_rest.repeat(n_batch,1,1)
            spectrum = torch.ones_like(wave)
            for i in range(n_order):
                wave_redshifted = - wave[:,i,:] * z[:,[i]] + wave[:,i,:]
                #spectrum[:,i,:] = cubic_transform(xx[i], spectrum_restframe[:,i,:], wave_redshifted)
                spectrum[:,i,:] = Interp1d()(xx[:,i,:], spectrum_restframe[:,i,:], wave_redshifted)
        return spectrum

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Combine spectrum encoder and decoder
class BaseAutoencoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 rv_estimator,
                 telluric,
                 fringe,
                 normalize=False,
                 activity_estimator=None,
                 doppler_model=None,
                ):

        super(BaseAutoencoder, self).__init__()
        if encoder is not None:
            assert encoder.n_latent == decoder.n_latent
        self.encoder = encoder
        self.decoder = decoder
        self.rv_estimator = rv_estimator
        self.telluric = telluric
        self.fringe = fringe
        self.normalize = normalize
        self.activity_estimator = activity_estimator
        self.doppler_model = doppler_model
        if not decoder.mlp is None:
            for p in self.decoder.mlp.parameters():
                torch.nn.init.normal_(p,std=1e-3)


    def encode(self, x, aux=None):
        return self.encoder(x, aux=aux)

    def decode(self, x):
        return self.decoder(x)

    def estimate_rv(self,x):
        rv_estimates = self.rv_estimator(x) # v and Var[v]
        rv = rv_estimates[:,[0]]
        log_err_square = rv_estimates[:,[1]]
        rv_err = torch.exp(log_err_square)**0.5 # predict err
        return rv, rv_err

    def estimate_v_act(self,x):
        return self.activity_estimator(x)

    def estimate_doppler_rv(self,x):
        return self.doppler_model(x)

    def _forward(self, s_star, z, instrument=None, aux=None):
        if instrument is None:
            instrument = self.encoder.instrument

        if self.decoder.spec_rest == None: baseline = 1.0
        else: baseline = self.decoder.spec_rest
        if s_star is not None:
            spectrum_activity = self.decode(s_star)
            spectrum_restframe = baseline+spectrum_activity
        else:
            spectrum_restframe = baseline.repeat(z.shape[0],1,1)
            spectrum_activity = torch.zeros_like(spectrum_restframe)

        spectrum_observed = self.decoder.transform(spectrum_restframe, z, instrument.wave_obs)

        return spectrum_activity, spectrum_restframe, spectrum_observed

    def forward(self, s, z, instrument=None, aux=None):
        spectrum_activity, spectrum_restframe, spectrum_observed = self._forward(s, z, instrument=instrument, aux=aux)
        return spectrum_observed

    def loss(self, x, w, s, z, instrument=None, aux=None, individual=False):
        spectrum_observed = self.forward(x, w, s, z, instrument=instrument, aux=aux)
        return self._loss(x, w, spectrum_observed, individual=individual)

    def _loss(self, x, w, spectrum_observed, individual=False):
        # loss = total squared deviation in units of variance
        # if the model is identical to observed spectrum (up to the noise),
        # then loss per object = D (number of non-zero bins)

        # to make it to order unity for comparing losses, divide out L (number of bins)
        # instead of D, so that spectra with more valid bins have larger impact
        #if w.dim()==1:w=w.unsqueeze(1)
        loss_ind = w * (x - spectrum_observed).pow(2)

        #topk_indices = torch.topk(loss_ind,3, dim=2).indices
        # Create a mask of the same shape as x
        #mask = torch.ones_like(loss_ind, dtype=torch.bool)
        # Set the top 3 indices in each row to False in the mask
        #mask.scatter_(2, topk_indices, False)
        # Use the mask to set the top 3 elements to zero
        #loss_ind = loss_ind * mask

        loss_ind = torch.sum(loss_ind, dim=-1) / torch.sum(w>1,dim=-1)
        if individual:
            return loss_ind
        #D = loss_ind.shape[1]
        return torch.sum(loss_ind)# / D

    def _normalization(self, x, m, w=None):
        # apply constant factor c that minimizes (c*m - x)^2
        if w is None:
            w = 1
        mw = m*w
        c = (mw * x).sum(dim=-1) / (mw * m).sum(dim=-1)
        return c.unsqueeze(-1)

    @property
    def n_parameter(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def wave_obs(self):
        return self.encoder.instrument.wave_obs

    @property
    def wave_rest(self):
        return self.decoder.wave_rest

class SpectrumAutoencoder(BaseAutoencoder):
    def __init__(self,
                 instrument,
                 wave_rest,
                 spec_rest=None,
                 weight_rest=None,
                 rv_estimator=None,
                 planet_params=None,
                 skymask=None,
                 n_latent=10,
                 n_telluric=5,
                 n_aux=0,
                 n_hidden=(),#n_hidden=(64, 256, 1024),
                 act=None,
                 normalize=False,
                 skip_encoding=False
                ):

        encoder = SpectrumEncoder(instrument, n_latent, n_aux=n_aux)

        decoder = SpectrumDecoder(
            wave_rest,
            spec_rest,
            weight_rest=weight_rest,
            n_latent=n_latent,
            n_order=instrument.wave_obs.shape[0],
            n_hidden=n_hidden,
            act=act,
        )

        telluric = None#TelluricModel(wave_rest,instrument,n_decoder=n_telluric)
        fringe = None#FringeModel(instrument)

        activity_estimator = ActivityEstimator(n_latent)
        doppler_model = PeriodicSplineRV(planet_params)

        if rv_estimator==None:
            #rv_estimator = NullRVEstimator()
            rv_estimator = RVEstimator(instrument.wave_obs.shape,sizes = [20,80])

        if skip_encoding:
            encoder = None
            decoder.mlp = None
            #rv_estimator = NullRVEstimator()

        super(SpectrumAutoencoder, self).__init__(
            encoder,
            decoder,
            rv_estimator,
            telluric,
            fringe,
            activity_estimator=activity_estimator,
            doppler_model=doppler_model,
            normalize=normalize,
        )
