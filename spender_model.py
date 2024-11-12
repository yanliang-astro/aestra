import numpy as np
import torch
from torch import nn
from torchinterp1d import Interp1d
from torchcubicspline import natural_cubic_spline_coeffs

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
        self.mlp = nn.ModuleList([MLP(n_in,n_out,n_hidden=n_hidden,act=act,dropout=dropout,bias=bias) for i in range(n_channel)])


    def forward(self, x):
        x = [mlp(x[:,i,:])[:,None,:] for i,mlp in enumerate(self.mlp)]
        x = torch.cat(x,dim=1)
        return x

#### MLP which infers activity RV from latent vectors ####
class ActivityEstimator(nn.Module):
    def __init__(self,
                 n_in,
                 n_hidden=(16, 16, 16),
                 act=(nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.Identity()),
                 dropout=0):
        super(ActivityEstimator, self).__init__()
        n_out = 1
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
                 act=(nn.PReLU(128),nn.PReLU(64),nn.PReLU(32), nn.Identity()),
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
        self.mlp = MLP(self.n_feature, 1, n_hidden=n_hidden, act=act, dropout=dropout)
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
        return torch.zeros((x.shape[0],1),device=x.device)
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
        #x = x.unsqueeze(1)
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

# define the order by order sinusoidal fringe model
class FringeModel(nn.Module):
    def __init__(self,
                 instrument,
                 n_latent=3,
                 n_knot=80,
                 n_sin=0,
                 fringe_length=1.9, # angstrom
                 fringe_scale=1e-2, # flux value
                 fringe_phase=torch.pi,
                ):

        super(FringeModel, self).__init__()
        if instrument.wave_obs.ndim==2:
            n_channel,n_spec = instrument.wave_obs.shape
        else:
            n_channel=1
            n_spec = instrument.wave_obs.shape[0]

        x = instrument.wave_obs
        x_min,x_max = x.min(),x.max()
        x_normalized = x - (x_max + x_min)/2

        x_fringe = torch.linspace(x_min,x_max,n_knot)
        self.n_latent = n_latent
        self.n_knot = n_knot
        self.n_sin = n_sin
        self.L = fringe_length
        self.scale = fringe_scale
        self.phi = fringe_phase
        self.register_buffer('x', x_normalized)
        self.register_buffer('x_fringe', x_fringe)
        self.encoder = SpectrumEncoder(instrument, n_latent)
        self.decoder = MultipleMLP(n_latent,n_knot+n_sin,
                                   act=(nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.Identity()),
                                   n_channel=n_channel)#,
                                   #n_hidden=(),act=(nn.Identity(),))

    def L_k(self, k, x, x_eval, y_knot):
        L_k = torch.ones((y_knot.size(0), x_eval.size(0)), device=x.device)
        for i in range(x.size(0)):
            if i != k:
                L_k *= (x_eval - x[i]) / (x[k] - x[i])
        return L_k

    def lagrange_polynomial(self, y_knot):
        n_order,n_spec = self.x.shape
        x_knot = torch.linspace(-1,1,self.n_knot,device=y_knot.device)
        x_eval = torch.linspace(-1,1,n_spec,device=y_knot.device)
        P_batch = torch.zeros((y_knot.size(0),n_order, n_spec), device=y_knot.device, dtype=torch.float32)
        for i in range(n_order):
            for k in range(self.n_knot):
                L_k = self.L_k(k, x_knot, x_eval, y_knot[:,i,:])
                P_batch[:,i,:] += y_knot[:,i, k].unsqueeze(1) * L_k
        return P_batch

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def polynomial(self,s):
        return self.lagrange_polynomial(s)

    def cubic_interpolation(self,y_knot,z):
        if self.x.ndim==2:
            n_order,n_spec = self.x.shape
        else:
            n_order = 1
            n_spec = self.x.shape[0]
        x_knot = torch.linspace(-1,1,self.n_knot,device=y_knot.device)
        x_eval = torch.linspace(-1,1,n_spec,device=y_knot.device)
        x_eval = x_eval.repeat(y_knot.size(0),1)
        spectrum = torch.zeros((y_knot.size(0),n_order, n_spec), device=y_knot.device, dtype=torch.float32)
        for i in range(n_order):
            x_shifted = - x_eval * z[:,[i]] + x_eval
            spectrum[:,i,:] = cubic_transform(x_knot, y_knot[:,i,:], x_shifted)
        return self.scale*spectrum

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


class TelluricModel(nn.Module):
    def __init__(self,
                 wave_rest,
                 instrument,
                 n_decoder=6,
                ):

        super(TelluricModel, self).__init__()
        if instrument.wave_obs.ndim==2:
            n_channel,n_spec = instrument.wave_obs.shape
        else:
            n_channel=1
            n_spec = instrument.wave_obs.shape[0]

        n_latent = n_decoder
        self.n_latent = n_latent
        self.n_decoder = n_decoder
        self.instrument = instrument
        self.encoder = SpectrumEncoder(instrument, n_latent)
        self.decoder = MultipleMLP(n_decoder,n_spec,
                                   n_channel=n_channel,
                                   n_hidden=(),
                                   act=(nn.Identity(),))
        self.lsf = None
        self.register_buffer('wave_rest', wave_rest)
        # initialize weights to avoid large fluctuation
        for p in self.decoder.parameters():torch.nn.init.normal_(p,std=1e-3)

    def rectify(self, x):
        return x

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        x = self.decoder(x)
        x = self.rectify(x)
        return x

    def forward(self, s, z, wave, skymask):
        if s is None: return 1.0
        x = self.decode(s)
        if skymask is not None: x[:,~skymask] = 0
        x = 1.0 - self.transform(x,z,wave)
        return x

    def _forward(self, s, z, wave):
        x_lines = self.decode(s)
        x = 1.0 - self.transform(x_lines,z,wave)
        return x_lines,x

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
        return spectrum

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

    def encode(self, x, aux=None):
        return self.encoder(x, aux=aux)

    def decode(self, x):
        return self.decoder(x)

    def estimate_rv(self,x):
        # estimate z
        return self.rv_estimator(x)

    def estimate_v_act(self,x):
        return self.activity_estimator(x)

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
        if w.dim()==1:w=w.unsqueeze(1)
        loss_ind = w * (x - spectrum_observed).pow(2)
        loss_ind = torch.sum(loss_ind, dim=-1) / torch.sum(w>1,dim=-1)
        if individual:
            return loss_ind
        D = loss_ind.shape[1]
        return torch.sum(loss_ind) / D

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
                 skymask=None,
                 n_latent=10,
                 n_telluric=5,
                 n_aux=0,
                 n_hidden=(64, 256, 1024),
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

        telluric = TelluricModel(wave_rest,instrument,n_decoder=n_telluric)
        fringe = FringeModel(instrument)

        activity_estimator = ActivityEstimator(n_latent)

        if rv_estimator==None:
            rv_estimator = RVEstimator(instrument.wave_obs.shape,sizes = [20,40])

        if skip_encoding:
            encoder = None
            decoder.mlp = None
            rv_estimator = NullRVEstimator()

        super(SpectrumAutoencoder, self).__init__(
            encoder,
            decoder,
            rv_estimator,
            telluric,
            fringe,
            activity_estimator=activity_estimator,
            normalize=normalize,
        )
