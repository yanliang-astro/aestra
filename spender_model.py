import numpy as np
import torch
from torch import nn
from torchinterp1d import Interp1d
from util import cubic_transform

#### Simple MLP ####
class MLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden=(16, 16, 16),
                 act=(nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()),
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

#### MLP with multiple output channels####
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
        n_order,n_in = input_shape

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
        self.n_order = instrument.wave_obs.shape[0]


        filters = [self.n_order, 128, 256, 512]
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


#### Spectrum decoder ####
#### Simple MLP but with explicit redshift and instrument path ####
class SpectrumDecoder(MultipleMLP):
    def __init__(self,
                 wave_rest,
                 spec_rest,
                 n_latent=5,
                 n_hidden=(64, 256, 1024),
                 act=None,
                 dropout=0,
                 datatag="mockdata",
                ):

        n_channel,n_spec = wave_rest.shape
        print("wave_rest:",wave_rest.shape)

        if act==None: 
            act = [nn.LeakyReLU() for i in range(len(n_hidden)+1)]
            #act = [SpeculatorActivation(n) for n in n_hidden]
            #act.append(SpeculatorActivation(len(wave_rest)))

        super(SpectrumDecoder, self).__init__(
            n_latent,
            n_spec,
            n_channel=n_channel,
            n_hidden=n_hidden,
            act=act,
            dropout=dropout,
            )

        self.n_latent = n_latent
        #self.decode_act = nn.Identity()
        self.decode_act = nn.LeakyReLU()
        # register wavelength tensors on the same device as the entire model
        if spec_rest is None:
            self.spec_rest= torch.nn.Parameter(torch.randn(wave_rest.shape))
        else: self.spec_rest= torch.nn.Parameter(spec_rest)
        self.register_buffer('wave_rest', wave_rest)

    def decode(self, s):
        x = super().forward(s)
        x = -self.decode_act(-x)
        return x

    def forward(self, s):
        return self.decode(s)

    def transform(self, spectrum_restframe, z, instrument=None):
        xx = self.wave_rest

        if instrument in [False, None]:
            wave_obs = self.wave_rest
        else:
            wave_obs = instrument.wave_obs

        n_order, n_spec = wave_obs.shape
        batch_size = spectrum_restframe.shape[0]

        spectrum = torch.zeros((batch_size,n_order, n_spec), device=wave_obs.device)
        for i in range(n_order):
            wave_redshifted = - wave_obs[i] * z + wave_obs[i]
            spectrum[:,i,:] = cubic_transform(xx[i], spectrum_restframe[:,i,:], wave_redshifted)

        # convolve with LSF
        if instrument.lsf is not None:
            spectrum = instrument.lsf(spectrum.unsqueeze(1)).squeeze(1)

        # apply calibration function to observed spectrum
        if instrument is not None and instrument.calibration is not None:
            spectrum = instrument.calibration(wave_obs, spectrum)

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
                 normalize=False,
                ):

        super(BaseAutoencoder, self).__init__()
        assert encoder.n_latent == decoder.n_latent
        self.encoder = encoder
        self.decoder = decoder
        self.rv_estimator = rv_estimator
        self.normalize = normalize

    def encode(self, x, aux=None):
        return self.encoder(x, aux=aux)

    def decode(self, x):
        return self.decoder(x)

    def estimate_rv(self,x):
        # estimate z
        return self.rv_estimator(x)

    def _forward(self, x, w, s, z, instrument=None, aux=None):
        if w.dim()==1:w=w.unsqueeze(1)

        if instrument is None:
            instrument = self.encoder.instrument

        if self.decoder.spec_rest == None: baseline = 1.0
        else: baseline = self.decoder.spec_rest
        spectrum_activity = self.decode(s)
        spectrum_restframe = baseline+spectrum_activity
        spectrum_observed = self.decoder.transform(spectrum_restframe, z, instrument=instrument)

        if self.normalize:
            c = self._normalization(x, spectrum_observed, w=w)
            spectrum_observed = spectrum_observed * c
            spectrum_restframe = spectrum_restframe * c

        return spectrum_activity, spectrum_restframe, spectrum_observed

    def forward(self, x, w, s, z, instrument=None, aux=None):
        spectrum_activity, spectrum_restframe, spectrum_observed = self._forward(x, w, s, z, instrument=instrument, aux=aux)
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
        loss_ind = torch.sum(w * (x - spectrum_observed).pow(2), dim=-1) / x.shape[-1]
        loss_ind = torch.mean(loss_ind,dim=-1)

        if individual:
            return loss_ind

        return torch.sum(loss_ind)

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
                 rv_estimator=None,
                 n_latent=10,
                 n_aux=0,
                 n_hidden=(64, 256, 1024),
                 act=None,
                 normalize=False,
                ):

        encoder = SpectrumEncoder(instrument, n_latent, n_aux=n_aux)

        decoder = SpectrumDecoder(
            wave_rest,
            spec_rest,
            n_latent,
            n_hidden=n_hidden,
            act=act,
        )

        if rv_estimator==None:
            rv_estimator = RVEstimator(instrument.wave_obs.shape,sizes = [20,40])
            #rv_estimator = NullRVEstimator()

        super(SpectrumAutoencoder, self).__init__(
            encoder,
            decoder,
            rv_estimator,
            normalize=normalize,
        )
