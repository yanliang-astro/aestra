"""Neural network models of AESTRA.

TelluricModel (tellurics_training.py) splits the spectra of one echelle order
into telluric lines, stellar variability and continuum. SpectrumAutoencoder
(stellar_training.py) learns the latent activity state of the merged spectra.
ActivityEstimator and PlanetRV are the two parts of the RV decomposition
(rv_decomposition.py). The encoder and decoder follow spender
(https://github.com/pmelchior/spender, P. Melchior et al. 2023).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import SpectralData
from util import T0, interp1d

TELLURIC_MIN_WAVE = 5015.0   # [A] no telluric lines are modeled in orders that end bluer than this


def gaussian_kernel_1d(sigma=20, kernel_size=51):
    """Normalized Gaussian convolution kernel, shape (1, 1, kernel_size)."""
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return (kernel / kernel.sum()).view(1, 1, -1)


def has_tellurics(wave):
    """Whether telluric lines are modeled in an order with wavelengths wave [A].

    Empirically, there are no telluric lines bluer than TELLURIC_MIN_WAVE: orders
    that end there are trained with all telluric lines masked, and their sky
    masks are all False.
    """
    return float(wave.max()) >= TELLURIC_MIN_WAVE


class MLP(nn.Module):
    """Fully connected network; act holds one activation per layer (default: LeakyReLU)."""

    def __init__(self, n_in, n_out, n_hidden=(16, 16, 16), act=None, dropout=0):
        super().__init__()
        n_ = [n_in, *n_hidden, n_out]
        if act is None:
            act = [nn.LeakyReLU() for _ in range(len(n_) - 1)]
        layer = []
        for i in range(len(n_) - 1):
            layer.append(nn.Linear(n_[i], n_[i + 1]))
            layer.append(act[i])
            layer.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layer)

    def forward(self, x):
        return self.mlp(x)


class MultipleMLP(nn.Module):
    """One MLP per output channel, all with the same input; output shape (B, n_channel, n_out)."""

    def __init__(self, n_in, n_out, n_channel=1, n_hidden=(16, 16, 16), act=None, dropout=0):
        super().__init__()
        self.mlp = nn.ModuleList([MLP(n_in, n_out, n_hidden=n_hidden, act=act, dropout=dropout)
                                  for _ in range(n_channel)])

    def forward(self, x):
        return torch.cat([mlp(x)[:, None, :] for mlp in self.mlp], dim=1)


class ActivityEstimator(nn.Module):
    """Activity RV estimator: MLP from the stellar latents to an RV [m/s]."""

    def __init__(self, n_in, n_hidden=(128, 8), act=None, dropout=0):
        super().__init__()
        if act is None:   # new modules per instance: a default tuple would share PReLU weights
            act = (nn.PReLU(), nn.PReLU(), nn.Identity())
        self.mlp = MLP(n_in, 1, n_hidden=n_hidden, act=act, dropout=dropout)

    def forward(self, x):
        return self.mlp(x)


class SpectrumEncoder(nn.Module):
    """Spectrum encoder of spender: CNN after Serra et al. (2018) with attention pooling
    after Geisler (2020), then an MLP to n_latent latents."""

    def __init__(self, instrument, n_latent, n_hidden=(128, 64, 32), act=None, dropout=0):
        super().__init__()
        if act is None:   # new modules per instance: a default tuple would share PReLU weights
            act = (nn.PReLU(128), nn.PReLU(64), nn.PReLU(32), nn.Identity())
        self.instrument = instrument
        self.n_latent = n_latent
        self.n_order = instrument.wave_obs.shape[0] if instrument.wave_obs.ndim == 2 else 1

        filters = [self.n_order, 64, 128, 256]
        sizes = [5, 11, 21]
        self.conv1, self.conv2, self.conv3 = self._conv_blocks(filters, sizes, dropout=dropout)
        self.n_feature = filters[-1] // 2

        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s, padding=s // 2) for s in sizes[:2])
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = MLP(self.n_feature, self.n_latent, n_hidden=n_hidden, act=act, dropout=dropout)

    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for f_in, f, s in zip(filters[:-1], filters[1:], sizes):
            conv = nn.Conv1d(in_channels=f_in, out_channels=f, kernel_size=s, padding=s // 2)
            convs.append(nn.Sequential(conv, nn.InstanceNorm1d(f), nn.PReLU(num_parameters=f),
                                       nn.Dropout(p=dropout)))
        return tuple(convs)

    def _downsample(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        # half of the channels are the attention values, the other half the keys
        C = x.shape[1] // 2
        return torch.split(x, [C, C], dim=1)

    def forward(self, x):
        h, a = self._downsample(x)
        x = torch.sum(h * self.softmax(a), dim=2)
        return self.mlp(x)


class StablePolynomialBackground(nn.Module):
    """Smooth curve over the pixels of an order, for a small fractional wavelength correction.

    A cubic Catmull-Rom spline through n_points control values evenly spaced over
    the pixels; the control values are scaled by 1e-8 to keep the correction small.
    """

    def __init__(self, n_points=7):
        super().__init__()
        coeffs = torch.empty(n_points).uniform_(-1, 1)
        self.raw_coeffs = nn.Parameter(coeffs - coeffs.mean())
        self.coeff_scale = 1e-8

    def spline(self, x, values):
        """Catmull-Rom spline through `values` at x in [0, 1); end points are clamped."""
        N = values.shape[0]
        x_scaled = x * N
        i = torch.floor(x_scaled).long() % N
        f = x_scaled - i.float()

        def get(offset):
            return values[torch.clamp(i + offset, 0, N - 1)]

        y0, y1, y2, y3 = get(-1), get(0), get(1), get(2)
        a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
        b = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3
        c = -0.5 * y0 + 0.5 * y2
        return ((a * f + b) * f + c) * f + y1

    def forward(self, x):
        """Correction at every pixel of x (only its length is used), shape (len(x),)."""
        x_norm = torch.linspace(0, 1, x.shape[0] + 1, device=x.device)
        return self.spline(x_norm[:-1], self.coeff_scale * self.raw_coeffs)


class TelluricModel(nn.Module):
    """Telluric model of one echelle order (tellurics_training.py).

    The encoder maps a residual spectrum to n_latent telluric latents, 3 stellar
    and 5 continuum latents, plus one additive offset. The spectrum model is

        spectrum = transform((1 - lines) (1 + continuum)) * (spec_rest + y_star) + 1e-3 offset

    with the telluric lines convolved with a learned line-spread function, the
    continuum smoothed and the stellar variability y_star high-pass filtered
    (both with a Gaussian of 25 pixels). transform Doppler-shifts the telluric
    part by z_sky (the barycentric velocity), from the Earth frame into the
    frame of the spectra.
    """

    def __init__(self, wave_rest, spec_rest, instrument, n_latent=1):
        super().__init__()
        n_spec = instrument.wave_obs.shape[-1]
        n_sky, n_star, n_continuum = n_latent, 3, 5

        self.n_latent = n_sky + n_star + n_continuum + 1
        self.n_sky = n_sky
        self.n_star = n_star
        self.instrument = instrument
        self.encoder = SpectrumEncoder(instrument, self.n_latent)
        self.sky_decoder = MLP(n_sky, n_spec, n_hidden=(), act=(nn.LeakyReLU(),))
        self.star_decoder = MLP(n_star, n_spec, n_hidden=(), act=(nn.Identity(),))
        self.continuum_decoder = MLP(n_continuum, n_spec, n_hidden=(), act=(nn.Identity(),))

        self.lines_act = nn.LeakyReLU()   # keeps the line-spread function non-negative
        self.instrument_poly = StablePolynomialBackground()

        self.register_buffer('wave_rest', wave_rest)
        self.register_buffer('broad_kernel', gaussian_kernel_1d(sigma=25, kernel_size=201))
        self.lsf_kernel = nn.Parameter(gaussian_kernel_1d(sigma=3, kernel_size=41))
        self.spec_rest = nn.Parameter(spec_rest.float())   # stellar template

        # Not used by the model; kept so that existing checkpoints load.
        self.overall_rv_offset = nn.Parameter(torch.rand(1).float() - 0.5)
        self.telluric_rv_offset = nn.Parameter(torch.rand(1).float() - 0.5)

        # small initial weights avoid large fluctuations early in training
        for decoder in (self.star_decoder, self.sky_decoder, self.continuum_decoder):
            for p in decoder.parameters():
                nn.init.normal_(p, std=1e-3)

    def evaluate_wavelength_polynomial(self, wave_raw, jd):
        """Fractional wavelength correction of every spectrum; nonzero before the shutdown (jd < 800)."""
        wavelength_shift = torch.zeros_like(wave_raw)
        spline = self.instrument_poly(wave_raw.mean(dim=0).float())
        wavelength_shift[jd.squeeze(1) < 800] += spline
        return wavelength_shift

    def convolve_kernel(self, x, kernel):
        return F.conv1d(x, kernel, padding=kernel.shape[-1] // 2)

    def encode(self, x):
        return self.encoder(x)

    def _forward(self, s, z_sky, wave, skymask=None):
        """(lines, continuum, y_star, spectrum) for latents s and telluric redshifts z_sky.

        Telluric lines are set to zero where skymask is False.
        """
        aux = s[:, -1:]

        lines = self.sky_decoder(s[:, :self.n_sky]).unsqueeze(1)
        if skymask is not None:
            lines[:, :, ~skymask] = 0
        lsf = self.lines_act(self.lsf_kernel)
        lines = self.convolve_kernel(lines, lsf / lsf.sum())

        y_star = self.star_decoder(s[:, self.n_sky:self.n_sky + self.n_star]).unsqueeze(1)
        y_star = y_star - self.convolve_kernel(y_star, self.broad_kernel)

        continuum = self.continuum_decoder(s[:, self.n_sky + self.n_star:-1]).unsqueeze(1)
        continuum = self.convolve_kernel(continuum, self.broad_kernel)

        x = (1 - lines) * (1 + continuum)
        x = self.transform(x, z_sky, wave)
        x *= (self.spec_rest + y_star).squeeze(1)
        x += 1e-3 * aux
        return lines, continuum, y_star.squeeze(1), x

    def transform(self, spectrum_restframe, z, wave):
        """Evaluate spectrum_restframe (B, 1, n_pix), given on wave_rest, at wave * (1 - z).

        wave: grid of shape (1, n_pix); z: shape (B, 1). Returns shape (B, n_pix).
        """
        xx = self.wave_rest.repeat(len(spectrum_restframe), 1)
        return interp1d(xx, spectrum_restframe[:, 0, :], -wave[0] * z + wave[0])


class SpectrumDecoder(MultipleMLP):
    """Decoder from the latents to the stellar variability, 1e-2 * MLP(s); holds the template spec_rest.

    wave_rest and spec_rest have shape (1, n_pix).
    """

    def __init__(self, wave_rest, spec_rest, n_latent=5, n_hidden=(64, 256, 1024), act=None, dropout=0):
        n_channel, n_spec = wave_rest.shape
        if act is None:
            # last layer: PReLU, which allows negative outputs
            act = [nn.LeakyReLU() for _ in n_hidden] + [nn.PReLU()]
        super().__init__(n_latent, n_spec, n_channel=n_channel, n_hidden=n_hidden, act=act, dropout=dropout)

        self.n_latent = n_latent
        self.spec_rest = nn.Parameter(spec_rest.float())
        self.register_buffer('wave_rest', wave_rest)

    def decode(self, s):
        return 1e-2 * super().forward(s).squeeze(1)

    def forward(self, s):
        return self.decode(s)


class PlanetRV(nn.Module):
    """Planet RVs: a sum of circular-orbit sinusoids with fixed periods and phases.

        v_planets(t) = sum_k K_k sin(2 pi ((t - T0) / P_k + phase_k))

    planet_params holds rows (K [m/s], P [d], phase [cycles]). The leading rows,
    as many as there are positive K, become the planets; only their amplitudes K
    are trained.
    """

    def __init__(self, planet_params):
        super().__init__()
        planet_params = np.atleast_2d(np.asarray(planet_params, dtype=np.float64))[:, :3]
        n_fill = int(np.sum(planet_params[:, 0] > 0))
        params = torch.from_numpy(planet_params[:n_fill]).float()
        self.K = nn.Parameter(params[:, 0].clone())
        self.register_buffer("logperiods", torch.log(params[:, 1]))
        self.register_buffer("phases", params[:, 2].clone())
        self.register_buffer("mask", self.K.detach() > 0)

    def get_amplitudes(self):
        return self.K[self.mask]

    def get_periods(self):
        return torch.exp(self.logperiods[self.mask])

    def get_phases(self):
        return self.phases[self.mask]

    def forward(self, t):
        """RV of every planet at times t [JD - 2459000] of shape (B,) or (B, 1); shape (n_planet, B)."""
        t = t[:, 0] if t.ndim == 2 and t.shape[1] == 1 else t.reshape(-1)
        P, K, phase = self.get_periods(), self.get_amplitudes(), self.get_phases()
        angle = (2.0 * math.pi) * (((t - T0)[None, :] / P[:, None]) + phase[:, None])
        return K[:, None] * torch.sin(angle)


class SpectrumAutoencoder(nn.Module):
    """Stellar activity autoencoder (stellar_training.py).

    encoder: residual spectrum -> n_latent latents; decoder: latents -> stellar
    variability, a single layer by default (n_hidden=()); spec_rest is the
    template of the merged spectra.
    """

    def __init__(self, instrument, wave_rest, spec_rest, n_latent=10, n_hidden=(), act=None):
        super().__init__()
        self.encoder = SpectrumEncoder(instrument, n_latent)
        self.decoder = SpectrumDecoder(wave_rest, spec_rest, n_latent=n_latent, n_hidden=n_hidden, act=act)
        for p in self.decoder.mlp.parameters():
            nn.init.normal_(p, std=1e-3)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, s):
        return self.decoder(s)

    def forward(self, x):
        return self.decode(self.encode(x))

    def _loss(self, x, w, model, individual=False):
        """chi^2 of model per spectrum, divided by the number of pixels with weight > 1."""
        loss = torch.sum(w * (x - model).pow(2), dim=-1) / torch.sum(w > 1, dim=-1)
        return loss if individual else torch.sum(loss)


# ------------------------------------------------------- trained models --

def load_sky_model(filename, instrument, device):
    """Rebuild a TelluricModel from a tellurics_training.py checkpoint."""
    # weights_only=False: the checkpoint also holds the loss history (numpy array)
    state = torch.load(filename, map_location=device, weights_only=False)["model"][0]
    n_sky = state["sky_decoder.mlp.0.weight"].shape[1]
    model = TelluricModel(state["wave_rest"], state["spec_rest"], instrument, n_latent=n_sky)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def build_stellar_model(wave, template, n_latent):
    """Stage-1 SpectrumAutoencoder on the merged wavelength grid, with the template as fixed baseline."""
    instrument = SpectralData(wave)
    return SpectrumAutoencoder(instrument, wave, spec_rest=template.float().clone(), n_latent=n_latent)


# Checkpoints of earlier versions also hold untrained RV-decomposition modules.
OLD_MODULES = ("activity_estimator.", "doppler_model.")


def load_stellar_checkpoint(filename, model):
    """Load a stellar_training.py checkpoint into model; returns the loss history."""
    # weights_only=False: the checkpoint also holds the loss history (numpy array)
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    state = {k: v for k, v in checkpoint["model"].items() if not k.startswith(OLD_MODULES)}
    model.load_state_dict(state, strict=True)
    return checkpoint["losses"]


def load_stellar_model(filename, wave, template):
    """Rebuild a trained stage-1 model for the merged grid `wave`; returns (model, losses)."""
    state = torch.load(filename, map_location="cpu", weights_only=False)["model"]
    if not torch.equal(state["encoder.instrument.wave_obs"].cpu(), wave.cpu()):
        raise ValueError(f"{filename} was trained on a different wavelength grid")
    n_latent = state["decoder.mlp.0.mlp.0.weight"].shape[1]   # decoder: one layer from the latents
    model = build_stellar_model(wave, template, n_latent)
    return model, load_stellar_checkpoint(filename, model)
