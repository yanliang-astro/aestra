# AESTRA II

Code for **AESTRA II: Generative Spectral Modeling of the Sun as a Star for Precise
Radial Velocities** (Y. Liang, J. N. Winn, P. Melchior, S. Lu, and Q. H. Tran;
accepted for publication in AJ; [arXiv:2606.13574](https://arxiv.org/abs/2606.13574)).
The code of the first AESTRA paper (Liang, Winn & Melchior 2024) is preserved at
the tag [`paper-I`](https://github.com/yanliang-astro/aestra/tree/paper-I).

AESTRA (Æstra) is a generative spectrum model for extreme-precision radial
velocities. Applied to the NEID Sun-as-a-star observations, it decomposes every
echelle order into telluric absorption, continuum variability and stellar
line-shape variability, without external atmospheric or stellar templates. After
the telluric and continuum components are removed, the orders are merged, and a
low-dimensional representation of the spectra is learned. This representation
gives the activity-driven part of the apparent radial velocities, which is
fitted jointly with candidate planetary signals.

This repository contains the code of every step of the method, as separate
scripts. It is not packaged as a single end-to-end run: each step reads the
output of the previous one, and the expensive steps (one telluric model per
echelle order, the stellar encoder-decoder) are trained step by step. The
detection statistic and its calibration on the 500-planet injection suite of the
paper (Sec. 5.4–5.6) are not included; `rv_decomposition.py` runs the
injection and recovery of one planet with a given period, amplitude and phase.

## How it works
<img width="8000" height="4500" alt="architecture_diagram" src="https://github.com/user-attachments/assets/0ef5152d-fc82-4c80-940a-2ab3f96c9bb7" />

*The AESTRA II workflow (Liang et al. 2026, Figure 2): spectral decomposition into stellar, telluric, and continuum components, followed by learned stellar-activity representations and radial-velocity modeling.*

## Results
<img width="3000" height="1200" alt="full_recovery_comparison" src="https://github.com/user-attachments/assets/c4fabdcf-8a43-4d60-942b-fc591c809079" />

Using **29,352 NEID Sun-as-a-star spectra**, we tested AESTRA II in **500 single-planet injection–recovery experiments**. With detection thresholds separately calibrated to yield **zero spurious selections in this test suite**, the AESTRA pipeline recovered **238/500 injected signals**, compared with **9/500** using traditional CCF-based activity-indicator detrending.

## Tutorial

[`tutorial.ipynb`](tutorial.ipynb) walks through the steps on a small demo
data set: two echelle orders (5011–5094 Å) of 1,000 spectra, with their
telluric models and a stellar activity model. It runs steps 2–7 in about a
minute; the RV decomposition (step 8), which needs a longer RV time series than
the demo, is described but not run. The demo data (116 MB) are attached to the
AESTRA II release of this repository; download and unpack them in the repository
root before running the tutorial:

```
curl -L https://github.com/yanliang-astro/aestra/releases/download/paper-II/aestra2_demo.tar.gz | tar xz
```

## Installation

The code is tested with Python 3.12 and

| package    | version |
|------------|---------|
| torch      | 2.12    |
| accelerate | 1.13    |
| numpy      | 2.4     |
| scipy      | 1.18    |
| astropy    | 8.0     |
| matplotlib | 3.10    |
| psutil     | 7.2     |

```
pip install torch accelerate numpy scipy astropy matplotlib psutil
```

The training scripts use a GPU when available. Run all scripts from the
repository root.

## Data

* **NEID solar spectra.** The NEID Level-2 files of the solar feed are public in
  the [NEID Solar Archive](https://neid.ipac.caltech.edu/search_solar.php).
  Download the files listed in `data/NEID_HIGH_QUALITY_OBSNAME.txt` (the first
  column is the archive path of each file) into one directory, with the
  archive's web interface or its Python package `pyneid`. `prepare_orders.py`
  uses the listed files it finds there, and draws a random subset with
  `--n_sample`.
* **`data/NEID_HIGH_QUALITY_OBSNAME.txt`**: the observations that pass the quality
  cuts of Sec. 2 (archive path, JD and NEID CCF RV per line). Those with
  JD < 2460600 are the 72,449 high-quality spectra of the paper, of which about
  30,000 were selected at random for the analysis.
* **`orders.config`**: the wavelength grid of every echelle order (start,
  end, and a quadratic polynomial for the pixel step).
* **NEID FSR mask** (`neidMaster_FSR_Mask20210218_v002.fits`): the free spectral
  range of every order. It is not included in this repository; we obtained it
  from the NEID team.

Most scripts find the prepared data in `--dir`, or in the directory given by the
environment variable `DATA_DIR`.

## Steps

Default names follow the paper's data set: per-order files
`newprod_order{N}_full*` and merged files `newprod_merged_full*`; the prefix
and suffix can be changed with `--data`/`--tag` and `--when`.

### 1. Per-order spectra — `prepare_orders.py` (Sec. 3.1–3.2, App. A.1)

Selects the observations (quality-cut list, JD < 2460600, files present,
shuffled with a seed), reads every NEID L2 file once, and writes the spectra of
each order: free-spectral-range pixels, flux divided by the blaze and its median,
inverse-variance weights, single-pixel spikes masked. Spectra with more than 100
negative pixels, or with a CCF RV more than 50 m/s from the mean of their batch,
are rejected. The wavelengths of each order are corrected for one bulk RV offset
before and one after the 2022 shutdown (the median RV against a first template),
and the template of the order is the median of 1000 spectra on its grid.

```
./prepare_orders.py --l2 /path/to/L2 --fsr_mask /path/to/neidMaster_FSR_Mask20210218_v002.fits \
    --dir data/ --n_sample 30000
```

Output in `--dir`, for every order N: batch files `newprod_order{N}_full_{k}.pt`,
`newprod_order{N}_full-template.pt` and `newprod_order{N}_full-param.pt`.
Each worker process needs about 2 GB of memory (`--processes`).

### 2. Telluric model — `tellurics_training.py` (Sec. 3.3, App. A.2)

Trains the decomposition of one order into telluric lines, continuum and stellar
variability, in two stages: a warm-up with the consistency loss, then the full
fit. Orders that end bluer than 5015 Å are trained without telluric lines.

```
./tellurics_training.py --order 51 --dir data/ --batch_size 1000 --batch_number 1 --rate 2e-2 \
    --iteration 1000 --consistency --shuffle -v newprod_order skymodel/sky_order51_full.pt
./tellurics_training.py --order 51 --dir data/ --batch_size 3000 --batch_number 1 --rate 1e-3 \
    --iteration 3000 --shuffle --clobber -v newprod_order skymodel/sky_order51_full.pt
```

`skytrain.sh` runs both stages and step 3 for one order, or for all 42 orders as
a SLURM job array (`DATA_DIR=/path/to/data bash skytrain.sh 51`).

### 3. Sky masks — `calculate_skymask.py` (Sec. 3.4)

Evaluates a trained telluric model on all spectra of its order, correlates the
telluric absorption with a telluric proxy (NEID water vapor, or the most
variable pixel), and masks the wavelengths around the most correlated lines.
Orders that end bluer than 5015 Å get an all-False mask.

```
./calculate_skymask.py skymodel/sky_order51_full.pt --dir data/ --plot
```

Output: `skymask/newprod_order51_full-skymask.pt` (boolean, True = telluric
pixel) and, with `--plot`, a diagnostic figure.

### 4. Cleaning and merging — `merge_orders.py` (Sec. 3.4)

Divides each order's telluric lines (inside its sky mask) and continuum out of
every spectrum, rejects outlying pixels, and joins the orders into one stellar
spectrum per observation; the template is the median of all merged spectra.

```
./merge_orders.py --dir data/      # orders 31-74 without 69 and 70
```

Output in `data/merged/`: `newprod_merged_full_{k}.pt`,
`newprod_merged_full-template.pt` and `newprod_merged_full-param.pt`.

### 5. Apparent RVs — `measure_apparent_rv.py` (Sec. 3.5)

Fits every merged spectrum with the Doppler-shifted template (cubic spline,
weighted χ²) and adds the apparent RV `v_app` and `chi2_app` of every
observation to `newprod_merged_full-param.pt`.

```
./measure_apparent_rv.py --dir data/
```

### 6. Stellar activity representation — `stellar_training.py`, `export_latents.py` (Sec. 4.1)

Trains the encoder-decoder on the merged spectra shifted to zero apparent RV;
the defaults are the settings of the paper (20 latents, 10,000 epochs of one
batch of 250 spectra). `export_latents.py` then encodes every spectrum.

```
./stellar_training.py --dir data/ -v stellar_model/stellar_s20.pt
./export_latents.py --dir data/ stellar_model/stellar_s20.pt     # -> stellar_model/stellar_s20-latents.pt
```

### 7. CCF activity indicators — `measure_ccf_indicators.py` (App. A.3)

Cross-correlates every merged spectrum with the template line profile and adds
the CCF RV, depth, FWHM, bisector span and offset to the merged param file.
They are used to detrend the RVs in step 8.

```
./measure_ccf_indicators.py --dir data/
```

### 8. Injection and recovery — `search_candidates.py`, `rv_decomposition.py` (Sec. 4.2, 5.1–5.3, App. A.5)

`rv_decomposition.py` injects a circular-orbit planet into the apparent RVs,

  v_meas = v_app + K sin(2π((t − t0)/P + φ)),  t = JD − 2459000, t0 = 800,

finds candidate periods with the traditional RV-level search of
`search_candidates.py` (CCF-indicator detrending, Lomb-Scargle periodograms in
several windows, sinusoid fits), and fits the activity RV estimator jointly with
the candidate amplitudes in four period blocks (P < 10 d, 10–100 d, 100–250 d,
> 250 d). The default example injects K = 0.5 m/s at P = 40 d:

```
./rv_decomposition.py --dir data/ -v stellar_model/stellar_s20-latents.pt recovery.pt
./rv_decomposition.py --dir data/ --amp 0.3 --period 12.5 --phase 0.25 stellar_model/stellar_s20-latents.pt recovery.pt
./search_candidates.py --dir data/ candidates.txt     # the candidate list alone
```

A candidate list can also be given with `--candidates` (one "K [m/s]  P [d]
phase" per line). The activity RV estimator is a network with 3,731 parameters
(`--hidden`, default 128 and 8 as in the paper), so the decomposition needs many
more observations than that, like the ~29,000 spectra of the paper; on a few
thousand observations it overfits.

The output holds the injection, the RVs, the candidates and,
for every period block, the fitted candidates (K, P, phase), the activity RVs
`v_act`, the planet RVs, `v_trend`, the model and its loss history.

## Modules

| file      | contents |
|-----------|----------|
| `data.py` | `SpectralData` (wavelength grid, batch-file loader), `load_batch` |
| `models.py` | `TelluricModel`, `SpectrumAutoencoder`, `ActivityEstimator`, `PlanetRV` |
| `util.py` | linear interpolation, resampling of spectra onto a model grid |

## Conventions and file formats

* **Orders** are indices 0–121 of the NEID arrays (index N is DRP order 173 − N).
  The paper uses indices 31–74 without 69 and 70 (DRP orders 142–99 without 104
  and 103).
* **Time** is t = JD − 2459000 (also the spectrum id, as float32); planet
  phases are in cycles at t0 = 800. RVs are in m/s, wavelengths in Å.
* **Batch files** (`*_{k}.pt`) are lists of per-spectrum tuples (wavelengths,
  flux, weight, barycentric correction SSBRV, id). Merged batch files have 0 in
  place of the wavelengths, which are those of the template.
* **Template files** (`*-template.pt`) are `[wave, flux, weight, 0, 888]`, each
  of shape (1, n_pix).
* **Param files** (`*-param.pt`) map every observation to its metadata, plus an
  `"info"` entry with the sample, the batch files and the template; load them with
  `torch.load(..., weights_only=False)`.

## Publications

- **AESTRA II:** Liang, Y., Winn, J. N., Melchior, P., Lu, S., & Tran, Q. (2026). *AESTRA II: Generative Spectral Modeling of the Sun as a Star for Precise Radial Velocities*. **Accepted in The Astronomical Journal.** [arXiv:2606.13574](https://arxiv.org/abs/2606.13574)
- **AESTRA I:** Liang, Y., Winn, J. N., & Melchior, P. (2024). *AESTRA: Deep Learning for Precise Radial Velocity Estimation in the Presence of Stellar Activity*. **The Astronomical Journal, 167, 23.** [DOI:10.3847/1538-3881/ad0e01](https://doi.org/10.3847/1538-3881/ad0e01)

## Citation

If you use this code, please cite the AESTRA II paper and the original AESTRA
paper above. The encoder and decoder follow spender (P. Melchior et al. 2023,
https://github.com/pmelchior/spender).
