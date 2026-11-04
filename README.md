# AESTRA

AESTRA (**A**uto-**E**ncoding **ST**ellar **R**adial-velocity and **A**ctivity) is a self-supervised machine-learning framework for separating stellar activity from planetary Doppler signals. **AESTRA II** extends the framework to real high-resolution spectra, modeling stellar line-shape variability, telluric absorption, and continuum variations in NEID Sun-as-a-star observations.

> **Update — October 2026:** **AESTRA II has been accepted for publication in *The Astronomical Journal*.** Updated code and a runnable demo will be added to this repository soon.
>
> **Code status:** The current `main` branch contains an earlier implementation based on simulated spectra; it does **not yet** include the AESTRA II pipeline.

## How it works
<img width="8000" height="4500" alt="architecture_diagram" src="https://github.com/user-attachments/assets/0ef5152d-fc82-4c80-940a-2ab3f96c9bb7" />

*The AESTRA II workflow (Liang et al. 2026, Figure 2): spectral decomposition into stellar, telluric, and continuum components, followed by learned stellar-activity representations and radial-velocity modeling.*

## Results
<img width="3000" height="1200" alt="full_recovery_comparison" src="https://github.com/user-attachments/assets/c4fabdcf-8a43-4d60-942b-fc591c809079" />

Using **29,352 NEID Sun-as-a-star spectra**, we tested AESTRA II in **500 single-planet injection–recovery experiments**. With detection thresholds separately calibrated to yield **zero spurious selections in this test suite**, the AESTRA pipeline recovered **238/500 injected signals**, compared with **9/500** using traditional CCF-based activity-indicator detrending.

## Publications

- **AESTRA II:** Liang, Y., Winn, J. N., Melchior, P., Lu, S., & Tran, Q. (2026). *AESTRA II: Generative Spectral Modeling of the Sun as a Star for Precise Radial Velocities*. **Accepted in The Astronomical Journal.** [arXiv:2606.13574](https://arxiv.org/abs/2606.13574)
- **AESTRA I:** Liang, Y., Winn, J. N., & Melchior, P. (2024). *AESTRA: Deep Learning for Precise Radial Velocity Estimation in the Presence of Stellar Activity*. **The Astronomical Journal, 167, 23.** [DOI:10.3847/1538-3881/ad0e01](https://doi.org/10.3847/1538-3881/ad0e01)

## License

[MIT](LICENSE)
