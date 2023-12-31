# AESTRA Architecture
AESTRA (Auto-Encoding STellar Radial-velocity and Activity) is a deep learning method for precise radial velocity measurements in the presence of stellar activity noise. 
The architecture combines a convolutional radial-velocity estimator and a spectrum auto-encoder called [spender](https://github.com/pmelchior/spender). For an in-depth understanding of the spectrum auto-encoder, see [Melchior et al. 2023](https://iopscience.iop.org/article/10.3847/1538-3881/ace0ff) and [Liang et al. 2023](https://iopscience.iop.org/article/10.3847/1538-3881/ace100). 

**Liang, Y., Winn, J. N., & Melchior, P.** (2023). *AESTRA: Deep Learning for Precise Radial Velocity Estimation in the Presence of Stellar Activity*. Manuscript submitted.

The input consists of a collection of hundreds or more of spectra of a single star, which span a variety of activity states and orbital motion phases of any potential planets.

![AESTRA_Diagram_R1](https://github.com/yanliang-astro/aestra/assets/71669502/f6d6f40f-98a7-4d9a-bc90-8db084545d2a)

Training of the AESTRA architecture does not require a spectral template or line list, or indeed any prior knowledge about the star.
The spectrum auto-encoder is trained with a fidelity loss that ensures accurate reconstruction of the activity.
The RV estimator network is trained with an RV-consistency loss that seeks to recover the injected velocity offset from an artificially Doppler-shifted "augment" spectrum.

