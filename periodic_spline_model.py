import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodicSplineRV(nn.Module):
    def __init__(self,
                 planet_params,
                 n_amp_basis=10,
                 n_phi_basis=5,
                 n_trend_basis=10,
                 trend_timescale_days=200,
                 n_planet=12,
                 tmin=200,
                 tmax=1500,
                 share_lengthscales=False):
        super(PeriodicSplineRV, self).__init__()

        # Initialize additional trainable parameters
        ampls = planet_params[:,0]
        
        n_fill = min(n_planet,len(ampls[ampls>0]))
        init = torch.zeros((n_planet,3))
        init[:n_fill] = torch.from_numpy(planet_params[:n_fill,:3])

        sel = [0,2]
        #planet_params = self.initialize_planets(init[:n_fill])
        planet_params = init[:n_fill,sel]

        self.planet_params = nn.Parameter(planet_params)
        #self.register_buffer('planet_params',planet_params)
        #self.register_buffer('periods',init[:n_fill,1])
        #self.periods = nn.Parameter(init[:n_fill,1])
        #self.logperiods = nn.Parameter(torch.log(init[:n_fill,1]))
        self.register_buffer('logperiods',torch.log(init[:n_fill,1]))

        # --- Amplitude modulation params ---
        # Coeffs per component for a small RBF basis over time

        self.amp_coef = nn.Parameter(
            0.01 * torch.randn(n_fill, n_amp_basis)
        )
        # lengthscales (in time units). Softplus to keep >0
        amp_timescale = torch.clone(init[:n_fill,1])
        amp_timescale[amp_timescale<50] = 50
        self.log_ell_amp = nn.Parameter(torch.log(amp_timescale))
        self.register_buffer('amp_basis_center',torch.linspace(tmin, tmax, n_amp_basis))

        # --- Phase drift params (in cycles) ---
        self.phi_coef = nn.Parameter(
            0.01 * torch.randn(n_fill, n_phi_basis)
        )
        #nn.Parameter(torch.tensor(60.))  # typically slower than amp
        ph_timescale = torch.clone(init[:n_fill,1])
        ph_timescale[ph_timescale<50] = 50
        self.log_ell_phi = nn.Parameter(torch.log(ph_timescale))
        self.register_buffer('phi_basis_center',torch.linspace(tmin, tmax, n_phi_basis))

        # Optional epoch reference
        self.t0 = nn.Parameter(torch.tensor(0.0), requires_grad=False)


    def get_periods(self):
        return torch.exp(self.logperiods)
        #return self.periods


    def initialize_planets(self, init, n_anchors=5, n_components=5):
        if init is None:
            periods = torch.rand(n_components,1)+0.5
            anchors = torch.rand(n_components, n_anchors)
            anchors -= anchors.mean()
            anchors /= anchors.std()

        else:
            n_components = init.shape[0]
            x = torch.linspace(0, 1, steps=n_anchors+1,device=init.device)[:-1]
            anchors = torch.zeros(n_components, n_anchors)
            ampls = init[:,0]
            phases = init[:,2]
            for i in range(n_components):
                anchors[i] = ampls[i] * torch.sin(2 * math.pi * (x - phases[i]))

        return anchors

    def sinusoidal_rv(self, t, ph_shift=0, t0=800):
        n_planet,n_param = self.planet_params.shape
        periods = self.get_periods()
        v_doppler = torch.zeros((n_planet,t.shape[0]),device=t.device)
        for i in range(n_planet):
            amp,phase_t0 = self.planet_params[i]
            per = periods[i]
            phase = (((t-t0)/per)+phase_t0+ph_shift)%1
            #phase = ((t/per)-phase_t0)%1
            v = amp*torch.sin(2*math.pi*phase)
            v_doppler[i] = v[:,0]
        return v_doppler

    # ---------- Small helpers ----------    
    @staticmethod
    def _make_rbf_basis(t, centers, ell, t_break=850,gate_width=5):
        """
        RBF columns are smoothly 'gated' to one side of t_break:
          - columns with center < t_break active only for t < t_break
          - columns with center > t_break active only for t > t_break
        Gate is a sigmoid with width ~gate_width (days).
        """
        # (T,1) - (1,M) -> (T,M)
        dt2 = (t.view(-1,1) - centers.view(1,-1))**2
        B = torch.exp(-0.5 * dt2 / (ell**2 + 1e-12))
        # Smooth gates per column
        left_cols  = (centers <= t_break).view(1, -1)   # [1, M]
        right_cols = ~left_cols

        # gates as functions of t (broadcast over columns)
        g_left  = torch.sigmoid((t_break - t).view(-1,1) / gate_width)   # high for t<<break, low for t>>break
        g_right = torch.sigmoid((t - t_break).view(-1,1) / gate_width)   # high for t>>break, low for t<<break

        G = g_left * left_cols + g_right * right_cols                    # [T, M]
        B = B * G
        return B

    def get_quasi_periodic_terms(self, t):
        n_components,n_param = self.planet_params.shape
        quasi_terms = torch.zeros((n_components,t.shape[0],2),device=t.device)

        for i in range(n_components):
            # --- Amplitude modulation A_i(t) ---
            ell_amp = torch.exp(self.log_ell_amp[i])
            B_amp = self._make_rbf_basis(t,self.amp_basis_center, ell_amp)
            A_t = torch.exp(B_amp @ self.amp_coef[i])  # [T,], positive & near 1 at init


            # --- Phase drift phi_i(t) in cycles (so we add directly to theta) ---
            ell_phi = torch.exp(self.log_ell_phi[i])
            B_phi = self._make_rbf_basis(t,self.phi_basis_center, ell_phi)
            phi_t = B_phi @ self.phi_coef[i]           # [T,], small at init
            quasi_terms[i,:,0] = A_t
            quasi_terms[i,:,1] = phi_t
        return quasi_terms
    
    def quasi_periodic_spline_rv(self, t):
        """
        t: time tensor, shape [T,1]
        Returns: RV(t), shape [n_components,T]
        """
        n_components,n_param = self.planet_params.shape
        periods = self.get_periods()
        anchor_amps = self.planet_params
        rv_total = torch.zeros((n_components,t.shape[0]),device=t.device)

        for i in range(n_components):
            P = periods[i]  # scalar
            anchors = anchor_amps[i]       # shape [N]
            # Base phase in cycles
            theta = ((t - self.t0) / P) % 1.0         # [T,]

            # --- Amplitude modulation A_i(t) ---
            ell_amp = torch.exp(self.log_ell_amp[i])
            B_amp = self._make_rbf_basis(t,self.amp_basis_center, ell_amp)
            A_t = torch.exp(B_amp @ self.amp_coef[i])  # [T,], positive & near 1 at init

            # --- Phase drift phi_i(t) in cycles (so we add directly to theta) ---
            ell_phi = torch.exp(self.log_ell_phi[i])
            B_phi = self._make_rbf_basis(t,self.phi_basis_center, ell_phi)
            phi_t = B_phi @ self.phi_coef[i]           # [T,], small at init

            # Shifted phase with drift
            theta_drift = (theta + phi_t.unsqueeze(1)) % 1.0

            # Carrier from your cubic spline over anchors
            rv_carrier = self.periodic_spline(theta_drift, anchors)  # [T,]
            # drop obvious activity signals?
            #if (A_t.max()-A_t.min())>1.0: continue
            # Apply amplitude envelope
            rv_total[i] = A_t * rv_carrier.squeeze()

        #ell_trend = torch.exp(self.log_ell_trend)
        #B_trend = self._make_rbf_basis(t,self.trend_basis_center, ell_trend)
        #rv_trend = B_trend @ self.trend_coef
        return rv_total

    def spline_rv(self, t):
        """
        t: time tensor, shape [T,1]
        Returns: RV(t), shape [n_components,T]
        """
        n_components,n_param = self.planet_params.shape
        periods = self.get_periods()
        anchor_amps = self.planet_params
        rv_total = torch.zeros((n_components,t.shape[0]),device=t.device)

        for i in range(n_components):
            P = periods[i]  # scalar
            anchors = anchor_amps[i]       # shape [N]
            x = (t[:,0] % P) / P
            rv = self.periodic_spline(x, anchors)
            rv_total[i] = rv
        return rv_total

    def periodic_spline(self, x, values):
        """
        Cubic Catmull-Rom spline interpolation on a circle.
        x: shape [T], in [0, 1)
        values: shape [N]
        """
        N = values.shape[0]
        x_scaled = x * N  # [0, N)
        i = torch.floor(x_scaled).long() % N
        f = x_scaled - i.float()

        def get(idx):
            return values[(i + idx) % N]

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

    def forward(self, t, ph_shift=0):
        #return self.quasi_periodic_spline_rv(t)
        #return self.spline_rv(t)
        return self.sinusoidal_rv(t, ph_shift=ph_shift)