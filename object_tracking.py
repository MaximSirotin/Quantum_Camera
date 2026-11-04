"""D=128 quantum telescope moving-object tracking figure.

universal classical receiver with all classical correlations = UCR

Numerical flow:
  1. Draw fresh frozen-flow Gaussian 1D pupil phase screens.
  2. Compute the full universal classical receiver likelihood and analytical
     centroid/relative-displacement likelihood derivatives.
  3. Estimate the UCR Fisher matrix with independent-half cross-products.
  4. Combine the FT Fisher coefficients with the ideal shifted-Y-exchange
     RFS differential FI and a 30% absolute-position branch.
  5. Generate the 1D residual-trajectory panel and the full-extent 2D
     trajectory plot from the Gaussian local-FI uncertainty model.


Requires: numpy, scipy, matplotlib.
To reproduce original reference numbers, use default screens=100000.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.fft import fft, ifft, rfft, irfft, fft2

MAS_PER_RAD = 180.0 * 3600.0 * 1000.0 / np.pi
Q_COLOR = '#AF258C'
C_COLOR = '#0BAA7A'
TRUTH_COLOR = '#222222'


# I. Optical two-photon FT FI from fresh Gaussian phase screens
@dataclass
class PairFIConfig:
    D: int = 512
    a_m: float = 0.0640625
    wavelength_nm: float = 550.0
    r0_m: float = 0.177
    L0_m: float = 22.0
    wind_m_s: float = 10.0
    lag_s: float = 0.020
    focal_bins: int = 1024
    screen_grid_points: int = 4096
    screens: int = 100000
    batch_size: int = 128
    seed: int = 20261019
    q_temporal_Y: float = 0.65
    q_absolute_anchor: float = 0.30
    visibility: float = 1.0
    quantum_pair_efficiency: float = 1.0


def _phase_power(cfg: PairFIConfig) -> tuple[np.ndarray, dict]:
    """Periodic stationary Gaussian phase *slice* with calibrated one-step D_phi."""
    L = cfg.screen_grid_points
    if L < 2*cfg.D or L % 2:
        raise ValueError('screen_grid_points must be even and >= 2*D')
    f = np.fft.rfftfreq(L, d=cfg.a_m)  # cycles per meter
    psd = (f*f + cfg.L0_m**(-2))**(-4/3)
    psd[0] = 0.0  # remove physically irrelevant common piston
    raw_cov = irfft(psd, n=L)
    d1_raw = 2*(raw_cov[0]-raw_cov[1])
    target = 6.88*(cfg.a_m/cfg.r0_m)**(5/3)
    psd *= target/d1_raw
    cov = irfft(psd,n=L)
    structure = {str(k): float(2*(cov[0]-cov[k])) for k in (1,2,3,5,8,16,32)}
    return np.sqrt(psd), {
        'single_pitch_phase_structure_rad2': float(target),
        'structure_function_at_grid_lags_rad2': structure,
        'screen_spectrum_description': '1D slice, (f^2+L0^-2)^(-4/3), one-pitch normalization',
    }


def _autocorr_modes(screen_phases: np.ndarray, D: int) -> np.ndarray:
    """Rowwise R_r = sum_k v[k+r] conjugate(v[k]) for -(D-1)<=r<=D-1."""
    field=np.exp(1j*screen_phases)
    f=fft(field, n=2*D, axis=-1)
    z=ifft(f*np.conj(f), axis=-1)[:,:D]
    return np.concatenate((np.conj(z[:,1:][:,::-1]),z),axis=-1)


def _to_prob_and_derivatives(C: np.ndarray, D: int, M: int):
    """2D Fourier transform of E[R1_r R2_s] and its phase derivatives."""
    rr=np.arange(-(D-1),D,dtype=np.int32)
    r=rr[:,None]
    s=rr[None,:]
    inds=rr % M
    def transform(coeff):
        # r modulo M handles coarse detection, including aliasing at M=D.
        canvas=np.zeros((M,M),dtype=np.complex128)
        np.add.at(canvas,(inds[:,None],inds[None,:]),coeff)
        return fft2(canvas).real/(D*M)**2
    p=transform(C)
    dc=transform(1j*(r+s)*C)
    dd=transform(0.5j*(s-r)*C)
    return p,dc,dd


def _estimate_fi(p: np.ndarray, dca: np.ndarray, dda: np.ndarray,
                 dcb: np.ndarray | None = None, ddb: np.ndarray | None = None):
    if dcb is None: dcb=dca
    if ddb is None: ddb=dda
    if p.min() <= 0:
        raise RuntimeError('Estimated likelihood has nonpositive probability; increase MC screens')
    Fcc=float(np.sum(dca*dcb/p))
    Fdd=float(np.sum(dda*ddb/p))
    Fcd=float(0.5*np.sum((dca*ddb+dda*dcb)/p))
    return [[Fcc,Fcd],[Fcd,Fdd]]


def reproduce(cfg: PairFIConfig, *, verbose: bool=True) -> dict:
    D,M,L=cfg.D,cfg.focal_bins,cfg.screen_grid_points
    if M < D or cfg.screens<4 or cfg.batch_size<1:
        raise ValueError('Need focal_bins >= D, screens >= 4, and batch_size >= 1')
    if cfg.q_temporal_Y+cfg.q_absolute_anchor>1+1e-12:
        raise ValueError('Allocated quantum pair fractions sum to more than 1')
    if cfg.wind_m_s*cfg.lag_s >= D*cfg.a_m:
        raise ValueError('Shift exceeds telescope pupil')
    sqrtpsd, phase_details = _phase_power(cfg)
    s=cfg.wind_m_s*cfg.lag_s/cfg.a_m
    phase_shift=np.exp(-2j*np.pi*np.fft.rfftfreq(L)*s)
    rng=np.random.default_rng(cfg.seed)
    C_halves=np.zeros((2,2*D-1,2*D-1),dtype=np.complex128)
    counts=[0,0]
    t0=time.time()
    for start in range(0,cfg.screens,cfg.batch_size):
        n=min(cfg.batch_size,cfg.screens-start)
        noise=rng.standard_normal((n,L))
        f=rfft(noise,axis=1)*sqrtpsd[None,:]
        chi1=irfft(f,n=L,axis=1)[:,:D]
        chi2=irfft(f*phase_shift[None,:],n=L,axis=1)[:,:D]
        R1=_autocorr_modes(chi1,D)
        R2=_autocorr_modes(chi2,D)
        # Deterministic contiguous halves, independent Gaussian screens.
        offset=0
        for half_start,half_end,h in ((0,cfg.screens//2,0),(cfg.screens//2,cfg.screens,1)):
            left=max(0,half_start-start)
            right=min(n,half_end-start)
            if right>left:
                C_halves[h]+=R1[left:right].T@R2[left:right]
                counts[h]+=right-left
    if verbose:
        print('Accumulated',counts,'phase-screen realizations in',round(time.time()-t0,1),'seconds')
    (p1,dc1,dd1),(p2,dc2,dd2)=[_to_prob_and_derivatives(C_halves[h]/counts[h],D,M) for h in range(2)]
    p=(p1+p2)/2
    dc=(dc1+dc2)/2
    dd=(dd1+dd2)/2
    Fcv=_estimate_fi(p,dc1,dd1,dc2,dd2)
    Fplug=_estimate_fi(p,dc,dd)
    # Each derivative of the joint probability should sum to zero exactly.
    checks={
        'likelihood_sum':float(p.sum()),
        'likelihood_min':float(p.min()),
        'derivative_c_sum':float(dc.sum()),
        'derivative_delta_sum':float(dd.sum()),
        'max_half_likelihood_difference':float(np.max(np.abs(p1-p2))),
        'ft_centroid_FI_plugin_upward_bias':float(Fplug[0][0]-Fcv[0][0]),
        'ft_displacement_FI_plugin_upward_bias':float(Fplug[1][1]-Fcv[1][1]),
    }
    overlap=1-s/D
    F_delta_q=(cfg.q_temporal_Y*cfg.quantum_pair_efficiency*
               cfg.visibility**2*overlap**4*(D*D-1)/6)
    # Independent check via the explicitly normalized ideal exchange-Y POVM.
    # Fractional shifts require interpolation/continuous-pupil modeling; this
    # sanity check rounds to the closest integer pixel shift only.
    n_overlap=D-int(round(s))
    m_sep=np.arange(1,n_overlap,dtype=np.float64)
    p_sep=2*(n_overlap-m_sep)/(n_overlap*n_overlap)
    F_delta_Y_integer_shift=(cfg.q_temporal_Y*cfg.quantum_pair_efficiency*
                            cfg.visibility**2*(n_overlap/D)**2*
                            float(np.sum(p_sep*m_sep*m_sep)))
    # The tracking plots use the diagonal FI model, even if offdiagonal
    # pair FI is present; report the offdiagonal for users to assess.
    ftc=Fcv[0][0];ftd=Fcv[1][1]
    if ftc<=0 or ftd<=0:
        raise RuntimeError('Cross-validated FI became nonpositive; increase --screens')
    return {
        'parameters': asdict(cfg),
        'full_pair_FI_cross_validated': Fcv,
        'full_pair_FI_plugin_biased': Fplug,
        'likelihood_checks': checks,
        'phase_model': phase_details,
        'summary': {
            'F_centroid_FT_per_pair': float(ftc),
            'F_delta_FT_per_pair': float(ftd),
            'F_centroid_delta_FT_per_pair': float(Fcv[0][1]),
            'F_delta_RFS_per_nominal_pair_including_Y_fraction': float(F_delta_q),
            'F_delta_RFS_explicit_Y_sum_nearest_integer_shift': float(F_delta_Y_integer_shift),
            'ideal_pure_pair_relative_QFI_no_allocation': float((D*D-1)/6),
        },
    }


# II. Joint position/velocity Fisher model and data
@dataclass(frozen=True)
class TrackingParameters:
    # Exact selected 1D reference benchmark
    D: int = 512
    pixel_pitch_m: float = 0.0640625
    wavelength_nm: float = 550.0
    wind_m_s: float = 10.0
    lag_s: float = 0.020
    T_obs_s: float = 0.25
    T_pred_s: float = 2.0
    photon_rate_s: float = 440.8
    visibility: float = 1.0
    quantum_pair_efficiency: float = 1.0
    q_temporal_Y: float = 0.65
    q_absolute_anchor: float = 0.30
    F_centroid_FT: float = 1.836
    F_delta_FT: float = 1.774

    # One dimension receives the full budget. In the 2D *illustration*,
    # x and y each receive half the available nominal photon-pair budget.
    coordinate_fraction_2d: float = 0.50

    # True small residual corrections (relative to known curved ephemeris).
    # Units: mas and mas/s. These parameters are estimated, NOT given to either
    # receiver; their values only determine the simulation's true trajectory.
    true_theta0_mas: float = 0.0
    true_v_res_mas_s: float = 0.0
    true_x0_mas: float = 0.0
    true_y0_mas: float = 0.0
    true_vx_res_mas_s: float = 0.0
    true_vy_res_mas_s: float = 0.0

    # Known nominal ephemeris: constant speed along gently curving arc.
    v_nominal_mas_s: float = 250.0
    turn_deg_over_obs: float = 4.0
    # A rotated, deliberately more curved KNOWN ephemeris is used only to
    # improve the visibility of both sky coordinates. The residual acceleration
    # is still assumed KNOWN/zero (no acceleration is estimated).
    initial_heading_deg: float = 30.0
    seed: int = 1  # legacy common seed fallback
    seed_quantum: int | None = None   # controls Q 1D and 2D stochastic fits only
    seed_classical: int | None = None # controls FT 1D and 2D stochastic fits only

    # None => full-extent plot includes the entire 2-second FT 1-sigma region.
    # Set explicit limits for a closer view, which will spatially clip that
    # region but never truncate its temporal horizon.
    sky_xlim_2d: tuple[float, float] | None = None
    sky_ylim_2d: tuple[float, float] | None = None
    show_classical_forecast_tube: bool = True
    # The full classical prediction tube is now drawn over ALL T_pred_s.
    # This optional limit is kept as a backwards-compatible field, but does not
    # restrict the tube. The plot may clip it if explicit sky limits are set.
    classical_forecast_tube_max_horizon_s: float = 2.0
    tube_grid_pixels: int = 340

    # Display of the full marginal position uncertainties, not new shots.
    observation_sigma_2d: float = 1.0
    forecast_sigma_2d: float = 1.0
    show_final_forecast_ellipses: bool = True
    show_stochastic_fits_2d: bool = True
    ellipse_future_horizons_s: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    y_exaggeration_2d: float = 1.0


def _validate(p: TrackingParameters) -> None:
    positives = {
        'D': p.D, 'pixel_pitch_m': p.pixel_pitch_m,
        'wavelength_nm': p.wavelength_nm, 'wind_m_s': p.wind_m_s,
        'lag_s': p.lag_s, 'T_obs_s': p.T_obs_s,
        'photon_rate_s': p.photon_rate_s, 'F_centroid_FT': p.F_centroid_FT,
        'F_delta_FT': p.F_delta_FT, 'coordinate_fraction_2d': p.coordinate_fraction_2d,
        'observation_sigma_2d': p.observation_sigma_2d,
        'forecast_sigma_2d': p.forecast_sigma_2d,
        'y_exaggeration_2d': p.y_exaggeration_2d,
    }
    for name, value in positives.items():
        if value <= 0:
            raise ValueError(f'{name} must be positive')
    if p.D < 2 or p.lag_s >= p.T_obs_s or p.T_pred_s < 0:
        raise ValueError('Need D>=2, 0<lag<T_obs, and T_pred>=0')
    for name, v in [('visibility', p.visibility),
                    ('quantum_pair_efficiency', p.quantum_pair_efficiency),
                    ('q_temporal_Y', p.q_temporal_Y),
                    ('q_absolute_anchor', p.q_absolute_anchor),
                    ('coordinate_fraction_2d', p.coordinate_fraction_2d)]:
        if not 0 < v <= 1:
            raise ValueError(f'{name} must lie in (0,1]')
    if p.q_temporal_Y + p.q_absolute_anchor > 1.0 + 1e-12:
        raise ValueError('RFS Y + FT-anchor branch fractions exceed unity')
    if p.wind_m_s * p.lag_s >= p.D * p.pixel_pitch_m:
        raise ValueError('The wind shift must be smaller than the pupil length')
    if p.tube_grid_pixels < 60:
        raise ValueError('tube_grid_pixels must be >= 60')
    if p.classical_forecast_tube_max_horizon_s < 0:
        raise ValueError('classical_forecast_tube_max_horizon_s must be >= 0')
    for nm, lim in [('sky_xlim_2d',p.sky_xlim_2d),
                    ('sky_ylim_2d',p.sky_ylim_2d)]:
        if lim is not None and (len(lim) != 2 or lim[1] <= lim[0]):
            raise ValueError(f'{nm} must be an ordered (min, max) tuple')


def _design(t: np.ndarray, p: TrackingParameters) -> np.ndarray:
    """Linear residual motion reference time = midpoint of observation."""
    t = np.asarray(t, dtype=float)
    return np.column_stack((np.ones(t.size), t - p.T_obs_s / 2.0))


def fisher_and_covariance(p: TrackingParameters, *, coord_fraction: float = 1.0
                          ) -> dict[str, Any]:
    """Joint FI and covariance, in (mas, mas/s), per one coordinate.

    q_anchor photons are detected directly; only q_Y photons incur the RFS
    memory/pairing efficiency. eta=1 reproduces the reference handover exactly.
    The F_{centroid,FT} and F_{delta,FT} terms are both kept for UCR.
    Cross terms are set to zero in the *centered, diagonal FI* toy model.
    """
    _validate(p)
    if not 0 < coord_fraction <= 1:
        raise ValueError('coord_fraction must be in (0,1]')

    N = p.photon_rate_s * p.T_obs_s * 0.5 * coord_fraction
    Vt = (p.T_obs_s - p.lag_s) ** 2 / 12.0
    overlap = 1.0 - p.wind_m_s * p.lag_s / (p.D * p.pixel_pitch_m)
    F_delta_RFS = (p.q_temporal_Y * p.quantum_pair_efficiency *
                   p.visibility**2 * overlap**4 * (p.D**2 - 1.0)/6.0)

    I0_q = N * p.q_absolute_anchor * p.F_centroid_FT
    I0_c = N * p.F_centroid_FT
    Iv_q = N * (p.q_absolute_anchor * p.F_centroid_FT * Vt +
                F_delta_RFS * p.lag_s**2)
    Iv_c = N * (p.F_centroid_FT * Vt + p.F_delta_FT * p.lag_s**2)

    kappa = (p.wavelength_nm * 1e-9 / (2*np.pi*p.pixel_pitch_m)
             * MAS_PER_RAD)
    # FI is for dimensionless phase slope k*a*theta. Angular covariance:
    # Cov[theta0_mas, v_mas/s] = kappa^2 * inv(F_phase).
    F_q = np.diag([I0_q, Iv_q])
    F_c = np.diag([I0_c, Iv_c])
    cov_q = kappa**2 * np.linalg.inv(F_q)
    cov_c = kappa**2 * np.linalg.inv(F_c)
    return {
        'F_Q_phase': F_q, 'F_FT_phase': F_c,
        'cov_Q': cov_q, 'cov_FT': cov_c,
        'N_nominal_pairs_per_coordinate': N,
        'overlap': overlap, 'Vt_s2': Vt,
        'F_delta_RFS_per_nominal_pair': F_delta_RFS,
        'mas_per_phase': kappa,
        'velocity_FI_ratio': Iv_q / Iv_c,
    }


def ephemeris_arc(t: np.ndarray, p: TrackingParameters) -> np.ndarray:
    """Known 2D circular arc with adjustable initial sky-plane heading.

    Changing heading/curvature rotates/bends the *known* ephemeris only and
    does not affect residual position/velocity FI. A 4-degree turn in 250 ms
    is intentionally illustrative, not an asteroid acceleration prediction.
    """
    t = np.asarray(t, dtype=float)
    omega = np.deg2rad(p.turn_deg_over_obs)/p.T_obs_s
    heading = np.deg2rad(p.initial_heading_deg)
    if abs(omega) < 1e-13:
        return p.v_nominal_mas_s*np.column_stack((t*np.cos(heading),
                                                   t*np.sin(heading)))
    a = heading+omega*t
    return p.v_nominal_mas_s/omega*np.column_stack(
        (np.sin(a)-np.sin(heading), np.cos(heading)-np.cos(a)))


def marginal_sigma(t: np.ndarray, cov: np.ndarray,
                   p: TrackingParameters) -> np.ndarray:
    X = _design(t, p)
    return np.sqrt(np.maximum(np.einsum('ti,ij,tj->t', X, cov, X), 0.0))


def make_tracking_data(p: TrackingParameters = TrackingParameters(),
                       *, n_observation: int = 160,
                       n_future: int = 320) -> dict[str, Any]:
    """Generate theory and one stochastic efficient-estimator realization."""
    _validate(p)
    if n_observation < 2 or n_future < 2:
        raise ValueError('n_observation and n_future must be >= 2')
    t_obs = np.linspace(0, p.T_obs_s, n_observation, endpoint=False)
    t_pred = np.linspace(p.T_obs_s, p.T_obs_s+p.T_pred_s, n_future)
    t = np.r_[t_obs, t_pred]
    X = _design(t, p)
    obs_mask = t < p.T_obs_s
    # Independent deterministic RNG streams: changing the RFS seed does NOT
    # alter the UCR realization, and vice versa. The extra stream ID
    # distinguishes the receivers even when seed_quantum == seed_classical.
    seed_q = p.seed if p.seed_quantum is None else p.seed_quantum
    seed_c = p.seed if p.seed_classical is None else p.seed_classical
    rngs = {
        'Q': np.random.default_rng(np.random.SeedSequence([int(seed_q), 111])),
        'FT': np.random.default_rng(np.random.SeedSequence([int(seed_c), 222])),
    }

    one = fisher_and_covariance(p, coord_fraction=1.0)
    two = fisher_and_covariance(p, coord_fraction=p.coordinate_fraction_2d)

    # 1D residual to known ephemeris. This is a *true* residual parameter,
    # which neither receiver receives as an oracle.
    beta_true_1d = np.array([p.true_theta0_mas, p.true_v_res_mas_s])
    residual_true = X @ beta_true_1d
    one_d = {'t_s': t, 'is_observation': obs_mask,
             'residual_true_mas': residual_true}
    for label, cov_key in [('Q', 'cov_Q'), ('FT', 'cov_FT')]:
        cov = one[cov_key]
        sampled_beta = rngs[label].multivariate_normal(beta_true_1d, cov)
        one_d[f'beta_hat_{label}'] = sampled_beta
        one_d[f'residual_hat_{label}_mas'] = X @ sampled_beta
        one_d[f'sigma_{label}_mas'] = marginal_sigma(t, cov, p)

    # 2D illustrative receiver: *each coordinate* is given half the pairs;
    # x and y have separate, identically distributed joint estimators.
    beta_true_2d = np.array([[p.true_x0_mas, p.true_vx_res_mas_s],
                             [p.true_y0_mas, p.true_vy_res_mas_s]])
    ephem = ephemeris_arc(t, p)
    true_2d = ephem + X @ beta_true_2d.T
    two_d = {'t_s': t, 'is_observation': obs_mask,
             'ephem_mas': ephem, 'xy_true_mas': true_2d,
             'beta_true_2d': beta_true_2d}
    for label, cov_key in [('Q', 'cov_Q'), ('FT', 'cov_FT')]:
        cov = two[cov_key]
        samples = np.stack([rngs[label].multivariate_normal(beta_true_2d[i], cov)
                            for i in range(2)], axis=0)
        two_d[f'beta_hat_{label}'] = samples
        two_d[f'xy_hat_{label}_mas'] = ephem + X @ samples.T
        two_d[f'sigma_{label}_mas'] = marginal_sigma(t, cov, p)

    return {'params': p, 'FI_1D': one, 'FI_2D': two,
            'one_d': one_d, 'two_d': two_d}


def _plot_confidence_tube(ax: plt.Axes, xy: np.ndarray,
                          sigma_mas: np.ndarray, *, n_sigma: float,
                          y_exag: float, color: str, alpha: float,
                          xlim: tuple[float, float],
                          ylim: tuple[float, float],
                          grid_pixels: int = 340,
                          draw_boundary: bool = True) -> None:
    """Plot the UNION of instantaneous isotropic n-sigma disks along a path.

    Unlike an ordinary transverse-only ribbon, this display also includes
    along-track uncertainty and rounded endcaps. It is the envelope of
    time-marginal disks, NOT a simultaneous confidence band for the entire
    stochastic fitted trajectory (the fit errors are time-correlated).
    The physical x/y noise is isotropic. y_exag changes display only.
    """
    gx = np.linspace(xlim[0], xlim[1], grid_pixels)
    gy = np.linspace(ylim[0], ylim[1], grid_pixels)
    X, Y = np.meshgrid(gx, gy)
    field = np.full(X.shape, -np.inf, dtype=float)
    # Subsample the time mesh purely for displaying a smooth confidence tube.
    for i in np.unique(np.round(np.linspace(0, len(xy)-1,
                                               min(175,len(xy)))).astype(int)):
        r2 = (float(n_sigma)*float(sigma_mas[i]))**2
        r2_minus_dist2 = (r2 - (X-xy[i,0])**2 -
                          (Y/y_exag - xy[i,1])**2)
        np.maximum(field, r2_minus_dist2, out=field)
    if field.max() <= 0:
        return
    lo = 0.0
    hi = max(float(field.max()),1.0)
    ax.contourf(gx, gy, field, levels=[lo,hi], colors=[color],
                alpha=alpha, antialiased=True, zorder=1)
    if draw_boundary and field.min() < 0:
        ax.contour(gx, gy, field, levels=[0], colors=[color],
                   linewidths=.75, alpha=min(.55, alpha*2.5),zorder=2)


def plot_tracking_2d(data: dict[str, Any], *, figsize=(7.6, 6.2),
                     dpi: int = 160) -> plt.Figure:
    """Shared-axis arc-style figure with complete 1-sigma FT forecast.

    The dashed ellipses at the final predicted location show the one-time
    marginal forecast uncertainty; the colored tubes show all future times.

    The actual observation is [0, T_obs], followed by a prediction lasting
    T_pred seconds. Curves are SOLID during observation, DASHED in prediction;
    truth is black solid (observation) / black dotted (prediction). The future
    colored confidence regions have NO stroke/boundary.

    With finite sky_xlim_2d/sky_ylim_2d the *full-in-time* FT tube is computed
    and rendered, but it may be spatially clipped by the user-specified axes.
    Set those limits to None to auto-include the entire classical 1-sigma tube.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    p, d = data['params'], data['two_d']
    t = d['t_s']
    truth = d['xy_true_mas']
    exag = p.y_exaggeration_2d
    observed = t <= p.T_obs_s + 1e-12
    future = t >= p.T_obs_s - 1e-12
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=.16, right=.97, bottom=.16, top=.94)

    # Auto limits show the COMPLETE classical +2-s 1-sigma region. The optional
    # fixed limits retain the earlier close-up view (potentially with clipping).
    ft_center = truth[future]
    ft_r = p.forecast_sigma_2d * d['sigma_FT_mas'][future]
    margin = 60.0
    xlim_full = (float(np.min(ft_center[:, 0] - ft_r) - margin),
                 float(np.max(ft_center[:, 0] + ft_r) + margin))
    ylim_full = (float(np.min(exag*(ft_center[:, 1] - ft_r)) - margin),
                 float(np.max(exag*(ft_center[:, 1] + ft_r)) + margin))
    xlim = xlim_full if p.sky_xlim_2d is None else p.sky_xlim_2d
    ylim = ylim_full if p.sky_ylim_2d is None else p.sky_ylim_2d
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Both tubes cover the FULL future duration, and have no outline.
    if p.show_classical_forecast_tube and np.any(future):
        _plot_confidence_tube(
            ax, truth[future], d['sigma_FT_mas'][future],
            n_sigma=p.forecast_sigma_2d, y_exag=exag,
            color=C_COLOR, alpha=.12, xlim=xlim, ylim=ylim,
            grid_pixels=p.tube_grid_pixels, draw_boundary=False)
    if np.any(future):
        _plot_confidence_tube(
            ax, truth[future], d['sigma_Q_mas'][future],
            n_sigma=p.forecast_sigma_2d, y_exag=exag,
            color=Q_COLOR, alpha=.31, xlim=xlim, ylim=ylim,
            grid_pixels=p.tube_grid_pixels, draw_boundary=False)

    # Dashed one-sigma reconstructed-midpoint observation ellipses, not
    # independent time-bin measurements.
    idx_mid = int(np.argmin(np.abs(t-p.T_obs_s/2)))
    for label, color in [('FT', C_COLOR), ('Q', Q_COLOR)]:
        sig = float(d[f'sigma_{label}_mas'][idx_mid])
        ax.add_patch(Ellipse(
            (truth[idx_mid, 0], exag*truth[idx_mid, 1]),
            width=2*p.observation_sigma_2d*sig,
            height=2*p.observation_sigma_2d*sig*exag,
            facecolor='none', edgecolor=color, linewidth=1,
            linestyle=(0, (4,3)), alpha=.80, zorder=6))

    # Endpoint forecast contours: instantaneous one-sigma marginal region at
    # t = T_obs + T_pred, centered on the true/expected trajectory endpoint.
    # Dashed outlines are distinct from the borderless time-union tubes.
    if p.show_final_forecast_ellipses and p.T_pred_s > 0:
        xy_final = truth[-1]
        for label, color in [('FT', C_COLOR), ('Q', Q_COLOR)]:
            if label == 'FT' and not p.show_classical_forecast_tube:
                continue
            radius = p.forecast_sigma_2d * float(d[f'sigma_{label}_mas'][-1])
            ax.add_patch(Ellipse(
                (xy_final[0], exag*xy_final[1]),
                width=2*radius, height=2*radius*exag,
                facecolor='none', edgecolor=color,
                linewidth=1.7, linestyle=(0, (5, 3)),
                alpha=0.95, zorder=12))

    # Black true curve: solid within the measured interval, dotted thereafter.
    ax.plot(truth[observed, 0], exag*truth[observed, 1],
            color=TRUTH_COLOR, linestyle='-', lw=1, zorder=10)
    ax.plot(truth[future, 0], exag*truth[future, 1],
            color=TRUTH_COLOR, linestyle=':', lw=1, zorder=10)

    # Optional stochastic global fits; normally omitted from manuscript panel.
    if p.show_stochastic_fits_2d:
        for label, color, lw, order in [('FT', C_COLOR, 1, 8),
                                        ('Q', Q_COLOR, 1, 11)]:
            fit = d[f'xy_hat_{label}_mas']
            ax.plot(fit[observed, 0], exag*fit[observed, 1],
                    color=color, linestyle='-', lw=lw, alpha=.97, zorder=order)
            ax.plot(fit[future, 0], exag*fit[future, 1],
                    color=color, linestyle='--', lw=lw, alpha=.97, zorder=order)

    # True observation endpoint diamond markers, drawn atop all curves.
    start = truth[0]
    end = truth[int(np.argmin(np.abs(t-p.T_obs_s)))]
    for xy, face in [(start, 'black'), (end, 'white')]:
        ax.plot(xy[0], exag*xy[1], marker='D', markersize=0,
                markerfacecolor=face, markeredgecolor='black',
                markeredgewidth=1.2, linestyle='none', zorder=20)

    ax.set_xlabel('Sky x [mas]')
    ax.set_ylabel('Sky y [mas]' + (f' \u00d7{exag:g} (display)' if exag != 1 else ''))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(direction='out')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    # ax.grid(color='#aaaaaa', alpha=.17, zorder=0)

    handles = [
        Line2D([0],[0], color=TRUTH_COLOR, linestyle='-', lw=2,
               label='True, observed'),
        Line2D([0],[0], color=TRUTH_COLOR, linestyle=':', lw=2,
               label='True, future'),
        Line2D([0],[0], color=Q_COLOR, lw=2.2, linestyle='-',
               label='RFS fit (solid / dashed)'),
        Line2D([0],[0], color=C_COLOR, lw=2.1, linestyle='-',
               label='FT fit (solid / dashed)'),
        Patch(facecolor=Q_COLOR, alpha=.31, edgecolor='none',
              label='RFS 1\u03c3 forecast'),
        Patch(facecolor=C_COLOR, alpha=.12, edgecolor='none',
              label='FT 1\u03c3 forecast'),
    ]
    # ax.legend(handles=handles, loc='upper left', frameon=False,
    #           handlelength=2.6, labelspacing=.32)
    return fig

# III. Original standalone 1D and complete 2D plot functions
def _data(source):
    return make_tracking_data(source) if isinstance(source, TrackingParameters) else source


def _style_context(rc: Mapping[str, Any] | None):
    # No global rcParams mutation. Caller can pass e.g. {'font.size': 16}.
    return plt.rc_context(rc=rc or {})


def _save(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight')
    return fig


def plot_along_track(source, *, figsize=(8.5, 4.5), dpi=160,
                     rc=None, ylim=None, xlim=None,
                     show_true=True, show_fits=False, show_bands=True,
                     band_alpha_q=0.30, band_alpha_ft=0.14,
                     linewidth=1.8, show_legend=True,
                     title=None, xlabel='Time since observation start (s)',
                     ylabel='Along-track residual position (mas)',
                     save_path=None):
    """One-axes 1D residual-position time plot; returns a Matplotlib Figure.

    `source` may be TrackingParameters or a precomputed make_tracking_data result.
    This is not the old three-panel 1D dashboard.
    """
    data = _data(source)
    p, d = data['params'], data['one_d']
    t, truth = d['t_s'], d['residual_true_mas']
    with _style_context(rc):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if show_bands:
            ax.fill_between(t, truth-d['sigma_FT_mas'], truth+d['sigma_FT_mas'],
                            color=C_COLOR, alpha=band_alpha_ft, linewidth=0,
                            label='classical: 1$\\sigma$')
            ax.fill_between(t, truth-d['sigma_Q_mas'], truth+d['sigma_Q_mas'],
                            color=Q_COLOR, alpha=band_alpha_q, linewidth=0,
                            label='Hybrid RFS: 1$\\sigma$')
        if show_true:
            obs = t <= p.T_obs_s
            fut = t >= p.T_obs_s
            ax.plot(t[obs], truth[obs], color=TRUTH_COLOR, lw=linewidth,
                    ls='-', label='True residual: observed')
            ax.plot(t[fut], truth[fut], color=TRUTH_COLOR, lw=linewidth,
                    ls='--', label='True residual: predicted')
        if show_fits:
            obs, fut = t <= p.T_obs_s, t >= p.T_obs_s
            for tag, color, label in [('FT', C_COLOR, 'Classical'),
                                      ('Q', Q_COLOR, 'Hybrid RFS')]:
                y = d[f'residual_hat_{tag}_mas']
                ax.plot(t[obs], y[obs], color=color, ls='-', lw=linewidth,
                        label=f'{label} fit')
                ax.plot(t[fut], y[fut], color=color, ls='--', lw=linewidth)
        ax.axvline(p.T_obs_s, color='#777777', lw=1, ls=':')
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if xlim is not None: ax.set_xlim(xlim)
        else: ax.set_xlim(0, p.T_obs_s+p.T_pred_s)
        if ylim is not None: ax.set_ylim(ylim)
        if show_legend: ax.legend(frameon=False, fontsize='small')
        return _save(fig, save_path)


def _plot_2d(source, *, figsize, dpi, rc, xlim, ylim,
             show_final_ellipses, show_fits, markersize,
             linewidth, title, save_path):
    data = _data(source)
    p = replace(data['params'], sky_xlim_2d=xlim, sky_ylim_2d=ylim,
                show_final_forecast_ellipses=show_final_ellipses,
                show_stochastic_fits_2d=show_fits)
    with _style_context(rc):
        fig = plot_tracking_2d({**data, 'params': p}, figsize=figsize, dpi=dpi)
        ax = fig.axes[0]
        if title is not None: ax.set_title(title)
        # The legacy code sets diamond markersize=0. Restore them here.
        for line in ax.lines:
            if line.get_marker() == 'D': line.set_markersize(markersize)
            elif linewidth is not None: line.set_linewidth(linewidth)
        return _save(fig, save_path)


def plot_2d_full(source, *, figsize=(8.5, 7.0), dpi=160, rc=None,
                 xlim=None, ylim=None, show_final_ellipses=True,
                 show_fits=False, markersize=5, linewidth=None,
                 title=None, save_path=None):
    """Full-extent 2D plot. By default includes entire FT future 1-sigma tube."""
    return _plot_2d(source, figsize=figsize, dpi=dpi, rc=rc,
                    xlim=xlim, ylim=ylim,
                    show_final_ellipses=show_final_ellipses,
                    show_fits=show_fits, markersize=markersize,
                    linewidth=linewidth, title=title, save_path=save_path)

# IV. Exact benchmark configuration and convenience functions


def default_pair_fi_config(*, screens: int = 100_000) -> PairFIConfig:
    """Exact D=128, 8.2-m, 40-ms reference settings from tracking figure."""
    return PairFIConfig(
        D=128, a_m=0.0640625, wavelength_nm=550.0,
        r0_m=0.177, L0_m=22.0, wind_m_s=10.0, lag_s=0.040,
        focal_bins=256, screen_grid_points=2048,
        screens=screens, batch_size=128, seed=20261059,
        q_temporal_Y=0.65, q_absolute_anchor=0.30,
        visibility=1.0, quantum_pair_efficiency=1.0,
    )


def tracking_parameters_from_fi(record: dict, *, horizon_s: float = 8.0
                                ) -> TrackingParameters:
    """Populate the original tracking model from a fresh optical FI result."""
    cfg = record['parameters']
    fi = record['summary']
    return replace(
        TrackingParameters(),
        D=int(cfg['D']), pixel_pitch_m=float(cfg['a_m']),
        wavelength_nm=float(cfg['wavelength_nm']),
        wind_m_s=float(cfg['wind_m_s']), lag_s=float(cfg['lag_s']),
        T_obs_s=0.25, T_pred_s=float(horizon_s),
        photon_rate_s=1600.0,  # 400 photons during 0.25 s
        q_temporal_Y=float(cfg['q_temporal_Y']),
        q_absolute_anchor=float(cfg['q_absolute_anchor']),
        visibility=float(cfg['visibility']),
        quantum_pair_efficiency=float(cfg['quantum_pair_efficiency']),
        F_centroid_FT=float(fi['F_centroid_FT_per_pair']),
        F_delta_FT=float(fi['F_delta_FT_per_pair']),
        coordinate_fraction_2d=0.5,
        v_nominal_mas_s=250.0, turn_deg_over_obs=0.4,
        initial_heading_deg=30.0,
        seed_quantum=3, seed_classical=7,
        observation_sigma_2d=5.0, forecast_sigma_2d=1.0,
        show_stochastic_fits_2d=False,
    )


def run_tracking(*, screens: int = 100_000, horizon_s: float = 8.0,
                 verbose: bool = True) -> tuple[dict, dict]:
    """Fresh 2-click FI followed by 1D and 2D tracking forecast model."""
    config = default_pair_fi_config(screens=screens)
    fi_record = reproduce(config, verbose=verbose)
    params = tracking_parameters_from_fi(fi_record, horizon_s=horizon_s)
    data = make_tracking_data(params)
    return fi_record, data


def tracking_forecast_summary(data: dict) -> dict:
    """RFS versus FT prediction uncertainty per 2D allocated coordinate."""
    p = data['params']
    fi = data['FI_2D']
    horizon_from_midpoint = p.T_pred_s + p.T_obs_s/2
    sigma_q = float(np.sqrt(fi['cov_Q'][0, 0] +
                            horizon_from_midpoint**2 * fi['cov_Q'][1, 1]))
    sigma_ft = float(np.sqrt(fi['cov_FT'][0, 0] +
                             horizon_from_midpoint**2 * fi['cov_FT'][1, 1]))
    return {
        'prediction_horizon_s': p.T_pred_s,
        'velocity_FI_ratio': float(fi['velocity_FI_ratio']),
        'prediction_photon_cost_ratio': (sigma_ft/sigma_q)**2,
        'sigma_RFS_mas': sigma_q,
        'sigma_classical_mas': sigma_ft,
    }

