#!/usr/bin/env python3
"""Binary-source separation simulations.

universal classical receiver with all classical correlations = UCR

  1) Power-spectrum QFI on the total-index-twirled pair state (equals atm. aware RFS).
  2) RFS FI (atm. blind).
  3) Exact joint two-click, product-Fourier (FT x FT) FI on the full pair state (universal classical receiver: direct imaging with all classical correlations).
  4) Mixed-SLD PPT FI upper bound on the *total-index-twirled* pair state.
  5) The modewise steady-state Kalman reference-learning model.

Fixed pupil pixel pitch a=0.0640625 m; L0=22m, r0=0.157921m, wind=10m/s.
t/t_corr=u/L0 for t_corr=L0/wind. All FI are PER TWO-PHOTON PAIR.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.special import gamma, kv
from scipy.sparse.linalg import cg, LinearOperator
from threadpoolctl import threadpool_limits

PITCH_M = A_REF = 0.0640625
L0_M = L0 = 22.0
R0_M = R0 = 0.157921
WIND_M_S = 10.0
RAYLEIGH_FRACTION = 0.10

POWER = 'power_QFI_pair'
RFS = 'RFS_FI_pair'
PRODUCT_FT = 'Product_FT_FI_pair'
PPT = 'PPT_upper_total_index_mixed_pair'
KALMAN = 'Kalman_mean_cadence_model_FI_pair'
POISSON_KEYS = (POWER, RFS, PRODUCT_FT, PPT)
FI_KEYS = (POWER, RFS, PRODUCT_FT, PPT, KALMAN)
KALMAN_SETTINGS = dict(beta_scale=1.0, beta_budget='renorm_sum',
                       rho_clip=1.-1e-12, ridge=1e-12, eig_tol=1e-12)


def rayleigh_separation(d: int, fraction: float = RAYLEIGH_FRACTION) -> float:
    """Full binary phase separation [rad/pixel] at the chosen Rayleigh fraction."""
    if int(d) < 2 or not 0 < fraction <= 1:
        raise ValueError('Need d>=2 and 0 < Rayleigh fraction <= 1')
    return 2.0 * np.pi * float(fraction) / int(d)


# =========================================================================
# Physical frozen-flow Gaussian atmosphere + quantum measurement
# =========================================================================

def corr(r, mode='physical'):
    r = np.asarray(r, dtype=float)
    nu = 5.0 / 6.0 if mode == 'physical' else 4.0 / 3.0
    z = 2 * np.pi * np.abs(r) / L0
    shape = np.ones_like(z)
    inds = z > 0
    shape[inds] = 2 ** (1 - nu) / gamma(nu) * z[inds] ** nu * kv(nu, z[inds])
    if mode == 'physical':
        c = -gamma(-nu) / (2 ** (2 * nu) * gamma(nu))
        var = 6.88 / (2 * c * (2 * np.pi * R0 / L0) ** (2 * nu))
    elif mode == 'legacy':
        zz = 2 * np.pi * A_REF / L0
        h = 2 ** (1 - nu) / gamma(nu) * zz ** nu * kv(nu, zz)
        var = 1.52942 / (2 * (1 - h))
    else:
        raise ValueError('unknown covariance mode')
    return var * shape


def cached_covariance(d, pitch_m, lag_m, cov_func):
    offsets = np.arange(-(d - 1), d, dtype=np.int32)
    c = np.asarray(cov_func(offsets * pitch_m), dtype=float)
    cross = np.asarray(cov_func(offsets * pitch_m + lag_m), dtype=float)
    return (c, cross)


def power_charge_blocks(d, delta, c, cross):
    zero = d - 1
    rho_blocks = []
    tangent_blocks = []
    for s in range(2 * d - 1):
        j = np.arange(max(0, s - d + 1), min(d - 1, s) + 1, dtype=np.int32)
        k = s - j
        jj = j[:, None]
        ll = j[None, :]
        kk = k[:, None]
        nn = k[None, :]
        dr = jj - ll
        var = 4 * (c[zero] - c[zero + dr]) + 2 * (cross[zero + jj - kk] - cross[zero + jj - nn] - cross[zero + ll - kk] + cross[zero + ll - nn])
        if np.min(var) < -1e-07:
            raise RuntimeError('Negative phase variance: check covariance or indices')
        base = np.exp(-0.5 * np.maximum(var, 0.0)) / (d * d)
        z = delta * dr
        rho = base * np.cos(0.5 * z) ** 2
        deriv = -0.5 * base * dr * np.sin(z)
        rho_blocks.append((rho + rho.T) * 0.5)
        tangent_blocks.append((deriv + deriv.T) * 0.5)
    return (rho_blocks, tangent_blocks)


def charge_qfi(d, rho_blocks, tangent_blocks, rel_tol=1e-10):
    eigdata = []
    global_max = 0.0
    min_eig = np.inf
    for rho, x in zip(rho_blocks, tangent_blocks):
        w, v = eigh(rho, check_finite=False, driver='evr')
        min_eig = min(min_eig, float(w[0]))
        global_max = max(global_max, float(w[-1]))
        eigdata.append((w, v.T @ x @ v))
    cutoff = rel_tol * max(global_max, 1.0 / (d * d))
    fi = 0.0
    omitted_num = 0.0
    positive_neg_eigs = 0
    for w, x_eig in eigdata:
        positive_neg_eigs += int(np.count_nonzero(w < -1e-12))
        w = np.maximum(w, 0.0)
        denom = w[:, None] + w[None, :]
        good = denom > cutoff
        omitted_num += int(np.size(denom) - np.count_nonzero(good))
        fi += 2.0 * np.sum((x_eig * x_eig)[good] / denom[good])
    return dict(F_pair=float(fi), cutoff=float(cutoff), min_eig=float(min_eig), omitted_terms=omitted_num, negative_eigs=positive_neg_eigs)


def rfs_from_cov(d, delta, pitch_m, lag_m, cov_func):
    m = np.arange(1, d, dtype=float)
    C0 = float(cov_func(0.0))
    Dphi = lambda r: 2 * (C0 - np.asarray(cov_func(r)))
    K = Dphi(m * pitch_m) + Dphi(lag_m) - 0.5 * (Dphi(m * pitch_m + lag_m) + Dphi(m * pitch_m - lag_m))
    g = np.exp(-np.maximum(K, 0.0))
    alpha = np.cos(0.5 * delta * m) ** 2
    derivative = -0.5 * m * np.sin(delta * m)
    contrast = g * alpha
    denom = np.maximum(1 - contrast ** 2, 1e-300)
    F = 2 / (d * d) * np.sum((d - m) * g * g * derivative * derivative / denom)
    return float(F)



# =========================================================================
# Classical reference-learning (Kalman) model
# =========================================================================

def gamma_res_modewise_vontrees_1d_EXACTBETA_CANONICAL_FIXED(d, *, cov_func, Delta, p1=0.5, dx=1.0, delta_t=1.0, v_pix_per_time=1.0, beta_scale=1.0, beta_budget='renorm_sum', rho_clip=0.01, ridge=1e-12, eig_tol=1e-12, return_full=True):
    d = int(d)
    dx = float(dx)
    delta_t = float(delta_t)
    v = float(v_pix_per_time) * dx
    s = v * delta_t
    x = np.arange(d, dtype=float) * dx
    K0_full = cov_func(np.abs(x[:, None] - x[None, :]))
    Ks_full = cov_func(np.abs(x[:, None] + s - x[None, :]))
    K0_full = 0.5 * (K0_full + K0_full.T)
    Ks_full = 0.5 * (Ks_full + Ks_full.T)
    one = np.ones((d, 1), dtype=float)
    Pp = np.eye(d) - one @ one.T / float(d)
    K0 = Pp @ K0_full @ Pp
    Ks = Pp @ Ks_full @ Pp
    K0 = 0.5 * (K0 + K0.T)
    Ks = 0.5 * (Ks + Ks.T)
    w, B = np.linalg.eigh(K0 + ridge * np.eye(d))
    keep = w > max(eig_tol, ridge)
    w = w[keep]
    B = B[:, keep]
    r = int(w.size)
    if r <= 0:
        raise ValueError('K0 has no support after piston projection; check cov_func / units.')
    sqrtw = np.sqrt(w)
    invsqrtw = 1.0 / sqrtw
    Ks_r = B.T @ Ks @ B
    M_r = invsqrtw[:, None] * Ks_r * invsqrtw[None, :]
    M_r = 0.5 * (M_r + M_r.T)
    rho_eigs, V_r = np.linalg.eigh(M_r)
    rho_k = np.clip(rho_eigs, -rho_clip, rho_clip)
    G = B * sqrtw[None, :] @ V_r
    rho_opt = build_two_source_plane_wave_rho(d, Delta, p=p1)
    Fphi = qfi_matrix_for_diagonal_phases(rho_opt, gauge='sum0')
    beta_k = np.einsum('ik,ij,jk->k', G, Fphi, G)
    beta_k = np.maximum(beta_k, 0.0)
    B_tot = 4.0 * float(beta_scale) * (1.0 - 1.0 / d)
    if beta_budget == 'renorm_sum':
        sumb = float(np.sum(beta_k))
        if sumb > 0:
            beta_k = beta_k * (B_tot / sumb)
    elif beta_budget == 'none':
        pass
    else:
        raise ValueError("beta_budget must be 'none' or 'renorm_sum'")
    Ppred = np.zeros(r)
    Ppost = np.zeros(r)
    for k in range(r):
        Ppred[k], Ppost[k] = _steady_state_scalar_kalman_ar1_pred_post_unitvar(rho_k[k], beta_k[k])
    Sigma_pred = G * Ppred[None, :] @ G.T
    Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred.T)
    Sigma_prior = K0
    if return_full:
        Sigma_post = G * Ppost[None, :] @ G.T
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    def cov_to_var_gamma(Sigma):
        dS = np.diag(Sigma)
        Var = dS[:, None] + dS[None, :] - 2.0 * Sigma
        Var = np.maximum(Var, 0.0)
        Gam = np.exp(-0.5 * Var)
        return (Var, Gam)
    Var_res, Gamma = cov_to_var_gamma(Sigma_pred)
    Var_prior, Gamma_prior = cov_to_var_gamma(Sigma_prior)
    out = dict(Gamma=Gamma, Var_res=Var_res, Gamma_prior=Gamma_prior, Var_prior=Var_prior, rho_k=rho_k, beta_k=beta_k, beta_total=float(np.sum(beta_k)), B_tot=B_tot, s_shift=s, d=d, r=r, beta_budget=beta_budget)
    if return_full:
        Var_post, Gamma_oracle = cov_to_var_gamma(Sigma_post)
        out.update(Gamma_oracle=Gamma_oracle, Var_oracle=Var_post, Ppred_k=Ppred, Ppost_k=Ppost, K0=K0_full, Ks=Ks_full, K0_proj=K0, Ks_proj=Ks, Fphi_exact_sum0=Fphi, Sigma_pred=Sigma_pred, Sigma_post=Sigma_post, Sigma_prior=Sigma_prior)
    return out


def build_two_source_plane_wave_rho(d, Delta, p=0.5):
    d = int(d)
    j = np.arange(d, dtype=float)
    phase = np.exp(2j * np.pi * j / d)
    zD = np.exp(2j * np.pi * j * float(Delta) / d)
    plus = np.ones(d, dtype=complex) / np.sqrt(d)
    psi_p = zD * plus
    psi_m = np.conj(zD) * plus
    rho = p * np.outer(psi_p, psi_p.conj()) + (1 - p) * np.outer(psi_m, psi_m.conj())
    rho = 0.5 * (rho + rho.conj().T)
    return rho


def qfi_matrix_for_diagonal_phases(rho, *, eps=1e-12, gauge='sum0'):
    rho = 0.5 * (rho + rho.conj().T)
    d = rho.shape[0]
    p, V = np.linalg.eigh(rho)
    p = np.maximum(p, 0.0)
    Vc = V.conj()
    F = np.zeros((d, d), dtype=float)
    pa = p[:, None]
    pb = p[None, :]
    denom = pa + pb
    mask = denom > eps
    coef = pb - pa
    for j in range(d):
        Mj = Vc[j, :, None] * V[j, None, :]
        drj = 1j * coef * Mj
        for k in range(j, d):
            Mk = Vc[k, :, None] * V[k, None, :]
            drk = 1j * coef * Mk
            term = 2.0 * (drj * np.conj(drk))
            val = np.sum(np.real(term[mask] / denom[mask]))
            F[j, k] = val
            F[k, j] = val
    if gauge is None:
        return F
    if gauge == 'sum0':
        P = np.eye(d) - np.ones((d, d)) / d
        return P @ F @ P
    if gauge == 'phi0':
        return F[1:, 1:]
    raise ValueError("gauge must be None, 'sum0', or 'phi0'")


def _steady_state_scalar_kalman_ar1_pred_post_unitvar(rho, beta, tol=1e-13, max_iter=100000):
    rho = float(np.clip(rho, -0.999999999999, 0.999999999999))
    beta = max(0.0, float(beta))
    rho2 = rho * rho
    Q = max(0.0, 1.0 - rho2)
    Ppost = 1.0
    for _ in range(max_iter):
        Ppred = rho2 * Ppost + Q
        if beta > 0.0:
            Ppost_new = 1.0 / (1.0 / max(Ppred, tol) + beta)
        else:
            Ppost_new = Ppred
        if abs(Ppost_new - Ppost) <= tol * (1.0 + abs(Ppost)):
            Ppost = Ppost_new
            break
        Ppost = Ppost_new
    Ppred = rho2 * Ppost + Q
    if beta > 0.0:
        Ppost = 1.0 / (1.0 / max(Ppred, tol) + beta)
    else:
        Ppost = Ppred
    return (Ppred, Ppost)


def small_delta_weight_x(x):
    x = np.asarray(x, dtype=float)
    w = 30.0 * x ** 4 * (1.0 - x)
    w[(x < 0.0) | (x > 1.0)] = 0.0
    return w


def ideal_qfi_small_delta_from_d(d, *, dx=1.0, p1=0.5):
    d = np.asarray(d, dtype=float)
    p2 = 1.0 - float(p1)
    sigma2 = float(dx) ** 2 * (d ** 2 - 1.0) / 12.0
    return 4.0 * float(p1) * p2 * sigma2


def baseline_average_from_pairwise_matrix(M, *, dx=1.0):
    M = np.asarray(M, dtype=float)
    d = M.shape[0]
    r_vals = np.arange(d, dtype=float) * float(dx)
    Kbar = np.empty(d, dtype=float)
    for m in range(d):
        diag = np.diag(M, k=m)
        Kbar[m] = np.mean(diag) if diag.size > 0 else np.nan
    return (r_vals, Kbar)


def Aeff_smallDelta_1c_kalman_vK(*, d, t, Delta_ref, dx, cov_func, v=1.0, p1=0.5, nx=2000, kalman_kwargs=None, kalman_mode='approx', x_eff=5.0 / 7.0):
    if 'gamma_res_modewise_vontrees_1d_EXACTBETA_CANONICAL_FIXED' not in globals():
        raise RuntimeError('gamma_res_modewise_vontrees_1d_EXACTBETA_CANONICAL_FIXED is not defined.')
    if kalman_kwargs is None:
        kalman_kwargs = {}
    out = gamma_res_modewise_vontrees_1d_EXACTBETA_CANONICAL_FIXED(d=int(d), cov_func=cov_func, Delta=float(Delta_ref), p1=float(p1), dx=float(dx), delta_t=float(t), v_pix_per_time=float(v), return_full=False, **kalman_kwargs)
    Var_res = np.asarray(out['Var_res'], dtype=float)
    r_vals, Kbar = baseline_average_from_pairwise_matrix(Var_res, dx=dx)
    D = int(d) * float(dx)
    if kalman_mode == 'approx':
        r_eff = float(x_eff) * D
        K_eff = np.interp(r_eff, r_vals, Kbar, left=Kbar[0], right=Kbar[-1])
        penalty = float(np.exp(-min(K_eff, 700.0)))
        penalty = np.clip(penalty, 1e-300, 1.0)
        Aeff = -np.log(penalty)
        return {'Aeff': Aeff, 'penalty': penalty, 'r_eff': r_eff, 'K_eff': K_eff, 'Var_res': Var_res, 'r_vals': r_vals, 'Kbar': Kbar, 'mode': 'approx'}
    if kalman_mode == 'full':
        x = np.linspace(0.0, 1.0, int(nx))
        w = small_delta_weight_x(x)
        r = D * x
        K_interp = np.interp(r, r_vals, Kbar, left=Kbar[0], right=Kbar[-1])
        penalty = np.trapz(w * np.exp(-np.minimum(K_interp, 700.0)), x)
        penalty = np.clip(penalty, 1e-300, 1.0)
        Aeff = -np.log(penalty)
        return {'Aeff': Aeff, 'penalty': penalty, 'x': x, 'r': r, 'K_ad': K_interp, 'Var_res': Var_res, 'r_vals': r_vals, 'Kbar': Kbar, 'mode': 'full'}
    raise ValueError("kalman_mode must be 'approx' or 'full'")


def full_separation_to_legacy_plane_wave_Delta(*, separation, d, dx=1.0):
    return float(separation) * int(d) * float(dx) / (4.0 * np.pi)


def old_smallDelta_kalman_mean_cadence(*, d, dx, mean_u, cov_func, separation, v_code_per_time, p1=0.5, kalman_kwargs=None, kalman_mode='full', nx=250, x_eff=5.0 / 7.0, return_details=False):
    d = int(d)
    dx = float(dx)
    mean_u = float(mean_u)
    v_code_per_time = float(v_code_per_time)
    if v_code_per_time <= 0:
        raise ValueError('v_code_per_time must be positive.')
    if kalman_kwargs is None:
        kalman_kwargs = {}
    mean_tau = mean_u / v_code_per_time
    v_pix_per_time = v_code_per_time / dx
    Delta_legacy = full_separation_to_legacy_plane_wave_Delta(separation=separation, d=d, dx=dx)
    outk = Aeff_smallDelta_1c_kalman_vK(d=d, t=mean_tau, Delta_ref=Delta_legacy, dx=dx, cov_func=cov_func, v=v_pix_per_time, p1=p1, nx=nx, kalman_kwargs=kalman_kwargs, kalman_mode=kalman_mode, x_eff=x_eff)
    F0 = float(ideal_qfi_small_delta_from_d(d=d, dx=dx, p1=p1))
    F = F0 * float(outk['penalty'])
    if not return_details:
        return float(F)
    result = dict(outk)
    result.update({'F': float(F), 'F0': F0, 'mean_u': mean_u, 'mean_tau': mean_tau, 'rate': 1.0 / mean_tau, 'separation_full': float(separation), 'Delta_legacy': float(Delta_legacy), 'v_code_per_time': v_code_per_time, 'v_pix_per_time': v_pix_per_time, 'cadence_model': 'mean-cadence'})
    return result



# =========================================================================
# Exact two-photon product-Fourier likelihood and Fisher information
# =========================================================================

def exact_full_product_ft_fi_two_copy_discrete(*, d, dx, u, separation, cov_func, return_details=False, eps=1e-300):
    d = int(d)
    dx = float(dx)
    u = float(u)
    separation = float(separation)
    if d < 2:
        raise ValueError('d must be >= 2')
    offsets = np.arange(-(d - 1), d, dtype=int)
    noff = len(offsets)
    r_off = offsets.astype(float) * dx
    mu = np.cos(0.5 * separation * r_off)
    dmu = -0.5 * r_off * np.sin(0.5 * separation * r_off)
    C0 = float(np.asarray(cov_func(0.0)))
    C_off = np.asarray(cov_func(np.abs(r_off)), dtype=float)
    D_off = 2.0 * (C0 - C_off)
    qmax = 3 * (d - 1)
    q_grid = np.arange(-qmax, qmax + 1, dtype=int)
    C_shift = np.asarray(cov_func(np.abs(q_grid.astype(float) * dx + u)), dtype=float)
    q0 = qmax

    def C_lookup(q):
        """
        q can be scalar or integer ndarray.
        Returns C(|q*dx + u|).
        """
        return C_shift[np.asarray(q, dtype=int) + q0]
    G = np.zeros((d, d), dtype=np.float64)
    dG = np.zeros((d, d), dtype=np.float64)
    b = offsets
    L_lo = np.maximum(0, b)
    L_hi = np.minimum(d - 1, d - 1 + b)
    c = offsets
    b2 = b[:, None]
    c2 = c[None, :]
    for ia, a in enumerate(offsets):
        J_lo = max(0, int(a))
        J_hi = min(d - 1, d - 1 + int(a))
        low = np.maximum(J_lo, L_lo[:, None] + c2)
        high = np.minimum(J_hi, L_hi[:, None] + c2)
        multiplicity = np.maximum(high - low + 1, 0).astype(float)
        cross = C_lookup(c2) - C_lookup(c2 + b2) - C_lookup(c2 - a) + C_lookup(c2 - a + b2)
        half_var = 0.5 * D_off[ia] + 0.5 * D_off[:, None] + cross
        half_var = np.maximum(half_var, 0.0)
        atmosphere = np.exp(-np.minimum(half_var, 700.0))
        atmospheric_sum = np.sum(multiplicity * atmosphere, axis=1)
        H_ab = mu[ia] * mu / float(d ** 2) * atmospheric_sum
        dH_ab = (dmu[ia] * mu + mu[ia] * dmu) / float(d ** 2) * atmospheric_sum
        a_mod = int(a % d)
        b_mod = np.mod(b, d)
        np.add.at(G[a_mod], b_mod, H_ab)
        np.add.at(dG[a_mod], b_mod, dH_ab)
    P = np.fft.fft2(G).real / float(d ** 2)
    dP = np.fft.fft2(dG).real / float(d ** 2)
    P_sum = float(np.sum(P))
    P_min = float(np.min(P))
    P_safe = np.maximum(P, eps)
    Fpq = dP ** 2 / P_safe
    F = float(np.sum(Fpq))
    if not return_details:
        return F
    return {'F': F, 'P': P, 'dP': dP, 'Fpq': Fpq, 'P_sum': P_sum, 'P_min': P_min, 'G': G, 'dG': dG, 'offsets': offsets, 'r_offsets': r_off, 'mu': mu, 'dmu': dmu, 'D_off': D_off}



# =========================================================================
# Total-index-twirled mixed-SLD/PPT upper certificate
# =========================================================================

def sym(a):
    return 0.5 * (np.asarray(a) + np.asarray(a).T)


@dataclass
class ChargeGeometry:
    d: int
    sums: list
    diffs: list
    s_from_d: list
    d_from_s: list
    s_from_d_flat: list
    d_from_s_flat: list

    @staticmethod
    def create(d):
        d = int(d)
        sums = []
        diffs = []
        for s in range(2 * d - 1):
            sums.append([(i, s - i) for i in range(d) if 0 <= s - i < d])
        for t in range(-(d - 1), d):
            diffs.append([(i, i - t) for i in range(d) if 0 <= i - t < d])
        sidx = {ij: (s, k) for s, group in enumerate(sums) for k, ij in enumerate(group)}
        didx = {ij: (t, k) for t, group in enumerate(diffs) for k, ij in enumerate(group)}
        s_from_d = []
        for group in diffs:
            si = np.zeros((len(group), len(group)), dtype=np.int32)
            ii = np.zeros_like(si)
            jj = np.zeros_like(si)
            for a, (i, j) in enumerate(group):
                for b, (k, l) in enumerate(group):
                    ss, rr = sidx[i, l]
                    ss2, cc = sidx[k, j]
                    assert ss == ss2
                    si[a, b] = ss
                    ii[a, b] = rr
                    jj[a, b] = cc
            s_from_d.append((si, ii, jj))
        d_from_s = []
        for group in sums:
            di = np.zeros((len(group), len(group)), dtype=np.int32)
            ii = np.zeros_like(di)
            jj = np.zeros_like(di)
            for a, (i, j) in enumerate(group):
                for b, (k, l) in enumerate(group):
                    dd, rr = didx[i, l]
                    dd2, cc = didx[k, j]
                    assert dd == dd2
                    di[a, b] = dd
                    ii[a, b] = rr
                    jj[a, b] = cc
            d_from_s.append((di, ii, jj))

        def flatten_maps(maps, src_groups):
            dims = np.asarray([len(group) for group in src_groups], dtype=np.int64)
            offsets = np.cumsum([0] + [len(group) ** 2 for group in src_groups])
            return [offsets[si] + ii * dims[si] + jj for si, ii, jj in maps]
        return ChargeGeometry(d, sums, diffs, s_from_d, d_from_s, flatten_maps(s_from_d, sums), flatten_maps(d_from_s, diffs))

    @staticmethod
    def _map(blocks, indices):
        source = np.concatenate([x.ravel() for x in blocks])
        return [source[idx] for idx in indices]

    def sum_to_diff(self, x):
        return self._map(x, self.s_from_d_flat)

    def diff_to_sum(self, x):
        return self._map(x, self.d_from_s_flat)

    def from_dense(self, a, which='sum'):
        groups = self.sums if which == 'sum' else self.diffs
        return [sym(a[np.ix_([i * self.d + j for i, j in g], [i * self.d + j for i, j in g])]) for g in groups]

    def to_dense(self, blocks, which='sum'):
        n = self.d * self.d
        out = np.zeros((n, n))
        groups = self.sums if which == 'sum' else self.diffs
        for g, b in zip(groups, blocks):
            inds = [i * self.d + j for i, j in g]
            out[np.ix_(inds, inds)] = b
        return out

    def make_rho_x(self, delta, lag_m, pitch_m=A_REF, mode='physical', local=False):
        """Construct atmospheric rho and tangent on sum charge blocks, O(D^3) memory."""
        d = self.d
        x = pitch_m * np.arange(d, dtype=float)
        c0 = float(corr(0.0, mode))

        def C(z):
            return corr(z, mode)
        rh = []
        tg = []
        for g in self.sums:
            a = np.array([i for i, j in g])[:, None]
            b = np.array([j for i, j in g])[:, None]
            h = a.T
            k = b.T
            variance = 4 * c0 - 2 * C(x[a] - x[h]) - 2 * C(x[b] - x[k]) + 2 * (C(x[a] - x[b] + lag_m) - C(x[a] - x[k] + lag_m) - C(x[h] - x[b] + lag_m) + C(x[h] - x[k] + lag_m))
            base = np.exp(-0.5 * np.maximum(variance, 0.0)) / d ** 2
            z1 = (a - h) * pitch_m / A_REF
            z2 = (b - k) * pitch_m / A_REF
            assert np.max(np.abs(z1 + z2)) < 1e-11
            if local:
                R = base
                X = -delta * 0.5 * z1 * z1 * base
            else:
                R = base * np.cos(delta * z1 / 2) ** 2
                X = -0.5 * z1 * np.sin(delta * z1) * base
            rh.append(sym(R))
            tg.append(sym(X))
        return (rh, tg)


def min_eig(blocks):
    return float(min((np.linalg.eigvalsh(sym(b))[0] for b in blocks)))


def norm_sq(blocks):
    return float(sum((np.sum(b * b) for b in blocks)))


def trace_mul(a, b):
    return float(sum((np.sum(x * y) for x, y in zip(a, b))))


def baseline_split(geom, rho):
    q = [0.5 * a for a in geom.sum_to_diff(rho)]
    p = [0.5 * a for a in rho]
    return (p, q)


def metric_apply(geom, L, P, Q):
    one = [0.5 * (p @ l + l @ p) for p, l in zip(P, L)]
    ptL = geom.sum_to_diff(L)
    two_diff = [0.5 * (q @ l + l @ q) for q, l in zip(Q, ptL)]
    two_sum = geom.diff_to_sum(two_diff)
    return [a + b for a, b in zip(one, two_sum)]


def estimate_bound(geom, rho, X, P, Q, rtol=2e-11, maxiter=2500):
    assert len(P) == len(rho) and len(Q) == len(geom.diffs)
    minp = min_eig(P)
    minq = min_eig(Q)
    mismatch = max((float(np.max(np.abs(a - b - c))) for a, b, c in zip(rho, P, geom.diff_to_sum(Q))))
    if min(minp, minq) < -2e-12 or mismatch > 1e-10:
        raise ValueError(f'Not a feasible split: minP={minp} minQ={minq}, mismatch={mismatch}')
    offsets = np.cumsum([0] + [b.size for b in P])
    shapes = [b.shape for b in P]

    def unpack(z):
        return [z[offsets[k]:offsets[k + 1]].reshape(shape) for k, shape in enumerate(shapes)]

    def pack(b):
        return np.concatenate([x.ravel() for x in b])
    op = LinearOperator((offsets[-1], offsets[-1]), matvec=lambda z: pack(metric_apply(geom, unpack(z), P, Q)), dtype=float)
    rhs = pack(X)
    answer, info = cg(op, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
    L = unpack(answer)
    ml = metric_apply(geom, L, P, Q)
    R = [x - y for x, y in zip(X, ml)]
    lower = 2 * trace_mul(X, L) - trace_mul(L, ml)
    residual = np.sqrt(norm_sq(R))
    mu = minp + minq - 1e-12 * max(1e-15, max((np.linalg.norm(p, 2) for p in P)), max((np.linalg.norm(q, 2) for q in Q)))
    upper = lower + residual ** 2 / mu if mu > 0 else None
    return dict(corrected_upper=float(upper) if upper is not None else None, variational_lower=float(lower), cg_info=int(info), residual_relative=float(residual / max(np.sqrt(norm_sq(X)), 1e-300)), residual_norm=float(residual), minP=minp, minQ=minq, metric_floor=float(mu), split_equality_residual=float(mismatch), metric_dimension=int(offsets[-1]))



# =========================================================================
# Poisson arrival average of the interpolated fixed-lag FI
# =========================================================================

def _linear_segment_weights(z):
    z = np.asarray(z, dtype=float)
    w_sum = -np.expm1(-z)
    right = np.empty_like(z)
    small = z < 0.001
    zz = z[small]
    right[small] = zz / 2 - zz ** 2 / 3 + zz ** 3 / 8 - zz ** 4 / 30 + zz ** 5 / 144 - zz ** 6 / 840
    large = ~small
    zz = z[large]
    right[large] = (-np.expm1(-zz) - zz * np.exp(-zz)) / zz
    left = w_sum - right
    return (left, right)


def exponential_interpolation_weights(lag_m, mean_lag_m):
    u = np.asarray(lag_m, dtype=float)
    mu = np.atleast_1d(np.asarray(mean_lag_m, dtype=float))
    if len(u) < 2 or not np.all(np.isfinite(u)) or np.any(u < 0):
        raise ValueError('Need >=2 finite, nonnegative fixed-lag anchors')
    if np.any(np.diff(u) <= 0):
        raise ValueError('Fixed-lag anchors must be strictly increasing')
    if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
        raise ValueError('Poisson means must be finite and positive')
    W = np.zeros((len(mu), len(u)), dtype=float)
    W[:, 0] += -np.expm1(-u[0] / mu)
    for i in range(len(u) - 1):
        z = (u[i + 1] - u[i]) / mu
        w_left, w_right = _linear_segment_weights(z)
        decay = np.exp(-u[i] / mu)
        W[:, i] += decay * w_left
        W[:, i + 1] += decay * w_right
    W[:, -1] += np.exp(-u[-1] / mu)
    return W


# ===========================================================================
# Fresh five-curve FI evaluations and Poisson time/dimension scans
# ===========================================================================

@lru_cache(maxsize=16)
def _charge_geometry(d: int):
    return ChargeGeometry.create(d)


def kalman_fi_per_pair(d: int, delta: float, mean_lag_m: float,
                       *, nx: int = 250) -> float:
    """Original steady-state Kalman model at fixed *mean* cadence."""
    with threadpool_limits(limits=1):
        k = old_smallDelta_kalman_mean_cadence(
            d=int(d), dx=1., mean_u=float(mean_lag_m)/PITCH_M,
            cov_func=lambda x: corr(PITCH_M*np.asarray(x)),
            separation=float(delta), v_code_per_time=WIND_M_S/PITCH_M,
            p1=.5, kalman_kwargs=KALMAN_SETTINGS.copy(),
            kalman_mode='full', nx=nx, return_details=True)
    return 2.*float(k['F'])


@lru_cache(maxsize=256)
def _fixed_lag_fi_cached(d: int, delta: float, lag_m: float,
                         include_ppt: bool, cg_rtol: float,
                         cg_maxiter: int) -> tuple[float, float, float, float]:
    """Fresh fixed-lag FI, cached only in RAM during the present kernel run."""
    if d < 2 or delta <= 0 or lag_m < 0:
        raise ValueError('Invalid optical dimension, source separation or lag')
    with threadpool_limits(limits=1):
        c, cross = cached_covariance(d, PITCH_M, lag_m, corr)
        blocks, tangents = power_charge_blocks(d, delta, c, cross)
        power_qfi = float(charge_qfi(d, blocks, tangents)['F_pair'])
        rfs = float(rfs_from_cov(d, delta, PITCH_M, lag_m, corr))
        ft = exact_full_product_ft_fi_two_copy_discrete(
            d=d, dx=1., u=lag_m/PITCH_M, separation=delta,
            cov_func=lambda x: corr(PITCH_M*np.asarray(x)),
            return_details=True)
        if abs(float(ft['P_sum'])-1.) > 3e-8 or float(ft['P_min']) < -1e-10:
            raise ArithmeticError('Invalid full-state FT x FT outcome probabilities')
        bound = np.nan
        if include_ppt:
            geom = _charge_geometry(d)
            rho, tangent = geom.make_rho_x(delta, lag_m, pitch_m=PITCH_M)
            positive, pt_positive = baseline_split(geom, rho)
            cert = estimate_bound(geom, rho, tangent, positive, pt_positive,
                                  rtol=cg_rtol, maxiter=cg_maxiter)
            bound = cert['corrected_upper']
            if (cert['cg_info'] != 0 or bound is None or not np.isfinite(bound)
                    or cert['metric_floor'] <= 0):
                raise ArithmeticError(f'PPT certificate did not converge: {cert}')
    return power_qfi, rfs, float(ft['F']), float(bound)


def fixed_lag_fi(d: int, delta: float, lag_m: float, *,
                 include_ppt: bool = True, cg_rtol: float = 2e-11,
                 cg_maxiter: int = 3000) -> dict:
    """Compute all four conditional FI values at one physical photon-pair lag."""
    # Same canonicalization as the original backend for stable PPT numerics.
    lag_m = float(f'{float(lag_m):.12g}')
    a = _fixed_lag_fi_cached(int(d), float(delta), lag_m,
                             bool(include_ppt), float(cg_rtol), int(cg_maxiter))
    return dict(zip(POISSON_KEYS, a))


def _validate_anchor_grid(x):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)) or np.any(a <= 0) or np.any(np.diff(a) <= 0):
        raise ValueError('Provide at least two ascending positive lag anchors')
    return a


def binary_time_data(*, d: int = 128, rayleigh_fraction: float = 0.10,
                     delta_override: float | None = None,
                     lag_anchor_x=None, mean_x=None, include_ppt: bool = True,
                     kalman_nx: int = 250, verbose: bool = True) -> dict:
    """Poisson-averaged binary FI versus mean t/t_corr = mean u/L0.

    `lag_anchor_x` determines the expensive fixed-lag evaluations.
    `mean_x` specifies the plot points, integrated through those anchors.
    The original 10-anchor grid is the default, preserving the historical
    figure's piecewise-linear Poisson approximation. The mean axis defaults
    to 11 points, without introducing extra conditional QFI evaluations.
    Every plotted Kalman point is freshly evaluated at its own MEAN cadence.

    `delta_override=0.005` exactly reproduces the historical D=128 time
    source; leaving it None enforces 10% Rayleigh exactly at every D.
    """
    if lag_anchor_x is None:
        lag_anchor_x = np.geomspace(1e-5, 1e2, 10)
    anchors = _validate_anchor_grid(lag_anchor_x)
    means = np.asarray(np.geomspace(1e-5, 1e2, 11) if mean_x is None else mean_x, dtype=float)
    if means.ndim != 1 or means.size == 0 or np.any(means <= 0) or not np.all(np.isfinite(means)):
        raise ValueError('Invalid mean interarrival-time grid')
    delta = (rayleigh_separation(d, rayleigh_fraction)
             if delta_override is None else float(delta_override))
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Source separation must be finite and positive')
    samples = []
    for i, x in enumerate(anchors, 1):
        if verbose:
            print(f'[Time {i}/{len(anchors)}] D={d}, u/L0={x:.4g}', flush=True)
        samples.append(fixed_lag_fi(d, delta, float(x)*L0_M,
                                    include_ppt=include_ppt))
    W = exponential_interpolation_weights(anchors*L0_M, means*L0_M)
    data = dict(D=int(d), Delta=float(delta),
                rayleigh_fraction=float(delta*d/(2*np.pi)),
                requested_rayleigh_fraction=float(rayleigh_fraction),
                mean_u_over_L0=means, fixed_u_over_L0=anchors,
                t_corr_s=L0_M/WIND_M_S)
    for key in POISSON_KEYS:
        y = np.asarray([s[key] for s in samples], dtype=float)
        data['Poisson_'+key] = W @ y if np.all(np.isfinite(y)) else np.full(len(means), np.nan)
    values=[]
    for i, x in enumerate(means, 1):
        if verbose and (i==1 or i==len(means) or i % max(1,len(means)//8)==0):
            print(f'[Time Kalman {i}/{len(means)}] mean u/L0={x:.4g}', flush=True)
        values.append(kalman_fi_per_pair(d, delta, float(x)*L0_M, nx=kalman_nx))
    data[KALMAN] = np.asarray(values, dtype=float)
    data['poisson_extrapolated_left_mass'] = -np.expm1(-anchors[0]/means)
    data['poisson_extrapolated_right_mass'] = np.exp(-anchors[-1]/means)
    return data


def binary_dimension_data(*, d_values=(8, 16, 32, 64, 128),
                          rayleigh_fraction: float = 0.10,
                          mean_u_over_L0: float = 1e-3,
                          lag_anchor_x=None, include_ppt: bool = True,
                          kalman_nx: int = 250, verbose: bool = True) -> dict:
    """Poisson-averaged binary FI versus D at the same mean photon-pair lag.

    Every D uses Delta(D)=2*pi*(Rayleigh fraction)/D.
    Kalman remains fixed-mean-cadence instead of a Poisson average.
    """
    ds=np.asarray(d_values, dtype=int)
    if ds.ndim != 1 or len(ds)==0 or np.any(ds<2) or np.any(np.diff(ds)<=0):
        raise ValueError('D values must be ascending integers >= 2')
    if lag_anchor_x is None:
        lag_anchor_x=np.geomspace(1e-5, 1.0, 10)
    anchors=_validate_anchor_grid(lag_anchor_x)
    if mean_u_over_L0 <= 0:
        raise ValueError('Mean lag must be >0')
    mean_lag_m=mean_u_over_L0*L0_M
    weights=exponential_interpolation_weights(anchors*L0_M, [mean_lag_m])[0]
    data=dict(D=ds, regime=np.full(len(ds),'rayleigh',dtype='U8'),
              Delta=np.asarray([rayleigh_separation(int(d),rayleigh_fraction) for d in ds]),
              Rayleigh_fraction=float(rayleigh_fraction),
              mean_u_over_L0=float(mean_u_over_L0),
              fixed_u_over_L0=anchors)
    columns={key:[] for key in FI_KEYS}
    for d,delta in zip(ds,data['Delta']):
        data_D=[]
        for i,x in enumerate(anchors,1):
            if verbose:
                print(f'[Dimension D={d}, {i}/{len(anchors)}] u/L0={x:.4g}',flush=True)
            data_D.append(fixed_lag_fi(int(d),float(delta),float(x)*L0_M,
                                       include_ppt=include_ppt))
        for key in POISSON_KEYS:
            vals=np.asarray([r[key] for r in data_D],dtype=float)
            columns[key].append(float(weights@vals) if np.all(np.isfinite(vals)) else np.nan)
        columns[KALMAN].append(kalman_fi_per_pair(int(d),float(delta),mean_lag_m,nx=kalman_nx))
    for key,value in columns.items():
        data[key]=np.asarray(value,dtype=float)
    return data
