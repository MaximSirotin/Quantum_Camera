from collections import deque

import numpy as np
import matplotlib.pyplot as plt


# ====================================================================
# FOURIER IMAGE AND SINGLE-PHOTON STATE
# ====================================================================

def phi_from_P_2d(P):
    P = np.asarray(P, float)
    d = P.shape[0]
    assert P.shape == (d, d)
    rx, ry = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
    theta_factor = 2.0 * np.pi / d
    phi = np.zeros((d, d), dtype=complex)
    for mx in range(d):
        for my in range(d):
            phase = np.exp(1j * theta_factor * (mx * rx + my * ry))
            phi[mx, my] = np.sum(P * phase)
    return phi


def P_from_phi(phi):
    d = phi.shape[0]
    P = np.fft.ifft2(np.conjugate(phi)).real
    P[P < 0] = 0.0
    s = P.sum()
    if s > 0:
        P /= s
    else:
        P[:, :] = 1.0 / (d * d)
    return P


def analytic_chis_from_phi(phi):
    phi = np.asarray(phi, complex)
    d = phi.shape[0]
    assert phi.shape == (d, d)
    alpha = np.abs(phi) ** 2
    chi_plain = alpha / d ** 2
    phi_xplus = np.roll(phi, -1, axis=0)
    phi_yplus = np.roll(phi, -1, axis=1)
    alpha_xplus = np.abs(phi_xplus) ** 2
    alpha_yplus = np.abs(phi_yplus) ** 2
    z_x = phi * np.conjugate(phi_xplus)
    z_y = phi * np.conjugate(phi_yplus)
    chiX_x = (alpha + alpha_xplus + 2 * z_x.real) / (2 * d ** 2)
    chiY_x = (alpha + alpha_xplus + 2 * z_x.imag) / (2 * d ** 2)
    chiX_y = (alpha + alpha_yplus + 2 * z_y.real) / (2 * d ** 2)
    chiY_y = (alpha + alpha_yplus + 2 * z_y.imag) / (2 * d ** 2)
    return (chi_plain, chiX_x, chiY_x, chiX_y, chiY_y)


def rho1_from_phi(phi):
    d = phi.shape[0]
    H = d * d
    rho1 = np.zeros((H, H), dtype=complex)

    def idx(kx, ky):
        return kx * d + ky
    for kx in range(d):
        for ky in range(d):
            for lx in range(d):
                for ly in range(d):
                    mx = (lx - kx) % d
                    my = (ly - ky) % d
                    rho1[idx(kx, ky), idx(lx, ly)] = phi[mx, my] / d ** 2
    return rho1


def rho1_from_phi_with_phase_screen(phi, phase_screen):
    phi = np.asarray(phi, complex)
    phase_screen = np.asarray(phase_screen, float)
    d = phi.shape[0]
    assert phi.shape == (d, d)
    assert phase_screen.shape == (d, d)
    rho1_plain = rho1_from_phi(phi)
    H = d * d
    D = np.zeros((H, H), dtype=complex)

    def idx_k(kx, ky):
        return kx * d + ky
    for kx in range(d):
        for ky in range(d):
            i = idx_k(kx, ky)
            D[i, i] = np.exp(1j * phase_screen[kx, ky])
    rho1_turb = D @ rho1_plain @ D.conj().T
    return rho1_turb


def rho2_from_rho1(rho1):
    return np.kron(rho1, rho1)



# ====================================================================
# TURBULENCE AND CLASSICAL SPECKLE
# ====================================================================

def kolmogorov_phase_screen(N, r0_pix, delta=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    fx = np.fft.fftfreq(N, d=delta)
    fy = np.fft.fftfreq(N, d=delta)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    k = np.sqrt(FX ** 2 + FY ** 2) + 1e-10
    psd_phi = 0.023 * r0_pix ** (-5.0 / 3.0) * k ** (-11.0 / 3.0)
    cn_real = rng.normal(size=(N, N))
    cn_imag = rng.normal(size=(N, N))
    cn = (cn_real + 1j * cn_imag) / np.sqrt(2.0)
    phi_ft = cn * np.sqrt(psd_phi) * (delta * N)
    phi_ft[0, 0] = 0.0
    phi = np.fft.ifft2(phi_ft).real
    phi -= phi.mean()
    return phi


def speckle_image_from_phase_and_P(P_true, phase_screen):
    P_true = np.asarray(P_true, float)
    phase_screen = np.asarray(phase_screen, float)
    d = P_true.shape[0]
    assert P_true.shape == (d, d)
    assert phase_screen.shape == (d, d)
    A = np.exp(1j * phase_screen)
    field_img = np.fft.fft2(A)
    PSF = np.abs(field_img) ** 2
    PSF /= PSF.sum()
    F_P = np.fft.fft2(P_true)
    F_PSF = np.fft.fft2(PSF)
    I_speckle = np.fft.ifft2(F_P * F_PSF).real
    I_speckle[I_speckle < 0] = 0.0
    s = I_speckle.sum()
    if s > 0:
        I_speckle /= s
    else:
        I_speckle[:, :] = 1.0 / (d * d)
    return I_speckle



# ====================================================================
# EXACT BLOCK-SWAP POVM PROBABILITIES ON THE DENSE TWO-COPY STATE
# ====================================================================

def _idx_site(d, x, y):
    return x * d + y


def block_swap_probs_plain_global_2d_fast(rho2, d):
    D2 = rho2.shape[0]
    D = int(round(np.sqrt(D2)))
    assert D * D == D2
    assert D == d * d
    p_plus = np.zeros((d, d), dtype=float)
    p_minus = np.zeros((d, d), dtype=float)
    c_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    c_minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
    for mx in range(d):
        for my in range(d):
            if mx == 0 and my == 0:
                continue
            p_p = 0.0
            p_m = 0.0
            for x in range(d):
                for y in range(d):
                    x2 = (x + mx) % d
                    y2 = (y + my) % d
                    a = _idx_site(d, x, y)
                    b = _idx_site(d, x2, y2)
                    i_ab = a * D + b
                    i_ba = b * D + a
                    idxs = [i_ab, i_ba]
                    sub = rho2[np.ix_(idxs, idxs)]
                    p_p += 0.5 * np.real(np.conjugate(c_plus) @ (sub @ c_plus))
                    p_m += 0.5 * np.real(np.conjugate(c_minus) @ (sub @ c_minus))
            p_plus[mx, my] = p_p
            p_minus[mx, my] = p_m
    return (p_plus, p_minus)


def block_swap_probs_rotated_X_global_2d_fast(rho2, d, direction='x'):
    D2 = rho2.shape[0]
    D = int(round(np.sqrt(D2)))
    assert D * D == D2
    assert D == d * d
    p_plus_X = np.zeros((d, d), dtype=float)
    p_minus_X = np.zeros((d, d), dtype=float)
    if direction == 'x':
        dx, dy = (1, 0)
    elif direction == 'y':
        dx, dy = (0, 1)
    else:
        raise ValueError("direction must be 'x' or 'y'")
    c_plus_generic = np.array([1, 1, 1, 1], dtype=complex) / 2.0
    c_minus_generic = np.array([1, -1, 1, -1], dtype=complex) / 2.0
    c_plus_anchor = np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2.0)
    c_minus_anchor = np.array([1, 0, -1, 0], dtype=complex) / np.sqrt(2.0)
    for mx in range(d):
        for my in range(d):
            nx = (mx + dx) % d
            ny = (my + dy) % d
            if (nx == 0 and ny == 0) and (not (mx == 0 and my == 0)):
                continue
            if mx == 0 and my == 0:
                c_plus = c_plus_anchor
                c_minus = c_minus_anchor
            else:
                c_plus = c_plus_generic
                c_minus = c_minus_generic
            p_p = 0.0
            p_m = 0.0
            for x in range(d):
                for y in range(d):
                    x2 = (x + mx) % d
                    y2 = (y + my) % d
                    a1 = _idx_site(d, x, y)
                    b1 = _idx_site(d, x2, y2)
                    i1 = a1 * D + b1
                    i2 = b1 * D + a1
                    x3 = (x + nx) % d
                    y3 = (y + ny) % d
                    a3 = _idx_site(d, x, y)
                    b3 = _idx_site(d, x3, y3)
                    i3 = a3 * D + b3
                    i4 = b3 * D + a3
                    idxs = [i1, i2, i3, i4]
                    sub = rho2[np.ix_(idxs, idxs)]
                    p_p += 0.5 * np.real(np.conjugate(c_plus) @ (sub @ c_plus))
                    p_m += 0.5 * np.real(np.conjugate(c_minus) @ (sub @ c_minus))
            p_plus_X[mx, my] = p_p
            p_minus_X[mx, my] = p_m
    return (p_plus_X, p_minus_X)


def block_swap_probs_rotated_Y_global_2d_fast(rho2, d, direction='x'):
    D2 = rho2.shape[0]
    D = int(round(np.sqrt(D2)))
    assert D * D == D2
    assert D == d * d
    p_plus_Y = np.zeros((d, d), dtype=float)
    p_minus_Y = np.zeros((d, d), dtype=float)
    if direction == 'x':
        dx, dy = (1, 0)
    elif direction == 'y':
        dx, dy = (0, 1)
    else:
        raise ValueError("direction must be 'x' or 'y'")
    c_plus_generic = np.array([1, 1, 1j, 1j], dtype=complex) / 2.0
    c_minus_generic = np.array([1, -1, 1j, -1j], dtype=complex) / 2.0
    c_plus_anchor = np.array([1, 0, 1j, 0], dtype=complex) / np.sqrt(2.0)
    c_minus_anchor = np.array([1, 0, -1j, 0], dtype=complex) / np.sqrt(2.0)
    for mx in range(d):
        for my in range(d):
            nx = (mx + dx) % d
            ny = (my + dy) % d
            if (nx == 0 and ny == 0) and (not (mx == 0 and my == 0)):
                continue
            if mx == 0 and my == 0:
                c_plus = c_plus_anchor
                c_minus = c_minus_anchor
            else:
                c_plus = c_plus_generic
                c_minus = c_minus_generic
            p_p = 0.0
            p_m = 0.0
            for x in range(d):
                for y in range(d):
                    x2 = (x + mx) % d
                    y2 = (y + my) % d
                    a1 = _idx_site(d, x, y)
                    b1 = _idx_site(d, x2, y2)
                    i1 = a1 * D + b1
                    i2 = b1 * D + a1
                    x3 = (x + nx) % d
                    y3 = (y + ny) % d
                    a3 = _idx_site(d, x, y)
                    b3 = _idx_site(d, x3, y3)
                    i3 = a3 * D + b3
                    i4 = b3 * D + a3
                    idxs = [i1, i2, i3, i4]
                    sub = rho2[np.ix_(idxs, idxs)]
                    p_p += 0.5 * np.real(np.conjugate(c_plus) @ (sub @ c_plus))
                    p_m += 0.5 * np.real(np.conjugate(c_minus) @ (sub @ c_minus))
            p_plus_Y[mx, my] = p_p
            p_minus_Y[mx, my] = p_m
    return (p_plus_Y, p_minus_Y)



# ====================================================================
# PHASE CHAINING, ANCHORS, AND RECONSTRUCTION
# ====================================================================

def reconstruct_phi_from_chis_2d_full(chi_plain, chiX_x, chiY_x, chiX_y, chiY_y, eps=1e-12):
    chi_plain = np.asarray(chi_plain, float)
    chiX_x = np.asarray(chiX_x, float)
    chiY_x = np.asarray(chiY_x, float)
    chiX_y = np.asarray(chiX_y, float)
    chiY_y = np.asarray(chiY_y, float)
    d = chi_plain.shape[0]
    assert chi_plain.shape == (d, d)
    alpha = d ** 2 * chi_plain
    phi_abs = np.sqrt(np.maximum(alpha, 0.0))
    phi_abs[0, 0] = 1.0
    alpha_xplus = np.roll(alpha, -1, axis=0)
    alpha_yplus = np.roll(alpha, -1, axis=1)
    Re_z_x = (2 * d ** 2 * chiX_x - alpha - alpha_xplus) / 2.0
    Im_z_x = (2 * d ** 2 * chiY_x - alpha - alpha_xplus) / 2.0
    Re_z_y = (2 * d ** 2 * chiX_y - alpha - alpha_yplus) / 2.0
    Im_z_y = (2 * d ** 2 * chiY_y - alpha - alpha_yplus) / 2.0
    z_x = Re_z_x + 1j * Im_z_x
    z_y = Re_z_y + 1j * Im_z_y
    z_y[0, 0] *= -1.0
    valid_x = np.ones((d, d), dtype=bool)
    valid_y = np.ones((d, d), dtype=bool)
    for mx in range(d):
        for my in range(d):
            nx = (mx + 1) % d
            ny = (my + 1) % d
            if phi_abs[mx, my] < eps or phi_abs[nx, my] < eps:
                valid_x[mx, my] = False
            if phi_abs[mx, my] < eps or phi_abs[mx, ny] < eps:
                valid_y[mx, my] = False
    valid_x[0, 0] = False
    valid_y[0, 0] = False
    valid_x[d - 1, 0] = False
    valid_y[0, d - 1] = False
    valid_x &= np.isfinite(z_x)
    valid_y &= np.isfinite(z_y)
    flat_idx = np.argsort(phi_abs.ravel())[::-1]
    root = None
    for idx in flat_idx:
        mx0 = idx // d
        my0 = idx % d
        if (mx0, my0) == (0, 0):
            continue
        if phi_abs[mx0, my0] < eps:
            continue
        if valid_x[mx0, my0] or valid_y[mx0, my0]:
            root = (mx0, my0)
            break
    if root is None:
        phi = phi_abs.astype(complex)
        phi[0, 0] = 1.0 + 0j
        return phi
    mx0, my0 = root
    phi = np.zeros((d, d), dtype=complex)
    visited = np.zeros((d, d), dtype=bool)
    phi[mx0, my0] = phi_abs[mx0, my0]
    visited[mx0, my0] = True
    q = deque([(mx0, my0)])
    print('valid_x(d-1,0) =', valid_x[d - 1, 0], ' valid_y(0,d-1) =', valid_y[0, d - 1], ' valid(0,0) =', valid_x[0, 0], valid_y[0, 0])
    while q:
        mx, my = q.popleft()
        phi_m = phi[mx, my]
        if abs(phi_m) < eps:
            continue
        if valid_x[mx, my]:
            nx, ny = ((mx + 1) % d, my)
            if not visited[nx, ny] and phi_abs[nx, ny] > eps:
                z = z_x[mx, my]
                phi_n = np.conjugate(z) / np.conjugate(phi_m)
                amp = np.abs(phi_n)
                if amp > eps:
                    phi_n *= phi_abs[nx, ny] / amp
                else:
                    phi_n = phi_abs[nx, ny]
                phi[nx, ny] = phi_n
                visited[nx, ny] = True
                q.append((nx, ny))
        if valid_y[mx, my]:
            nx, ny = (mx, (my + 1) % d)
            if not visited[nx, ny] and phi_abs[nx, ny] > eps:
                z = z_y[mx, my]
                phi_n = np.conjugate(z) / np.conjugate(phi_m)
                amp = np.abs(phi_n)
                if amp > eps:
                    phi_n *= phi_abs[nx, ny] / amp
                else:
                    phi_n = phi_abs[nx, ny]
                phi[nx, ny] = phi_n
                visited[nx, ny] = True
                q.append((nx, ny))
    for mx in range(d):
        for my in range(d):
            if not visited[mx, my]:
                phi[mx, my] = phi_abs[mx, my]
    phi00 = phi[0, 0]
    if np.abs(phi00) < eps:
        phi[0, 0] = 1.0 + 0j
    else:
        g = np.exp(-1j * np.angle(phi00))
        phi *= g
        phi[0, 0] = 1.0 + 0j
    return phi


def enforce_hermitian_phi_with_amplitudes(phi, chi_plain, eps=1e-12):
    phi = np.asarray(phi, complex)
    chi_plain = np.asarray(chi_plain, float)
    d = phi.shape[0]
    assert phi.shape == (d, d)
    assert chi_plain.shape == (d, d)
    alpha = d ** 2 * chi_plain
    phi_abs = np.sqrt(np.maximum(alpha, 0.0))
    phi_abs[0, 0] = 1.0
    phi_sym = np.zeros_like(phi, dtype=complex)
    phi_sym[0, 0] = 1.0 + 0j
    for mx in range(d):
        for my in range(d):
            if mx == 0 and my == 0:
                continue
            nx = -mx % d
            ny = -my % d
            if nx < mx or (nx == mx and ny < my):
                continue
            if mx == nx and my == ny:
                amp = phi_abs[mx, my]
                theta = np.angle(phi[mx, my])
                sign = 1.0 if np.cos(theta) >= 0 else -1.0
                phi_sym[mx, my] = sign * amp
            else:
                a = phi[mx, my]
                b = np.conjugate(phi[nx, ny])
                v = a + b
                if np.abs(v) < eps:
                    if np.abs(a) < eps:
                        theta = 0.0
                    else:
                        theta = np.angle(a)
                else:
                    theta = np.angle(v)
                amp_m = phi_abs[mx, my]
                amp_nm = phi_abs[nx, ny]
                phi_sym[mx, my] = amp_m * np.exp(1j * theta)
                phi_sym[nx, ny] = amp_nm * np.exp(-1j * theta)
    return phi_sym


def anchor_povm_offdiagonal_2d(rho, d, lam=1.0):
    rho = np.asarray(rho, complex)
    D = rho.shape[0]
    assert rho.shape == (D, D)
    assert D == d * d, f'D={D} must equal d*d={d * d}'
    Sx = 0.0 + 0j
    Sy = 0.0 + 0j
    for x in range(d):
        xp = (x + 1) % d
        for y in range(d):
            yp = (y + 1) % d
            j = x * d + y
            j_xp_y = xp * d + y
            j_x_yp = x * d + yp
            Sx += rho[j_xp_y, j]
            Sy += rho[j_x_yp, j]
    Sx_exp = Sx
    Sy_exp = Sy
    Cx_exp = Sx_exp.real
    Dx_exp = Sx_exp.imag
    Cy_exp = Sy_exp.real
    Dy_exp = Sy_exp.imag

    def probs_from_expectation(exp):
        o_mean = exp / lam
        p_plus = 0.5 * (1.0 + o_mean)
        p_minus = 0.5 * (1.0 - o_mean)
        return (float(p_plus), float(p_minus))
    pCx_plus, pCx_minus = probs_from_expectation(Cx_exp)
    pDx_plus, pDx_minus = probs_from_expectation(Dx_exp)
    pCy_plus, pCy_minus = probs_from_expectation(Cy_exp)
    pDy_plus, pDy_minus = probs_from_expectation(Dy_exp)
    probs = {'Cx': (pCx_plus, pCx_minus), 'Dx': (pDx_plus, pDx_minus), 'Cy': (pCy_plus, pCy_minus), 'Dy': (pDy_plus, pDy_minus)}
    return {'Sx': Sx_exp, 'Sy': Sy_exp, 'phi10_anchor': Sx_exp, 'phi01_anchor': Sy_exp, 'Cx': Cx_exp, 'Dx': Dx_exp, 'Cy': Cy_exp, 'Dy': Dy_exp, 'probs': probs}


def fix_ac_sign_with_anchor_and_brightness_prior(phi, chi_plain, P_from_phi_func, phi10_anchor=None, phi01_anchor=None, eps_anchor=1e-05, k_bright=10, prior='dark'):
    phi = np.asarray(phi, complex)
    d = phi.shape[0]
    assert phi.shape == (d, d)
    chi_plain = np.asarray(chi_plain, float)
    assert chi_plain.shape == (d, d)
    targets = []
    if phi10_anchor is not None and abs(phi10_anchor) > eps_anchor and (abs(phi[1, 0]) > eps_anchor):
        targets.append((phi[1, 0], np.conjugate(phi10_anchor)))
    if phi01_anchor is not None and abs(phi01_anchor) > eps_anchor and (abs(phi[0, 1]) > eps_anchor):
        targets.append((phi[0, 1], np.conjugate(phi01_anchor)))
    if targets:
        cost_plus = 0.0
        cost_minus = 0.0
        for phi_est, phi_tgt in targets:
            cost_plus += abs(phi_est - phi_tgt) ** 2
            cost_minus += abs(-phi_est - phi_tgt) ** 2
        if cost_minus < cost_plus:
            phi_flipped = phi.copy()
            for mx in range(d):
                for my in range(d):
                    if mx == 0 and my == 0:
                        continue
                    phi_flipped[mx, my] *= -1.0
            return phi_flipped
        else:
            return phi

    def bright_score(P, k=k_bright):
        flat = np.sort(P.ravel())[::-1]
        k = min(k, flat.size)
        return float(np.sum(flat[:k]))
    P_plus = P_from_phi_func(phi)
    score_plus = bright_score(P_plus)
    phi_minus = phi.copy()
    for mx in range(d):
        for my in range(d):
            if mx == 0 and my == 0:
                continue
            phi_minus[mx, my] *= -1.0
    P_minus = P_from_phi_func(phi_minus)
    score_minus = bright_score(P_minus)
    prior = prior.lower()
    if prior not in ('dark', 'bright'):
        raise ValueError("prior must be 'dark' or 'bright'")
    if prior == 'dark':
        if score_minus > score_plus:
            return phi_minus
        else:
            return phi
    elif score_minus < score_plus:
        return phi_minus
    else:
        return phi


def gs_P_from_chi_plain_2d(chi_plain, n_iters=500, n_restarts=5, rng=None, phi_init=None):
    chi_plain = np.asarray(chi_plain, float)
    d = chi_plain.shape[0]
    assert chi_plain.shape == (d, d)
    if rng is None:
        rng = np.random.default_rng()
    chi = chi_plain.copy()
    chi[0, 0] = 1.0 / (d * d)
    alpha = d ** 2 * chi
    phi_abs = np.sqrt(np.maximum(alpha, 0.0))
    for mx in range(d):
        for my in range(d):
            px, py = (-mx % d, -my % d)
            avg = 0.5 * (phi_abs[mx, my] + phi_abs[px, py])
            phi_abs[mx, my] = phi_abs[px, py] = avg

    def gs_once(phi_init_local=None):
        phi = np.zeros((d, d), dtype=complex)
        phi[0, 0] = phi_abs[0, 0]
        if phi_init_local is not None:
            phi_phase = np.angle(phi_init_local)
            for mx in range(d):
                for my in range(d):
                    if mx == 0 and my == 0:
                        continue
                    px, py = (-mx % d, -my % d)
                    if mx > px or (mx == px and my > py):
                        continue
                    amp = phi_abs[mx, my]
                    if amp < 1e-15:
                        val = 0.0
                    else:
                        ang = phi_phase[mx, my]
                        val = amp * np.exp(1j * ang)
                    phi[mx, my] = val
                    phi[px, py] = np.conjugate(val)
        else:
            rng_phases = rng.uniform(0, 2 * np.pi, size=(d, d))
            for mx in range(d):
                for my in range(d):
                    if mx == 0 and my == 0:
                        continue
                    px, py = (-mx % d, -my % d)
                    if mx > px or (mx == px and my > py):
                        continue
                    amp = phi_abs[mx, my]
                    if amp < 1e-15:
                        val = 0.0
                    else:
                        ang = rng_phases[mx, my]
                        val = amp * np.exp(1j * ang)
                    phi[mx, my] = val
                    phi[px, py] = np.conjugate(val)
        for _ in range(n_iters):
            P = np.fft.ifft2(phi).real
            P[P < 0] = 0.0
            s = P.sum()
            if s == 0:
                P[:, :] = 1.0 / (d * d)
            else:
                P /= s
            phi_tilde = np.fft.fft2(P)
            phi_new = np.zeros_like(phi)
            phi_new[0, 0] = phi_abs[0, 0]
            for mx in range(d):
                for my in range(d):
                    if mx == 0 and my == 0:
                        continue
                    px, py = (-mx % d, -my % d)
                    if mx > px or (mx == px and my > py):
                        continue
                    amp = phi_abs[mx, my]
                    if amp < 1e-15:
                        val = 0.0
                    else:
                        ang = np.angle(phi_tilde[mx, my])
                        val = amp * np.exp(1j * ang)
                    phi_new[mx, my] = val
                    phi_new[px, py] = np.conjugate(val)
            phi = phi_new
        P_final = np.fft.ifft2(phi).real
        P_final[P_final < 0] = 0.0
        s = P_final.sum()
        if s == 0:
            P_final[:, :] = 1.0 / (d * d)
        else:
            P_final /= s
        return P_final
    if phi_init is not None:
        return gs_once(phi_init_local=phi_init)
    best_P = None
    best_cost = np.inf
    for _ in range(n_restarts):
        P = gs_once()
        phi_tilde = np.fft.fft2(P)
        cost = np.sum((np.abs(phi_tilde) - phi_abs) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_P = P
    return best_P


def orient_P_using_phi_reference_2d(P_gs, P_ref, chi_plain, chiX_x, chiY_x, chiX_y, chiY_y):
    P_gs = np.asarray(P_gs, float)
    chi_plain = np.asarray(chi_plain, float)
    chiX_x = np.asarray(chiX_x, float)
    chiY_x = np.asarray(chiY_x, float)
    chiX_y = np.asarray(chiX_y, float)
    chiY_y = np.asarray(chiY_y, float)
    d = P_gs.shape[0]
    assert P_gs.shape == (d, d)
    P_inv = np.zeros_like(P_gs)
    for x in range(d):
        for y in range(d):
            P_inv[x, y] = P_gs[-x % d, -y % d]
    best_cost = np.inf
    second_best = np.inf
    best_label = None
    best_P = None
    for sx in range(d):
        for sy in range(d):
            cand = np.roll(np.roll(P_gs, sx, axis=0), sy, axis=1)
            cost = np.sum((cand - P_ref) ** 2)
            if cost < best_cost:
                second_best = best_cost
                best_cost = cost
                best_label = ((sx, sy), +1)
                best_P = cand
            elif cost < second_best:
                second_best = cost
            cand = np.roll(np.roll(P_inv, sx, axis=0), sy, axis=1)
            cost = np.sum((cand - P_ref) ** 2)
            if cost < best_cost:
                second_best = best_cost
                best_cost = cost
                best_label = ((sx, sy), -1)
                best_P = cand
            elif cost < second_best:
                second_best = cost
    if np.isinf(second_best) or best_cost <= 0:
        confidence = 1.0
    else:
        confidence = (second_best - best_cost) / (second_best + 1e-12)
        confidence = float(np.clip(confidence, 0.0, 1.0))
    P_est = best_P.copy()
    return (P_est, best_label, confidence, P_ref)



# ====================================================================
# IMAGING FIGURE PANELS
# ====================================================================

def plot_P_true_est_speckle(P_true, P_est, P_est2, I_speckle, titles=('P_true', 'P_est', 'P_est2', 'Speckle image'), cmap='hot', origin='upper'):
    P_true = np.asarray(P_true, float)
    P_est = np.asarray(P_est, float)
    P_est2 = np.asarray(P_est2, float)
    I_speckle = np.asarray(I_speckle, float)
    vmin = min(P_true.min(), P_est.min(), P_est2.min(), I_speckle.min())
    vmax = max(P_true.max(), P_est.max(), P_est2.min(), I_speckle.max())
    fig, axs = plt.subplots(1, 4, figsize=(12, 4), sharex=True, sharey=True, dpi=150)
    ims = []
    ims.append(axs[0].imshow(P_true, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin))
    axs[0].set_title(titles[0])
    axs[0].set_xlabel('y')
    axs[0].set_ylabel('x')
    axs[0].axis('off')
    ims.append(axs[1].imshow(P_est, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin))
    axs[1].set_title(titles[1])
    axs[1].set_xlabel('y')
    axs[1].axis('off')
    ims.append(axs[2].imshow(P_est2, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin))
    axs[2].set_title(titles[2])
    axs[2].set_xlabel('y')
    axs[2].axis('off')
    ims.append(axs[3].imshow(I_speckle, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin))
    axs[3].set_title(titles[3])
    axs[3].set_xlabel('y')
    axs[3].axis('off')
    cbar = fig.colorbar(ims[0], ax=axs.ravel().tolist(), location='right', pad=0.01, shrink=0.6)
    cbar.set_label('Brightness', labelpad=-5)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels(['0', '1'])
    plt.show()
    return (fig, axs)


def plot_P_true_est_speckle_2(P_true, P_est, I_speckle, titles=('P_true', 'P_est', 'Speckle image'), cmap='hot', origin='upper'):
    P_true = np.asarray(P_true, float)
    P_est = np.asarray(P_est, float)
    I_speckle = np.asarray(I_speckle, float)
    vmin = min(P_true.min(), P_est.min(), I_speckle.min())
    vmax = max(P_true.max(), P_est.max(), I_speckle.max())
    fig, axs = plt.subplots(1, 3, figsize=(9, 4), sharex=True, sharey=True, dpi=150)
    ims = []
    ims.append(axs[0].imshow(P_true, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin))
    axs[0].set_title(titles[0])
    axs[0].set_xlabel('y')
    axs[0].set_ylabel('x')
    axs[0].axis('off')
    ims.append(axs[1].imshow(P_est, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin))
    axs[1].set_title(titles[1])
    axs[1].set_xlabel('y')
    axs[1].axis('off')
    ims.append(axs[2].imshow(I_speckle, vmin=vmin, vmax=vmax, cmap=cmap, origin=origin))
    axs[2].set_title(titles[2])
    axs[2].set_xlabel('y')
    axs[2].axis('off')
    cbar = fig.colorbar(ims[0], ax=axs.ravel().tolist(), location='right', pad=0.01, shrink=0.6)
    cbar.set_label('Brightness', labelpad=-5)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels(['0', '1'])
    plt.show()
    return (fig, axs)


__all__ = [
    'phi_from_P_2d',
    'P_from_phi',
    'analytic_chis_from_phi',
    'rho1_from_phi',
    'rho1_from_phi_with_phase_screen',
    'rho2_from_rho1',
    'kolmogorov_phase_screen',
    'speckle_image_from_phase_and_P',
    'block_swap_probs_plain_global_2d_fast',
    'block_swap_probs_rotated_X_global_2d_fast',
    'block_swap_probs_rotated_Y_global_2d_fast',
    'reconstruct_phi_from_chis_2d_full',
    'enforce_hermitian_phi_with_amplitudes',
    'anchor_povm_offdiagonal_2d',
    'fix_ac_sign_with_anchor_and_brightness_prior',
    'gs_P_from_chi_plain_2d',
    'orient_P_using_phi_reference_2d',
    'plot_P_true_est_speckle',
    'plot_P_true_est_speckle_2',
]
