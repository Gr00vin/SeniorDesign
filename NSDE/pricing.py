"""
pricing.py — Fully-vectorised, GPU-accelerated option pricing via
             Heston / Bates characteristic functions + Gauss-Laguerre quadrature.

All heavy lifting uses PyTorch tensors (complex128 on CUDA when available).
"""

import math
import numpy as np
import torch
from config import DEVICE, DTYPE, GL_NUM_POINTS


# ── helpers ──────────────────────────────────────────────────────────────────

COMPLEX_DTYPE = torch.complex128          # matches float64 real part


def _to(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is on the correct device and dtype."""
    return x.to(device=DEVICE, dtype=DTYPE)


# ── Gauss-Laguerre nodes & weights ──────────────────────────────────────────

def generate_gauss_laguerre(n: int = GL_NUM_POINTS):
    """
    Compute Gauss-Laguerre abscissas and weights (total weight = w_j·exp(x_j)).
    Returns torch tensors on DEVICE.
    """
    def _nchoosek(n, r):
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

    x_np, _ = np.polynomial.laguerre.laggauss(n)
    w_np = np.zeros(n, dtype=np.float64)
    dL = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        for k in range(n):
            dL[k, j] = ((-1) ** (k + 1) / math.factorial(k)
                         * _nchoosek(n, k + 1) * x_np[j] ** k)
        w_np[j] = 1.0 / x_np[j] / np.sum(dL[:, j]) ** 2
        w_np[j] *= np.exp(x_np[j])

    x_t = torch.tensor(x_np, dtype=DTYPE, device=DEVICE)
    w_t = torch.tensor(w_np, dtype=DTYPE, device=DEVICE)
    return x_t, w_t


# pre-compute once at import time
GL_X, GL_W = generate_gauss_laguerre(GL_NUM_POINTS)


# ── Characteristic functions (vectorised over batch AND quadrature) ─────────

def heston_cf(
    phi: torch.Tensor,       # (Q,)           quadrature nodes (complex)
    kappa: torch.Tensor,     # (B, 1)
    theta: torch.Tensor,
    xi: torch.Tensor,
    v0: torch.Tensor,
    rho: torch.Tensor,
    S: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:           # (B, Q)
    """
    Heston (1993) characteristic function f₂ — 'Little Trap' formulation.
    All param tensors are real (B,1); phi is complex (Q,).
    Returns complex tensor (B, Q).
    """
    i = torch.tensor(1j, dtype=COMPLEX_DTYPE, device=DEVICE)

    # broadcast:  params -> (B,1),  phi -> (1,Q)  →  result (B,Q)
    phi = phi.unsqueeze(0)                            # (1, Q)
    x = torch.log(S.to(COMPLEX_DTYPE))                # (B, 1)

    a     = (kappa * theta).to(COMPLEX_DTYPE)          # (B,1)
    u     = -0.5
    b     = kappa.to(COMPLEX_DTYPE)                    # (lambd = 0 absorbed)
    sigma = xi.to(COMPLEX_DTYPE)
    rho_c = rho.to(COMPLEX_DTYPE)
    v0_c  = v0.to(COMPLEX_DTYPE)
    r_c   = r.to(COMPLEX_DTYPE)
    q_c   = q.to(COMPLEX_DTYPE)
    T_c   = T.to(COMPLEX_DTYPE)

    d = torch.sqrt((rho_c * sigma * i * phi - b) ** 2
                   - sigma ** 2 * (2 * u * i * phi - phi ** 2))

    g = (b - rho_c * sigma * i * phi + d) / (b - rho_c * sigma * i * phi - d)
    c = 1.0 / g

    D = ((b - rho_c * sigma * i * phi - d) / sigma ** 2
         * (1.0 - torch.exp(-d * T_c)) / (1.0 - c * torch.exp(-d * T_c)))

    G = (1.0 - c * torch.exp(-d * T_c)) / (1.0 - c)

    C = ((r_c - q_c) * i * phi * T_c
         + a / sigma ** 2 * ((b - rho_c * sigma * i * phi - d) * T_c
                             - 2.0 * torch.log(G)))

    f2 = torch.exp(C + D * v0_c + i * phi * x)        # (B, Q)
    return f2


def jump_cf(
    phi: torch.Tensor,       # (Q,) complex
    lam: torch.Tensor,       # (B,1) real
    muJ: torch.Tensor,
    sigmaJ: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:           # (B, Q) complex
    """Merton jump-diffusion characteristic function multiplier."""
    i = torch.tensor(1j, dtype=COMPLEX_DTYPE, device=DEVICE)
    phi = phi.unsqueeze(0)

    lam_c    = lam.to(COMPLEX_DTYPE)
    muJ_c    = muJ.to(COMPLEX_DTYPE)
    sigmaJ_c = sigmaJ.to(COMPLEX_DTYPE)
    T_c      = T.to(COMPLEX_DTYPE)

    jcf = torch.exp(
        -lam_c * muJ_c * i * phi * T_c
        + lam_c * T_c * (
            (1.0 + muJ_c) ** (i * phi)
            * torch.exp(0.5 * sigmaJ_c ** 2 * i * phi * (i * phi - 1.0))
            - 1.0
        )
    )
    return jcf


# ── Gauss-Laguerre integrand for P1 / P2 ───────────────────────────────────

def _sv_integrand(
    phi_real: torch.Tensor,  # (Q,) real nodes
    params_dict: dict,       # batched tensors
    Pnum: int,               # 1 or 2
    use_jumps: bool,
) -> torch.Tensor:           # (B, Q) real
    """
    Compute the SV integrand at all quadrature nodes for the whole batch.
    """
    i_unit = torch.tensor(1j, dtype=COMPLEX_DTYPE, device=DEVICE)
    phi_c = phi_real.to(COMPLEX_DTYPE)                 # (Q,)

    S = params_dict["S"]; K = params_dict["K"]
    r = params_dict["r"]; q = params_dict["q"]; T = params_dict["T"]
    kappa = params_dict["kappa"]; theta = params_dict["theta"]
    xi = params_dict["xi"]; v0 = params_dict["v0"]; rho = params_dict["rho"]

    if Pnum == 2:
        f = heston_cf(phi_c, kappa, theta, xi, v0, rho, S, r, q, T)
        if use_jumps:
            f = f * jump_cf(phi_c, params_dict["lam"],
                            params_dict["muJ"], params_dict["sigmaJ"], T)
        logK = torch.log(K.to(COMPLEX_DTYPE))           # (B,1)
        phi_broad = phi_c.unsqueeze(0)                   # (1,Q)
        integ = (torch.exp(-i_unit * phi_broad * logK) * f
                 / (i_unit * phi_broad)).real             # (B, Q)

    else:  # Pnum == 1
        phi_m_i = phi_c - i_unit                         # (Q,)
        fnum = heston_cf(phi_m_i, kappa, theta, xi, v0, rho, S, r, q, T)
        fden = heston_cf(-i_unit.expand_as(phi_c), kappa, theta, xi, v0, rho,
                         S, r, q, T)
        if use_jumps:
            fnum = fnum * jump_cf(phi_m_i, params_dict["lam"],
                                  params_dict["muJ"], params_dict["sigmaJ"], T)
            fden = fden * jump_cf(-i_unit.expand_as(phi_c), params_dict["lam"],
                                  params_dict["muJ"], params_dict["sigmaJ"], T)
        logK = torch.log(K.to(COMPLEX_DTYPE))
        phi_broad = phi_c.unsqueeze(0)
        integ = (torch.exp(-i_unit * phi_broad * logK)
                 * fnum / (i_unit * phi_broad) / fden).real

    return integ  # (B, Q)


# ── Vectorised SV price ────────────────────────────────────────────────────

def sv_price_batch(
    params_dict: dict,
    put_call: str = "C",
    use_jumps: bool = True,
    gl_x: torch.Tensor = GL_X,
    gl_w: torch.Tensor = GL_W,
) -> torch.Tensor:
    """
    Price a *batch* of European options via Gauss-Laguerre quadrature
    of the Heston (or Bates) characteristic function.

    params_dict must contain tensors of shape (B, 1):
        S, K, T, r, q, kappa, theta, xi, v0, rho
        and optionally lam, muJ, sigmaJ  (if use_jumps).

    Returns
    -------
    prices : (B,) real tensor — call or put prices.
    """
    S = params_dict["S"]; K = params_dict["K"]
    r = params_dict["r"]; q = params_dict["q"]; T = params_dict["T"]

    # integrand matrices  (B, Q)
    int1 = gl_w.unsqueeze(0) * _sv_integrand(gl_x, params_dict, Pnum=1,
                                              use_jumps=use_jumps)
    int2 = gl_w.unsqueeze(0) * _sv_integrand(gl_x, params_dict, Pnum=2,
                                              use_jumps=use_jumps)

    I1 = int1.sum(dim=1, keepdim=True)   # (B,1)
    I2 = int2.sum(dim=1, keepdim=True)

    P1 = 0.5 + I1 / math.pi
    P2 = 0.5 + I2 / math.pi

    call = S * torch.exp(-q * T) * P1 - K * torch.exp(-r * T) * P2

    if "P" in put_call.upper():
        price = call - S * torch.exp(-q * T) + K * torch.exp(-r * T)
    else:
        price = call

    return price.squeeze(1)              # (B,)


# ── Convenience wrapper used by the trainer ─────────────────────────────────

def price_options_from_features(
    X_batch: torch.Tensor,
    predicted_params: torch.Tensor,
    feature_names: list[str],
    use_jumps: bool = True,
) -> torch.Tensor:
    """
    Given a feature matrix and NN-predicted Heston/Bates parameters,
    return model call prices for the batch.

    Parameters
    ----------
    X_batch          (B, F)  market features
    predicted_params (B, 5 or 8)  [theta, rho, v0, xi, kappa, (muJ, sigmaJ, lam)]
    feature_names    list of column names in X_batch
    """
    from data import get_column_index

    idx_S = get_column_index(feature_names, ["Spot", "S0", "Underlying"])
    idx_K = get_column_index(feature_names, ["Strike", "K"])
    idx_T = get_column_index(feature_names, ["Tau", "Time", "Days", "T"])
    idx_r = get_column_index(feature_names, ["r", "Rate", "Interest"])
    idx_q = get_column_index(feature_names, ["q", "Div", "Yield"])

    B = X_batch.shape[0]
    reshape = lambda t: t.reshape(B, 1)

    params_dict = dict(
        S     = reshape(X_batch[:, idx_S]),
        K     = reshape(X_batch[:, idx_K]),
        T     = reshape(X_batch[:, idx_T]),
        r     = reshape(X_batch[:, idx_r]),
        q     = reshape(X_batch[:, idx_q]),
        theta = reshape(predicted_params[:, 0]),
        rho   = reshape(predicted_params[:, 1]),
        v0    = reshape(predicted_params[:, 2]),
        xi    = reshape(predicted_params[:, 3]),
        kappa = reshape(predicted_params[:, 4]),
    )

    if use_jumps and predicted_params.shape[1] == 8:
        params_dict["muJ"]    = reshape(predicted_params[:, 5])
        params_dict["sigmaJ"] = reshape(predicted_params[:, 6])
        params_dict["lam"]    = reshape(predicted_params[:, 7])

    return sv_price_batch(params_dict, put_call="C", use_jumps=use_jumps)
