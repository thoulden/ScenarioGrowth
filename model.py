from __future__ import annotations

import numpy as np


# -----------------------------
# Helpers: CES price / unit cost
# -----------------------------
def ces_price(p1: float, p2: float, share: float, sigma: float) -> float:
    """
    CES unit cost / price index (two inputs):
        p = [ share*p1^(1-sigma) + (1-share)*p2^(1-sigma) ]^(1/(1-sigma))

    Notes:
    - This matches the dual expressions in your writeup.
    - If sigma ~ 1, we use the log-CES limit (geometric mean) for stability.
    """
    share = float(share)
    sigma = float(sigma)
    p1 = float(p1)
    p2 = float(p2)

    if not (0.0 < share < 1.0):
        raise ValueError(f"CES share must be in (0,1). Got {share}.")

    if abs(sigma - 1.0) < 1e-8:
        # log-CES limit
        return (p1 ** share) * (p2 ** (1.0 - share))

    one_minus = 1.0 - sigma
    return (share * (p1 ** one_minus) + (1.0 - share) * (p2 ** one_minus)) ** (1.0 / one_minus)


def final_unit_cost(r: float, p_eff: float, A: float, alpha: float) -> float:
    """
    Cobb–Douglas final good: Y = A*K^alpha * L^(1-alpha)
    Unit cost:
      c = (1/A) * r^alpha * p_eff^(1-alpha) / [ alpha^alpha (1-alpha)^(1-alpha) ]
    """
    r = float(r)
    p_eff = float(p_eff)
    A = float(A)
    alpha = float(alpha)

    if not (A > 0.0):
        raise ValueError("A must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    denom = (alpha ** alpha) * ((1.0 - alpha) ** (1.0 - alpha))
    return (1.0 / A) * (r ** alpha) * (p_eff ** (1.0 - alpha)) / denom


# -----------------------------
# Household: labor supply from "labor budget"
# -----------------------------
def household_labor_supply(wc: float, wp: float, Lbar: float, tau: float) -> tuple[float, float]:
    """
    Max wc*Lc + wp*Lp  s.t.  Lc^tau + Lp^tau = Lbar^tau, tau>1.
    Interior implies:
      Lc/Lp = (wc/wp)^(1/(tau-1))
    """
    wc = float(wc)
    wp = float(wp)
    Lbar = float(Lbar)
    tau = float(tau)

    if tau <= 1.0:
        raise ValueError("tau must be > 1.")
    if wc <= 0.0 or wp <= 0.0:
        raise ValueError("Wages must be positive.")
    if Lbar <= 0.0:
        raise ValueError("Lbar must be positive.")

    ratio = (wc / wp) ** (1.0 / (tau - 1.0))
    Lp = Lbar * (1.0 / (ratio ** tau + 1.0)) ** (1.0 / tau)
    Lc = ratio * Lp
    return float(Lc), float(Lp)


# -----------------------------
# Exogenous automation capability path
# -----------------------------
def mu_auto_path(mu0: float, kappa: float, T: int, mu_max: float = 0.999999) -> np.ndarray:
    """
    Automation capability share mu_t with exponentially declining remaining tasks:
      1 - mu_t = (1 - mu0) * exp(-kappa * t)

    Here mu_t is the "automatable share" (increasing over time).
    We clip away from 0/1 for numerical stability.
    """
    mu0 = float(mu0)
    kappa = float(kappa)
    if not (0.0 <= mu0 < 1.0):
        raise ValueError("mu0 must be in [0,1).")
    if kappa < 0.0:
        raise ValueError("kappa must be >= 0.")
    t = np.arange(int(T))
    mu = 1.0 - (1.0 - mu0) * np.exp(-kappa * t)
    return np.clip(mu, 1e-9, mu_max)


# -----------------------------
# Static within-period equilibrium solver
# -----------------------------
def static_equilibrium_one_sector(
    *,
    K_t: float,
    A_t: float,
    Xc_t: float,
    Xp_t: float,
    params: dict,
    x0_guess: tuple[float, float, float] = (0.05, 1.0, 1.0),
    max_iter: int = 80,
    tol: float = 1e-10,
) -> dict:
    """
    Solve for (r, w_c, w_p) from 3 conditions:
      (i)  Zero profit in final good: unit cost c(r, p_eff)=1
      (ii) Human cognitive labor market clears: Lhc_d = Lc_s
      (iii) Human physical labor market clears: Lhp_d = Lp_s

    Uses damped Newton with finite-difference Jacobian (no SciPy required).
    """
    # unpack params
    alpha = float(params["alpha"])
    theta = float(params["theta"])
    eps = float(params["eps"])
    sig_c = float(params["sig_c"])
    sig_p = float(params["sig_p"])
    phi_c = float(params["phi_c"])
    phi_p = float(params["phi_p"])
    Lbar = float(params["Lbar"])
    tau = float(params["tau"])

    # time-varying CES weights are included in params dict when needed
    mu_c = float(params.get("mu_c", 0.3))  # weight on human in cognitive nest
    mu_p = float(params.get("mu_p", 0.3))  # weight on human in physical nest

    # sanity
    if K_t <= 0:
        raise ValueError("K_t must be positive.")
    if Xc_t <= 0 or Xp_t <= 0:
        raise ValueError("Xc_t and Xp_t must be positive.")
    if phi_c <= 0 or phi_p <= 0:
        raise ValueError("phi_c and phi_p must be positive.")
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0,1).")
    if not (0.0 < mu_c < 1.0) or not (0.0 < mu_p < 1.0):
        raise ValueError("mu_c and mu_p must be in (0,1).")

    def residuals(logx: np.ndarray) -> np.ndarray:
        r = float(np.exp(logx[0]))
        wc = float(np.exp(logx[1]))
        wp = float(np.exp(logx[2]))

        qc = r / (phi_c * Xc_t)
        qr = r / (phi_p * Xp_t)

        p_cog = ces_price(wc, qc, mu_c, sig_c)
        p_phys = ces_price(wp, qr, mu_p, sig_p)
        p_eff = ces_price(p_cog, p_phys, theta, eps)

        # (i) zero profit: unit cost == 1
        c = final_unit_cost(r, p_eff, A_t, alpha)

        # conditional demands per unit Y
        KY_perY = alpha / r
        Leff_perY = (1.0 - alpha) / p_eff

        Lcog_perY = theta * (p_cog / p_eff) ** (-eps) * Leff_perY
        Lphys_perY = (1.0 - theta) * (p_phys / p_eff) ** (-eps) * Leff_perY

        Lhc_perY = mu_c * (wc / p_cog) ** (-sig_c) * Lcog_perY
        Lhp_perY = mu_p * (wp / p_phys) ** (-sig_p) * Lphys_perY
        LAI_perY = (1.0 - mu_c) * (qc / p_cog) ** (-sig_c) * Lcog_perY
        LR_perY = (1.0 - mu_p) * (qr / p_phys) ** (-sig_p) * Lphys_perY

        KAI_perY = LAI_perY / (phi_c * Xc_t)
        KR_perY = LR_perY / (phi_p * Xp_t)

        kappaK = float(KY_perY + KAI_perY + KR_perY)
        Y = float(K_t / kappaK)

        # factor demands (levels)
        Lhc_d = float(Lhc_perY * Y)
        Lhp_d = float(Lhp_perY * Y)

        # household supply
        Lc_s, Lp_s = household_labor_supply(wc, wp, Lbar, tau)

        return np.array([
            np.log(c),                  # c = 1
            np.log(Lhc_d / Lc_s),        # Lhc_d = Lc_s
            np.log(Lhp_d / Lp_s),        # Lhp_d = Lp_s
        ], dtype=float)

    # ---- damped Newton in log-space ----
    logx = np.log(np.array(x0_guess, dtype=float))
    damping = 0.6

    for it in range(max_iter):
        F = residuals(logx)
        normF = float(np.max(np.abs(F)))
        if normF < tol:
            break

        # finite-difference Jacobian
        J = np.zeros((3, 3))
        h = 1e-6
        for j in range(3):
            step = np.zeros(3)
            step[j] = h
            Fp = residuals(logx + step)
            Fm = residuals(logx - step)
            J[:, j] = (Fp - Fm) / (2.0 * h)

        # solve J * dx = -F
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -F, rcond=None)[0]

        # line search
        success = False
        for ls in range(20):
            cand = logx + (damping ** ls) * dx
            Fcand = residuals(cand)
            if np.max(np.abs(Fcand)) < normF:
                logx = cand
                success = True
                break
        if not success:
            logx = logx + 0.1 * dx

    # ---- recover at solution ----
    r = float(np.exp(logx[0]))
    wc = float(np.exp(logx[1]))
    wp = float(np.exp(logx[2]))

    qc = r / (phi_c * Xc_t)
    qr = r / (phi_p * Xp_t)

    p_cog = ces_price(wc, qc, mu_c, sig_c)
    p_phys = ces_price(wp, qr, mu_p, sig_p)
    p_eff = ces_price(p_cog, p_phys, theta, eps)

    # demands per unit Y
    KY_perY = alpha / r
    Leff_perY = (1.0 - alpha) / p_eff
    Lcog_perY = theta * (p_cog / p_eff) ** (-eps) * Leff_perY
    Lphys_perY = (1.0 - theta) * (p_phys / p_eff) ** (-eps) * Leff_perY

    Lhc_perY = mu_c * (wc / p_cog) ** (-sig_c) * Lcog_perY
    Lhp_perY = mu_p * (wp / p_phys) ** (-sig_p) * Lphys_perY
    LAI_perY = (1.0 - mu_c) * (qc / p_cog) ** (-sig_c) * Lcog_perY
    LR_perY = (1.0 - mu_p) * (qr / p_phys) ** (-sig_p) * Lphys_perY

    KAI_perY = LAI_perY / (phi_c * Xc_t)
    KR_perY = LR_perY / (phi_p * Xp_t)
    kappaK = float(KY_perY + KAI_perY + KR_perY)
    Y = float(K_t / kappaK)

    # levels
    KY = float(KY_perY * Y)
    KAI = float(KAI_perY * Y)
    KR = float(KR_perY * Y)

    Lhc_d = float(Lhc_perY * Y)
    Lhp_d = float(Lhp_perY * Y)
    LAI = float(LAI_perY * Y)
    LR = float(LR_perY * Y)

    # household supply
    Lc_s, Lp_s = household_labor_supply(wc, wp, Lbar, tau)

    out = dict(
        r=r, wc=wc, wp=wp, qc=qc, qr=qr,
        p_cog=float(p_cog), p_phys=float(p_phys), p_eff=float(p_eff),
        Y=Y,
        KY=KY, KAI=KAI, KR=KR,
        Lhc_d=Lhc_d, Lhp_d=Lhp_d, LAI=LAI, LR=LR,
        Lc_s=float(Lc_s), Lp_s=float(Lp_s),
        iters=int(it + 1),
        residual_max=float(np.max(np.abs(residuals(logx)))),
    )
    return out


# -----------------------------
# Transition simulation
# -----------------------------
def simulate_transition(
    *,
    T: int,
    K0: float,
    A0: float,
    gA: float,
    Xc0: float,
    gXc: float,
    Xp0: float,
    gXp: float,
    s: float,
    delta: float,
    params: dict,
    mu_c0: float = 0.2,
    kappa_mu_c: float = 0.02,
    mu_p0: float = 0.2,
    kappa_mu_p: float = 0.02,
    guess: tuple[float, float, float] = (0.06, 1.0, 1.0),
) -> dict:
    """
    Exogenous paths:
      A_t  = A0  * exp(gA * t)
      Xc_t = Xc0 * exp(gXc * t)
      Xp_t = Xp0 * exp(gXp * t)

    Capital:
      K_{t+1} = (1-delta)K_t + s*Y_t
      C_t = (1-s)Y_t

    Automation capability (exogenous):
      mu_auto,t = 1 - (1 - mu0)*exp(-kappa*t)

    In the CES nests, params["mu_c"] and params["mu_p"] are *human* weights,
    so we convert:
      mu_human,t = 1 - mu_auto,t
    """
    T = int(T)
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (0.0 <= s < 1.0):
        raise ValueError("s must be in [0,1).")
    if not (0.0 <= delta < 1.0):
        raise ValueError("delta must be in [0,1).")

    mu_c_auto = mu_auto_path(mu_c0, kappa_mu_c, T)
    mu_p_auto = mu_auto_path(mu_p0, kappa_mu_p, T)

    K = np.zeros(T + 1)
    Y = np.zeros(T)
    C = np.zeros(T)
    r = np.zeros(T)
    wc = np.zeros(T)
    wp = np.zeros(T)
    qc = np.zeros(T)
    qr = np.zeros(T)

    Lhc = np.zeros(T)
    Lhp = np.zeros(T)
    LAI = np.zeros(T)
    LR = np.zeros(T)

    KY = np.zeros(T)
    KAI = np.zeros(T)
    KR = np.zeros(T)

    mu_c = np.zeros(T)
    mu_p = np.zeros(T)

    K[0] = float(K0)
    current_guess = guess

    # diagnostics
    sufficient_savings_ok = True
    residual_max_overall = 0.0

    for t in range(T):
        A_t = float(A0 * np.exp(gA * t))
        Xc_t = float(Xc0 * np.exp(gXc * t))
        Xp_t = float(Xp0 * np.exp(gXp * t))

        mu_c[t] = float(mu_c_auto[t])
        mu_p[t] = float(mu_p_auto[t])

        mu_c_human = float(np.clip(1.0 - mu_c[t], 1e-6, 1.0 - 1e-6))
        mu_p_human = float(np.clip(1.0 - mu_p[t], 1e-6, 1.0 - 1e-6))

        params_t = dict(params)
        params_t["mu_c"] = mu_c_human
        params_t["mu_p"] = mu_p_human

        sol = static_equilibrium_one_sector(
            K_t=float(K[t]),
            A_t=A_t,
            Xc_t=Xc_t,
            Xp_t=Xp_t,
            params=params_t,
            x0_guess=current_guess,
        )

        Y[t] = sol["Y"]
        r[t] = sol["r"]
        wc[t] = sol["wc"]
        wp[t] = sol["wp"]
        qc[t] = sol["qc"]
        qr[t] = sol["qr"]

        Lhc[t] = sol["Lhc_d"]
        Lhp[t] = sol["Lhp_d"]
        LAI[t] = sol["LAI"]
        LR[t] = sol["LR"]

        KY[t] = sol["KY"]
        KAI[t] = sol["KAI"]
        KR[t] = sol["KR"]

        C[t] = (1.0 - s) * Y[t]
        K[t + 1] = (1.0 - delta) * K[t] + s * Y[t]

        # diagnostics
        sufficient_savings_ok = sufficient_savings_ok and (qc[t] < wc[t]) and (qr[t] < wp[t])
        residual_max_overall = max(residual_max_overall, float(sol["residual_max"]))

        current_guess = (sol["r"], sol["wc"], sol["wp"])

    return dict(
        K=K, Y=Y, C=C,
        r=r, wc=wc, wp=wp,
        qc=qc, qr=qr,
        Lhc=Lhc, Lhp=Lhp, LAI=LAI, LR=LR,
        KY=KY, KAI=KAI, KR=KR,
        mu_c=mu_c, mu_p=mu_p,
        diagnostics=dict(
            sufficient_savings_ok=bool(sufficient_savings_ok),
            residual_max=float(residual_max_overall),
        ),
    )
