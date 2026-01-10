"""
Two-Sector Static Equilibrium Simulator

- Final good is numeraire (P=1) and aggregates sector outputs S (services), G (goods)
- Two sectors differ in cognitive intensity: theta_S > theta_G
- Within each sector: effective labor nest:
      (cog, phys) CES with epsilon
  cog nest: (human cog, AI cog) CES with sigma_c and task-cutoff mu_h_c
  phys nest:(human phys, robots) CES with sigma_p and task-cutoff mu_h_p
- Total human labor L is exogenous; split into (Hc, Hp) endogenously:
      Hc/Hp = kappa * (wc/wp)^omega
- AI_cog and R_phys are exogenous factor supplies each year
- Automation frontier: bar_auto_* is "max automatable share"
      => min human task share mu_h_*_min = 1 - bar_auto_*
  Start at frontier (mu_h = mu_min). If wc<qc (humans cheaper than AI),
  back off automation by increasing mu_h until wc=qc. Same for physical.
- Unknowns solved by Newton in log space: wc, wp, qc, qr
"""

from __future__ import annotations

import numpy as np
import math

TINY = 1e-12


def ces_price(p1: float, p2: float, mu: float, sigma: float) -> float:
    """
    Dual price index consistent with CES nests.
    p = [ mu p1^(1-sigma) + (1-mu) p2^(1-sigma) ]^(1/(1-sigma))
    """
    mu = float(np.clip(mu, 1e-12, 1 - 1e-12))
    sigma = float(sigma)
    p1 = float(p1)
    p2 = float(p2)
    if abs(sigma - 1.0) < 1e-10:
        return (p1 ** mu) * (p2 ** (1 - mu))
    one_minus = 1.0 - sigma
    return (mu * (p1 ** one_minus) + (1.0 - mu) * (p2 ** one_minus)) ** (1.0 / one_minus)


def cd_unit_cost(r: float, p_eff: float, A: float, alpha: float) -> float:
    """Cobb-Douglas unit cost for sector production."""
    denom = (alpha ** alpha) * ((1 - alpha) ** (1 - alpha))
    return (1.0 / A) * (r ** alpha) * (p_eff ** (1 - alpha)) / denom


def normalize_prices_numeraire(P_S: float, P_G: float, nu: float, eta: float) -> tuple[float, float, float]:
    """
    Compute P = [nu P_S^(1-eta) + (1-nu) P_G^(1-eta)]^(1/(1-eta)),
    then rescale prices so P=1.
    """
    P_S = float(P_S)
    P_G = float(P_G)
    nu = float(nu)
    eta = float(eta)

    if abs(eta - 1.0) < 1e-10:
        P = (P_S ** nu) * (P_G ** (1 - nu))
    else:
        P = (nu * (P_S ** (1 - eta)) + (1 - nu) * (P_G ** (1 - eta))) ** (1.0 / (1 - eta))

    return P_S / P, P_G / P, P


def sector_outputs_from_prices(Y: float, P_S: float, P_G: float, nu: float, eta: float) -> tuple[float, float]:
    """
    With final good as numeraire (P=1):
      S = nu * (P_S)^(-eta) * Y
      G = (1-nu) * (P_G)^(-eta) * Y
    """
    Y = float(Y)
    nu = float(nu)
    eta = float(eta)
    S = nu * (P_S ** (-eta)) * Y
    G = (1 - nu) * (P_G ** (-eta)) * Y
    return float(S), float(G)


def human_split(L: float, wc: float, wp: float, kappa: float, omega: float) -> tuple[float, float, float]:
    """
    Human labor supply split: Hc/Hp = kappa * (wc/wp)^omega
    """
    L = float(L)
    wc = float(wc)
    wp = float(wp)
    kappa = float(kappa)
    omega = float(omega)

    z = max(wc / wp, TINY)
    x = kappa * (z ** omega)
    ell = x / (1 + x)
    Hc = ell * L
    Hp = (1 - ell) * L
    return float(Hc), float(Hp), float(ell)


def sector_block(
    *,
    Y: float,
    K: float,
    L: float,
    AI_cog: float,
    R_phys: float,
    mu_h_c: float,
    mu_h_p: float,
    wc: float,
    wp: float,
    qc: float,
    qr: float,
    params: dict,
) -> dict:
    """
    Compute sector prices and factor demands given factor prices and task shares.
    """
    alpha = params["alpha"]
    nu = params["nu"]
    eta = params["eta"]
    theta_S = params["theta_S"]
    theta_G = params["theta_G"]
    eps = params["eps"]
    sig_c = params["sig_c"]
    sig_p = params["sig_p"]
    kappa = params["kappa"]
    omega = params["omega"]

    # human supply split
    Hc_s, Hp_s, ell_c = human_split(L, wc, wp, kappa, omega)

    # rental rate from capital market (aggregate)
    r = alpha * Y / K

    # within-nest prices (same for both sectors given common factor prices)
    p_cog = ces_price(wc, qc, mu_h_c, sig_c)
    p_phys = ces_price(wp, qr, mu_h_p, sig_p)

    # sector effective labor prices
    p_eff_S = ces_price(p_cog, p_phys, theta_S, eps)
    p_eff_G = ces_price(p_cog, p_phys, theta_G, eps)

    # unit costs -> sector prices (A_S=A_G=1 here)
    P_S = cd_unit_cost(r, p_eff_S, A=1.0, alpha=alpha)
    P_G = cd_unit_cost(r, p_eff_G, A=1.0, alpha=alpha)

    # normalize final good price index to 1
    P_S, P_G, P_index = normalize_prices_numeraire(P_S, P_G, nu, eta)

    # sector outputs (final demand)
    S, G = sector_outputs_from_prices(Y, P_S, P_G, nu, eta)

    # Sector spending on effective labor: (1-alpha) * revenue / p_eff
    def sector_demands(Yi, Pi, p_eff_i, theta_i):
        Leff = (1 - alpha) * Pi * Yi / p_eff_i

        # split Leff into cog vs phys
        Lcog = theta_i * (p_cog / p_eff_i) ** (-eps) * Leff
        Lphys = (1 - theta_i) * (p_phys / p_eff_i) ** (-eps) * Leff

        # split cog into human vs AI
        Hc_d = mu_h_c * (wc / p_cog) ** (-sig_c) * Lcog
        AI_d = (1 - mu_h_c) * (qc / p_cog) ** (-sig_c) * Lcog

        # split phys into human vs robots
        Hp_d = mu_h_p * (wp / p_phys) ** (-sig_p) * Lphys
        R_d = (1 - mu_h_p) * (qr / p_phys) ** (-sig_p) * Lphys

        return Hc_d, Hp_d, AI_d, R_d

    Hc_S, Hp_S, AI_S, R_S = sector_demands(S, P_S, p_eff_S, theta_S)
    Hc_G, Hp_G, AI_G, R_G = sector_demands(G, P_G, p_eff_G, theta_G)

    Hc_d = Hc_S + Hc_G
    Hp_d = Hp_S + Hp_G
    AI_d = AI_S + AI_G
    R_d = R_S + R_G

    return dict(
        r=r,
        P_S=P_S,
        P_G=P_G,
        P_index=P_index,
        S=S,
        G=G,
        p_cog=p_cog,
        p_phys=p_phys,
        p_eff_S=p_eff_S,
        p_eff_G=p_eff_G,
        Hc_s=Hc_s,
        Hp_s=Hp_s,
        ell_c=ell_c,
        Hc_d=Hc_d,
        Hp_d=Hp_d,
        AI_d=AI_d,
        R_d=R_d,
    )


def solve_prices_given_mu(
    Y: float,
    K: float,
    L: float,
    AI_cog: float,
    R_phys: float,
    mu_h_c: float,
    mu_h_p: float,
    params: dict,
    x0: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    tol: float = 1e-10,
    max_iter: int = 80,
) -> dict:
    """
    Newton solver for wc, wp, qc, qr given mu_h_c, mu_h_p.
    Market clearing: Hc_d = Hc_s, Hp_d = Hp_s, AI_d = AI_supply, R_d = R_supply
    """
    logx = np.log(np.array(x0, dtype=float))

    def residuals(logx):
        wc, wp, qc, qr = np.exp(logx)
        out = sector_block(
            Y=Y,
            K=K,
            L=L,
            AI_cog=AI_cog,
            R_phys=R_phys,
            mu_h_c=mu_h_c,
            mu_h_p=mu_h_p,
            wc=wc,
            wp=wp,
            qc=qc,
            qr=qr,
            params=params,
        )
        F = np.array(
            [
                np.log(out["Hc_d"] / max(out["Hc_s"], TINY)),
                np.log(out["Hp_d"] / max(out["Hp_s"], TINY)),
                np.log(out["AI_d"] / max(AI_cog, TINY)),
                np.log(out["R_d"] / max(R_phys, TINY)),
            ],
            dtype=float,
        )
        return F, out

    damping = 0.6
    out = None
    F = None
    for it in range(max_iter):
        F, out = residuals(logx)
        normF = float(np.max(np.abs(F)))
        if normF < tol:
            wc, wp, qc, qr = np.exp(logx)
            out.update(
                dict(
                    wc=float(wc),
                    wp=float(wp),
                    qc=float(qc),
                    qr=float(qr),
                    residual_max=normF,
                    iters=it + 1,
                )
            )
            return out

        # Jacobian
        J = np.zeros((4, 4))
        h = 1e-6
        for j in range(4):
            step = np.zeros(4)
            step[j] = h
            Fp, _ = residuals(logx + step)
            Fm, _ = residuals(logx - step)
            J[:, j] = (Fp - Fm) / (2 * h)

        dx = np.linalg.lstsq(J, -F, rcond=None)[0]

        # line search
        success = False
        for ls in range(20):
            cand = logx + (damping ** ls) * dx
            Fcand, _ = residuals(cand)
            if np.max(np.abs(Fcand)) < normF:
                logx = cand
                success = True
                break
        if not success:
            logx = logx + 0.1 * dx

    wc, wp, qc, qr = np.exp(logx)
    if out is not None:
        out.update(
            dict(
                wc=float(wc),
                wp=float(wp),
                qc=float(qc),
                qr=float(qr),
                residual_max=float(np.max(np.abs(F))) if F is not None else 1.0,
                iters=max_iter,
            )
        )
    return out


def bisect_root(func, lo: float, hi: float, tol: float = 1e-10, max_iter: int = 80) -> float:
    """Bisection root finder for automation margin adjustment."""
    flo = func(lo)
    fhi = func(hi)
    if abs(flo) < tol:
        return lo
    if abs(fhi) < tol:
        return hi
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    a, b, fa, fb = lo, hi, flo, fhi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = func(m)
        if abs(fm) < tol or (b - a) < 1e-12:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def solve_one_year(
    Y: float,
    K: float,
    L: float,
    AI_cog: float,
    R_phys: float,
    bar_auto_c: float,
    bar_auto_p: float,
    params: dict,
    guess: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> dict:
    """
    Solve one-year equilibrium with endogenous automation margins.

    If wages are below machine rentals, the automation margin backs off
    until wc=qc (or wp=qr).
    """
    barc = float(np.clip(bar_auto_c, 0, 1 - 1e-12))
    barp = float(np.clip(bar_auto_p, 0, 1 - 1e-12))

    # max automatable share => min human task share
    mu_h_c_min = float(np.clip(1.0 - barc, 1e-9, 1 - 1e-9))
    mu_h_p_min = float(np.clip(1.0 - barp, 1e-9, 1 - 1e-9))

    # start at frontier
    mu_c, mu_p = mu_h_c_min, mu_h_p_min
    base = solve_prices_given_mu(Y, K, L, AI_cog, R_phys, mu_c, mu_p, params, x0=guess)

    need_c = base["wc"] < base["qc"]
    need_p = base["wp"] < base["qr"]

    # coordinate bisection to find equilibrium margins
    for _ in range(25):
        prev = (mu_c, mu_p)

        if need_c:

            def f(mu):
                o = solve_prices_given_mu(Y, K, L, AI_cog, R_phys, mu, mu_p, params, x0=guess)
                return math.log(o["wc"] / o["qc"])

            mu_c = bisect_root(f, mu_h_c_min, 1 - 1e-9)

        if need_p:

            def g(mu):
                o = solve_prices_given_mu(Y, K, L, AI_cog, R_phys, mu_c, mu, params, x0=guess)
                return math.log(o["wp"] / o["qr"])

            mu_p = bisect_root(g, mu_h_p_min, 1 - 1e-9)

        if max(abs(mu_c - prev[0]), abs(mu_p - prev[1])) < 1e-10:
            break

    out = solve_prices_given_mu(Y, K, L, AI_cog, R_phys, mu_c, mu_p, params, x0=guess)
    out.update(
        dict(
            mu_h_c_min=mu_h_c_min,
            mu_h_p_min=mu_h_p_min,
            mu_h_c=mu_c,
            mu_h_p=mu_p,
            regime_c="endogenous" if need_c else "frontier",
            regime_p="endogenous" if need_p else "frontier",
        )
    )

    # profit check (share should be ~0)
    labor_bill = out["wc"] * out["Hc_s"] + out["wp"] * out["Hp_s"] + out["qc"] * AI_cog + out["qr"] * R_phys
    cap_bill = out["r"] * K
    out["profit"] = Y - labor_bill - cap_bill
    out["profit_share"] = out["profit"] / Y

    return out


def mu_auto_path(mu0: float, kappa: float, T: int, mu_max: float = 0.999999) -> np.ndarray:
    """
    Automation capability share mu_t with exponentially declining remaining tasks:
      1 - mu_t = (1 - mu0) * exp(-kappa * t)
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


def simulate_transition_2sector(
    *,
    T: int,
    K0: float,
    Y0: float,
    L0: float,
    AI0: float,
    R0: float,
    gY: float,
    gL: float,
    gAI: float,
    gR: float,
    s: float,
    delta: float,
    params: dict,
    mu_c0: float = 0.2,
    kappa_mu_c: float = 0.02,
    mu_p0: float = 0.2,
    kappa_mu_p: float = 0.02,
    guess: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> dict:
    """
    Simulate a two-sector transition economy.

    Exogenous paths (exponential growth):
      Y_t = Y0 * exp(gY * t)
      L_t = L0 * exp(gL * t)
      AI_t = AI0 * exp(gAI * t)
      R_t = R0 * exp(gR * t)

    Capital:
      K_{t+1} = (1-delta)*K_t + s*Y_t

    Automation capability:
      bar_auto_c = 1 - (1-mu_c0)*exp(-kappa_c*t)
      bar_auto_p = 1 - (1-mu_p0)*exp(-kappa_p*t)
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

    # Initialize arrays
    K = np.zeros(T + 1)
    Y = np.zeros(T)
    C = np.zeros(T)
    S_out = np.zeros(T)
    G_out = np.zeros(T)
    r = np.zeros(T)
    wc = np.zeros(T)
    wp = np.zeros(T)
    qc = np.zeros(T)
    qr = np.zeros(T)
    P_S = np.zeros(T)
    P_G = np.zeros(T)
    Hc_d = np.zeros(T)
    Hp_d = np.zeros(T)
    AI_d = np.zeros(T)
    R_d = np.zeros(T)
    mu_h_c = np.zeros(T)
    mu_h_p = np.zeros(T)
    regime_c = []
    regime_p = []
    profit_share = np.zeros(T)

    K[0] = float(K0)
    current_guess = guess

    residual_max_overall = 0.0

    for t in range(T):
        Y_t = float(Y0 * np.exp(gY * t))
        L_t = float(L0 * np.exp(gL * t))
        AI_t = float(AI0 * np.exp(gAI * t))
        R_t = float(R0 * np.exp(gR * t))

        bar_auto_c_t = float(mu_c_auto[t])
        bar_auto_p_t = float(mu_p_auto[t])

        sol = solve_one_year(
            Y=Y_t,
            K=float(K[t]),
            L=L_t,
            AI_cog=AI_t,
            R_phys=R_t,
            bar_auto_c=bar_auto_c_t,
            bar_auto_p=bar_auto_p_t,
            params=params,
            guess=current_guess,
        )

        Y[t] = Y_t
        S_out[t] = sol["S"]
        G_out[t] = sol["G"]
        r[t] = sol["r"]
        wc[t] = sol["wc"]
        wp[t] = sol["wp"]
        qc[t] = sol["qc"]
        qr[t] = sol["qr"]
        P_S[t] = sol["P_S"]
        P_G[t] = sol["P_G"]
        Hc_d[t] = sol["Hc_d"]
        Hp_d[t] = sol["Hp_d"]
        AI_d[t] = sol["AI_d"]
        R_d[t] = sol["R_d"]
        mu_h_c[t] = sol["mu_h_c"]
        mu_h_p[t] = sol["mu_h_p"]
        regime_c.append(sol["regime_c"])
        regime_p.append(sol["regime_p"])
        profit_share[t] = sol["profit_share"]

        C[t] = (1.0 - s) * Y_t
        K[t + 1] = (1.0 - delta) * K[t] + s * Y_t

        residual_max_overall = max(residual_max_overall, float(sol.get("residual_max", 0)))

        current_guess = (sol["wc"], sol["wp"], sol["qc"], sol["qr"])

    return dict(
        K=K,
        Y=Y,
        C=C,
        S=S_out,
        G=G_out,
        r=r,
        wc=wc,
        wp=wp,
        qc=qc,
        qr=qr,
        P_S=P_S,
        P_G=P_G,
        Hc=Hc_d,
        Hp=Hp_d,
        AI=AI_d,
        R=R_d,
        mu_h_c=mu_h_c,
        mu_h_p=mu_h_p,
        regime_c=regime_c,
        regime_p=regime_p,
        profit_share=profit_share,
        diagnostics=dict(
            residual_max=float(residual_max_overall),
        ),
    )
