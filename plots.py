from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model import household_labor_supply


def results_dataframe(sim: dict) -> pd.DataFrame:
    """
    Long-ish tidy table for exporting/inspection.
    Note: K has length T+1; most others have length T.
    """
    T = len(sim["Y"])
    t = np.arange(T)

    df = pd.DataFrame({
        "t": t,
        "Y": sim["Y"],
        "C": sim["C"],
        "r": sim["r"],
        "w_c": sim["wc"],
        "w_p": sim["wp"],
        "q_c": sim["qc"],
        "q_r": sim["qr"],
        "Lhc (human cog demand)": sim["Lhc"],
        "LAI (AI services)": sim["LAI"],
        "Lhp (human phys demand)": sim["Lhp"],
        "LR (robot services)": sim["LR"],
        "KY": sim["KY"],
        "KAI": sim["KAI"],
        "KR": sim["KR"],
        "mu_c (AI automatable share)": sim["mu_c"],
        "mu_p (robot automatable share)": sim["mu_p"],
        "K_t": sim["K"][:-1],
        "K_{t+1}": sim["K"][1:],
    })
    return df


def _fig_ax():
    fig, ax = plt.subplots()
    return fig, ax


def make_figures(sim: dict, *, params: dict, exog: dict) -> list[tuple[str, plt.Figure]]:
    """
    Streamlit-friendly version of your plot_paths(): returns matplotlib figures instead of showing them.
    """
    K  = sim["K"]     # T+1
    Y  = sim["Y"]     # T
    r  = sim["r"]     # T
    wc = sim["wc"]    # T
    wp = sim["wp"]    # T
    qc = sim["qc"]
    qr = sim["qr"]

    T = len(Y)
    t = np.arange(T)
    tK = np.arange(T + 1)

    figs: list[tuple[str, plt.Figure]] = []

    # Plot 1: wages
    fig, ax = _fig_ax()
    ax.plot(t, wp, label="w_p (physical wage)")
    ax.plot(t, wc, label="w_c (cognitive wage)")
    ax.set_xlabel("t")
    ax.set_ylabel("w")
    ax.set_title("Wages over time")
    ax.legend()
    fig.tight_layout()
    figs.append(("Wages", fig))

    # Plot 2: machine rentals
    fig, ax = _fig_ax()
    ax.plot(t, qc, label="q^c (AI services rental)")
    ax.plot(t, qr, label="q^r (robot services rental)")
    ax.set_xlabel("t")
    ax.set_ylabel("q")
    ax.set_title("Rental rates for AI and robot services")
    ax.legend()
    fig.tight_layout()
    figs.append(("AI/robot rental rates", fig))

    # Plot 3: capital stock
    fig, ax = _fig_ax()
    ax.plot(tK, K, label="K")
    ax.set_xlabel("t")
    ax.set_ylabel("Capital")
    title = "Capital over time"
    if "s" in exog and "delta" in exog:
        title += f"  (s={exog['s']}, δ={exog['delta']})"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    figs.append(("Aggregate capital K", fig))

    # Plot 4: output
    fig, ax = _fig_ax()
    ax.plot(t, Y, label="Y")
    ax.set_xlabel("t")
    ax.set_ylabel("Output")
    ax.set_title("Output over time")
    ax.legend()
    fig.tight_layout()
    figs.append(("Output Y", fig))

    # Plot 5: physical decomposition
    fig, ax = _fig_ax()
    ax.plot(t, sim["Lhp"], label="Human physical labor L_p^H")
    ax.plot(t, sim["LR"], label="Robot services L^R")
    ax.set_xlabel("t")
    ax.set_ylabel("Quantity")
    ax.set_title("Physical tasks: human vs robots")
    ax.legend()
    fig.tight_layout()
    figs.append(("Physical tasks decomposition", fig))

    # Plot 6: cognitive decomposition
    fig, ax = _fig_ax()
    ax.plot(t, sim["Lhc"], label="Human cognitive labor L_c^H")
    ax.plot(t, sim["LAI"], label="AI services L^{AI}")
    ax.set_xlabel("t")
    ax.set_ylabel("Quantity")
    ax.set_title("Cognitive tasks: human vs AI")
    ax.legend()
    fig.tight_layout()
    figs.append(("Cognitive tasks decomposition", fig))

    # Plot 7: capital allocation
    fig, ax = _fig_ax()
    ax.plot(t, sim["KY"],  label="K^Y (final goods)")
    ax.plot(t, sim["KAI"], label="K_AI (AI services capital)")
    ax.plot(t, sim["KR"],  label="K_R (robot capital)")
    ax.set_xlabel("t")
    ax.set_ylabel("Capital")
    ax.set_title("Capital allocation across uses")
    ax.legend()
    fig.tight_layout()
    figs.append(("Capital allocation", fig))

    # Plot 8: automation capability paths
    fig, ax = _fig_ax()
    ax.plot(t, sim["mu_c"], label="mu_c (AI automatable share)")
    ax.plot(t, sim["mu_p"], label="mu_p (robot automatable share)")
    ax.set_xlabel("t")
    ax.set_ylabel("Share")
    ax.set_title("Exogenous automation capability μ_t")
    ax.legend()
    fig.tight_layout()
    figs.append(("Automation capability paths", fig))

    # Plot 9: household labor supply (computed from wages)
    Lc_s = np.zeros(T)
    Lp_s = np.zeros(T)
    for i in range(T):
        Lc_s[i], Lp_s[i] = household_labor_supply(wc[i], wp[i], params["Lbar"], params["tau"])

    fig, ax = _fig_ax()
    ax.plot(t, Lp_s, label="Household supply: L_p^H")
    ax.plot(t, Lc_s, label="Household supply: L_c^H")
    ax.set_xlabel("t")
    ax.set_ylabel("Human labor supplied")
    ax.set_title("Household supplied labor: cognitive vs physical")
    ax.legend()
    fig.tight_layout()
    figs.append(("Household labor supply", fig))

    # Plot 10: sufficient savings condition check q < w  <=>  r < phi*X*w
    Xc0, gXc = float(exog["Xc0"]), float(exog["gXc"])
    Xp0, gXp = float(exog["Xp0"]), float(exog["gXp"])
    Xc = Xc0 * np.exp(gXc * t)
    Xp = Xp0 * np.exp(gXp * t)
    rhs_c = params["phi_c"] * Xc * wc
    rhs_p = params["phi_p"] * Xp * wp

    fig, ax = _fig_ax()
    ax.plot(t, r, label="r")
    ax.plot(t, rhs_c, label="phi_c * Xc * w_c")
    ax.plot(t, rhs_p, label="phi_p * Xp * w_p")
    ax.set_xlabel("t")
    ax.set_ylabel("Level")
    ax.set_title("Sufficient-savings condition: require q< w ⇔ r < ϕ·X·w")
    ax.legend()
    fig.tight_layout()
    figs.append(("Sufficient-savings condition", fig))

    return figs
