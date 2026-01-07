import streamlit as st
import pandas as pd

from src.model import simulate_transition
from src.plots import make_figures, results_dataframe

st.set_page_config(page_title="AI + Robots Growth Model", layout="wide")

st.title("AI + Robots Growth Model (One-sector)")
st.caption(
    "Calibrate parameters, run the transition, and view the plots. "
    "Starting capital is fixed at K₀ = 1."
)

# -----------------------------
# Defaults (edit as you like)
# -----------------------------
DEFAULTS = dict(
    # horizon
    T=20,

    # exogenous paths
    A0=1.0, gA=0.00,
    Xc0=1.0, gXc=0.03,
    Xp0=1.0, gXp=0.05,

    # savings & depreciation
    s=0.20, delta=0.20,

    # technology / nesting
    alpha=0.30,
    theta=0.50,
    eps=0.40,      # substitution between cog/phys in L_eff (complements if < 1)
    sig_c=0.50,    # substitution within cognitive nest (human vs AI; complements if < 1)
    sig_p=0.50,    # substitution within physical nest (human vs robots; complements if < 1)

    # machine-service productivity (maps capital -> services)
    phi_c=1.0,
    phi_p=1.0,

    # household labor supply / reallocation elasticity
    Lbar=1.0,
    tau=1.50,      # must be > 1

    # automation capability paths (shares automatable by AI/robots)
    mu_c0=0.20, kappa_mu_c=0.02,
    mu_p0=0.20, kappa_mu_p=0.02,

    # solver warm start
    r0=0.06, wc0=1.0, wp0=1.0,
)

def _num(name, **kwargs):
    return st.sidebar.number_input(name, **kwargs)

with st.sidebar:
    st.header("Run settings")

    T = int(_num("Horizon T", min_value=1, max_value=400, value=int(DEFAULTS["T"]), step=1,
                 help="Number of periods to simulate."))

    st.divider()
    st.subheader("Initial condition")
    st.text_input("K₀ (fixed)", value="1", disabled=True,
                  help="Starting capital is fixed at 1, as requested.")

    st.divider()
    st.subheader("Exogenous paths (exponential for now; take estimates from AIFP)")
    A0  = _num("A₀ (TFP level)", value=float(DEFAULTS["A0"]), step=0.1,
              help="TFP level in period 0. Must be positive.")
    gA  = _num("gA (TFP growth rate)", value=float(DEFAULTS["gA"]), step=0.005,
              help="Continuous growth rate for A_t = A₀·exp(gA·t). Can be negative or positive.")

    Xc0 = _num("Xc₀ (compute capability)", value=float(DEFAULTS["Xc0"]), step=0.1,
              help="Compute/cognitive capability index in period 0. Must be positive.")
    gXc = _num("gXc (compute growth rate)", value=float(DEFAULTS["gXc"]), step=0.005,
              help="Continuous growth rate for Xc_t = Xc₀·exp(gXc·t).")

    Xp0 = _num("Xp₀ (robotics capability)", value=float(DEFAULTS["Xp0"]), step=0.1,
              help="Robotics/physical capability index in period 0. Must be positive.")
    gXp = _num("gXp (robotics growth rate)", value=float(DEFAULTS["gXp"]), step=0.005,
              help="Continuous growth rate for Xp_t = Xp₀·exp(gXp·t).")

    st.divider()
    st.subheader("Savings & depreciation")
    s = _num("s (savings rate)", min_value=0.0, max_value=0.999, value=float(DEFAULTS["s"]), step=0.01,
             help="Fixed savings rate in K_{t+1}=(1-δ)K_t + s·Y_t. Must be in [0,1).")
    delta = _num("δ (depreciation)", min_value=0.0, max_value=0.999, value=float(DEFAULTS["delta"]), step=0.01,
                 help="Capital depreciation rate. Must be in [0,1).")

    st.divider()
    st.subheader("Technology (shares)")
    alpha = _num("α (capital share)", min_value=0.0, max_value=0.999, value=float(DEFAULTS["alpha"]), step=0.01,
                 help="Capital share in Cobb–Douglas Y = A·K^α·L_eff^(1-α). Must be in (0,1).")
    theta = _num("θ (cognitive task weight)", min_value=0.0, max_value=0.999, value=float(DEFAULTS["theta"]), step=0.01,
                 help="Weight on cognitive tasks in the effective labor CES aggregator. Must be in (0,1).")

    st.divider()
    st.subheader("Technology (substitution elasticities)")
    eps = _num("ε (cog vs phys)", value=float(DEFAULTS["eps"]), step=0.05,
              help="CES substitution parameter between cognitive and physical task aggregates in L_eff. "
                   "For complements, set ε < 1. Avoid exactly 1.")
    sig_c = _num("σc (human vs AI)", value=float(DEFAULTS["sig_c"]), step=0.05,
                help="CES substitution parameter within the cognitive nest. For complements, set σc < 1. Avoid exactly 1.")
    sig_p = _num("σp (human vs robots)", value=float(DEFAULTS["sig_p"]), step=0.05,
                help="CES substitution parameter within the physical nest. For complements, set σp < 1. Avoid exactly 1.")

    st.divider()
    st.subheader("Machine productivity (services per unit capital)")
    phi_c = _num("ϕc (AI services scale)", min_value=1e-9, value=float(DEFAULTS["phi_c"]), step=0.1,
                help="Maps compute capital to AI services: L_AI = ϕc·Xc_t·K_AI. Must be positive.")
    phi_p = _num("ϕp (robot services scale)", min_value=1e-9, value=float(DEFAULTS["phi_p"]), step=0.1,
                help="Maps robot capital to robot services: L_R = ϕp·Xp_t·K_R. Must be positive.")

    st.divider()
    st.subheader("Household labor")
    Lbar = _num("L̄ (labor budget)", min_value=1e-9, value=float(DEFAULTS["Lbar"]), step=0.1,
               help="Total labor endowment in the CES labor constraint. Must be positive.")
    tau = _num("τ (labor curvature)", min_value=1.0001, value=float(DEFAULTS["tau"]), step=0.05,
              help="Household constraint: Lc^τ + Lp^τ = L̄^τ with τ>1 to keep both supplies interior.")

    st.divider()
    st.subheader("Automation capability paths")
    mu_c0 = _num("μc₀ (AI automatable share at t=0)", min_value=0.0, max_value=0.999999,
                value=float(DEFAULTS["mu_c0"]), step=0.05,
                help="Initial fraction of cognitive tasks that are automatable by AI (increases over time). Must be in [0,1).")
    kappa_mu_c = _num("κc (AI automation speed)", min_value=0.0, value=float(DEFAULTS["kappa_mu_c"]), step=0.01,
                     help="Speed of automation expansion for AI: 1-μ_t = (1-μ0)·exp(-κ·t). Must be ≥ 0.")

    mu_p0 = _num("μp₀ (robot automatable share at t=0)", min_value=0.0, max_value=0.999999,
                value=float(DEFAULTS["mu_p0"]), step=0.05,
                help="Initial fraction of physical tasks automatable by robots (increases over time). Must be in [0,1).")
    kappa_mu_p = _num("κp (robot automation speed)", min_value=0.0, value=float(DEFAULTS["kappa_mu_p"]), step=0.01,
                     help="Speed of automation expansion for robots: 1-μ_t = (1-μ0)·exp(-κ·t). Must be ≥ 0.")

    st.divider()
    st.subheader("Solver warm start")
    r0  = _num("r guess", min_value=1e-8, value=float(DEFAULTS["r0"]), step=0.01,
              help="Initial guess for the rental rate of capital in the within-period solver.")
    wc0 = _num("w_c guess", min_value=1e-8, value=float(DEFAULTS["wc0"]), step=0.1,
              help="Initial guess for the cognitive human wage.")
    wp0 = _num("w_p guess", min_value=1e-8, value=float(DEFAULTS["wp0"]), step=0.1,
              help="Initial guess for the physical human wage.")

# quick validation guidance (non-blocking)
warnings = []
for name, val in [("ε", eps), ("σc", sig_c), ("σp", sig_p)]:
    if abs(val - 1.0) < 1e-6:
        warnings.append(f"{name} is extremely close to 1; CES formulas become log-CES and the solver may be unstable.")
    if val >= 1.0:
        warnings.append(f"{name} ≥ 1. You said you want complements; consider setting {name} < 1.")

if warnings:
    st.warning("Parameter notes:\n- " + "\n- ".join(warnings))

params = dict(
    alpha=float(alpha),
    theta=float(theta),
    eps=float(eps),
    sig_c=float(sig_c),
    sig_p=float(sig_p),
    phi_c=float(phi_c),
    phi_p=float(phi_p),
    Lbar=float(Lbar),
    tau=float(tau),
)

run = st.button("Run simulation", type="primary", use_container_width=True)

if run:
    sim = simulate_transition(
        T=T,
        K0=1.0,
        A0=float(A0), gA=float(gA),
        Xc0=float(Xc0), gXc=float(gXc),
        Xp0=float(Xp0), gXp=float(gXp),
        s=float(s), delta=float(delta),
        params=params,
        mu_c0=float(mu_c0), kappa_mu_c=float(kappa_mu_c),
        mu_p0=float(mu_p0), kappa_mu_p=float(kappa_mu_p),
        guess=(float(r0), float(wc0), float(wp0)),
    )

    df = results_dataframe(sim)

    # ---- Summary header ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("K_T", f"{sim['K'][-1]:.4f}")
    col2.metric("Y_{T-1}", f"{sim['Y'][-1]:.4f}")
    col3.metric("r_{T-1}", f"{sim['r'][-1]:.4f}")
    col4.metric("Solver residual (max)", f"{sim['diagnostics']['residual_max']:.2e}")

    if not sim["diagnostics"]["sufficient_savings_ok"]:
        st.error(
            "Sufficient-savings condition violated in at least one period: "
            "q_c < w_c and/or q_r < w_p failed. See Plot 10."
        )

    # ---- Table + download ----
    st.subheader("Simulation results table")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", data=csv, file_name="simulation_results.csv",
                       mime="text/csv", use_container_width=True)

    st.divider()

    st.subheader("Plots")
    figs = make_figures(sim, params=params, exog=dict(Xc0=Xc0, gXc=gXc, Xp0=Xp0, gXp=gXp, s=s, delta=delta))
    for title, fig in figs:
        st.markdown(f"#### {title}")
        st.pyplot(fig, clear_figure=True)

else:
    st.info("Set parameters in the sidebar, then click **Run simulation**.")
