# Economic Model: Technical Writeup

## 1. Overview

This model projects the economic consequences of AI and robotics adoption over a 16-year horizon (2025–2040). It combines a nested CES production function with endogenous capital accumulation, labor supply, two-sector price dynamics, a consumption-side land market, an optional trusted-labor sector, and a full income distribution module.

All quantities are in **real terms** — output $Y$ serves as the numéraire (composite good), and sector prices are expressed relative to it.

---

## 2. Factor Inputs

The model tracks five time-varying factor inputs, provided exogenously in the scenario table:

| Symbol | Description |
|--------|-------------|
| $N$ | Working-age population (WAP) |
| $H$ | Human labor force (endogenous or exogenous) |
| $A^{\text{AI}}_c$ | AI cognitive worker-equivalents |
| $R_p$ | Robot physical worker-equivalents |
| $K$ | Physical capital stock (endogenous via accumulation) |

The scenario table provides annual values for WAP, labor force, AI copies, robots, and background output/capital paths. When **predicted output** mode is active (the standard mode), $Y$ and $K$ are solved endogenously rather than read from the table.

---

## 3. Production Function

### 3.1 Effective Labor: Three-Level CES Nesting

Labor services are aggregated through a three-level CES nest that captures substitution between humans and machines at each task type.

**Level 1 — Sub-task nests.** Cognitive and physical tasks each combine a human input with its machine substitute:

$$L_{\text{cog}} = \left[\mu_{h,c}^{1/\sigma_c} \cdot H_c^{\,\rho_c} + (1-\mu_{h,c})^{1/\sigma_c} \cdot (A^{\text{AI}}_c)^{\rho_c}\right]^{1/\rho_c}$$

$$L_{\text{phys}} = \left[\mu_{h,p}^{1/\sigma_p} \cdot H_p^{\,\rho_p} + (1-\mu_{h,p})^{1/\sigma_p} \cdot (R_p^L)^{\rho_p}\right]^{1/\rho_p}$$

where $\rho_j = (\sigma_j - 1)/\sigma_j$, and $\mu_{h,j}$ is the endogenous human share (see §4).

$R_p^L = \varphi \cdot R_p$ is the fraction of robots allocated to the labor channel (see §3.3).

**Level 2 — Task aggregator.** Cognitive and physical labor combine into effective labor:

$$L_{\text{eff}} = \left[\theta^{1/\varepsilon} \cdot L_{\text{cog}}^{\,\rho_\varepsilon} + (1-\theta)^{1/\varepsilon} \cdot L_{\text{phys}}^{\,\rho_\varepsilon}\right]^{1/\rho_\varepsilon}$$

where $\theta$ is the cognitive task weight and $\varepsilon$ is the cross-task elasticity.

**Human occupational split.** Total labor $L$ divides between cognitive ($H_c$) and physical ($H_p$) tasks based on the equilibrium wage ratio $z = w_c / w_p$:

$$\frac{H_c}{H_p} = \kappa \cdot z^{\,\omega}$$

where $\kappa$ is a calibrated scale parameter and $\omega$ captures occupational mobility. Given total labor $L = H_c + H_p$:

$$\ell_c \equiv \frac{H_c}{L} = \frac{\kappa \, z^\omega}{\kappa \, z^\omega + 1}$$

### 3.2 Effective Capital: K_eff Nesting

Robots can substitute for capital as well as labor. A fraction $(1-\varphi)$ of robot worker-equivalents is allocated to the capital channel via a CES nest:

$$K_{\text{eff}} = K_{Y,0} \cdot \left[\nu_K^{1/\sigma_{K_{\text{eff}}}} \cdot \left(\frac{K_Y}{K_{Y,0}}\right)^{\rho_{K_{\text{eff}}}} + (1-\nu_K)^{1/\sigma_{K_{\text{eff}}}} \cdot \left(\frac{R_p^K}{R_{p,0}}\right)^{\rho_{K_{\text{eff}}}}\right]^{1/\rho_{K_{\text{eff}}}}$$

where:
- $K_Y$ is productive capital (capital allocated to output production, as opposed to AI/robot capital)
- $R_p^K = (1-\varphi) \cdot R_p$ is the capital-allocated robot fraction
- $K_{Y,0}$ and $R_{p,0}$ are base-year normalization values
- $\nu_K$ is the capital share within this nest
- $\sigma_{K_{\text{eff}}}$ is the capital–robot substitution elasticity

The normalization ensures that at base year, $K_{\text{eff}} \approx K_Y$ (since robots are negligible initially). $R_{p,0}$ is set to base-year $H_p$ so that when robots reach human-equivalent physical labor, $R_p^K / R_{p,0} \approx 1$.

### 3.3 Robot Split: $\varphi$ Allocation

Robots can contribute through two channels: labor (replacing physical workers) or capital (augmenting productive capital). The split $\varphi$ is determined by equating the marginal product across both channels:

$$\underbrace{\frac{\partial Y}{\partial L_{\text{eff}}} \cdot \frac{\partial L_{\text{eff}}}{\partial L_{\text{phys}}} \cdot \frac{\partial L_{\text{phys}}}{\partial R_p^L}}_{\text{labor-channel MP}} = \underbrace{\frac{\partial Y}{\partial K_{\text{eff}}} \cdot \frac{\partial K_{\text{eff}}}{\partial R_p^K}}_{\text{capital-channel MP}}$$

This is solved numerically by bisection on $\varphi \in [0,1]$ at each time step.

### 3.4 Top-Level Production Function

Output is Cobb-Douglas in effective capital and effective labor:

$$Y = A \cdot K_{\text{eff}}^{\,\alpha} \cdot L_{\text{eff}}^{\,1-\alpha}$$

where $A$ is total factor productivity (calibrated at base year) and $\alpha$ is the capital share.

**TFP Calibration.** At base year $t=0$:

$$A = \frac{Y_0}{K_{\text{eff},0}^{\,\alpha} \cdot L_{\text{eff},0}^{\,1-\alpha}}$$

TFP is held constant across all years — output growth is driven entirely by factor accumulation (capital deepening, AI/robot adoption) and the changing composition of labor.

---

## 4. Endogenous Automation: $\mu$ Determination

The human shares $\mu_{h,c}$ and $\mu_{h,p}$ are bounded below by the automation frontier:

$$\mu_{h,j} \geq 1 - \bar{a}_j$$

where $\bar{a}_j \in [0,1]$ is the maximum automatable share of task type $j$ (given exogenously by the scenario).

**Regime selection.** For each task type, the solver checks whether the human wage exceeds the machine marginal product at the frontier:

- If $w_j > q_j$ at $\mu_{h,j} = 1 - \bar{a}_j$: **Frontier regime.** All automatable tasks are automated. $\mu_{h,j} = 1 - \bar{a}_j$.
- If $w_j < q_j$ at the frontier: **Endogenous regime.** $\mu_{h,j}$ is found by bisection so that $w_j = q_j$ (no-arbitrage between human and machine labor).

At full automation ($\bar{a}_j \to 1$), human wages are capped at the machine marginal product to enforce no-arbitrage.

---

## 5. Factor Prices

All factor prices are derived as marginal products through the chain rule of the nested CES structure. For the cognitive human wage:

$$w_c = P_Y \cdot \frac{\partial Y}{\partial K_{\text{eff}}} \bigg|_{\text{chain}} \cdot \frac{\partial L_{\text{eff}}}{\partial L_{\text{cog}}} \cdot \frac{\partial L_{\text{cog}}}{\partial H_c}$$

More explicitly:

$$w_c = P_Y \cdot \underbrace{(1-\alpha)\frac{Y}{L_{\text{eff}}}}_{\text{MP of } L_{\text{eff}}} \cdot \underbrace{\frac{\partial L_{\text{eff}}}{\partial L_{\text{cog}}}}_{\text{task agg}} \cdot \underbrace{\frac{\partial L_{\text{cog}}}{\partial H_c}}_{\text{sub-nest}}$$

Similarly for $w_p$, $q_c$ (AI marginal product), and $q_r$ (robot marginal product via the labor channel). The CES derivative for a two-input aggregator $Q = [\mu^{1/\sigma} x^\rho + (1-\mu)^{1/\sigma} y^\rho]^{1/\rho}$ with respect to its first input is:

$$\frac{\partial Q}{\partial x} = Q^{1-\rho} \cdot \mu^{1/\sigma} \cdot x^{\rho-1}$$

$P_Y$ is the shadow price of output from the trust layer (§8); it equals 1 when the trust layer is inactive.

**Interest rate.** The return on productive capital $r$ is determined by no-arbitrage in the capital market. When K_eff nesting is active:

$$r = \frac{\partial Y}{\partial K_{\text{eff}}} \cdot \frac{\partial K_{\text{eff}}}{\partial K_Y} = \alpha \frac{Y}{K_{\text{eff}}} \cdot \frac{\partial K_{\text{eff}}}{\partial K_Y}$$

### 5.1 Capital Market Clearing

Total capital $K$ is allocated across three uses:

$$K = K_Y + K_C + K_R$$

where:
- $K_Y$: productive capital (in production function)
- $K_C$: capital embodied in AI systems
- $K_R$: capital embodied in robots

The AI and robot capital stocks are pinned by no-arbitrage between investing in physical capital (earning $r$) and investing in AI/robots (earning their marginal product minus depreciation, net of tax):

$$K_C = \frac{(1-\tau_{\text{AI}}) \cdot q_c \cdot A^{\text{AI}}_c}{(1-\tau_K) \cdot r - \delta_K + \delta_C}$$

$$K_R = \frac{(1-\tau_R) \cdot q_r \cdot R_p}{(1-\tau_K) \cdot r - \delta_K + \delta_R}$$

The interest rate $r$ is found by bisection over the market-clearing condition $K_Y(r) + K_C(r) + K_R(r) = K$.

---

## 6. Capital Accumulation

Capital evolves via a Solow-style accumulation equation:

$$K_{t+1} = s \cdot Y_t + (1 - \delta_K) \cdot K_t$$

where $s$ is the gross savings rate and $\delta_K$ is the depreciation rate. Initial capital is $K_0 = (\text{K/Y ratio}) \times Y_0$.

---

## 7. Endogenous Labor Supply

Labor force participation responds to wages and UBI through a logistic participation model:

$$\text{LFP}(\bar{w}, \bar{w}_0, u) = F\!\left(\frac{\bar{w}}{\bar{w}_0}\right) \cdot D\!\left(\frac{u}{\bar{w}}\right)$$

**Baseline curve** $F$: A logistic function on $\log(w/w_0)$, auto-calibrated so $F(1) = \text{LFP}_{\text{target}}$:

$$F(w_{\text{mult}}) = \frac{1}{1 + \exp\!\bigl(-s_L(\ln w_{\text{mult}} - \text{shift})\bigr)}$$

where $\text{shift} = \ln(1/\text{LFP}_{\text{target}} - 1) / s_L$.

**UBI dampener** $D$: Captures the participation-reducing effect of unconditional transfers:

$$D = \exp\!\left(-k_{\text{ubi}} \cdot \frac{u}{\bar{w}}\right)$$

The average wage $\bar{w}$ is:

$$\bar{w} = w_c \cdot \ell_c + w_p \cdot (1 - \ell_c)$$

The endogenous labor force is $L = \text{LFP} \times N$, solved jointly with output via nested fixed-point iteration (inner loop on $L$, outer loop on $Y$).

---

## 8. Trusted Labor Sector

A small sector of "trusted labor" — jobs where human judgment, accountability, or fiduciary trust are essential — sits atop the main production function as a CES complement:

$$\tilde{Y} = \left[s_T \cdot X_T^{\,\rho_T} + (1 - s_T) \cdot Y^{\,\rho_T}\right]^{1/\rho_T}$$

where $X_T = C_T(t) \cdot H_T$ is the efficiency-adjusted trusted labor input, $C_T(t)$ is a time-varying efficiency multiplier, and $\sigma_T$ is the elasticity of substitution between trusted labor and everything else (typically low, making trusted labor a bottleneck).

**Trust supply.** $H_T$ is determined by an inelastic supply curve:

$$H_T = H_{T,0} \cdot \left(\frac{w_T}{\bar{w}}\right)^{\varepsilon_T}$$

where $\varepsilon_T \approx 0.2$ (inelastic — reflecting licensing and credential constraints), and $w_T$ is the marginal product of trusted labor:

$$w_T = \frac{\partial \tilde{Y}}{\partial X_T} \cdot C_T$$

The shadow price of upstream output becomes $P_Y = \partial \tilde{Y} / \partial Y$, which multiplies all factor prices.

**Calibration.** $s_T$ is calibrated at the trust activation year so that trusted workers earn the specified initial total income.

---

## 9. Two-Sector Price Decomposition (Goods vs. Services)

The economy produces two sectors — services (cognitive-intensive) and goods (physical-intensive) — that differ only in their cognitive task weight $\theta_j$:

$$\theta_S > \theta > \theta_G$$

Since sectors share common factor markets, sector prices are derived from factor prices post-equilibrium.

### 9.1 Sector Price Construction

**Sub-nest prices** (dual CES price indices, common across sectors):

$$p_{\text{cog}} = \left[\mu_{h,c} \cdot w_c^{1-\sigma_c} + (1-\mu_{h,c}) \cdot q_c^{1-\sigma_c}\right]^{1/(1-\sigma_c)}$$

$$p_{\text{phys}} = \left[\mu_{h,p} \cdot w_p^{1-\sigma_p} + (1-\mu_{h,p}) \cdot q_r^{1-\sigma_p}\right]^{1/(1-\sigma_p)}$$

**Sector effective labor prices:**

$$p_{\text{eff},j} = \left[\theta_j \cdot p_{\text{cog}}^{1-\varepsilon} + (1-\theta_j) \cdot p_{\text{phys}}^{1-\varepsilon}\right]^{1/(1-\varepsilon)}$$

**Relative sector prices:** Since sectors share the same capital cost $r$ and TFP $A$:

$$P_j = \left(\frac{p_{\text{eff},j}}{p_{\text{eff,agg}}}\right)^{1-\alpha}$$

normalized so the CES price aggregator equals 1.

### 9.2 CES Demand

Consumer demand across sectors follows CES preferences:

$$S = \nu \cdot P_S^{-\eta} \cdot Y, \qquad G = (1-\nu) \cdot P_G^{-\eta} \cdot Y$$

where $\eta$ is the demand elasticity and $\nu$ is calibrated from base-year service spending share $a_S$:

$$\nu = \frac{a_S \cdot P_{G,0}^{1-\eta}}{a_S \cdot P_{G,0}^{1-\eta} + (1-a_S) \cdot P_{S,0}^{1-\eta}}$$

As AI automates cognitive tasks, $p_{\text{cog}}$ falls, $P_S$ falls relative to $P_G$, and the service share evolves endogenously (Baumol's cost disease reversal).

---

## 10. Consumption-Side Land Market

Land is modeled on the consumption side, not in the production function. Total expenditure $E_t = (1-s) \cdot Y_t$ is allocated between non-land consumption and land services.

### 10.1 Three Endogenous Land Categories

For each endogenous category $i \in \{\text{urban}, \text{rural}, \text{agricultural}\}$:

**Demand:**
$$M_i^D = A_i \cdot E_t^{\,\beta_i} \cdot v_i^{-\eta_{D,i}}$$

**Supply:**
$$M_i^S = B_i \cdot v_i^{\,\eta_{S,i}}$$

where $v_i$ is the rent per hectare, $M_i$ is hectares, $\beta_i$ is the income elasticity, $\eta_{D,i}$ is the demand price elasticity, and $\eta_{S,i}$ is the supply elasticity.

**Equilibrium.** Setting $M^D = M^S$ and solving:

$$v_i = \left(\frac{A_i}{B_i}\right)^{1/(\eta_{D,i}+\eta_{S,i})} \cdot E_t^{\,\beta_i / (\eta_{D,i}+\eta_{S,i})}$$

$$M_i = B_i \cdot v_i^{\,\eta_{S,i}}$$

**Total land rent:**
$$R_{\text{total}} = \sum_i v_i \cdot M_i$$

### 10.2 Land Constraints

Two buffer categories (commercial/industrial, wilderness) have protection floors. Wilderness is 100% protected until a deregulation year, then ramps linearly to a target protection rate. If endogenous categories would exceed available land (total minus protected), all are scaled proportionally, and rents are re-derived from the demand curve at constrained acreage (ensuring scarcity raises prices).

### 10.3 Calibration

At base year, $A_i$ and $B_i$ are calibrated so that:
- Category spending matches the observed category expenditure share
- Category acreage matches the observed physical area

$$v_{i,0} = \frac{\text{cat\_exp\_share}_i \cdot \text{land\_exp\_share} \cdot E_0}{M_{i,0}}$$

$$A_i = \frac{M_{i,0}}{E_0^{\,\beta_i} \cdot v_{i,0}^{-\eta_{D,i}}}, \qquad B_i = \frac{M_{i,0}}{v_{i,0}^{\,\eta_{S,i}}}$$

---

## 11. Taxation and Universal Basic Income

A tax-and-transfer system is activated at a user-specified start year:

**Tax base:** Proportional taxes on capital income ($\tau_K$), AI income ($\tau_{\text{AI}}$), and robot income ($\tau_R$):

$$T = \tau_K \cdot r \cdot K_Y + \tau_{\text{AI}} \cdot q_c \cdot A^{\text{AI}}_c + \tau_R \cdot q_r \cdot R_p$$

**UBI:** A fraction of tax revenue is distributed equally to all working-age persons:

$$u = \frac{T \cdot \phi_{\text{UBI}}}{N}$$

where $\phi_{\text{UBI}}$ is the share of tax revenue allocated to UBI (region-specific). Before the start year, tax rates are zero.

---

## 12. Income Distribution

### 12.1 Household Structure

The population is divided into 109 percentile buckets: 99 buckets of 1% each (percentiles 1–99) plus 10 fine-grained buckets of 0.1% each (99.1–100), providing high resolution at the top.

Each household $h$ is characterized by:
- **Skill** $s_h$: relative wage-earning ability (population-weighted mean = 1)
- **Ownership shares**: $\omega_{K,h}$, $\omega_{\text{AI},h}$, $\omega_{R,h}$, $\omega_{\text{Land},h}$

### 12.2 Parameterization via Lorenz Curves and Multiplier Curves

**Ownership shares** are derived from empirical Lorenz curves. For each asset type, 7 knot points define a piecewise-linear Lorenz curve $L(p)$ at percentiles $\{1, 25, 50, 75, 90, 99, 99.9\}$. The ownership share of bucket $h$ spanning $[p_{\text{lo}}, p_{\text{hi}}]$ is:

$$\omega_h = L(p_{\text{hi}}) - L(p_{\text{lo}})$$

**Wage skills** use the same Lorenz approach: the slope $dL/dp$ at each bucket's midpoint gives relative income, normalized to mean 1.

**Smooth distribution** (for charts): Income multiplier curves provide a PCHIP interpolator in log-space at 9 anchor percentiles. At percentile $p$, per-person income from source $j$ is:

$$y_{j}(p) = f_j(p) \cdot \bar{y}_j$$

where $f_j(p)$ is the multiplier and $\bar{y}_j$ is mean per-person income from source $j$.

### 12.3 Per-Person Income

For each year and bucket $h$, per-working-age-person income is:

$$y_h = \underbrace{w_h}_{\text{wage}} + \underbrace{k_h}_{\text{capital}} + \underbrace{a_h}_{\text{AI}} + \underbrace{r_h}_{\text{robot}} + \underbrace{l_h}_{\text{land}} + \underbrace{t_h}_{\text{trust}} + \underbrace{u}_{\text{UBI}}$$

where:
- **Wages** are allocated to working households proportional to skill $\times$ population fraction
- **Asset income** is proportional to ownership share
- **UBI** is uniform across all working-age persons
- **Trust income** is distributed like wages (proportional to skill among workers)

Labor force participation determines which households work: the top LFP$\times N$ households by skill are employed.

### 12.4 Inequality Metrics

**Gini coefficient** (weighted): Computed from per-person values and population fractions using cumulative-weight trapezoidal integration:

$$G = 1 - 2 \sum_k w_k \cdot \left(S_k - \frac{y_k}{2}\right)$$

where $S_k$ is the cumulative income share through bucket $k$.

**Poverty headcount rates**: For absolute lines (World Bank: \$2.15, \$3.65, \$6.85/day) and relative lines (50% and 30% of base-year median income):

$$\text{Poverty rate} = \sum_{h : y_h < \text{threshold}} \text{pop\_fraction}_h$$

---

## 13. Solver Architecture

The model uses nested numerical solvers, all based on bisection:

1. **Outer loop (Y):** Fixed-point iteration on output. Given a guess $Y$, solve the inner equilibrium, compute predicted $Y$ from the production function, update with damping.

2. **Inner loop (L):** When labor is endogenous, iterate on labor supply $L$ given $Y$. Compute equilibrium wages → participation → new $L$, with damped updates.

3. **$\mu$ solver:** For each task type, if the endogenous regime is triggered, bisect on $\mu_{h,j}$ to find the no-arbitrage point $w_j = q_j$.

4. **$z$ solver:** Fixed-point iteration on the wage ratio $z = w_c/w_p$. Given $z$, compute human split → prices → new $z$.

5. **$\varphi$ solver:** Bisect on robot split to equate labor-channel and capital-channel marginal products.

6. **$r$ solver:** Bisect on interest rate to clear the capital market ($K_Y + K_C + K_R = K$).

7. **Trust co-iteration:** When trusted labor is active, $H_T$ is co-iterated with the $z$ solver using the inelastic supply curve.

Convergence tolerance is typically $10^{-4}$ for prices and $10^{-3}$ for output.

---

## 14. Summary of Key Parameters

| Parameter | Symbol | Typical Value (US) | Description |
|-----------|--------|-------------------|-------------|
| Capital share | $\alpha$ | 0.43 | Output elasticity of capital |
| Cognitive weight | $\theta$ | 0.56 | Weight on cognitive tasks in task aggregator |
| Cog/phys ratio scale | $\kappa$ | 1.27 | Occupational mobility parameter |
| Depreciation | $\delta_K$ | 0.053 | Annual capital depreciation rate |
| Savings rate | $s$ | 0.19 | Gross savings / GDP |
| Initial K/Y | — | 3.12 | Capital-output ratio at base year |
| Service share | $a_S$ | 0.77 | Base-year services share of GDP |
| LFP target | — | 0.62 | Base-year labor force participation rate |
| Land expenditure share | — | 0.068 | Base-year land spending / total expenditure |
| Land cap rate | — | 4.5% | Capitalization rate for land valuation |

All parameters are region-specific (US, China, World) with calibrated defaults from PWT, BLS, WDI, and other standard sources.
