# Scenario Growth Model: Technical Documentation

## 1. What This Model Does

This is a **scenario-driven growth model** that projects economic outcomes under different assumptions about AI and robotics adoption. You provide a scenario — how capable AI and robots become, and how many are deployed — and the model computes the resulting economic equilibrium at each time step: output, wages, factor prices, income distribution, and more.

The model answers questions like:
- What happens to wages if AI automates 80% of cognitive tasks by 2035?
- How does the capital share evolve as robots substitute for physical labor?
- What does income distribution look like under different UBI policies?

---

## 2. The Scenario: AI and Robot Trajectories

The scenario defines **two dimensions for each technology**:

### AI Systems

| Parameter | What It Means |
|-----------|---------------|
| **AI copies deployed** | Number of AI instances running (e.g., 100M AI agents) |
| **Human-equivalents per AI copy** | How productive each AI is (e.g., 0.1 = 10 AIs needed to match one human; 10.0 = each AI does 10× a human's cognitive work) |
| **Cognitive automation frontier** | Maximum share of cognitive tasks that *can* be automated (0–100%) — the technological ceiling |

**Effective AI labor** = (AI copies) × (human-equivalents per copy)

The automation frontier is separate from efficiency: an AI might be very efficient (high human-equivalents) but still only applicable to 60% of cognitive tasks. The frontier says which tasks are automatable *in principle*; efficiency says how well the AI performs those tasks.

### Robots

| Parameter | What It Means |
|-----------|---------------|
| **Robots deployed** | Physical robots in operation |
| **Human-equivalents per robot** | Productivity per robot (e.g., 1.0 = robot matches one human; 50.0 = robot does 50× a human's physical work) |
| **Physical automation frontier** | Maximum share of physical tasks that *can* be automated (0–100%) |

**Effective robot labor** = (robots) × (human-equivalents per robot)

### The Two Dimensions

Think of it this way:
- **Efficiency** (human-equivalents): How good are the machines at tasks they can do?
- **Capability** (automation frontier): What fraction of tasks can they do at all?

A scenario with frontier at 50% and efficiency at 100× means: AI can do half the cognitive tasks, but does those tasks 100× faster than humans. A scenario with frontier at 100% and efficiency at 1× means: AI can do everything humans can, but no faster.

These interact in the model — actual automation depends on both capability (can the machine do it?) and cost (is the machine cheaper than a human?).

---

## 3. Model Structure: From Scenario to Output

### 3.1 The Core Equation

Output is Cobb-Douglas in effective capital and effective labor:

$$Y = A \cdot K_{\text{eff}}^{\,\alpha} \cdot L_{\text{eff}}^{\,1-\alpha}$$

where:
- $A$ is TFP (calibrated at base year, held constant — all growth comes from factor accumulation)
- $K_{\text{eff}}$ is **effective capital** (physical capital + robot capital services)
- $L_{\text{eff}}$ is **effective labor** (human + AI labor services)
- $\alpha \approx 0.4$ is the capital share

### 3.2 Effective Labor: Nested CES Aggregation

Labor services aggregate through nested CES functions:

**Level 1 — Task-specific nests.** Cognitive and physical tasks each combine humans with machines:

$$L_{\text{cog}} = \text{CES}(H_c, A^{\text{AI}}; \mu_c, \sigma_c)$$
$$L_{\text{phys}} = \text{CES}(H_p, R^L; \mu_p, \sigma_p)$$

Here $A^{\text{AI}}$ is effective AI labor (copies × efficiency) and $R^L$ is effective robot labor allocated to the labor channel.

$\mu_c$ and $\mu_p$ are the **human task shares** — the fraction of each task type performed by humans. These are **endogenous**, solved based on relative costs (see §4).

**Level 2 — Task aggregator.** Cognitive and physical labor combine:

$$L_{\text{eff}} = \text{CES}(L_{\text{cog}}, L_{\text{phys}}; \theta, \varepsilon)$$

where $\theta \approx 0.56$ is the cognitive task weight.

### 3.3 Effective Capital: Robot Augmentation

Robots can substitute for both labor (replacing physical workers) and capital (augmenting productive machinery). A fraction $\varphi$ goes to the labor channel; the rest $(1-\varphi)$ augments capital:

$$K_{\text{eff}} = \text{CES}(K_Y, R^K)$$

The split $\varphi$ is solved to equalize marginal products across channels — robots flow to wherever their marginal value is highest.

### 3.4 Human Occupational Choice

Human labor divides between cognitive and physical tasks based on relative wages:

$$\frac{H_c}{H_p} = \kappa \cdot \left(\frac{w_c}{w_p}\right)^{\omega}$$

Workers flow toward higher-paying task types.

---

## 4. Endogenous Automation (The Heart of the Model)

The human task shares $\mu_c$ and $\mu_p$ are **solved endogenously** — they're not inputs. This is the key mechanism.

### The Automation Frontier as a Constraint

The scenario provides the automation frontier $\bar{a}_j$ — the maximum machine share *technologically possible*:

$$1 - \mu_j \leq \bar{a}_j$$

Within this constraint, the model finds equilibrium:

- **If humans are cheaper than machines at the frontier**: Not all automatable tasks get automated. $\mu_j$ adjusts so $w_j = q_j$ (human wage = machine marginal product). Machines only take tasks where they're cost-competitive.

- **If machines are cheaper even at full frontier**: All automatable tasks are automated. $\mu_j = 1 - \bar{a}_j$.

### Why This Matters

The automation frontier is a *capability bound*, not a prediction of actual automation. A scenario might say "AI can automate 90% of cognitive tasks" — but if AI is expensive (low deployment, low efficiency), actual automation might be lower. The model solves for what actually happens given costs.

**Example dynamics:**
- **Early AI era**: Few AI copies, low efficiency → AI is expensive per-unit → actual automation below frontier → human wages hold up
- **Scaling AI**: More copies, higher efficiency → AI marginal product falls → automation catches up to frontier → human wages start tracking machine costs
- **Full automation**: Frontier at 100%, abundant AI → human wages capped at machine marginal product (no-arbitrage)

---

## 5. Factor Prices

All prices are marginal products, computed via chain rule through the nested CES:

$$w_c = \frac{\partial Y}{\partial H_c}, \quad w_p = \frac{\partial Y}{\partial H_p}, \quad q_c = \frac{\partial Y}{\partial A^{\text{AI}}}, \quad q_r = \frac{\partial Y}{\partial R}$$

**Interest rate.** Capital earns its marginal product, and the rate clears the capital market:

$$K = K_Y + K_{\text{AI}} + K_R$$

where AI and robot capital stocks are pinned by no-arbitrage (equal risk-adjusted returns across asset types).

---

## 6. Capital Accumulation

Capital evolves via Solow-style accumulation:

$$K_{t+1} = s \cdot Y_t + (1-\delta) \cdot K_t$$

Initial capital is set by the capital-output ratio (~3.1× for the US). The savings rate $s$ and depreciation $\delta$ are calibrated parameters.

---

## 7. Extensions

### 7.1 Trusted Labor Sector

A small sector of jobs requiring human judgment (fiduciary, legal, medical) sits atop the main production function:

$$\tilde{Y} = \text{CES}(X_T, Y; \sigma_T)$$

with low substitution elasticity. Even with abundant AI, output is constrained by trusted labor supply.

### 7.2 Two-Sector Prices (Goods vs. Services)

Services are cognitive-intensive; goods are physical-intensive. As AI automates cognitive tasks, service prices fall relative to goods — reversing Baumol's cost disease.

### 7.3 Taxation and UBI

Proportional taxes on capital, AI, and robot income fund UBI transfers.

---

## 8. Income Distribution

109 percentile buckets. Each receives:
- **Wages**: proportional to skill × employment
- **Asset income** (capital, AI, robots, land): proportional to ownership shares
- **UBI**: uniform

Gini coefficients and percentile incomes computed each year.

---

## 9. Solver

Six nested equilibrium conditions, all solved by bisection:

1. **Output** $Y$: production function fixed-point
2. **Labor** $L$: participation equilibrium
3. **Human task shares** $\mu$: no-arbitrage with machines
4. **Occupational split** $z$: wage ratio equilibrium
5. **Robot split** $\varphi$: equalize marginal products
6. **Interest rate** $r$: clear capital market

---

## 10. Key Parameters

| Parameter | Symbol | US Default | Description |
|-----------|--------|------------|-------------|
| Capital share | $\alpha$ | 0.43 | Output elasticity of capital |
| Cognitive weight | $\theta$ | 0.56 | Task aggregator weight |
| Savings rate | $s$ | 0.19 | Gross savings / GDP |
| Depreciation | $\delta$ | 0.053 | Annual capital depreciation |
| K/Y ratio | — | 3.12 | Initial capital-output ratio |

---

## 11. Regional Differences: US, China, World

The model supports three regions with distinct calibrations reflecting their economic structures.

### Why Different Regions?

- **US**: High-income services economy, high cognitive task weight, lower savings
- **China**: Manufacturing powerhouse, higher capital share, very high savings rate
- **World**: GDP-weighted global average, most heterogeneous baseline

### Parameter Comparison

| Parameter | US | China | World | Notes |
|-----------|------|-------|-------|-------|
| Capital share (α) | 0.43 | 0.50 | 0.40 | China's investment-led growth model |
| Cognitive weight (θ) | 0.56 | 0.50 | 0.60 | US more service-oriented |
| Savings rate (s) | 0.19 | 0.43 | 0.26 | China's exceptionally high savings |
| K/Y ratio | 3.1 | 3.4 | 3.5 | Capital intensity |
| Services share | 77% | 57% | 65% | Sectoral composition |
| LFP target | 62% | 65% | 69% | Labor force participation |
| Land exp. share | 6.8% | 9.0% | 10% | Housing cost burden |

### Background Trajectories

Each region has distinct projections (2025-2040) for:
- **Working-age population**: US growing slowly, China declining, World growing
- **Baseline output**: Pre-AI GDP/GWP trajectories
- **Capital stock**: Follows from savings and depreciation

These come from UN Population Prospects, Penn World Tables, and World Bank WDI.

### Data Sources

- **Penn World Table 10.01**: Capital share, depreciation, K/Y ratios
- **World Bank WDI**: Savings rates, services share, GDP
- **ILOSTAT**: Labor force participation, cognitive/physical breakdown
- **Bai, Hsieh & Qian (2006)**: China capital share adjustments

---

## 12. Using the Model

### Running the Model

1. **Click "Run All Regions"**: Computes US, China, and World simultaneously
2. **Switch regions instantly**: After running, toggle between regions without re-running
3. **Adjust scenario**: Modify AI/robot paths in the table, then run again
4. **Upload custom CSV**: Override scenario with your own projections

### What the Charts Show

- **Output**: GDP/GWP over time (forecast vs. model prediction)
- **GWP Breakdown**: All three regions on one chart
- **Factor shares**: How income splits between labor, capital, AI, robots
- **Wages**: Cognitive and physical wages over time
- **Distribution**: Income by percentile, Gini coefficients

**The scenario is the lever.** Change the AI/robot paths or automation frontiers, see the economic consequences.
