# Scenario Growth Model - Technical Documentation

This document provides a comprehensive overview of the economic model implemented in `index.html`.

## Table of Contents
1. [Model Overview](#model-overview)
2. [Core Production Structure](#core-production-structure)
3. [Labor Market](#labor-market)
4. [Capital Market](#capital-market)
5. [Model Modes](#model-modes)
6. [Simulation Algorithm](#simulation-algorithm)
7. [Parameters](#parameters)
8. [Known Issues & Calibration Notes](#known-issues--calibration-notes)

---

## Model Overview

The model simulates an economy with:
- **Human labor** (cognitive and physical tasks)
- **AI/Machine labor** (cognitive tasks)
- **Robotic labor** (physical tasks)
- **Capital** (output capital K_Y, AI capital K_C, robot capital K_R)
- **Optional: Land** as a fixed factor

The key feature is that AI and robots can substitute for human labor in their respective task domains, subject to an **automation frontier** that limits how much can be automated.

---

## Core Production Structure

### Effective Labor (L_eff)

Labor services are aggregated in a nested CES structure:

```
L_cog = [μ_c^(1/σ_c) × H_c^ρ_c + (1-μ_c)^(1/σ_c) × AI^ρ_c]^(1/ρ_c)
L_phys = [μ_p^(1/σ_p) × H_p^ρ_p + (1-μ_p)^(1/σ_p) × R^ρ_p]^(1/ρ_p)
L_eff = [θ^(1/ε) × L_cog^ρ_ε + (1-θ)^(1/ε) × L_phys^ρ_ε]^(1/ρ_ε)
```

Where:
- `ρ = (σ-1)/σ` for each CES aggregator
- `H_c, H_p` = human cognitive and physical labor
- `AI` = AI cognitive services
- `R` = robot physical services
- `μ_c, μ_p` = human task shares (endogenous, subject to automation frontier)
- `σ_c, σ_p` = elasticity of substitution (human vs machine)
- `ε` = elasticity between cognitive and physical tasks
- `θ` = weight on cognitive tasks

### Production Function

**Without Land Mode:**
```
Y = A × K_Y^α × L_eff^(1-α)
```

**With Land Mode:**
```
Y_hat = A × K_Y^α × L_eff^(1-α)
Y = [ζ × (C_t × M)^ρ_L + (1-ζ) × Y_hat^ρ_L]^(1/ρ_L)
```

Where:
- `A` = Total Factor Productivity (calibrated from base year)
- `K_Y` = capital allocated to output production (NOT total capital K)
- `α` = capital share (default 0.30)
- `M` = land (fixed at base year output Y_0)
- `C_t` = land productivity multiplier (time-varying)
- `ζ` = land share parameter (s_L)
- `ρ_L = (σ_L - 1) / σ_L`
- `σ_L` = elasticity of substitution between land and other inputs

**Land Productivity (C_t):**
```
g_c = ln(C_2040) / 15
C_t = e^(g_c × years_since_base_year)
```
- `C_t = 1.0` in base year (2025)
- `C_t = C_2040` in year 2040
- If `C_2040 < 1`, land productivity declines over time (bottleneck tightens)
- If `C_2040 > 1`, land productivity improves over time (bottleneck loosens)

**IMPORTANT**: The production function uses K_Y (output capital), not total capital K. Total capital K = K_Y + K_C + K_R.

---

## Labor Market

### Human Labor Allocation

Humans allocate between cognitive and physical tasks based on relative wages:

```
L_c / L_p = κ × (w_c / w_p)^ω
```

Where:
- `κ` = scale factor (default 1.6)
- `ω` = labor allocation elasticity (default 0.1)
- `w_c, w_p` = cognitive and physical wages

This gives cognitive share: `ℓ_c = L_c / L = κz^ω / (κz^ω + 1)` where `z = w_c/w_p`

### Endogenous Labor Supply (Optional)

When enabled, total labor supply responds to wages:

```
L = B × w̄^(1/ψ)
```

Where:
- `w̄ = w_c × ℓ_c + w_p × (1 - ℓ_c)` (average wage)
- `ψ` = labor supply elasticity (default 1.0)
- `B` = calibrated from base year: `B = L_0 / w̄_0^(1/ψ)`
- Labor is capped at Working Age Population

### Task Shares (μ)

The human task shares `μ_c` and `μ_p` are determined by:

1. **Automation Frontier**: `μ ≥ 1 - bar_auto` (minimum human share)
2. **Interior Solution**: If wages > machine prices, μ adjusts so `w = q` (no-arbitrage)
3. **Frontier Solution**: If wages ≤ machine prices at frontier, μ stays at minimum

The model solves for equilibrium μ values using bisection on the condition `log(w/q) = 0`.

---

## Capital Market

### Capital Allocation

Total capital K is split between three uses:

```
K = K_Y + K_C + K_R
```

**Without Taxes:**
```
K_Y = α × Y / r
K_C = q_c × L_AI / (r - δ_K + δ_C)
K_R = q_p × L_R / (r - δ_K + δ_R)
```

**With Taxes:**
```
K_Y = α × Y / r
K_C = (1 - τ_AI) × q_c × L_AI / [(1 - τ_k) × r - δ_K + δ_C]
K_R = (1 - τ_R) × q_p × L_R / [(1 - τ_k) × r - δ_K + δ_R]
```

Where:
- `r` = interest rate (solved from capital market clearing)
- `q_c, q_p` = prices of AI and robot services
- `δ_K, δ_C, δ_R` = depreciation rates
- `τ_k, τ_AI, τ_R` = tax rates on capital, AI services, robot services

### Interest Rate Determination

The interest rate `r` is found by solving the capital market clearing condition:

```
K = K_Y(r) + K_C(r) + K_R(r)
```

This is solved using bisection search.

### Capital Accumulation (Predicted Output Mode)

When "Predict Output & Capital" is enabled:

```
K_{t+1} = s × Y_t + (1 - δ_K) × K_t
```

**Base year:** `K_0 = (K/Y ratio) × Y_0`

Where:
- `s` = savings rate (default 0.20)
- `δ_K` = depreciation rate (default 0.10)
- `K/Y ratio` = capital-to-output ratio for base year (default 3.71)

---

## Model Modes

### 1. Baseline Mode (All checkboxes unchecked)
- Uses Y and K from spreadsheet
- Exogenous labor supply (from spreadsheet)
- No land constraint
- No taxes

### 2. Land Bottlenecks Mode
- Adds land M as fixed factor (set to Y_0)
- Production becomes CES of land and Cobb-Douglas output
- Affects factor prices through shadow price P_Y

### 3. Endogenous Labor Supply Mode
- Labor supply responds to wages: `L = B × w̄^(1/ψ)`
- Calibrates B from base year
- Iterates until labor market equilibrium

### 4. Taxation Mode
- Adds taxes on capital income (τ_k), AI services (τ_AI), robot services (τ_R)
- Modifies capital demand equations
- Computes transfers: `T = τ_k × r × K_Y + τ_AI × q_c × L_AI + τ_R × q_p × L_R`
- Shows household income chart with wage + transfer breakdown

### 5. Predict Output & Capital Mode
- Predicts Y and K instead of using spreadsheet values
- Uses production function and capital accumulation
- Calibrates TFP (A) from base year
- Shows comparison chart: Forecast vs Model Prediction

---

## Simulation Algorithm

### Step 1: Load Data and Parameters
- Parse CSV with year columns
- Extract: Y, K, H (labor), AI_cog, R_phys, bar_auto_c, bar_auto_p
- Get user parameters from UI

### Step 2: Prepare Land Parameters (if enabled)
```
M = Y_0  (Base year output as fixed land)
zeta = s_L  (Land share parameter)
```

### Step 3: Base Year Calibration

**If Predict Output & Capital mode:**
```
K_0 = (K/Y ratio) × Y_0
Solve equilibrium with Y_0, K_0 → get K_Y_0, L_eff_0
A = Y_0 / (K_Y_0^α × L_eff_0^(1-α))
```

**If Endogenous Labor mode:**
```
Solve base year equilibrium → get w_c, w_p, ℓ_c
w̄_0 = w_c × ℓ_c + w_p × (1 - ℓ_c)
B = L_0 / (w̄_0^(1/ψ))
```

### Step 4: Solve Each Year

**For each year t:**

**A. Compute Capital (if predicting):**
```
if (t > 0):
    K_t = s × Y_{t-1} + (1 - δ_K) × K_{t-1}
```

**B. Solve Equilibrium (solveMuOneYear):**

1. **Initialize** at automation frontier (μ = 1 - bar_auto)

2. **Solve for wage ratio z = w_c/w_p:**
   - Fixed-point iteration on z
   - Given z, compute human split (H_c, H_p) using: ℓ_c = κz^ω / (κz^ω + 1)
   - Compute L_cog, L_phys, L_eff from CES aggregators
   - Compute wages w_c, w_p from marginal products
   - Update z until w_c/w_p = z

3. **Check if interior solution needed:**
   - If w_c < q_c at frontier → need to adjust μ_c upward
   - If w_p < q_p at frontier → need to adjust μ_p upward

4. **Bisect on μ values:**
   - Find μ such that log(w/q) = 0 for each task type
   - Coordinate iteration between μ_c and μ_p

5. **Solve for interest rate r:**
   - Capital market clearing: K = K_Y(r) + K_C(r) + K_R(r)
   - Bisection search for r

6. **Compute capital allocations:**
   - K_Y = α × Y / r
   - K_C, K_R from their respective formulas

**C. Predict Output (if enabled):**
```
Iterate until production function satisfied:
    Solve equilibrium given current Y guess → get K_Y, L_eff

    if (land_mode):
        Y_hat = A × K_Y^α × L_eff^(1-α)
        Y_pred = [ζ × M^ρ + (1-ζ) × Y_hat^ρ]^(1/ρ)
    else:
        Y_pred = A × K_Y^α × L_eff^(1-α)

    Update Y guess: Y = 0.5 × Y + 0.5 × Y_pred

until |Y_pred - Y| / Y < tolerance
```

**D. Endogenous Labor (if enabled):**
```
Nested iteration: outer on Y, inner on L

Outer loop (Y):
    Inner loop (L):
        Solve equilibrium with current Y, L
        w̄ = w_c × ℓ_c + w_p × (1 - ℓ_c)
        L_new = min(B × w̄^(1/ψ), WorkingAgePop)
        L = 0.5 × L + 0.5 × L_new
    until L converges

    Compute Y_pred from production function
    Y = 0.5 × Y + 0.5 × Y_pred
until Y converges
```

### Step 5: Compute Derived Quantities

For each year, compute:
- Factor income shares (capital, human cognitive, human physical, AI, robot, land)
- Tax transfers per worker (if taxation enabled)
- Technology levels: X_C = L_AI / K_C, X_P = L_R / K_R

### Step 6: Generate Charts and Tables

---

## Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Capital share | α | 0.30 | Cobb-Douglas capital exponent |
| Cog vs Phys elasticity | ε | 0.70 | Substitution between task types |
| Human vs AI elasticity | σ_c | 0.80 | Substitution in cognitive tasks |
| Human vs Robot elasticity | σ_p | 0.60 | Substitution in physical tasks |
| Labor allocation elasticity | ω | 0.10 | Response of L_c/L_p to wage ratio |
| Labor ratio scale | κ | 1.6 | Scale factor in labor allocation |
| Cognitive weight | θ | 0.68 | Weight on cognitive in L_eff |
| Labor supply elasticity | ψ | 1.0 | L = B × w̄^(1/ψ) |
| Land substitution | σ_L | 0.65 | Elasticity with land |
| Land share | s_L | 0.01 | Base-year land income share |
| Land productivity 2040 | C_2040 | 2.0 | Land productivity multiplier by 2040 |
| Output capital depreciation | δ_K | 0.10 | Depreciation rate |
| AI capital depreciation | δ_C | 0.10 | Depreciation rate |
| Robot capital depreciation | δ_R | 0.10 | Depreciation rate |
| Capital tax | τ_k | 0 | Tax on capital income |
| AI tax | τ_AI | 0 | Tax on AI services |
| Robot tax | τ_R | 0 | Tax on robot services |
| Savings rate | s | 0.20 | Fraction of output saved |
| Initial K/Y ratio | - | 3.71 | For computing K_0 = (K/Y) × Y_0 |

---

## Known Issues & Calibration Notes

### Capital-to-Output Ratio

The model uses **K/Y ratio** (capital-to-output) to compute initial capital:
```
K_0 = (K/Y ratio) × Y_0
```

With default K/Y = 3.71 and Y_0 = 29.7T:
```
K_0 = 3.71 × 29.7T ≈ 110T
```

This matches typical capital-to-output ratios in developed economies.

### Steady State Considerations

The capital accumulation equation `K_{t+1} = s × Y_t + (1 - δ_K) × K_t` has a steady state:
```
K*/Y* = s / δ_K = 0.20 / 0.10 = 2.0
```

If your data implies a different K/Y ratio (e.g., 3.71), the model will transition toward the steady state over time. To maintain a higher K/Y in steady state, either:
- Increase savings rate s
- Decrease depreciation rate δ_K

### Recommendations

1. **Set K/Y ratio to match your data**: If your spreadsheet shows K ≈ 110T when Y ≈ 30T, use K/Y ≈ 3.7

2. **Verify units**: All monetary values should be in same units (e.g., trillions)

3. **Compare outputs**: Use the forecast vs prediction chart to identify divergence

4. **Start simple**: Test with baseline mode (checkbox unchecked) first, verify equilibrium looks reasonable

5. **Check capital dynamics**: If predicted K diverges from forecast K, adjust s and δ_K to match your assumed growth path

---

## CSV Format

Required rows:
- Human Working Age Population (optional, for endogenous labor)
- Human Labor Force
- AI Cognitive
- Robotic Physical
- Output
- Capital
- Cognitive Automation Frontier (%)
- Physical Automation Frontier (%)

Columns: Years (e.g., 2025, 2026, ..., 2040)

---

## File Structure

```
ScenarioGrowth/
├── index.html          # Main web application
├── README.md           # This documentation
├── template.csv        # Example data template
└── *.csv               # User uploaded data files
```
