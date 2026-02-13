# Comprehensive Changelog: Original → Current

Changes from the original `thoulden/ScenarioGrowth` (main branch) to the current version, ordered from most important core modelling changes to smaller parameter/UI tweaks.

---

## I. Major Production Function & Model Architecture Changes

### 1. Production function: Cobb-Douglas → CES for capital-labor substitution
- **Original:** `Y = A × K_Y^α × L_eff^(1-α)` (Cobb-Douglas top level, unit elasticity between K and L)
- **Current:** `Y = A × CES(K_Y, L_eff, α, σ_K)` with new parameter `σ_K` (default 1.5)
- When `σ_K = 1` it reduces to the original Cobb-Douglas; `σ_K > 1` means capital and labor are more substitutable than unit elasticity
- Interest rate solver updated to handle CES via bisection on K_Y when `σ_K ≠ 1`

### 2. Land removed from production function → consumption-side equilibrium
- **Original:** Land was a **production bottleneck** — a fixed factor `M` nested in the production function via CES: `Y = [ζ(C_t·M)^ρ + (1-ζ)·Ŷ^ρ]^{1/ρ}` with `M = Y_0`, `s_L = 0.01`, `σ_L = 0.65`
- **Current:** Land is **not in the production function at all**. Instead it's modeled as a consumer spending equilibrium:
  - 5 land categories: urban, rural, agricultural, commercial, wilderness
  - 3 endogenous categories with supply/demand curves: `M_D = A·E^β·v^{-η_D}`, `M_S = B·v^{η_S}`
  - 2 buffer categories (commercial, wilderness) with protection mechanisms
  - Land expenditure share (~7-10% of household budget) determines rents endogenously
  - Wilderness/commercial protection with deregulation ramp (year, duration, floor)
  - Category-specific elasticities (β income, η_D demand, η_S supply) configurable in a table

### 3. Added "Trusted Labor" sector as optional CES layer
- **Original:** No trust/institutional quality modeling
- **Current:** Optional nested CES: `Y_final = CES(X_trust, Y_core, s_trust, σ_trust)`
  - Trusted labor = sector that can't be easily automated (e.g., governance, oversight)
  - Parameters: start year (2029), initial workers, avg wage, efficiency growth to 2040, `σ_trust = 0.9`
  - Mode-specific defaults: US 100K workers/$150K wage, China 100K/$80K, World 200K/$115K
  - Trust income distributed like wages in the income distribution system

### 4. Added full income distribution system (109 households)
- **Original:** No income distribution — only aggregate factor shares and a simple per-worker income chart
- **Current:** Complete Lorenz-curve-based distribution:
  - 109 households: 99 at 1-percentile width + 10 ultra-wealthy at 0.1% each
  - 5 asset ownership classes (K, AI, Robots, Land) + wage skill distribution
  - 7 income sources tracked per household: wage, capital, AI, robot, land, trust, UBI
  - CDF breakpoints (7 knots) configurable per mode (US/China/World)
  - Smooth distribution via PCHIP (shape-preserving cubic Hermite) interpolation in log-space
  - Income multiplier curves at 9 percentiles (p5 through p99.9)
  - Weighted Gini coefficients for both income and wealth

### 5. LFP model replacing simple labor supply
- **Original:** `L = B × w̄^{1/ψ}` with `ψ = 1.0` (power-law labor supply, calibrated from base year)
- **Current:** Logistic LFP model (when income distribution is on):
  - `LFP = F(w/w₀) × D(UBI/wage)`
  - `F` = logistic on log wage-multiple, auto-calibrated so `F(1) = lfp_target`
  - `D` = UBI dampener: `exp(-k_ubi × UBI/wage)` with `k_ubi = 1.13`
  - Region-specific `lfp_target`: US 0.62, China 0.648, World 0.686
  - Legacy power-law mode (`ψ = 0.5`) still available when distribution is off

### 6. Added China as country mode (3-way region toggle)
- **Original:** No multi-region support — single scenario, implicit US/global
- **Current:** Radio button toggle: Global / US / China
  - Region-specific defaults for: base Y, K, L, WAP, CDF breakpoints, LFP target, land expenditure share, land areas, trust labor parameters
  - Scenario table has editable share rows (US/China Share of Compute, US/China Share of Robots)
  - Triple model runs in country mode: US, China, RoW each solved independently
  - GWP Breakdown stacked area chart shows each region's output share
  - GDP/GWP label switching in chart titles

---

## II. Solver & Numerical Changes

### 7. Bisection sign-change check for mu root-finding
- **Original:** Always bisects for mu (no guard for non-existence of root)
- **Current:** Checks `f_lo * f_hi < 0` before bisecting. If no sign change (machines already cheaper across the board), stays at automation frontier `μ = μ_h_min` instead of diverging

### 8. No-arbitrage wage cap at full automation
- **Original:** No explicit enforcement that human wage ≤ machine wage post full automation
- **Current:** At `bar_auto ≥ 1.0 - 1e-6`, human wage is capped: `wc = min(wc, qc)` and `wp = min(wp, qr)`. Only triggers at essentially 100% automation

### 9. Interest rate solver handles CES production (σ_K ≠ 1)
- **Original:** K_Y derived analytically from Cobb-Douglas: `K_Y = α·Y/r` (closed-form)
- **Current:** When `σ_K = 1`, still uses the same closed-form `K_Y = α·Y/r`. When `σ_K ≠ 1`, the CES first-order condition has no clean analytical inverse, so K_Y is solved via bisection on the CES marginal product condition

---

## III. Scenario Table & Input Changes

### 10. Scenario table decomposition: AI Copies × Equivs/Copy
- **Original:** Single rows for "AI Cognitive" and "Robotic Physical" directly in the CSV
- **Current:** Decomposed into editable rows:
  - AI Copies × Human-equivs per AI Copy → derived AI Cognitive
  - AI Copies per H100e (informational)
  - Robots × Human-equivs per Robot → derived Robotic Physical

### 11. Share rows for country-mode allocation
- **Original:** None
- **Current:** Editable rows for "US Share of Compute (%)", "US Share of Robots (%)", and China equivalents — visible only in country modes

### 12. Default scenario CSV embedded
- **Original:** Required CSV upload to run the model
- **Current:** `default_scenario.csv` loaded by default so the model runs immediately on page load

---

## IV. Code Architecture Changes

### 13. Monolithic index.html → modular file structure
- **Original:** Single ~2,450-line `index.html` containing all CSS, HTML, and JavaScript
- **Current:** Modular structure:
  - `model/` directory: `constants.js`, `ces.js`, `production.js`, `land.js`, `labor.js`, `distribution.js`, `solvers.js`
  - `charts/` directory: `theme.js`, `createCharts.js`, `createDistributionCharts.js`, + 22 individual chart files
  - `index.html` for HTML/CSS and orchestration logic

---

## V. UI & Visualization Changes

### 14. Charts: ~13 → 22
- **Original:** ~10-13 charts (wages, mu, ell_c, capital, interest, factor income, tech, output, conditional labor/income)
- **Current:** 22 charts in user-specified order:
  1. Output: Forecast vs Model Prediction
  2. GWP Breakdown by Region (new)
  3. Factor Income Shares of GWP/GDP
  4. Wages per Human Equivalent Worker
  5. Automated Task Shares
  6. Labor Force and WAP
  7. Human Labor Allocation
  8. Labor Productivity (new)
  9. Rental Costs
  10. Productive Capital Allocation Shares
  11. Returns to Capital
  12. Capital Efficiency
  13. Land Rents per Hectare (new)
  14. Land Use Shares (new)
  15. Consumer Budget (new)
  16. Household Expenditure Split (new)
  17. Wage Income per Working-Age Person
  18. UBI Transfers per Working-Age Person
  19. Income Distribution (new)
  20. Income Composition by Percentile (new)
  21. Net Worth Distribution (new)
  22. Gini Coefficients (new)

### 15. Chart theming system
- **Original:** Inline chart options per chart, inconsistent styling
- **Current:** Centralized `CHART_THEME` in `theme.js` with consistent colors, formatters (`$`, `$exp`, `%`, `count`), tooltip styles, axis formatting. Helper functions `ds()` and `makeChartConfig()` standardize all charts

### 16. Chart download buttons
- **Original:** No download functionality
- **Current:** Each chart has a PNG download button (⬇) in a toolbar overlay

### 17. Side panel for advanced parameters
- **Original:** All parameters in a single vertical column above charts
- **Current:** Advanced parameters moved to a collapsible right side panel with sticky Run/Reset buttons. Main view shows only key parameters

### 18. Consumer budget percentile selector
- **Original:** N/A
- **Current:** Dropdown to view consumer budget split at different percentiles (mean, p10, p90, p99.5)

---

## VI. Parameter Default Changes

### 19. α (capital share): 0.30 → 0.36

### 20. σ_K (capital-labor elasticity): new parameter, default 1.5
- **Original:** Implicit `σ_K = 1` (Cobb-Douglas)

### 21. ω (labor allocation elasticity): 0.1 → 0.2

### 22. δ_K (output capital depreciation): 0.10 → 0.05

### 23. δ_C (AI capital depreciation): 0.10 → 0.30
- Reflects faster obsolescence of compute hardware

### 24. ψ (labor supply elasticity): 1.0 → 0.5
- Only used in legacy non-LFP mode

### 25. Land parameters: production-side → consumption-side
- **Original:** `s_L = 0.01` (production function land share), `σ_L = 0.65`, `C_2040 = 20`
- **Current:** `land_exp_share` = 0.068 (US) / 0.090 (China) / 0.100 (World) — consumer expenditure share, completely different concept

### 26. K/Y initial ratio: 3.71 → 3.0

### 27. Base year macro values now region-specific
- **Original:** Single set: Y₀ = 29.7T, L₀ = 168.6M, K = 110T
- **Current:**
  - US: Y₀ = 30.6T, L₀ = 168.6M, K₀ = 91.8T, WAP₀ = 270.7M
  - China: Y₀ = 20.7T, L₀ = 637M, K₀ = 70T, WAP₀ = 983M
  - World: Y₀ = 117T, L₀ = 3.5B, K₀ = 410T, WAP₀ = 5.10B

### 28. Taxation defaults: all zero → AI/robot taxes on
- **Original:** τ_k = 0, τ_AI = 0, τ_R = 0
- **Current:** τ_k = 0, τ_AI = 0.10, τ_R = 0.10, UBI start year 2033, UBI share 50% to US / 30% to World ex-China

---

## VII. New Parameter Groups (no original equivalent)

### 29. Income distribution CDF knots
- 7 breakpoints × 5 asset classes (wage, capital, AI, robot, land) per region

### 30. LFP parameters
- `lfp_target`, `logistic_s = 0.7`, `k_ubi = 1.13`, `labor_damp = 0.5`

### 31. Land category elasticities table
- 3×3 table of (β income, η_D demand, η_S supply) for urban, rural, agricultural

### 32. Trust labor parameters
- `trust_start_year`, `trust_num_workers`, `trust_avg_wage`, `trust_sigma`, `trust_C_2040` — mode-specific defaults

### 33. Income multiplier curves
- 5 sources × 9 percentiles table controlling per-person income shape

---

## VIII. Minor / Cosmetic

### 34. Chart y-axis formatting: scientific notation → human-readable
- Labor productivity chart uses `$K`, `$M`, `$B`, `$T` instead of scientific notation

### 35. GDP/GWP label switching
- Charts dynamically show "GDP" in US/China modes and "GWP" in Global mode

### 36. GATE benchmark line label
- Switches between "GATE GDP" and "GATE GWP" based on region mode
