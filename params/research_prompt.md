# Research Prompt: Calibrating Region-Specific Economic Parameters

## Context

We have an economic growth model that simulates how AI and robots affect labor, capital, and output across three regions: **US**, **China**, and **World** (global aggregate). The model was originally calibrated for the US. We now need empirically-grounded values for China and World for the parameters listed below.

For each parameter, we need:
1. A **point estimate** (best single value)
2. A **plausible range** (low–high)
3. The **specific data source** (with year and table/series number where possible)

## Parameters to Calibrate

### 1. Capital Share of Income (alpha)

**Definition:** Capital's share in a Cobb-Douglas production function: Y = A * K^alpha * L^(1-alpha). This is the fraction of national income going to capital owners (profits, rents, interest) vs. labor (wages, salaries, benefits).

**Current values:**
- US: 0.36 (source: BEA NIPA Table 1.12, factor income shares)

**What to find:**
- **China:** Look up in Penn World Table 10.01 (variable `labsh` = labor share; alpha = 1 - labsh). The Bai, Hsieh & Qian (2006) adjustment for China is important because official NBS data misclassifies some mixed income. PWT gives China's labor share around 0.48-0.55, implying alpha ~ 0.45-0.52. Also check Feenstra, Inklaar & Timmer (2015) PWT documentation.
- **World:** GDP-weighted average across all PWT countries. Can compute from PWT as: sum(alpha_i * GDP_i) / sum(GDP_i). Expected range: ~0.38-0.42.

**Key sources:**
- Penn World Table 10.01: https://www.rug.nl/ggdc/productivity/pwt/
- Variable: `labsh` (labor share at current national prices)
- Bai, Hsieh & Qian (2006) "The Return to Capital in China" - Brookings Papers
- World Bank World Development Indicators (for GDP weights)


### 2. Cognitive Task Weight (theta)

**Definition:** The fraction of effective labor that is cognitive (white-collar, knowledge work) vs. physical (manual, blue-collar). In our CES labor composite: L_eff = CES(L_cog, L_phys, theta, epsilon).

**Current values:**
- US: 0.68 (source: BLS occupational employment data, ~65-70% in cognitive occupations)

**What to find:**
- **China:** Fraction of workers in cognitive vs. physical/manual occupations. China has a larger manufacturing, agriculture, and construction sector. ILO data (ILOSTAT) classifies occupations using ISCO-08. Look at major groups: Managers (1), Professionals (2), Technicians (3), Clerical (4) = broadly cognitive. Plant/machine operators (8), Elementary occupations (9), Craft workers (7), Agricultural workers (6) = broadly physical. Services/sales (5) is mixed. Expected: ~0.45-0.55 for China.
- **World:** Employment-weighted global average. ILO Global Employment Trends reports breakdown. Expected: ~0.55-0.65 (pulled down by developing countries).

**Key sources:**
- ILOSTAT: https://ilostat.ilo.org/data/ — Employment by occupation (ISCO-08)
- China: NBS China Statistical Yearbook, "Employed Persons by Sector"
- World: ILO "World Employment and Social Outlook"


### 3. Labor Ratio Scale Factor (kappa)

**Definition:** Scale factor in the labor allocation equation: L_cog / L_phys = kappa * (w_cog / w_phys)^omega. At baseline equilibrium with equal wages, kappa = L_cog / L_phys. So kappa = theta / (1 - theta) approximately.

**Current values:**
- US: 1.6 (consistent with theta = 0.68: 0.68/0.32 ≈ 2.1, but adjusted for wage ratio)

**What to find:**
- Derive from theta: kappa ≈ theta / (1 - theta), adjusted if the baseline cog/phys wage ratio is known to differ from 1.
- **China:** If theta_china = 0.55, then kappa ~ 0.55/0.45 = 1.22
- **World:** If theta_world = 0.60, then kappa ~ 0.60/0.40 = 1.50


### 4. Capital Depreciation Rate (delta_K)

**Definition:** Annual depreciation rate for productive (non-AI, non-robot) capital. Reflects how fast machines, buildings, and equipment lose value.

**Current values:**
- US: 0.05 (5%) — source: BEA Fixed Asset Tables, aggregate depreciation/capital ratio

**What to find:**
- **China:** China's capital stock is newer on average (massive investment boom post-2000), and construction/infrastructure has a large share. PWT reports `delta` (depreciation rate) or it can be computed from capital stock changes: delta = (Investment - Delta_K) / K. Some estimates suggest 4-7% for China. Zhang, Wan & Jin (2007) estimate 5.9%. Newer capital should have slightly higher depreciation. Look at NBS investment/depreciation statistics. Expected: 0.05-0.07.
- **World:** GDP-weighted average. Mix of old capital in advanced economies and newer capital in developing ones. Expected: 0.05-0.06.

**Key sources:**
- Penn World Table 10.01: variable `delta` (depreciation rate)
- Zhang, Wan & Jin (2007) "China's Economic Growth: What We Know, What We Don't Know"
- BEA Fixed Asset Tables (US benchmark)


### 5. Savings Rate (savings_rate)

**Definition:** Gross national savings as a fraction of GDP. Determines how fast capital accumulates: K_{t+1} = s * Y_t + (1 - delta) * K_t.

**Current values:**
- US: 0.20 (source: World Bank WDI, BEA)

**What to find:**
- **China:** World Bank reports China's gross savings rate at 43-46% of GDP (consistently one of the highest globally). This includes household, corporate, and government savings. Check WDI indicator NY.GNS.ICTR.ZS. Expected: 0.43-0.46.
- **World:** Global gross savings rate ~25-28% of GDP. Check WDI global aggregate. Expected: 0.25-0.28.

**Key sources:**
- World Bank WDI: NY.GNS.ICTR.ZS (Gross savings % of GDP)
- IMF World Economic Outlook database
- For China specifically: NBS national accounts


### 6. Capital-Output Ratio (initial_KY_ratio)

**Definition:** Ratio of total productive capital stock to annual GDP (K/Y) in the base year (2025).

**Current values:**
- US: 3.00 (source: BEA, PWT)

**What to find:**
- **China:** PWT variable `ck` (capital stock at current PPPs) divided by `rgdpe` (expenditure-side real GDP). Various estimates put China K/Y at 3.0-3.5 as of 2020s. The high investment rate pushes it up. Expected: 3.2-3.5.
- **World:** Global K/Y ratio is roughly the GDP-weighted average. Expected: 3.3-3.7.

**Key sources:**
- Penn World Table 10.01: `ck` / `rgdpe`
- IMF Investment and Capital Stock Dataset (ICSD)


### 7. Services Share of GDP (a_services)

**Definition:** Nominal share of GDP produced by services (vs. goods/manufacturing/agriculture).

**Current values:**
- US: 0.77 (source: BEA)
- China: 0.53 (currently in model, from NBS)
- World: 0.65 (currently in model, from World Bank)

**What to verify/update:**
- Confirm these are still accurate for 2024-2025.
- **China:** Check NBS China Statistical Yearbook, GDP by industry. Services (tertiary) sector has been rising — was ~53% in 2023.
- **World:** Check World Bank WDI: NV.SRV.TOTL.ZS (services value added % of GDP).

**Key sources:**
- World Bank WDI: NV.SRV.TOTL.ZS
- NBS China Statistical Yearbook: "GDP by Industry"


### 8. Land Capitalization Rate (land_cap_rate)

**Definition:** The rate used to convert an annual land rent flow into an asset value: Asset_Value = Annual_Rent / cap_rate. A lower cap rate means land is valued more highly relative to its rental income (i.e., lower yield). This is used in the model's land module to translate equilibrium rents into land asset values for wealth distribution calculations.

**Current values:**
- US: 4.5% (source: REIT yields, commercial real estate data)

**What to find:**
- **China:** Chinese real estate is known for very low rental yields — especially residential. Tier-1 city residential yields are often 1.5-2.5%. Commercial yields are somewhat higher. An aggregate cap rate blending residential, commercial, and agricultural land might be 2-3%. Expected: 2.0-3.0%.
- **World:** Blended global average. Advanced economies ~4-5%, emerging markets variable. Expected: 3.5-4.5%.

**Key sources:**
- For China: China Index Academy (中指研究院), CBRE China cap rate reports, Knight Frank global yield data
- For World: MSCI Global Property Index, JLL Global Real Estate Transparency Index
- USDA for agricultural land (US benchmark comparison)


## Output Format

Please provide results in this format for each parameter:

```
Parameter: [name]
China: [value] (range: [low]-[high])
  Source: [specific source, year, table/variable]
  Notes: [any caveats or adjustments]

World: [value] (range: [low]-[high])
  Source: [specific source, year, table/variable]
  Notes: [any caveats or adjustments]
```

## Important Notes

- Use **PPP-adjusted** GDP where relevant (PWT `rgdpe` or similar), not market exchange rate GDP
- For "World" parameters, use **GDP-weighted** averages across countries, not simple averages
- Prefer the **most recent** data available (2023 or 2024 vintage)
- If a parameter has well-known measurement issues for China (e.g., capital share), note the adjustment method used
- Where the parameter is not directly observable, explain the derivation clearly
