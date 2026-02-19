// ========================================
// US Region Parameters
// ========================================
// Only parameters with clear empirical cross-country variation are
// region-specific. All other model parameters stay as global inputs.

var PARAMS_US = {
    // --- Production Function ---
    alpha: 0.43,            // Capital share (PWT 11.0 labsh via FRED, 2023: 1-0.5683=0.4317; range 0.41-0.45)
    theta: 0.56,            // Cognitive task weight (BLS Table 1.1 occupation mapping; range 0.52-0.60)
    kappa: 1.27,            // Cog/phys ratio scale (derived from theta; range 1.08-1.50)

    // --- Capital ---
    delta_K: 0.053,         // Capital depreciation (BEA fixed assets via FRED, M1TTOTL1ES000/K1TTOTL1ES000; range 0.050-0.058)
    savings_rate: 0.19,     // Gross savings / GDP (WDI NY.GNS.ICTR.ZS; range 0.17-0.22)
    initial_KY_ratio: 3.12, // Capital-output ratio (BEA net stock / nominal GDP 2024; range 2.9-3.4)

    // --- Sectoral ---
    a_services: 0.77,       // Services share of GDP (BEA/WDI NV.SRV.TOTL.ZS; range 0.74-0.80)

    // --- Land ---
    land_cap_rate: 4.5,     // Cap rate (NCREIF NPI ~4.6% 2025; range 3.5-5.5%)

    // --- Background Time Series (2025-2040) ---
    background: {
        "Human Working Age Population": ["270.7M","273.4M","276.1M","278.9M","281.7M","284.5M","287.3M","290.2M","293.1M","296.0M","299.0M","302.0M","305.0M","308.1M","311.1M","314.3M"],
        "Human Labor Force": ["168.6M","168.4M","168.0M","167.4M","166.5M","164.0M","159.0M","151.0M","140.0M","126.0M","108.0M","88.0M","66.0M","47.0M","37.0M","33.3M"],
        "Output": ["30.6T","31.2T","32T","33T","33.5T","34.5T","36.5T","39.5T","45.5T","63T","120T","205T","373T","710T","1375T","2710T"],
        "Capital": ["91.8T","93T","94T","95T","96T","97T","98T","99T","101T","104T","110T","125T","154T","210T","320T","538T"]
    },

    // ========================================
    // Parameters below were already region-specific in the original code.
    // They are included here for completeness / single-file editing.
    // ========================================

    // --- Trusted Labor ---
    trust_num_workers: 100000,
    trust_avg_wage: 150000,

    // --- Labor Supply ---
    lfp_target: 0.62,
    frisch_elasticity: 0.17,
    logistic_s: 0.7,

    // --- Land (already differentiated) ---
    land_exp_share: 0.068,
    land_areas: {
        agricultural: 424.3e6,
        urban: 30.2e6,
        rural: 15.5e6,
        commercial: 22.0e6,
        wilderness: 422.7e6
    },
    total_land_ha: 914.7e6,
    cat_exp_shares: { urban: 0.892, rural: 0.073, agricultural: 0.036 },

    // --- Income Distribution: CDF Breakpoints ---
    cdf: {
        wage:    [0.00006, 0.034, 0.1146, 0.3011, 0.5064, 0.7756, 0.8863],
        capital: [0.0, 0.0, 0.025, 0.10, 0.326, 0.69, 0.861],
        ai:      [0.0, 0.0005, 0.003, 0.015, 0.05, 0.25, 0.50],
        robot:   [0.0, 0.001, 0.005, 0.025, 0.08, 0.35, 0.60],
        land:    [0.0, 0.015, 0.098, 0.30, 0.559, 0.865, 0.96]
    },

    // --- Income Distribution: Multiplier Curves ---
    mult: {
        wage:    { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.04, 0.10, 0.28, 0.62, 1.20, 2.10, 3.30, 8.50, 40.0] },
        capital: { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0,  0.0,  0.02, 0.10, 0.45, 1.80, 4.00, 12.0, 80.0] },
        ai:      { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0,  0.0,  0.005,0.02, 0.10, 0.50, 2.50, 10.0, 60.0] },
        robot:   { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0,  0.0,  0.005,0.02, 0.10, 0.50, 2.50, 10.0, 60.0] },
        land:    { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.02, 0.06, 0.20, 0.55, 1.10, 2.00, 3.50, 8.00, 30.0] }
    }
};
