// ========================================
// World (Global) Region Parameters
// ========================================
// GDP-weighted global averages. Sources: PWT 10.01, ILOSTAT, World Bank WDI, CBRE

var PARAMS_WORLD = {
    // --- Production Function ---
    alpha: 0.40,            // Global capital share (PWT GDP-weighted labsh; range 0.38-0.42)
    theta: 0.60,            // Global cognitive task weight (ILOSTAT ISCO-08; range 0.55-0.65)
    kappa: 1.50,            // Cog/phys ratio scale (derived from theta; range 1.22-1.86)

    // --- Capital ---
    delta_K: 0.055,         // Capital depreciation (PWT/cross-country; range 0.05-0.06)
    savings_rate: 0.262,    // Gross savings / GDP (WDI NY.GNS.ICTR.ZS world aggregate; range 0.255-0.275)
    initial_KY_ratio: 3.50, // Capital-output ratio (PWT ck/rgdpe; range 3.3-3.8)

    // --- Sectoral ---
    a_services: 0.65,       // Services share of global GDP (WDI NV.SRV.TOTL.ZS; range 0.62-0.68)

    // --- Land ---
    land_cap_rate: 4.0,     // Blended global cap rate (CBRE cross-region surveys; range 3.5-4.5%)

    // --- Background Time Series (2025-2040) ---
    background: {
        "Human Working Age Population": ["5.10B","5.12B","5.15B","5.18B","5.21B","5.24B","5.28B","5.32B","5.36B","5.40B","5.44B","5.49B","5.53B","5.58B","5.63B","5.68B"],
        "Human Labor Force": ["3.50B","3.50B","3.49B","3.47B","3.44B","3.38B","3.30B","3.18B","3.02B","2.80B","2.50B","2.15B","1.75B","1.30B","0.95B","0.70B"],
        "Output": ["117T","120T","124T","128T","132T","137T","144T","155T","179T","249T","475T","811T","1477T","2804T","5438T","10715T"],
        "Capital": ["410T","414T","418T","422T","427T","432T","438T","444T","452T","465T","490T","556T","685T","932T","1420T","2384T"]
    },

    // ========================================
    // Parameters below were already region-specific in the original code.
    // ========================================

    // --- Trusted Labor ---
    trust_num_workers: 200000,
    trust_avg_wage: 115000,

    // --- Labor Supply ---
    lfp_target: 0.686,
    frisch_elasticity: 0.17,
    logistic_s: 0.7,

    // --- Land (already differentiated) ---
    land_exp_share: 0.100,
    land_areas: {
        agricultural: 4316e6,
        urban: 117e6,
        rural: 52e6,
        commercial: 39e6,
        wilderness: 8479e6
    },
    total_land_ha: 13003e6,
    cat_exp_shares: { urban: 0.694, rural: 0.184, agricultural: 0.122 },

    // --- Income Distribution: CDF Breakpoints ---
    cdf: {
        wage:    [0.00020, 0.030, 0.080, 0.250, 0.470, 0.800, 0.900],
        capital: [0.00000, 0.0005, 0.005, 0.020, 0.150, 0.450, 0.650],
        ai:      [0.00000, 0.00002, 0.00010, 0.0005, 0.002, 0.050, 0.200],
        robot:   [0.00000, 0.00005, 0.00030, 0.003, 0.010, 0.100, 0.250],
        land:    [0.00020, 0.040, 0.120, 0.350, 0.600, 0.850, 0.920]
    },

    // --- Income Distribution: Multiplier Curves ---
    mult: {
        wage:    { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0136, 0.0407, 0.1086, 0.2715, 0.8144, 2.0360, 3.3933, 8.1439, 33.9328] },
        capital: { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0000, 0.0000, 0.0073, 0.0291, 0.1163, 0.7267, 2.9070, 11.6279, 87.2093] },
        ai:      { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0000, 0.0000, 0.0015, 0.0077, 0.0307, 0.2299, 1.5327, 9.1961, 122.6148] },
        robot:   { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0000, 0.0000, 0.0033, 0.0165, 0.0661, 0.4133, 2.4797, 11.5722, 99.1899] },
        land:    { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0261, 0.0652, 0.1564, 0.3910, 0.9123, 2.0854, 3.3887, 7.1685, 26.0671] }
    }
};
