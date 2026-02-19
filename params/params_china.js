// ========================================
// China Region Parameters
// ========================================
// Sources: PWT 10.01, ILOSTAT, World Bank WDI, NBS China, CBRE Asia Pacific

var PARAMS_CHINA = {
    // --- Production Function ---
    alpha: 0.50,            // Capital share (PWT labsh; Bai, Hsieh & Qian 2006; range 0.45-0.55)
    theta: 0.50,            // Cognitive task weight (ILOSTAT ISCO-08; range 0.45-0.55)
    kappa: 1.00,            // Cog/phys ratio scale (derived from theta; range 0.82-1.22)

    // --- Capital ---
    delta_K: 0.06,          // Capital depreciation (China capital-stock literature; range 0.05-0.07)
    savings_rate: 0.428,    // Gross savings / GDP (WDI NY.GNS.ICTR.ZS 2024; range 0.426-0.448)
    initial_KY_ratio: 3.4,  // Capital-output ratio (PWT ck/rgdpe; range 3.1-3.7)

    // --- Sectoral ---
    a_services: 0.567,      // Services share of GDP (WDI NV.SRV.TOTL.ZS 2024; range 0.53-0.58)

    // --- Land ---
    land_cap_rate: 2.5,     // Cap rate (CBRE Asia Pacific survey; range 2.0-3.0%)

    // --- Background Time Series (2025-2040) ---
    background: {
        "Human Working Age Population": ["983M","977M","972M","966M","961M","955M","948M","941M","934M","927M","920M","913M","906M","899M","892M","885M"],
        "Human Labor Force": ["637M","635M","633M","631M","629M","620M","600M","570M","525M","470M","400M","320M","235M","165M","133M","123M"],
        "Output": ["20.7T","21.3T","22T","22.8T","23.5T","24.5T","26T","28T","32T","44T","84T","144T","262T","497T","963T","1897T"],
        "Capital": ["70T","71T","72T","73T","74T","75T","76T","77T","78.5T","81T","85T","97T","119T","163T","248T","417T"]
    },

    // ========================================
    // Parameters below were already region-specific in the original code.
    // ========================================

    // --- Trusted Labor ---
    trust_num_workers: 100000,
    trust_avg_wage: 80000,

    // --- Labor Supply ---
    lfp_target: 0.648,
    frisch_elasticity: 0.17,
    logistic_s: 0.7,

    // --- Land (already differentiated) ---
    land_exp_share: 0.090,
    land_areas: {
        agricultural: 455e6,
        urban: 12e6,
        rural: 17e6,
        commercial: 12e6,
        wilderness: 464e6
    },
    total_land_ha: 960e6,
    cat_exp_shares: { urban: 0.688, rural: 0.293, agricultural: 0.019 },

    // --- Income Distribution: CDF Breakpoints ---
    cdf: {
        wage:    [0.00010, 0.050, 0.137, 0.405, 0.567, 0.843, 0.930],
        capital: [0.00000, 0.001, 0.010, 0.050, 0.200, 0.550, 0.720],
        ai:      [0.00000, 0.0003, 0.001, 0.005, 0.020, 0.220, 0.420],
        robot:   [0.00000, 0.0005, 0.002, 0.010, 0.050, 0.300, 0.550],
        land:    [0.00050, 0.060, 0.180, 0.450, 0.700, 0.920, 0.960]
    },

    // --- Income Distribution: Multiplier Curves ---
    mult: {
        wage:    { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0550, 0.1101, 0.3210, 0.6879, 1.1465, 1.8344, 2.4765, 5.0447, 18.3444] },
        capital: { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0000, 0.0000, 0.0229, 0.0915, 0.4002, 1.7153, 3.4305, 10.2916, 57.1755] },
        ai:      { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0000, 0.0000, 0.0058, 0.0348, 0.1393, 0.8125, 3.4823, 13.9292, 69.6460] },
        robot:   { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0000, 0.0000, 0.0142, 0.0569, 0.2135, 1.1388, 3.5587, 12.8114, 64.0569] },
        land:    { pcts: [5,10,25,50,75,90,95,99,99.9], vals: [0.0487, 0.1169, 0.3410, 0.7793, 1.2177, 1.8509, 2.3380, 4.3838, 14.6128] }
    }
};
