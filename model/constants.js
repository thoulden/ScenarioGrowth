// Model constants — shared between model code and chart files

const TINY = 1e-12;

// ========================================
// 5-CATEGORY LAND CONSTANTS (consumption-side model)
// ========================================
// All 5 categories for accounting and charts
var LAND_CATEGORIES = ['agricultural', 'urban', 'rural', 'commercial', 'wilderness'];
// Only these 3 have supply-demand equilibria
var LAND_ENDOGENOUS_CATEGORIES = ['agricultural', 'urban', 'rural'];
// Commercial and wilderness are buffer categories with protection floors
var LAND_BUFFER_CATEGORIES = ['commercial', 'wilderness'];
var LAND_CATEGORY_LABELS = {
    agricultural: 'Agricultural',
    urban: 'Urban/Residential',
    rural: 'Rural/Residential',
    commercial: 'Commercial/Industrial',
    wilderness: 'Uninhabited/Wilderness'
};

// Physical areas in hectares (base year 2025)
const LAND_AREAS_US = {
    agricultural: 424.3e6,
    urban: 30.2e6,
    rural: 15.5e6,
    commercial: 22.0e6,
    wilderness: 422.7e6
};
const LAND_AREAS_WORLD = {
    agricultural: 4316e6,
    urban: 117e6,
    rural: 52e6,
    commercial: 39e6,
    wilderness: 8479e6
};
const TOTAL_LAND_HA_US = 914.7e6;
const TOTAL_LAND_HA_WORLD = 13003e6;

// Per-category share of total land expenditure at base year (endogenous types only)
// Derived from old GDP shares: US total was 0.0684
//   urban = 0.049/0.0684 ≈ 0.716, rural = 0.004/0.0684 ≈ 0.058, ag = 0.002/0.0684 ≈ 0.029
//   (commercial 0.013/0.0684 ≈ 0.197 is excluded — not endogenous)
//   Renormalized endogenous shares (urban+rural+ag) to sum to 1.0
var CAT_EXP_SHARES_US = { urban: 0.892, rural: 0.073, agricultural: 0.036 };
// World: urban 0.034, rural 0.009, ag 0.006 → total endo = 0.049
var CAT_EXP_SHARES_WORLD = { urban: 0.694, rural: 0.184, agricultural: 0.122 };
var DEFAULT_LAND_EXP_SHARE = 0.068;

// ========================================
// MODE-SPECIFIC DEFAULTS: Income Distribution, LFP, Land
// ========================================
// CDF breakpoints: [L(0.01), L(0.25), L(0.50), L(0.75), L(0.90), L(0.99), L(0.999)]
// Each value = cumulative share of total income/wealth held by bottom p%

var CDF_DEFAULTS_US = {
    wage:    [0.00006, 0.034, 0.1146, 0.3011, 0.5064, 0.7756, 0.8863],
    capital: [0.0, 0.0, 0.025, 0.10, 0.326, 0.69, 0.861],
    ai:      [0.0, 0.0005, 0.003, 0.015, 0.05, 0.25, 0.50],
    robot:   [0.0, 0.001, 0.005, 0.025, 0.08, 0.35, 0.60],
    land:    [0.0, 0.015, 0.098, 0.30, 0.559, 0.865, 0.96]
};

var CDF_DEFAULTS_WORLD = {
    wage:    [0.00002, 0.015, 0.100, 0.280, 0.480, 0.780, 0.900],
    capital: [0.00000, 0.0001, 0.010, 0.070, 0.200, 0.550, 0.700],
    ai:      [0.0, 0.00001, 0.0005, 0.003, 0.015, 0.12, 0.30],
    robot:   [0.0, 0.00005, 0.001, 0.008, 0.03, 0.18, 0.40],
    land:    [0.00000, 0.001, 0.010, 0.060, 0.250, 0.600, 0.800]
};

var CDF_DEFAULTS_CHINA = {
    wage:    [0.00008, 0.030, 0.125, 0.335, 0.530, 0.805, 0.905],
    capital: [0.00000, 0.0002, 0.020, 0.100, 0.300, 0.650, 0.800],
    ai:      [0.0, 0.00001, 0.0005, 0.003, 0.015, 0.12, 0.30],
    robot:   [0.0, 0.0005, 0.003, 0.015, 0.05, 0.20, 0.45],
    land:    [0.00000, 0.003, 0.035, 0.150, 0.400, 0.720, 0.860]
};

var LFP_DEFAULTS_US    = { lfp_target: 0.62, logistic_s: 0.7 };
var LFP_DEFAULTS_WORLD = { lfp_target: 0.686, logistic_s: 0.7 };
var LFP_DEFAULTS_CHINA = { lfp_target: 0.648, logistic_s: 0.7 };

var LAND_EXP_SHARE_US    = 0.068;
var LAND_EXP_SHARE_WORLD = 0.100;
var LAND_EXP_SHARE_CHINA = 0.090;

// Physical areas in hectares — China (base year 2025, from Third National Land Survey)
const LAND_AREAS_CHINA = {
    agricultural: 455e6,
    urban: 12e6,
    rural: 17e6,
    commercial: 12e6,
    wilderness: 464e6
};
const TOTAL_LAND_HA_CHINA = 960e6;
// Endogenous categories renormalized: urban 5.4/7.85≈0.688, rural 2.3/7.85≈0.293, ag 0.15/7.85≈0.019
var CAT_EXP_SHARES_CHINA = { urban: 0.688, rural: 0.293, agricultural: 0.019 };

// Chart colors for 5 categories
var LAND_COLORS = {
    agricultural: { border: '#005000', bg: 'rgba(0,80,0,0.7)' },
    urban:        { border: '#6B3A00', bg: 'rgba(107,58,0,0.7)' },
    rural:        { border: '#006060', bg: 'rgba(0,96,96,0.7)' },
    commercial:   { border: '#8B008B', bg: 'rgba(139,0,139,0.7)' },
    wilderness:   { border: '#556B2F', bg: 'rgba(85,107,47,0.7)' }
};

// H100 equivalents required per AI copy (from Plan A&B CSV row 5)
const H100E_PER_AI_COPY = [0.05, 0.08, 0.14, 0.23, 0.39, 0.65, 1.08, 1.80, 3.00, 5.00, 8.34, 13.91, 23.21, 38.71, 64.58, 107.72];

// ========================================
// INCOME DISTRIBUTION FUNCTIONS
// ========================================

// Standard normal quantile (inverse CDF) — Peter Acklam's rational approximation
function normInv(p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    if (Math.abs(p - 0.5) < 1e-15) return 0;

    var a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
              1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0];
    var b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
              6.680131188771972e1, -1.328068155288572e1];
    var c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
             -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
    var d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
             3.754408661907416e0];

    var p_low = 0.02425;
    var p_high = 1 - p_low;
    var q, r;

    if (p < p_low) {
        q = Math.sqrt(-2 * Math.log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    } else {
        q = Math.sqrt(-2 * Math.log(1 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
}

// Number of distribution households: 99 percentiles + 10 ultra-wealthy buckets = 109
var N_HOUSEHOLDS = 109;

// Percentile boundaries for 109 households: [0, 0.01, ..., 0.99, 0.991, ..., 1.0]
var PERCENTILE_BOUNDARIES = (function() {
    var b = [0];
    for (var i = 1; i <= 99; i++) b.push(i / 100);
    for (var i = 1; i <= 10; i++) b.push((990 + i) / 1000);
    return b; // length 110, giving 109 intervals
})();
