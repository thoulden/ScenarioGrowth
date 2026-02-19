// Model constants — shared between model code and chart files
// Region-specific parameters now live in params/params_{us,china,world}.js
// Variables below are backward-compatible aliases.

const TINY = 1e-12;

// ========================================
// 5-CATEGORY LAND CONSTANTS (consumption-side model)
// ========================================
var LAND_CATEGORIES = ['agricultural', 'urban', 'rural', 'commercial', 'wilderness'];
var LAND_ENDOGENOUS_CATEGORIES = ['agricultural', 'urban', 'rural'];
var LAND_BUFFER_CATEGORIES = ['commercial', 'wilderness'];
var LAND_CATEGORY_LABELS = {
    agricultural: 'Agricultural',
    urban: 'Urban/Residential',
    rural: 'Rural/Residential',
    commercial: 'Commercial/Industrial',
    wilderness: 'Uninhabited/Wilderness'
};

// Backward-compatible aliases — now drawn from param files
var LAND_AREAS_US    = PARAMS_US.land_areas;
var LAND_AREAS_CHINA = PARAMS_CHINA.land_areas;
var LAND_AREAS_WORLD = PARAMS_WORLD.land_areas;
var TOTAL_LAND_HA_US    = PARAMS_US.total_land_ha;
var TOTAL_LAND_HA_CHINA = PARAMS_CHINA.total_land_ha;
var TOTAL_LAND_HA_WORLD = PARAMS_WORLD.total_land_ha;
var CAT_EXP_SHARES_US    = PARAMS_US.cat_exp_shares;
var CAT_EXP_SHARES_CHINA = PARAMS_CHINA.cat_exp_shares;
var CAT_EXP_SHARES_WORLD = PARAMS_WORLD.cat_exp_shares;
var DEFAULT_LAND_EXP_SHARE = PARAMS_US.land_exp_share;

// CDF, multiplier, LFP, and land expenditure share aliases
var CDF_DEFAULTS_US    = PARAMS_US.cdf;
var CDF_DEFAULTS_CHINA = PARAMS_CHINA.cdf;
var CDF_DEFAULTS_WORLD = PARAMS_WORLD.cdf;
var MULT_DEFAULTS_US    = PARAMS_US.mult;
var MULT_DEFAULTS_CHINA = PARAMS_CHINA.mult;
var MULT_DEFAULTS_WORLD = PARAMS_WORLD.mult;
var LFP_DEFAULTS_US    = { lfp_target: PARAMS_US.lfp_target, logistic_s: PARAMS_US.logistic_s, frisch_elasticity: PARAMS_US.frisch_elasticity };
var LFP_DEFAULTS_CHINA = { lfp_target: PARAMS_CHINA.lfp_target, logistic_s: PARAMS_CHINA.logistic_s, frisch_elasticity: PARAMS_CHINA.frisch_elasticity };
var LFP_DEFAULTS_WORLD = { lfp_target: PARAMS_WORLD.lfp_target, logistic_s: PARAMS_WORLD.logistic_s, frisch_elasticity: PARAMS_WORLD.frisch_elasticity };
var LAND_EXP_SHARE_US    = PARAMS_US.land_exp_share;
var LAND_EXP_SHARE_CHINA = PARAMS_CHINA.land_exp_share;
var LAND_EXP_SHARE_WORLD = PARAMS_WORLD.land_exp_share;

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
