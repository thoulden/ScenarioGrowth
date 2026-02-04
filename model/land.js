// Consumption-side land model and trust sector helpers

// ========================================
// CONSUMPTION-SIDE LAND EQUILIBRIUM
// ========================================

// Calibrate base-year land parameters A_i and B_i.
// At base year, for each endogenous type i:
//   v_i(0) = cat_exp_share_i * land_exp_share * E_0 / M_i(0)
//   A_i so that: M_i^D = A_i * E_0^beta_i * v_i^(-eta_D_i) = M_i(0)
//   B_i so that: M_i^S = B_i * v_i^(eta_S_i) = M_i(0)
function calibrateLandBaseYear(E_0, M_observed, params) {
    var A = {}, B = {}, v_0 = {};
    var cats = LAND_ENDOGENOUS_CATEGORIES;
    var total_land_exp = params.land_exp_share * E_0;

    for (var c = 0; c < cats.length; c++) {
        var cat = cats[c];
        var cat_exp = params.cat_exp_share[cat] * total_land_exp;
        v_0[cat] = cat_exp / M_observed[cat];

        // A_i = M_i / (E_0^beta_i * v_i^(-eta_D_i))
        A[cat] = M_observed[cat] / (Math.pow(E_0, params.beta[cat]) * Math.pow(v_0[cat], -params.eta_D[cat]));

        // B_i = M_i / v_i^(eta_S_i)
        B[cat] = M_observed[cat] / Math.pow(v_0[cat], params.eta_S[cat]);
    }

    return { A: A, B: B, v_0: v_0 };
}

// Solve consumption-side land equilibrium for a given total expenditure E_t.
// For each endogenous type i, equating demand = supply gives:
//   v_i = (A_i/B_i)^(1/(eta_D+eta_S)) * E_t^(beta/(eta_D+eta_S))
//   M_i = B_i * v_i^(eta_S)
//   R_i = v_i * M_i
// Buffer categories (commercial, wilderness) have protection floors.
// Wilderness is 100% protected until land_dereg_year, then protect_wilderness applies.
// Unprotected buffer land is available for endogenous expansion.
function solveLandEquilibrium(E_t, landParams, year) {
    var cats = LAND_ENDOGENOUS_CATEGORIES;
    var v = {}, M = {}, R = {};
    var R_total = 0;
    var M_total_endogenous = 0;

    for (var c = 0; c < cats.length; c++) {
        var cat = cats[c];
        var beta_i = landParams.beta[cat];
        var eta_D_i = landParams.eta_D[cat];
        var eta_S_i = landParams.eta_S[cat];
        var A_i = landParams.A[cat];
        var B_i = landParams.B[cat];

        var denom = eta_D_i + eta_S_i;

        // Equilibrium price
        var price_coeff = Math.pow(A_i / B_i, 1.0 / denom);
        v[cat] = price_coeff * Math.pow(Math.max(E_t, TINY), beta_i / denom);

        // Equilibrium acreage
        M[cat] = B_i * Math.pow(v[cat], eta_S_i);

        // Land rent for this type
        R[cat] = v[cat] * M[cat];
        R_total += R[cat];
        M_total_endogenous += M[cat];
    }

    // Buffer land: protected portions are fixed floors, unprotected is available for expansion
    // Wilderness is 100% protected until deregulation year, then ramps down to protect_wilderness
    var dereg_year = landParams.land_dereg_year || 9999;
    var ramp_years = landParams.land_dereg_ramp || 20;
    var eff_protect_wild;
    if (!year || year < dereg_year) {
        eff_protect_wild = 1.0;
    } else if (ramp_years <= 0 || year >= dereg_year + ramp_years) {
        eff_protect_wild = landParams.protect_wilderness;
    } else {
        // Linear ramp from 1.0 at dereg_year to protect_wilderness at dereg_year + ramp_years
        var t = (year - dereg_year) / ramp_years;
        eff_protect_wild = 1.0 + (landParams.protect_wilderness - 1.0) * t;
    }
    var protected_wilderness = eff_protect_wild * landParams.M_wilderness_init;
    var protected_commercial = landParams.protect_commercial * landParams.M_commercial_init;
    var max_endogenous = landParams.total_land - protected_wilderness - protected_commercial;

    // If endogenous types would exceed available land, scale all proportionally.
    // Each type keeps its equilibrium-determined share of total endogenous acreage.
    // Price is re-read from the DEMAND curve (not supply) at constrained acreage:
    //   M = A * E^beta * v^(-eta_D)  →  v = (A * E^beta / M)^(1/eta_D)
    // This ensures scarcity raises price (demand-side willingness to pay).
    if (M_total_endogenous > max_endogenous && M_total_endogenous > 0) {
        var scale = max_endogenous / M_total_endogenous;
        R_total = 0;
        M_total_endogenous = 0;
        for (var c = 0; c < cats.length; c++) {
            var cat = cats[c];
            M[cat] = M[cat] * scale;
            // Price from demand curve: v = (A * E^beta / M)^(1/eta_D)
            var demand_numerator = landParams.A[cat] * Math.pow(Math.max(E_t, TINY), landParams.beta[cat]);
            v[cat] = Math.pow(demand_numerator / M[cat], 1.0 / landParams.eta_D[cat]);
            R[cat] = v[cat] * M[cat];
            R_total += R[cat];
            M_total_endogenous += M[cat];
        }
    }

    // Residual land split between commercial and wilderness proportionally to their unprotected shares
    var residual = landParams.total_land - M_total_endogenous;
    var unprotected_wilderness = landParams.M_wilderness_init * (1 - eff_protect_wild);
    var unprotected_commercial = landParams.M_commercial_init * (1 - landParams.protect_commercial);
    var total_unprotected = unprotected_wilderness + unprotected_commercial;

    if (total_unprotected > 0 && residual > protected_wilderness + protected_commercial) {
        // Distribute residual beyond protected floors proportionally
        var extra = residual - protected_wilderness - protected_commercial;
        var wild_share = unprotected_wilderness / total_unprotected;
        var comm_share = unprotected_commercial / total_unprotected;
        M['wilderness'] = protected_wilderness + extra * wild_share;
        M['commercial'] = protected_commercial + extra * comm_share;
    } else {
        // Endogenous types hit the protection floors
        M['wilderness'] = protected_wilderness;
        M['commercial'] = protected_commercial;
    }

    // Buffer categories have zero rent (not priced in consumption equilibrium)
    v['wilderness'] = 0;
    R['wilderness'] = 0;
    v['commercial'] = 0;
    R['commercial'] = 0;

    var C_nonland = E_t - R_total;
    var land_exp_share = E_t > TINY ? R_total / E_t : 0;

    // Build physical_land map for charts
    var physical_land = {};
    for (var c = 0; c < LAND_CATEGORIES.length; c++) {
        physical_land[LAND_CATEGORIES[c]] = M[LAND_CATEGORIES[c]] || 0;
    }

    return {
        v: v,
        M: M,
        R: R,
        R_total: R_total,
        C_nonland: C_nonland,
        E_t: E_t,
        land_exp_share: land_exp_share,
        physical_land: physical_land,
        rent_per_ha: v
    };
}

// ========================================
// TRUSTED LABOR SECTOR HELPERS
// ========================================

// Efficiency multiplier for trusted labor (linear ramp like land C_2040)
function computeTrustEfficiency(year, startYear, C_2040) {
    var yearsSinceStart = year - startYear;
    if (yearsSinceStart <= 0) return 1.0;
    var yearsTo2040 = 2040 - startYear;
    if (yearsTo2040 <= 0) return C_2040;
    // Linear interpolation: 1.0 at startYear, C_2040 at 2040, extrapolate beyond
    return 1.0 + (C_2040 - 1.0) * yearsSinceStart / yearsTo2040;
}

// Calibrate s_trust so that trusted sector earns target_income at given Y_hat and X_trust
function calibrateTrustShare(Y_hat, X_trust, target_income, sigma) {
    Y_hat = Math.max(Y_hat, TINY);
    X_trust = Math.max(X_trust, TINY);
    var target_share = Math.max(TINY, Math.min(1.0 - TINY, target_income / Y_hat));
    if (Math.abs(sigma - 1.0) < 1e-10) {
        return target_share;
    }
    var rho = (sigma - 1.0) / sigma;
    var Xr = Math.pow(X_trust, rho);
    var Yr = Math.pow(Y_hat, rho);
    var s = target_share * Yr / (target_share * Yr + (1.0 - target_share) * Xr);
    return Math.max(TINY, Math.min(1.0 - TINY, s));
}
