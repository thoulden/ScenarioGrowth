// Two-sector (Goods vs Services) decomposition
// Sectors differ only in θ (cognitive vs physical task weight).
// Common factor markets → post-solver decomposition from equilibrium prices.

// CES dual price index (unit cost): p = [mu*p1^(1-sig) + (1-mu)*p2^(1-sig)]^(1/(1-sig))
// This is the dual of cesTaskAgg which uses mu^(1/sigma) primal weights.
// When sigma = 1 (Cobb-Douglas): p = p1^mu * p2^(1-mu)
function cesDualPrice(p1, p2, mu, sigma) {
    p1 = Math.max(p1, TINY);
    p2 = Math.max(p2, TINY);
    mu = Math.max(TINY, Math.min(mu, 1.0 - TINY));
    if (Math.abs(sigma - 1.0) < 1e-10) {
        return Math.pow(p1, mu) * Math.pow(p2, 1.0 - mu);
    }
    var e = 1.0 - sigma;
    return Math.pow(mu * Math.pow(p1, e) + (1.0 - mu) * Math.pow(p2, e), 1.0 / e);
}

// Calibrate ν from base-year sector prices and empirical service spending share.
// ν = a * P_G^(1-η) / [a * P_G^(1-η) + (1-a) * P_S^(1-η)]
function calibrateSectorNu(P_S_0, P_G_0, a_services, eta) {
    var a = Math.max(TINY, Math.min(a_services, 1.0 - TINY));
    if (Math.abs(eta - 1.0) < 1e-10) {
        // Cobb-Douglas limit: ν = a (service share = weight)
        return a;
    }
    var e = 1.0 - eta;
    var PG_e = Math.pow(Math.max(P_G_0, TINY), e);
    var PS_e = Math.pow(Math.max(P_S_0, TINY), e);
    return a * PG_e / (a * PG_e + (1.0 - a) * PS_e);
}

// Main two-sector decomposition.
// Takes solver result (common factor prices) and sector parameters.
// Returns sector prices, outputs, and shares.
//
// sectorParams: { theta_S, theta_G, theta_agg, eta, nu, eps, sig_c, sig_p, alpha }
function computeSectorDecomposition(result, sectorParams) {
    var wc = result.wc, wp = result.wp, qc = result.qc, qr = result.qr;
    var mu_h_c = result.mu_h_c, mu_h_p = result.mu_h_p;
    var Y = result.Y_predicted || result.Y || 1.0;

    var theta_S = sectorParams.theta_S;
    var theta_G = sectorParams.theta_G;
    var theta_agg = sectorParams.theta_agg;
    var eta = sectorParams.eta;
    var nu = sectorParams.nu;
    var eps = sectorParams.eps;
    var sig_c = sectorParams.sig_c;
    var sig_p = sectorParams.sig_p;
    var alpha = sectorParams.alpha;

    // Sub-nest prices (common across sectors)
    var p_cog = cesDualPrice(wc, qc, mu_h_c, sig_c);
    var p_phys = cesDualPrice(wp, qr, mu_h_p, sig_p);

    // Sector-specific effective labor prices
    var p_eff_S = cesDualPrice(p_cog, p_phys, theta_S, eps);
    var p_eff_G = cesDualPrice(p_cog, p_phys, theta_G, eps);
    var p_eff_agg = cesDualPrice(p_cog, p_phys, theta_agg, eps);

    // Relative sector prices: P_j = (p_eff_j / p_eff_agg)^(1-α)
    // Capital cost r and TFP A cancel in the ratio since sectors share them.
    var P_S_raw = Math.pow(p_eff_S / Math.max(p_eff_agg, TINY), 1.0 - alpha);
    var P_G_raw = Math.pow(p_eff_G / Math.max(p_eff_agg, TINY), 1.0 - alpha);

    // Normalize so CES price aggregator = 1 exactly (ensures expenditure shares sum to 1)
    var P_agg = cesDualPrice(P_S_raw, P_G_raw, nu, eta);
    var P_S = P_S_raw / Math.max(P_agg, TINY);
    var P_G = P_G_raw / Math.max(P_agg, TINY);

    // CES demand → sector outputs (P_agg ≡ 1 by normalization)
    var S = nu * Math.pow(Math.max(P_S, TINY), -eta) * Y;
    var G = (1.0 - nu) * Math.pow(Math.max(P_G, TINY), -eta) * Y;

    var nominal_S = P_S * S;
    var nominal_G = P_G * G;

    return {
        P_S: P_S,
        P_G: P_G,
        S: S,
        G: G,
        nominal_S: nominal_S,
        nominal_G: nominal_G,
        service_share: nominal_S / Math.max(Y, TINY),
        price_ratio: P_S / Math.max(P_G, TINY),
        p_cog: p_cog,
        p_phys: p_phys,
        p_eff_S: p_eff_S,
        p_eff_G: p_eff_G
    };
}
