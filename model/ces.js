// CES (Constant Elasticity of Substitution) functions
// 2-input land CES, K-L CES, and task-level CES

// Two-input CES: Z = [s × X1^ρ + (1-s) × X2^ρ]^(1/ρ)
// X1 = land input, X2 = upstream aggregate
function ces2(X_land, X_agg, s, sigma) {
    X_land = Math.max(X_land, TINY);
    X_agg = Math.max(X_agg, TINY);
    s = Math.max(TINY, Math.min(s, 1.0 - TINY));
    if (Math.abs(sigma - 1.0) < 1e-10) {
        return Math.pow(X_land, s) * Math.pow(X_agg, 1.0 - s);
    }
    var rho = (sigma - 1.0) / sigma;
    var term1 = s * Math.pow(X_land, rho);
    var term2 = (1.0 - s) * Math.pow(X_agg, rho);
    return Math.pow(term1 + term2, 1.0 / rho);
}

// Marginal product of land input in a 2-input CES: dZ/dX_land = s × (X_land)^(ρ-1) × Z^(1-ρ)
function ces2_dZ_dXland(Z, X_land, s, sigma) {
    Z = Math.max(Z, TINY);
    X_land = Math.max(X_land, TINY);
    s = Math.max(TINY, Math.min(s, 1.0 - TINY));
    if (Math.abs(sigma - 1.0) < 1e-10) {
        return s * Z / X_land;
    }
    var rho = (sigma - 1.0) / sigma;
    return s * Math.pow(X_land, rho - 1.0) * Math.pow(Z, 1.0 - rho);
}

// Marginal product of upstream aggregate: dZ/dX_agg = (1-s) × (X_agg)^(ρ-1) × Z^(1-ρ)
function ces2_dZ_dXagg(Z, X_agg, s, sigma) {
    Z = Math.max(Z, TINY);
    X_agg = Math.max(X_agg, TINY);
    s = Math.max(TINY, Math.min(s, 1.0 - TINY));
    if (Math.abs(sigma - 1.0) < 1e-10) {
        return (1.0 - s) * Z / X_agg;
    }
    var rho = (sigma - 1.0) / sigma;
    return (1.0 - s) * Math.pow(X_agg, rho - 1.0) * Math.pow(Z, 1.0 - rho);
}

function cesKL(x, y, alpha, sigma) {
    x = Math.max(x, TINY);
    y = Math.max(y, TINY);
    alpha = Math.max(TINY, Math.min(alpha, 1.0 - TINY));

    if (Math.abs(sigma - 1.0) < 1e-10) {
        return Math.pow(x, alpha) * Math.pow(y, 1.0 - alpha);
    }

    const rho = (sigma - 1.0) / sigma;
    const term1 = alpha * Math.pow(x, rho);
    const term2 = (1.0 - alpha) * Math.pow(y, rho);
    return Math.pow(term1 + term2, 1.0 / rho);
}

// Derivative of cesKL wrt first input x (weight α)
// dQ/dx = Q^(1-ρ) × α × x^(ρ-1)
function dCesKL_dx(Q, x, alpha, sigma) {
    x = Math.max(x, TINY);
    Q = Math.max(Q, TINY);

    if (Math.abs(sigma - 1.0) < 1e-10) {
        return alpha * Q / x;
    }

    const rho = (sigma - 1.0) / sigma;
    return Math.pow(Q, 1.0 - rho) * alpha * Math.pow(x, rho - 1.0);
}

// Derivative of cesKL wrt second input y (weight 1-α)
// dQ/dy = Q^(1-ρ) × (1-α) × y^(ρ-1)
function dCesKL_dy(Q, y, alpha, sigma) {
    y = Math.max(y, TINY);
    Q = Math.max(Q, TINY);

    if (Math.abs(sigma - 1.0) < 1e-10) {
        return (1.0 - alpha) * Q / y;
    }

    const rho = (sigma - 1.0) / sigma;
    return Math.pow(Q, 1.0 - rho) * (1.0 - alpha) * Math.pow(y, rho - 1.0);
}

// CES task aggregator (used for labor nests — uses μ^(1/σ) weights)
function cesTaskAgg(x, y, mu, sigma) {
    x = Math.max(x, TINY);
    y = Math.max(y, TINY);
    mu = Math.max(TINY, Math.min(mu, 1.0 - TINY));

    if (Math.abs(sigma - 1.0) < 1e-10) {
        // Cobb-Douglas limit
        return Math.pow(x, mu) * Math.pow(y, 1.0 - mu);
    }

    const rho = (sigma - 1.0) / sigma;
    const term1 = Math.pow(mu, 1.0 / sigma) * Math.pow(x, rho);
    const term2 = Math.pow(1.0 - mu, 1.0 / sigma) * Math.pow(y, rho);
    return Math.pow(term1 + term2, 1.0 / rho);
}

// Derivative wrt first input (weight mu)
function dQdxFirst(Q, x, mu, sigma) {
    x = Math.max(x, TINY);
    Q = Math.max(Q, TINY);
    mu = Math.max(TINY, Math.min(mu, 1.0 - TINY));

    if (Math.abs(sigma - 1.0) < 1e-10) {
        return mu * Q / x;
    }

    const rho = (sigma - 1.0) / sigma;
    return Math.pow(Q, 1.0 - rho) * Math.pow(mu, 1.0 / sigma) * Math.pow(x, rho - 1.0);
}

// Derivative wrt second input (weight 1-mu)
function dQdySecond(Q, y, mu, sigma) {
    y = Math.max(y, TINY);
    Q = Math.max(Q, TINY);
    mu = Math.max(TINY, Math.min(mu, 1.0 - TINY));

    if (Math.abs(sigma - 1.0) < 1e-10) {
        return (1.0 - mu) * Q / y;
    }

    const rho = (sigma - 1.0) / sigma;
    return Math.pow(Q, 1.0 - rho) * Math.pow(1.0 - mu, 1.0 / sigma) * Math.pow(y, rho - 1.0);
}
