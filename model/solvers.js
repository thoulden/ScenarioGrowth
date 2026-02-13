// Numerical solvers: root-finding, capital allocation, price solving, and main equilibrium

function bisectRoot(func, lo, hi, maxIter = 30, tol = 1e-6) {
    let flo = func(lo);
    let fhi = func(hi);

    if (Math.abs(flo) < tol) return lo;
    if (Math.abs(fhi) < tol) return hi;

    if (flo * fhi > 0) {
        return Math.abs(flo) < Math.abs(fhi) ? lo : hi;
    }

    let a = lo, b = hi, fa = flo, fb = fhi;

    for (let i = 0; i < maxIter; i++) {
        const m = 0.5 * (a + b);
        const fm = func(m);

        if (Math.abs(fm) < tol || (b - a) < 1e-14) {
            return m;
        }

        if (fa * fm <= 0) {
            b = m;
            fb = fm;
        } else {
            a = m;
            fa = fm;
        }
    }

    return 0.5 * (a + b);
}

// Solve for interest rate from capital market clearing
// With CES (σ_K≠1): K_Y found by bisection where A × dQ/dK(K_Y, L_eff) = r
// With Cobb-Douglas (σ_K=1): K_Y = alpha*Y/r
// When K_eff nesting is enabled, K_eff = CES(K_Y, Rp) replaces K_Y as capital input
function solveROneT(Kt, Yt, q_c, q_p, LAI, LR, params, L_eff, A_tfp, Rp, phi) {
    const { alpha, delta_K, delta_C, delta_R, tau_k, tau_AI, tau_R } = params;
    const sigma_K = params.sigma_K || 1.0;
    const K0_base = params.K0_base || 0;
    const Leff0_base = params.Leff0_base || 0;
    const useKeffNest = params.enable_K_eff_nest;
    const one_minus_tau_k = 1.0 - tau_k;
    // Rp_K: capital-allocated robots (only these go into K_eff)
    var Rp_K = (useKeffNest && phi != null && phi < 1.0) ? Math.max((1.0 - phi) * Rp, TINY) : 0;

    // Helper: find K_Y given r
    function KY_from_r(r) {
        // K_eff nesting: bisect for K_Y where alpha*Y/K_eff(K_Y,Rp_K) * dKeff/dK = r
        if (useKeffNest && Rp_K > TINY) {
            var K_Y_base = params.K_Y_base || Kt;
            var Rp_base = params.Rp_base || 1.0;
            return bisectRoot(function(Ktest) {
                var K_norm = Ktest / Math.max(K_Y_base, TINY);
                var Rp_norm = Rp_K / Math.max(Rp_base, TINY);
                var Q_norm = cesTaskAgg(K_norm, Rp_norm, params.nu_K, params.sigma_Keff);
                var K_eff = Q_norm * K_Y_base;
                var mp_Keff = alpha * Yt / K_eff;
                var dKeff_dK = dQdxFirst(Q_norm, K_norm, params.nu_K, params.sigma_Keff);
                return mp_Keff * dKeff_dK - r;
            }, TINY, Kt * 10, 120, 1e-12);
        }
        // CES (σ_K ≠ 1): bisect for K_Y where A * dQ/dK = r
        if (Math.abs(sigma_K - 1.0) >= 1e-10) {
            const K0 = (K0_base > 0) ? K0_base : 1.0;
            const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
            return bisectRoot(function(Ktest) {
                var K_norm = Ktest / K0;
                var Q = cesKL(K_norm, L_norm, alpha, sigma_K);
                return A_tfp * dCesKL_dx(Q, K_norm, alpha, sigma_K) / K0 - r;
            }, TINY, Kt * 10, 120, 1e-12);
        }
        // Cobb-Douglas: K_Y = alpha*Y/r
        return alpha * Yt / r;
    }

    // Minimum r to keep denominators positive
    const r_min_C = (delta_K - delta_C) / one_minus_tau_k + 1e-8;
    const r_min_R = (delta_K - delta_R) / one_minus_tau_k + 1e-8;
    const r_min = Math.max(1e-8, r_min_C, r_min_R);

    function RHS(r) {
        const termY = KY_from_r(r);
        let termC = 0.0;
        if (LAI > 0 && isFinite(q_c)) {
            const denom_C = one_minus_tau_k * r - delta_K + delta_C;
            if (denom_C > TINY) {
                termC = (1.0 - tau_AI) * q_c * LAI / denom_C;
            }
        }
        let termR = 0.0;
        if (LR > 0 && isFinite(q_p)) {
            const denom_R = one_minus_tau_k * r - delta_K + delta_R;
            if (denom_R > TINY) {
                termR = (1.0 - tau_R) * q_p * LR / denom_R;
            }
        }
        return termY + termC + termR;
    }

    function H(r) {
        return RHS(r) - Kt;
    }

    // Find upper bound where H(r_hi) < 0
    let r_hi = Math.max(1.0, r_min * 10);
    for (let i = 0; i < 80; i++) {
        if (H(r_hi) < 0) break;
        r_hi *= 2.0;
    }

    // Bisection
    let a = r_min, b = r_hi;
    let fa = H(a), fb = H(b);

    if (!isFinite(fa)) {
        a = r_min * 1.01;
        fa = H(a);
    }

    for (let i = 0; i < 150; i++) {
        const m = 0.5 * (a + b);
        const fm = H(m);
        if (Math.abs(fm) < 1e-12) {
            a = b = m;
            break;
        }
        if (fa * fm > 0) {
            a = m;
            fa = fm;
        } else {
            b = m;
            fb = fm;
        }
    }
    return 0.5 * (a + b);
}

// Compute capital allocations given r
// CES, Cobb-Douglas, or K_eff nesting paths for K_Y
function computeCapitalAllocations(Kt, Yt, r, q_c, q_p, LAI, LR, params, L_eff, A_tfp, Rp, phi) {
    const { alpha, delta_K, delta_C, delta_R, tau_k, tau_AI, tau_R } = params;
    const sigma_K = params.sigma_K || 1.0;
    const K0_base = params.K0_base || 0;
    const Leff0_base = params.Leff0_base || 0;
    const useKeffNest = params.enable_K_eff_nest;
    const one_minus_tau_k = 1.0 - tau_k;
    var Rp_K = (useKeffNest && phi != null && phi < 1.0) ? Math.max((1.0 - phi) * Rp, TINY) : 0;

    // Find K_Y given r
    let K_Y;
    if (useKeffNest && Rp_K > TINY) {
        // K_eff nesting: bisect for K_Y using only capital-allocated robots
        var K_Y_base = params.K_Y_base || Kt;
        var Rp_base = params.Rp_base || 1.0;
        K_Y = bisectRoot(function(Ktest) {
            var K_norm = Ktest / Math.max(K_Y_base, TINY);
            var Rp_norm = Rp_K / Math.max(Rp_base, TINY);
            var Q_norm = cesTaskAgg(K_norm, Rp_norm, params.nu_K, params.sigma_Keff);
            var K_eff = Q_norm * K_Y_base;
            var mp_Keff = alpha * Yt / K_eff;
            var dKeff_dK = dQdxFirst(Q_norm, K_norm, params.nu_K, params.sigma_Keff);
            return mp_Keff * dKeff_dK - r;
        }, TINY, Kt * 10, 120, 1e-12);
    } else if (Math.abs(sigma_K - 1.0) >= 1e-10) {
        // CES: bisect for K_Y
        const K0 = (K0_base > 0) ? K0_base : 1.0;
        const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
        K_Y = bisectRoot(function(Ktest) {
            var K_norm = Ktest / K0;
            var Q = cesKL(K_norm, L_norm, alpha, sigma_K);
            return A_tfp * dCesKL_dx(Q, K_norm, alpha, sigma_K) / K0 - r;
        }, TINY, Kt * 10, 120, 1e-12);
    } else {
        // Cobb-Douglas
        K_Y = alpha * Yt / r;
    }
    let K_C = 0.0;
    if (LAI > 0 && isFinite(q_c)) {
        const denom_C = one_minus_tau_k * r - delta_K + delta_C;
        if (denom_C > TINY) {
            K_C = (1.0 - tau_AI) * q_c * LAI / denom_C;
        }
    }
    let K_R = 0.0;
    if (LR > 0 && isFinite(q_p)) {
        const denom_R = one_minus_tau_k * r - delta_K + delta_R;
        if (denom_R > TINY) {
            K_R = (1.0 - tau_R) * q_p * LR / denom_R;
        }
    }
    return {
        K_Y,
        K_C,
        K_R,
        share_KY: K_Y / Kt,
        share_KC: K_C / Kt,
        share_KR: K_R / Kt
    };
}

// Human split given wage ratio z = wc/wp
// L_c/L_p = kappa * z^omega => ell_c = kappa*z^omega / (kappa*z^omega + 1)
function humanSplit(L, z, kappa, omega) {
    z = Math.max(z, TINY);
    const x = kappa * Math.pow(z, omega);
    const ell = x / (x + 1.0);
    const Hc = ell * L;
    const Hp = (1.0 - ell) * L;
    return { Hc: Math.max(Hc, TINY), Hp: Math.max(Hp, TINY), ell_c: ell };
}

// Prices given mu and wage ratio z (which determines Hc, Hp via human split)
// PY is the shadow price of final output (used in trust mode)
// Supports CES (σ_K≠1), Cobb-Douglas (σ_K=1), and optional K_eff nesting
// Solve for the robot split φ where labor-channel MP = capital-channel MP.
// φ = fraction of Rp going to labor; (1-φ) goes to K_eff.
// Called once per φ-iteration (not per z-step), so speed is fine.
function solveRobotPhi(K, Hp, Rp, L_cog, mu_h_p, sig_p, theta, eps, params, Y, alpha) {
    if (Rp < TINY || !params.enable_K_eff_nest) return 1.0;
    var K_Y_base = params.K_Y_base || K;
    var Rp_base = params.Rp_base || 1.0;
    var nu_K = params.nu_K;
    var sigma_Keff = params.sigma_Keff;

    return bisectRoot(function(phi) {
        var Rp_L = Math.max(phi * Rp, TINY);
        var Rp_K = Math.max((1.0 - phi) * Rp, TINY);

        // Labor channel: L_phys → L_eff → Y
        var L_phys = cesTaskAgg(Hp, Rp_L, mu_h_p, sig_p);
        var L_eff = cesTaskAgg(L_cog, L_phys, theta, eps);
        var mp_Leff = (1.0 - alpha) * Y / L_eff;
        var dLeff_dLphys = dQdySecond(L_eff, L_phys, theta, eps);
        var dLphys_dRpL = dQdySecond(L_phys, Rp_L, mu_h_p, sig_p);
        var mp_labor = mp_Leff * dLeff_dLphys * dLphys_dRpL;

        // Capital channel: K_eff → Y
        var Rp_K_norm = Rp_K / Math.max(Rp_base, TINY);
        var K_norm = K / Math.max(K_Y_base, TINY);
        var Q_norm = cesTaskAgg(K_norm, Rp_K_norm, nu_K, sigma_Keff);
        var K_eff = Q_norm * K_Y_base;
        var mp_Keff = alpha * Y / K_eff;
        var dKeff_dRpK = (K_Y_base / Math.max(Rp_base, TINY)) * dQdySecond(Q_norm, Rp_K_norm, nu_K, sigma_Keff);
        var mp_capital = mp_Keff * dKeff_dRpK;

        return Math.log(Math.max(mp_labor, TINY)) - Math.log(Math.max(mp_capital, TINY));
    }, 1e-4, 1.0 - 1e-4, 30, 1e-4);
}

// Prices given mu and wage ratio z (which determines Hc, Hp via human split)
// PY is the shadow price of final output (used in trust mode)
// Supports CES (σ_K≠1), Cobb-Douglas (σ_K=1), and optional K_eff nesting
// phi: robot split — fraction of Rp going to labor (rest to K_eff). Default 1.0.
function pricesGivenMuAndZ(Y, K, L, AIc, Rp, mu_h_c, mu_h_p, z, params, PY, phi) {
    PY = PY || 1.0;
    phi = (phi != null) ? phi : 1.0;
    const { alpha, theta, eps, sig_c, sig_p, kappa, omega } = params;
    const sigma_K = params.sigma_K || 1.0;

    // Compute human split from wage ratio
    const split = humanSplit(L, z, kappa, omega);
    const Hc = split.Hc;
    const Hp = split.Hp;

    // Robot split: when K_eff nesting is active and φ < 1, split Rp
    const useKeffNest = params.enable_K_eff_nest;
    var Rp_L, Rp_K;
    if (useKeffNest && phi < 1.0 - 1e-9) {
        Rp_L = Math.max(phi * Rp, TINY);
        Rp_K = Math.max((1.0 - phi) * Rp, TINY);
    } else {
        Rp_L = Rp;
        Rp_K = 0;
    }

    // Nests — L_phys uses only the labor-allocated robots
    const L_cog = cesTaskAgg(Hc, AIc, mu_h_c, sig_c);
    const L_phys = cesTaskAgg(Hp, Rp_L, mu_h_p, sig_p);
    const L_eff = cesTaskAgg(L_cog, L_phys, theta, eps);

    // Compute K_eff — only the capital-allocated robots go in
    let K_eff, K_norm_keff, Rp_K_norm_keff, Q_norm_keff;
    let K_Y_base_val, Rp_base_val;
    if (useKeffNest && Rp_K > TINY) {
        K_Y_base_val = params.K_Y_base || K;
        Rp_base_val = params.Rp_base || 1.0;
        K_norm_keff = K / Math.max(K_Y_base_val, TINY);
        Rp_K_norm_keff = Rp_K / Math.max(Rp_base_val, TINY);
        Q_norm_keff = cesTaskAgg(K_norm_keff, Rp_K_norm_keff, params.nu_K, params.sigma_Keff);
        K_eff = Q_norm_keff * K_Y_base_val;
    } else {
        K_eff = K;
    }

    // Compute A, r_raw, mp_Leff depending on production function mode
    const K0_base = params.K0_base || 0;
    const Leff0_base = params.Leff0_base || 0;
    let A, r_raw, mp_Leff;

    if (useKeffNest && Rp_K > TINY) {
        // K_eff nesting: always Cobb-Douglas at top level with K_eff
        A = Y / (Math.pow(K_eff, alpha) * Math.pow(L_eff, 1.0 - alpha));
        const mp_Keff = alpha * Y / K_eff;
        mp_Leff = (1.0 - alpha) * Y / L_eff;

        // dKeff/dK via chain rule (base factors cancel)
        const dKeff_dK = dQdxFirst(Q_norm_keff, K_norm_keff, params.nu_K, params.sigma_Keff);
        r_raw = mp_Keff * dKeff_dK;
    } else if (Math.abs(sigma_K - 1.0) < 1e-10) {
        // Cobb-Douglas (no nesting or φ=1)
        A = Y / (Math.pow(K, alpha) * Math.pow(L_eff, 1.0 - alpha));
        r_raw = alpha * Y / K;
        mp_Leff = (1.0 - alpha) * Y / L_eff;
    } else {
        // CES with normalized inputs (old prod func mode)
        const K_norm = (K0_base > 0) ? K / K0_base : K;
        const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
        const Q = cesKL(K_norm, L_norm, alpha, sigma_K);
        A = Y / Q;
        r_raw = A * dCesKL_dx(Q, K_norm, alpha, sigma_K) / ((K0_base > 0) ? K0_base : 1.0);
        mp_Leff = A * dCesKL_dy(Q, L_norm, alpha, sigma_K) / ((Leff0_base > 0) ? Leff0_base : 1.0);
    }
    const r = r_raw;

    // Chain rule derivatives through labor nests (using Rp_L, not total Rp)
    const dLeff_dLcog = dQdxFirst(L_eff, L_cog, theta, eps);
    const dLeff_dLphys = dQdySecond(L_eff, L_phys, theta, eps);

    const dLcog_dHc = dQdxFirst(L_cog, Hc, mu_h_c, sig_c);
    const dLcog_dAIc = dQdySecond(L_cog, AIc, mu_h_c, sig_c);

    const dLphys_dHp = dQdxFirst(L_phys, Hp, mu_h_p, sig_p);
    const dLphys_dRp = dQdySecond(L_phys, Rp_L, mu_h_p, sig_p);

    // Robot wage: single channel through labor nest (no dual-channel!)
    // At equilibrium φ, this equals the capital channel marginal product.
    const qr_per_unit = mp_Leff * dLeff_dLphys * dLphys_dRp;

    // Scale wages by PY (shadow price from trust layer)
    const wc = PY * mp_Leff * dLeff_dLcog * dLcog_dHc;
    const qc = PY * mp_Leff * dLeff_dLcog * dLcog_dAIc;
    const wp = PY * mp_Leff * dLeff_dLphys * dLphys_dHp;
    const qr = PY * qr_per_unit;

    const profit = Y - (r * K + wc * Hc + wp * Hp + qc * AIc + qr * Rp);

    return { A, r, wc, wp, qc, qr, Hc, Hp, ell_c: split.ell_c, profit, L_eff, L_cog, L_phys, K_eff, phi, Rp_L, Rp_K };
}

// Inner solver: fixed-point iteration for z = wc/wp
// When trustParams is provided, co-iterates on H_trust using inelastic supply curve:
//   H_trust_target = H_trust_base × (trust_wage / wbar)^eps_trust
// eps_trust (supply elasticity) is small (0.1-0.3) → trust supply is inelastic,
// capturing occupational licensing / credential constraints.
function solveZForMu(Y, K, L_total, AIc, Rp, mu_h_c, mu_h_p, params, z0, damp, maxIter, tol, PY_fixed, trustParams, bar_auto_c, bar_auto_p, phi) {
    z0 = z0 || 1.0;
    damp = damp || 0.6;
    maxIter = maxIter || 30;
    tol = tol || 1e-4;
    PY_fixed = PY_fixed || 1.0;
    bar_auto_c = bar_auto_c || 0;
    bar_auto_p = bar_auto_p || 0;

    var trustActive = trustParams && trustParams.active;
    var H_trust_base = trustActive ? (trustParams.H_trust_seed || L_total * 0.01) : 0;
    var H_trust = trustActive ? (trustParams.prev_H_trust || H_trust_base) : 0;
    var eps_trust = trustActive ? (trustParams.eps_trust || 0.2) : 0;
    var trust_wage = 0;
    var trust_income = 0;
    var X_trust = 0;
    var PY_trust = 1.0;
    var trustDamp = 0.3;

    let logz = Math.log(z0);

    for (let i = 0; i < maxIter; i++) {
        // Current L after trust subtraction
        var L = trustActive ? Math.max(L_total - H_trust, TINY) : L_total;

        // Compute PY from trust layer
        var PY;
        if (trustActive) {
            PY_trust = 1.0;
            if (H_trust > TINY) {
                X_trust = trustParams.C_trust * H_trust;
                var Y_at = ces2(X_trust, Y, trustParams.s_trust, trustParams.sigma_trust);
                PY_trust = ces2_dZ_dXagg(Y_at, Y, trustParams.s_trust, trustParams.sigma_trust);
                var trust_rent = ces2_dZ_dXland(Y_at, X_trust, trustParams.s_trust, trustParams.sigma_trust);
                trust_income = trust_rent * X_trust;
                trust_wage = trust_rent * trustParams.C_trust;
            } else {
                trust_wage = 0;
                trust_income = 0;
                X_trust = 0;
            }
            PY = PY_trust;
        } else {
            PY = PY_fixed;
        }

        const z = Math.exp(logz);
        const out = pricesGivenMuAndZ(Y, K, L, AIc, Rp, mu_h_c, mu_h_p, z, params, PY, phi);

        // z-iteration uses unpinned wages
        const z_hat = out.wc / out.wp;
        const logz_hat = Math.log(z_hat);

        // Co-iterate on H_trust: inelastic supply curve
        var z_converged = Math.abs(logz_hat - logz) < tol;
        var trust_converged = true;

        if (trustActive) {
            // Compute effective wbar (blending toward AI prices near full automation)
            var wc_eff = out.wc;
            if (bar_auto_c >= 0.90 && out.wc > out.qc) {
                var bc = Math.min(1.0, (bar_auto_c - 0.90) / 0.10);
                wc_eff = out.wc * (1.0 - bc) + out.qc * bc;
            }
            var wp_eff = out.wp;
            if (bar_auto_p >= 0.90 && out.wp > out.qr) {
                var bp = Math.min(1.0, (bar_auto_p - 0.90) / 0.10);
                wp_eff = out.wp * (1.0 - bp) + out.qr * bp;
            }
            var wbar = computeAverageWage(wc_eff, wp_eff, out.ell_c);

            if (wbar > TINY && trust_wage > TINY) {
                // Inelastic supply curve: H_target = H_base × (trust_wage / wbar)^eps_trust
                // At eps_trust=0: fully fixed at H_base (exogenous)
                // At eps_trust→∞: free equalization (old behavior)
                // At eps_trust=0.2: 100% premium → ~15% more workers (inelastic)
                var premium_ratio = trust_wage / wbar;
                var H_trust_target = H_trust_base * Math.pow(premium_ratio, eps_trust);
                H_trust_target = Math.max(1, Math.min(H_trust_target, L_total - 1));

                var log_H = Math.log(Math.max(H_trust, 1));
                var log_H_target = Math.log(Math.max(H_trust_target, 1));
                trust_converged = Math.abs(log_H_target - log_H) < tol;
                H_trust = Math.exp((1.0 - trustDamp) * log_H + trustDamp * log_H_target);
                H_trust = Math.max(1, Math.min(H_trust, L_total - 1));
            } else if (wbar <= TINY && trust_wage > TINY) {
                // Regular wages collapsed → trust absorbs as much as supply allows
                var H_trust_target = H_trust_base * Math.pow(1000, eps_trust);  // large premium
                H_trust = Math.min(H_trust_target, L_total - 1);
                trust_converged = false;
            } else {
                H_trust = H_trust_base;  // No signal → stay at baseline
                trust_converged = true;
            }
        }

        if (z_converged && trust_converged) {
            out.z = Math.exp(logz_hat);
            out.H_trust = H_trust;
            out.trust_wage_raw = trust_wage;
            out.trust_income_raw = trust_income;
            out.X_trust = X_trust;
            out.PY_trust = PY_trust;
            return out;
        }

        logz = (1.0 - damp) * logz + damp * logz_hat;
    }

    // Return last result even if not fully converged
    var L_final = trustActive ? Math.max(L_total - H_trust, TINY) : L_total;
    var PY_final = trustActive ? PY_trust : PY_fixed;
    const z = Math.exp(logz);
    const out = pricesGivenMuAndZ(Y, K, L_final, AIc, Rp, mu_h_c, mu_h_p, z, params, PY_final, phi);
    out.z = z;
    out.z_converged = false;
    out.H_trust = H_trust;
    out.trust_wage_raw = trust_wage;
    out.trust_income_raw = trust_income;
    out.X_trust = X_trust;
    out.PY_trust = PY_trust;
    return out;
}

// Main solver for one year
// trustParams: { s_trust, sigma_trust, C_trust, eps_trust, H_trust_seed, active } - trusted labor sector
// H_trust determined by inelastic supply curve inside solveZForMu.
function solveMuOneYear(row, params, trustParams) {
    trustParams = trustParams || null;

    // Gate taxes on start year: zero tax rates before ubi_start_year
    var ubi_start = params.ubi_start_year || 9999;
    var year = row.year || 0;
    if (year < ubi_start) {
        params = Object.assign({}, params, { tau_k: 0, tau_AI: 0, tau_R: 0 });
    }

    const { Y, K, H_cog, AI_cog, R_phys, bar_auto_c, bar_auto_p } = row;
    const trustActive = trustParams && trustParams.active;
    const AIc = AI_cog;
    const Rp = R_phys;

    // bar_auto is max automatable share => min human share = 1 - bar_auto
    const mu_h_c_min = Math.max(1e-9, Math.min(1.0 - 1e-9, 1.0 - Math.max(0, Math.min(bar_auto_c, 1.0 - 1e-12))));
    const mu_h_p_min = Math.max(1e-9, Math.min(1.0 - 1e-9, 1.0 - Math.max(0, Math.min(bar_auto_p, 1.0 - 1e-12))));

    // Start at frontier (automate as much as possible)
    let mu_h_c = mu_h_c_min;
    let mu_h_p = mu_h_p_min;

    // When trust is active, solveZForMu co-iterates on H_trust and recomputes
    // PY from trust layer each step. When trust is NOT active, PY = 1.0.
    var PY_pre = 1.0;
    var _tp = trustActive ? trustParams : null;
    const useKeffNest = params.enable_K_eff_nest;

    // Robot split iteration: when K_eff nesting is active, iterate on φ (fraction
    // of Rp going to labor vs capital). φ=1.0 when nesting is off.
    // Each φ iteration runs the full z + mu solve with a fixed φ, then updates φ.
    var phi_est = 1.0;  // start with all robots in labor
    var maxPhiIter = useKeffNest ? 4 : 1;

    var final, mu_c_cur, mu_p_cur;
    var need_c, need_p;

    for (var phiIter = 0; phiIter < maxPhiIter; phiIter++) {

        let out = solveZForMu(Y, K, H_cog, AIc, Rp, mu_h_c, mu_h_p, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p, phi_est);

        // Determine if bisection is needed for each sector.
        need_c = out.wc < out.qc || bar_auto_c >= 0.50;
        need_p = out.wp < out.qr || bar_auto_p >= 0.50;

        // Coordinate bisection on mu's
        mu_c_cur = mu_h_c;
        mu_p_cur = mu_h_p;

        for (let iter = 0; iter < 30; iter++) {
            const prev_c = mu_c_cur;
            const prev_p = mu_p_cur;

            if (need_c) {
                const f = (mu) => {
                    const o = solveZForMu(Y, K, H_cog, AIc, Rp, mu, mu_p_cur, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p, phi_est);
                    return Math.log(o.wc / o.qc);
                };
                var f_lo = f(mu_h_c_min);
                var f_hi = f(1.0 - 1e-9);
                if (f_lo * f_hi < 0) {
                    mu_c_cur = bisectRoot(f, mu_h_c_min, 1.0 - 1e-9);
                } else {
                    mu_c_cur = mu_h_c_min;
                }
            }

            if (need_p) {
                const g = (mu) => {
                    const o = solveZForMu(Y, K, H_cog, AIc, Rp, mu_c_cur, mu, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p, phi_est);
                    return Math.log(o.wp / o.qr);
                };
                var g_lo = g(mu_h_p_min);
                var g_hi = g(1.0 - 1e-9);
                if (g_lo * g_hi < 0) {
                    mu_p_cur = bisectRoot(g, mu_h_p_min, 1.0 - 1e-9);
                } else {
                    mu_p_cur = mu_h_p_min;
                }
            }

            if (Math.max(Math.abs(mu_c_cur - prev_c), Math.abs(mu_p_cur - prev_p)) < 1e-4) {
                break;
            }
        }

        // Final solve with converged mu values at current φ
        final = solveZForMu(Y, K, H_cog, AIc, Rp, mu_c_cur, mu_p_cur, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p, phi_est);

        // If K_eff nesting active, update φ from equilibrium condition
        if (useKeffNest && phiIter < maxPhiIter - 1) {
            var phi_new = solveRobotPhi(K, final.Hp, Rp, final.L_cog, mu_p_cur, params.sig_p, params.theta, params.eps, params, Y, params.alpha);
            if (Math.abs(phi_new - phi_est) < 1e-3) {
                phi_est = phi_new;
                // Re-do final solve with converged φ
                final = solveZForMu(Y, K, H_cog, AIc, Rp, mu_c_cur, mu_p_cur, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p, phi_est);
                break;
            }
            phi_est = 0.5 * phi_est + 0.5 * phi_new;  // damped update
        }
    }

    // No-arbitrage: at full automation, human wage cannot exceed machine wage
    if (bar_auto_c >= 1.0 - 1e-6 && final.wc > final.qc) {
        final.wc = final.qc;
    }
    if (bar_auto_p >= 1.0 - 1e-6 && final.wp > final.qr) {
        final.wp = final.qr;
    }

    // Extract trust outputs from the converged solver
    const H_trust = final.H_trust || 0;
    const L = Math.max(H_cog - H_trust, TINY);
    const trust_wage_raw = final.trust_wage_raw || 0;
    const trust_income_raw = final.trust_income_raw || 0;
    const X_trust_val = final.X_trust || 0;
    const PY_trust = final.PY_trust || 1.0;
    const PY = PY_trust;

    // Compute Y_final from trust layer (for income shares)
    let Y_after_trust = Y;
    if (trustActive && H_trust > TINY) {
        Y_after_trust = ces2(X_trust_val, Y, trustParams.s_trust, trustParams.sigma_trust);
    }
    let Y_final = Y_after_trust;

    // Solve for interest rate from capital market clearing
    // Pass Rp for K_eff nesting bisection
    const r = solveROneT(K, Y, final.qc, final.qr, AIc, Rp, params, final.L_eff, final.A, Rp, phi_est);
    const capAlloc = computeCapitalAllocations(K, Y, r, final.qc, final.qr, AIc, Rp, params, final.L_eff, final.A, Rp, phi_est);

    // Factor income shares
    const alpha = params.alpha;
    const sigma_K_share = params.sigma_K || 1.0;
    let ces_capital_share;
    if (useKeffNest) {
        // With K_eff nesting, capital share = r * K_Y / Y (from solved prices)
        ces_capital_share = Math.max(0, Math.min(1, r * capAlloc.K_Y / Math.max(Y, TINY)));
    } else if (Math.abs(sigma_K_share - 1.0) < 1e-10) {
        // Pure Cobb-Douglas: capital share = alpha
        ces_capital_share = alpha;
    } else {
        // CES: capital share varies with K/L ratio
        const K0_base_share = params.K0_base || 0;
        const Leff0_base_share = params.Leff0_base || 0;
        const rho = (sigma_K_share - 1.0) / sigma_K_share;
        const K_n = (K0_base_share > 0) ? K / K0_base_share : K;
        const L_n = (Leff0_base_share > 0) ? final.L_eff / Leff0_base_share : final.L_eff;
        const termK = alpha * Math.pow(K_n, rho);
        const termL = (1.0 - alpha) * Math.pow(L_n, rho);
        ces_capital_share = termK / (termK + termL);
    }
    const ces_labor_share = 1.0 - ces_capital_share;
    const human_cog_val = final.wc * final.Hc;
    const human_phys_val = final.wp * final.Hp;
    const ai_val = final.qc * AIc;
    const robot_val = final.qr * Rp;
    const total_labor_val = human_cog_val + human_phys_val + ai_val + robot_val;

    const { tau_k, tau_AI, tau_R } = params;
    const T_capital = tau_k * r * capAlloc.K_Y;
    const T_AI = tau_AI * final.qc * AIc;
    const T_robot = tau_R * final.qr * Rp;

    const WAP = row.WorkingAgePop || L;
    const wage_per_worker = (human_cog_val + human_phys_val) / WAP;
    const T_capital_per_worker = T_capital / WAP;
    const T_AI_per_worker = T_AI / WAP;
    const T_robot_per_worker = T_robot / WAP;

    const total_income = Y_final;
    const yhat_to_yfinal = (trustActive && Y_final > TINY) ? PY * Y / Y_final : 1.0;
    const cap_share_final = ces_capital_share * yhat_to_yfinal;
    const lab_share_final = ces_labor_share * yhat_to_yfinal;
    const labor_frac = (total_labor_val > TINY) ? 1.0 / total_labor_val : 0;

    const trust_income_final = trust_income_raw;
    const trust_income_share = trustActive ? trust_income_final / Math.max(Y_final, TINY) : 0;

    const result = {
        ...final,
        r,
        ...capAlloc,
        mu_h_c: mu_c_cur,
        mu_h_p: mu_p_cur,
        mu_h_c_min,
        mu_h_p_min,
        H_cog_allocated: final.Hc,
        H_phys_allocated: final.Hp,
        regime_c: need_c ? "endogenous" : "frontier",
        regime_p: need_p ? "endogenous" : "frontier",
        capital_share: cap_share_final,
        human_cog_share: lab_share_final * human_cog_val * labor_frac,
        human_phys_share: lab_share_final * human_phys_val * labor_frac,
        ai_share: lab_share_final * ai_val * labor_frac,
        robot_share: lab_share_final * robot_val * labor_frac,
        trust_income: trust_income_final,
        trust_income_share: trust_income_share,
        trust_wage: trust_wage_raw,
        H_trust: H_trust,
        X_trust: X_trust_val,
        wage_per_worker,
        T_capital_per_worker,
        T_AI_per_worker,
        T_robot_per_worker,
        T_total_per_worker: T_capital_per_worker + T_AI_per_worker + T_robot_per_worker,
        phi: phi_est,
        Rp_labor: phi_est * Rp,
        Rp_capital: (1.0 - phi_est) * Rp
    };

    if (trustActive) {
        result.Y_final = Y_final;
        result.PY = PY;
    }

    // Per-year K_eff debug logging
    if (params._debug_keff && useKeffNest) {
        console.log('[K_eff yr=' + year + '] phi=' + phi_est.toFixed(4) +
            ' Rp_L=' + (phi_est * Rp / 1e6).toFixed(2) + 'M Rp_K=' + ((1 - phi_est) * Rp / 1e6).toFixed(2) + 'M' +
            ' | wp=' + final.wp.toFixed(2) + ' qr=' + final.qr.toFixed(2) +
            ' | r=' + r.toFixed(6) + ' K_Y=' + (capAlloc.K_Y / 1e12).toFixed(4) + 'T' +
            ' K_eff=' + (final.K_eff / 1e12).toFixed(4) + 'T');
    }

    return result;
}
