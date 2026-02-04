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
// With CES: K_Y found by bisection where A x dQ/dK(K_Y, L_eff) = r
// With Cobb-Douglas: K_Y = alpha*Y/r
function solveROneT(Kt, Yt, q_c, q_p, LAI, LR, params, L_eff, A_tfp) {
    const { alpha, delta_K, delta_C, delta_R, tau_k, tau_AI, tau_R } = params;
    const sigma_K = params.sigma_K || 1.0;
    const K0_base = params.K0_base || 0;
    const Leff0_base = params.Leff0_base || 0;
    const one_minus_tau_k = 1.0 - tau_k;

    // Helper: find K_Y given r (CES requires bisection)
    function KY_from_r(r) {
        if (Math.abs(sigma_K - 1.0) < 1e-10) {
            return alpha * Yt / r;
        }
        // Bisect: find K where A_tfp * dQ/d(K/K₀) / K₀ = r
        const K0 = (K0_base > 0) ? K0_base : 1.0;
        const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
        return bisectRoot(function(Ktest) {
            var K_norm = Ktest / K0;
            var Q = cesKL(K_norm, L_norm, alpha, sigma_K);
            return A_tfp * dCesKL_dx(Q, K_norm, alpha, sigma_K) / K0 - r;
        }, TINY, Kt * 10, 120, 1e-12);
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
// With CES: K_Y found by bisection where A x dQ/dK(K_Y, L_eff) = r
// With Cobb-Douglas: K_Y = alpha*Y/r
function computeCapitalAllocations(Kt, Yt, r, q_c, q_p, LAI, LR, params, L_eff, A_tfp) {
    const { alpha, delta_K, delta_C, delta_R, tau_k, tau_AI, tau_R } = params;
    const sigma_K = params.sigma_K || 1.0;
    const K0_base = params.K0_base || 0;
    const Leff0_base = params.Leff0_base || 0;
    const one_minus_tau_k = 1.0 - tau_k;

    // Find K_Y given r (CES requires bisection)
    let K_Y;
    if (Math.abs(sigma_K - 1.0) < 1e-10) {
        K_Y = alpha * Yt / r;
    } else {
        const K0 = (K0_base > 0) ? K0_base : 1.0;
        const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
        K_Y = bisectRoot(function(Ktest) {
            var K_norm = Ktest / K0;
            var Q = cesKL(K_norm, L_norm, alpha, sigma_K);
            return A_tfp * dCesKL_dx(Q, K_norm, alpha, sigma_K) / K0 - r;
        }, TINY, Kt * 10, 120, 1e-12);
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
function pricesGivenMuAndZ(Y, K, L, AIc, Rp, mu_h_c, mu_h_p, z, params, PY) {
    PY = PY || 1.0;
    const { alpha, theta, eps, sig_c, sig_p, kappa, omega } = params;
    const sigma_K = params.sigma_K || 1.0;

    // Compute human split from wage ratio
    const split = humanSplit(L, z, kappa, omega);
    const Hc = split.Hc;
    const Hp = split.Hp;

    // Nests
    const L_cog = cesTaskAgg(Hc, AIc, mu_h_c, sig_c);
    const L_phys = cesTaskAgg(Hp, Rp, mu_h_p, sig_p);
    const L_eff = cesTaskAgg(L_cog, L_phys, theta, eps);

    // Infer A and marginal products using CES or Cobb-Douglas
    const K0_base = params.K0_base || 0;
    const Leff0_base = params.Leff0_base || 0;
    let A, r_raw, mp_Leff;
    if (Math.abs(sigma_K - 1.0) < 1e-10) {
        // Cobb-Douglas
        A = Y / (Math.pow(K, alpha) * Math.pow(L_eff, 1.0 - alpha));
        r_raw = alpha * Y / K;
        mp_Leff = (1.0 - alpha) * Y / L_eff;
    } else {
        // CES with normalized inputs
        const K_norm = (K0_base > 0) ? K / K0_base : K;
        const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
        const Q = cesKL(K_norm, L_norm, alpha, sigma_K);
        A = Y / Q;
        r_raw = A * dCesKL_dx(Q, K_norm, alpha, sigma_K) / ((K0_base > 0) ? K0_base : 1.0);
        mp_Leff = A * dCesKL_dy(Q, L_norm, alpha, sigma_K) / ((Leff0_base > 0) ? Leff0_base : 1.0);
    }
    const r = r_raw;

    // Chain rule derivatives
    const dLeff_dLcog = dQdxFirst(L_eff, L_cog, theta, eps);
    const dLeff_dLphys = dQdySecond(L_eff, L_phys, theta, eps);

    const dLcog_dHc = dQdxFirst(L_cog, Hc, mu_h_c, sig_c);
    const dLcog_dAIc = dQdySecond(L_cog, AIc, mu_h_c, sig_c);

    const dLphys_dHp = dQdxFirst(L_phys, Hp, mu_h_p, sig_p);
    const dLphys_dRp = dQdySecond(L_phys, Rp, mu_h_p, sig_p);

    // Scale wages by PY (shadow price from trust layer)
    const wc = PY * mp_Leff * dLeff_dLcog * dLcog_dHc;
    const qc = PY * mp_Leff * dLeff_dLcog * dLcog_dAIc;
    const wp = PY * mp_Leff * dLeff_dLphys * dLphys_dHp;
    const qr = PY * mp_Leff * dLeff_dLphys * dLphys_dRp;

    const profit = Y - (r * K + wc * Hc + wp * Hp + qc * AIc + qr * Rp);

    return { A, r, wc, wp, qc, qr, Hc, Hp, ell_c: split.ell_c, profit, L_eff, L_cog, L_phys };
}

// Inner solver: fixed-point iteration for z = wc/wp
// When trustParams is provided, co-iterates on H_trust to equalize trust_wage and wbar.
function solveZForMu(Y, K, L_total, AIc, Rp, mu_h_c, mu_h_p, params, z0, damp, maxIter, tol, PY_fixed, trustParams, bar_auto_c, bar_auto_p) {
    z0 = z0 || 1.0;
    damp = damp || 0.6;
    maxIter = maxIter || 30;
    tol = tol || 1e-4;
    PY_fixed = PY_fixed || 1.0;
    bar_auto_c = bar_auto_c || 0;
    bar_auto_p = bar_auto_p || 0;

    var trustActive = trustParams && trustParams.active;
    var trustAnchored = trustActive && trustParams.anchor_H_trust;
    var H_trust = trustActive ? (trustParams.prev_H_trust || trustParams.H_trust_seed || L_total * 0.01) : 0;
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
        const out = pricesGivenMuAndZ(Y, K, L, AIc, Rp, mu_h_c, mu_h_p, z, params, PY);

        // z-iteration uses unpinned wages
        const z_hat = out.wc / out.wp;
        const logz_hat = Math.log(z_hat);

        // Co-iterate on H_trust: adjust toward wage equalization
        var z_converged = Math.abs(logz_hat - logz) < tol;
        var trust_converged = true;

        if (trustActive) {
            if (trustAnchored) {
                trust_converged = true;
            } else {
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
                var tw = trust_wage;

                if (wbar > TINY && tw > TINY) {
                    var ratio = tw / wbar;
                    var log_H = Math.log(Math.max(H_trust, 1));
                    var log_H_target = log_H + trustParams.sigma_trust * Math.log(ratio);
                    var H_trust_new = Math.exp(log_H_target);
                    H_trust_new = Math.max(1, Math.min(H_trust_new, L_total - 1));
                    trust_converged = Math.abs(Math.log(Math.max(H_trust_new, 1)) - log_H) < tol;
                    H_trust = Math.exp((1.0 - trustDamp) * log_H + trustDamp * Math.log(Math.max(H_trust_new, 1)));
                    H_trust = Math.max(1, Math.min(H_trust, L_total - 1));
                } else if (wbar <= TINY && tw > TINY) {
                    H_trust = L_total - 1;
                    trust_converged = false;
                } else {
                    H_trust = 0;
                    trust_converged = true;
                }
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
    const out = pricesGivenMuAndZ(Y, K, L_final, AIc, Rp, mu_h_c, mu_h_p, z, params, PY_final);
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
// trustParams: { s_trust, sigma_trust, C_trust, active } - trusted labor sector
// H_trust is solved endogenously inside solveZForMu via co-iteration with z.
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

    let out = solveZForMu(Y, K, H_cog, AIc, Rp, mu_h_c, mu_h_p, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p);

    // Determine if bisection is needed for each sector.
    // At the frontier (mu_min), if wc < qc humans are cheaper => bisect up to find wc = qc.
    // At the frontier, if wc >= qc humans are more expensive => at full automation this is
    // expected (scarce humans have high marginal product). We must STILL bisect to find the
    // equilibrium mu where wc = qc, since that's the no-arbitrage condition.
    // Only skip bisection when automation is low AND humans are already more expensive.
    const need_c = out.wc < out.qc || bar_auto_c >= 0.50;
    const need_p = out.wp < out.qr || bar_auto_p >= 0.50;

    // Coordinate bisection on mu's
    let mu_c_cur = mu_h_c;
    let mu_p_cur = mu_h_p;

    for (let iter = 0; iter < 30; iter++) {
        const prev_c = mu_c_cur;
        const prev_p = mu_p_cur;

        if (need_c) {
            const f = (mu) => {
                const o = solveZForMu(Y, K, H_cog, AIc, Rp, mu, mu_p_cur, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p);
                return Math.log(o.wc / o.qc);
            };
            var f_lo = f(mu_h_c_min);
            var f_hi = f(1.0 - 1e-9);
            if (f_lo * f_hi < 0) {
                mu_c_cur = bisectRoot(f, mu_h_c_min, 1.0 - 1e-9);
            } else {
                // No sign change — stay at frontier
                mu_c_cur = mu_h_c_min;
            }
        }

        if (need_p) {
            const g = (mu) => {
                const o = solveZForMu(Y, K, H_cog, AIc, Rp, mu_c_cur, mu, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p);
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

    // Final solve with converged mu values
    const final = solveZForMu(Y, K, H_cog, AIc, Rp, mu_c_cur, mu_p_cur, params, 1.0, 0.6, 30, 1e-4, PY_pre, _tp, bar_auto_c, bar_auto_p);

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
    const r = solveROneT(K, Y, final.qc, final.qr, AIc, Rp, params, final.L_eff, final.A);
    const capAlloc = computeCapitalAllocations(K, Y, r, final.qc, final.qr, AIc, Rp, params, final.L_eff, final.A);

    // Factor income shares
    const alpha = params.alpha;
    const sigma_K_share = params.sigma_K || 1.0;
    const K0_base_share = params.K0_base || 0;
    const Leff0_base_share = params.Leff0_base || 0;
    let ces_capital_share;
    if (Math.abs(sigma_K_share - 1.0) < 1e-10) {
        ces_capital_share = alpha;
    } else {
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
        T_total_per_worker: T_capital_per_worker + T_AI_per_worker + T_robot_per_worker
    };

    if (trustActive) {
        result.Y_final = Y_final;
        result.PY = PY;
    }

    return result;
}
