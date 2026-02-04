// Labor supply, LFP model, UBI computation, and endogenous labor solvers

// Compute average wage: w̄ = w_c × ℓ_c + w_p × (1 - ℓ_c)
function computeAverageWage(wc, wp, ell_c) {
    return wc * ell_c + wp * (1.0 - ell_c);
}

// Legacy labor supply function (used when distribution mode is off)
function laborSupply(B, wbar, psi, workingAgePop) {
    const L_uncapped = B * Math.pow(Math.max(wbar, TINY), 1.0 / psi);
    return Math.min(L_uncapped, workingAgePop);
}

// Legacy calibration (used when distribution mode is off)
function calibrateLaborSupplyB(L_0, wbar_0, psi) {
    return L_0 / Math.pow(Math.max(wbar_0, TINY), 1.0 / psi);
}

// ---- New LFP model: logistic baseline × UBI dampener ----

// Logistic function on log(w_multiple), auto-calibrated so F(1) = lfp_target
function lfpBaselineCurve(w_multiple, lfp_target, logistic_s) {
    var shift = Math.log(1.0 / lfp_target - 1.0) / logistic_s;
    var z = logistic_s * (Math.log(Math.max(w_multiple, TINY)) - shift);
    return 1.0 / (1.0 + Math.exp(-z));
}

// UBI dampener: D(ratio) = exp(-k_ubi * ratio)
function lfpUBIDampener(ubi_per_person, wbar, k_ubi) {
    if (ubi_per_person <= 0 || wbar <= 0) return 1.0;
    var ratio = ubi_per_person / wbar;
    return Math.exp(-k_ubi * ratio);
}

// Combined LFP: participation_rate = F(w_multiple) × D(UBI/wage)
function computeLFP(wbar, wbar0, ubi_per_person, params) {
    var w_multiple = wbar / Math.max(wbar0, TINY);
    var p_base = lfpBaselineCurve(w_multiple, params.lfp_target, params.logistic_s);
    var dampener = lfpUBIDampener(ubi_per_person, wbar, params.k_ubi);
    return p_base * dampener;
}

// Compute UBI per person from equilibrium result
function computeUBIPerPerson(year, eqResult, params, workingAgePop) {
    var d_ubi_start = params.dist_ubi_start_year || 9999;
    if (year < d_ubi_start) return 0;
    var d_tau_k = params.dist_tau_k || 0;
    var d_tau_AI = params.dist_tau_AI || 0;
    var d_tau_R = params.dist_tau_R || 0;
    var d_ubi_share_us = params.dist_ubi_share_us || 0;
    var T_total = d_tau_k * eqResult.r * eqResult.K_Y +
                  d_tau_AI * eqResult.qc * eqResult.AI_cog +
                  d_tau_R * eqResult.qr * eqResult.R_phys;
    return T_total * d_ubi_share_us / Math.max(workingAgePop, 1);
}

// Solve for predicted output WITH endogenous labor supply
// Outer loop: iterate on Y until production function holds
// Inner loop: iterate on L until labor supply converges
function solveYearWithPredictedOutputAndEndogenousLabor(row, params, A, B, workingAgePop, Y_guess, trustParams, maxOuterIter, maxInnerIter, tol) {
    maxOuterIter = maxOuterIter || 10;
    maxInnerIter = maxInnerIter || 8;
    tol = tol || 1e-3;
    var useLFP = params.wbar0 != null;
    var damp = params.labor_damp || 0.5;
    var Y = Y_guess;

    for (var outerIter = 0; outerIter < maxOuterIter; outerIter++) {
        var L = row.H_data || workingAgePop * 0.6;

        for (var innerIter = 0; innerIter < maxInnerIter; innerIter++) {
            var rowWithYL = Object.assign({}, row, { Y: Y, H_cog: L, H_phys: L });
            var result = solveMuOneYear(rowWithYL, params, trustParams);
            var wbar = computeAverageWage(result.wc, result.wp, result.ell_c);

            var L_new;
            if (useLFP) {
                var merged = Object.assign({}, row, result);
                var ubi = computeUBIPerPerson(row.year, merged, params, workingAgePop);
                var lfp = Math.max(0, computeLFP(wbar, params.wbar0, ubi, params));
                L_new = Math.min(lfp * workingAgePop, workingAgePop);
            } else {
                L_new = laborSupply(B, wbar, params.psi, workingAgePop);
            }

            if (Math.abs(L_new - L) / Math.max(L, TINY) < tol) {
                L = L_new;
                break;
            }
            L = (1 - damp) * L + damp * L_new;
        }

        var rowWithYL2 = Object.assign({}, row, { Y: Y, H_cog: L, H_phys: L });
        var result2 = solveMuOneYear(rowWithYL2, params, trustParams);
        var K_Y = result2.K_Y;
        var L_eff = result2.L_eff;
        var trustParamsResolved2 = (trustParams && trustParams.active && result2.X_trust > 0)
            ? Object.assign({}, trustParams, { X_trust: result2.X_trust }) : trustParams;
        var Y_pred = productionFunctionFull(A, K_Y, L_eff, params.alpha, params.sigma_K, params.K0_base, params.Leff0_base, trustParamsResolved2);

        var relError = Math.abs(Y_pred - Y) / Math.max(Y, TINY);
        if (relError < tol) {
            var finalRow = Object.assign({}, row, { Y: Y_pred, H_cog: L, H_phys: L });
            var finalResult = solveMuOneYear(finalRow, params, trustParams);
            finalResult.Y_predicted = Y_pred;
            finalResult.Y_forecast = row.Y;
            finalResult.L_endogenous = L;
            finalResult.wbar = computeAverageWage(finalResult.wc, finalResult.wp, finalResult.ell_c);
            finalResult.effective_wage = finalResult.wbar;
            finalResult.participation_rate = L / workingAgePop;
            if (useLFP) {
                var mergedFinal = Object.assign({}, row, finalResult);
                finalResult.ubi_per_person = computeUBIPerPerson(row.year, mergedFinal, params, workingAgePop);
            }
            return finalResult;
        }
        Y = 0.5 * Y + 0.5 * Y_pred;
    }

    var finalRow2 = Object.assign({}, row, { Y: Y, H_cog: L, H_phys: L });
    var finalResult2 = solveMuOneYear(finalRow2, params, trustParams);
    finalResult2.Y_predicted = Y;
    finalResult2.Y_forecast = row.Y;
    finalResult2.L_endogenous = L;
    finalResult2.wbar = computeAverageWage(finalResult2.wc, finalResult2.wp, finalResult2.ell_c);
    finalResult2.effective_wage = finalResult2.wbar;
    finalResult2.participation_rate = L / workingAgePop;
    finalResult2.Y_converged = false;
    if (useLFP) {
        var mergedFinal2 = Object.assign({}, row, finalResult2);
        finalResult2.ubi_per_person = computeUBIPerPerson(row.year, mergedFinal2, params, workingAgePop);
    }
    return finalResult2;
}

function solveWithEndogenousLabor(row, params, B, workingAgePop, trustParams, maxIter, tol) {
    maxIter = maxIter || 10;
    tol = tol || 1e-3;
    var useLFP = params.wbar0 != null;
    var damp = params.labor_damp || 0.5;

    var L = row.H_data || workingAgePop * 0.6;

    for (var iter = 0; iter < maxIter; iter++) {
        var rowWithL = Object.assign({}, row, { H_cog: L, H_phys: L });
        var result = solveMuOneYear(rowWithL, params, trustParams);
        var wbar = computeAverageWage(result.wc, result.wp, result.ell_c);

        var L_new;
        if (useLFP) {
            var merged = Object.assign({}, row, result);
            var ubi = computeUBIPerPerson(row.year, merged, params, workingAgePop);
            var lfp = Math.max(0, computeLFP(wbar, params.wbar0, ubi, params));
            L_new = Math.min(lfp * workingAgePop, workingAgePop);
        } else {
            L_new = laborSupply(B, wbar, params.psi, workingAgePop);
        }

        if (Math.abs(L_new - L) / Math.max(L, TINY) < tol) {
            L = L_new;
            break;
        }
        L = (1 - damp) * L + damp * L_new;
    }

    // Final solve with converged L
    var finalRow = Object.assign({}, row, { H_cog: L, H_phys: L });
    var finalResult = solveMuOneYear(finalRow, params, trustParams);
    finalResult.L_endogenous = L;
    finalResult.wbar = computeAverageWage(finalResult.wc, finalResult.wp, finalResult.ell_c);
    finalResult.effective_wage = finalResult.wbar;
    finalResult.participation_rate = L / workingAgePop;
    if (useLFP) {
        var mergedFinal = Object.assign({}, row, finalResult);
        finalResult.ubi_per_person = computeUBIPerPerson(row.year, mergedFinal, params, workingAgePop);
    }
    return finalResult;
}
