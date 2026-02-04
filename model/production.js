// Production functions, TFP calibration, and predicted output solver

// ========================================
// OUTPUT PREDICTION FUNCTIONS
// ========================================

// Production function: CES core Y_hat = A × cesTaskAgg(K̃, L̃, α, σ_K)
// where K̃ = K/K₀, L̃ = L_eff/L_eff₀ (normalized by base-year values)
// When σ_K ≈ 1, reduces to Cobb-Douglas Y_hat = A × K_Y^α × L_eff^(1-α)
function productionFunction(A, K_Y, L_eff, alpha, sigma_K, K0_base, Leff0_base) {
    sigma_K = sigma_K || 1.0;
    if (Math.abs(sigma_K - 1.0) < 1e-10) {
        return A * Math.pow(K_Y, alpha) * Math.pow(L_eff, 1.0 - alpha);
    }
    // Normalize by base-year values so CES share = α at base year
    const K_norm = (K0_base > 0) ? K_Y / K0_base : K_Y;
    const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
    return A * cesKL(K_norm, L_norm, alpha, sigma_K);
}

// Full production function with optional trust layer
// Land is consumption-side, not in the production function.
function productionFunctionFull(A, K_Y, L_eff, alpha, sigma_K, K0_base, Leff0_base, trustParams) {
    let Y = productionFunction(A, K_Y, L_eff, alpha, sigma_K, K0_base, Leff0_base);
    if (trustParams && trustParams.active) {
        Y = ces2(trustParams.X_trust, Y, trustParams.s_trust, trustParams.sigma_trust);
    }
    return Y;
}

// Calibrate TFP (A) from base year data
// CES with normalization: A = Y / cesTaskAgg(K/K₀, L_eff/L_eff₀, α, σ_K)
// At base year: K/K₀ = 1, L_eff/L_eff₀ = 1, so A = Y / CES(1,1) = Y
// Cobb-Douglas: A = Y / (K_Y^α × L_eff^(1-α))
function calibrateTFP(Y, K_Y, L_eff, alpha, sigma_K, K0_base, Leff0_base) {
    sigma_K = sigma_K || 1.0;
    if (Math.abs(sigma_K - 1.0) < 1e-10) {
        return Y / (Math.pow(K_Y, alpha) * Math.pow(L_eff, 1.0 - alpha));
    }
    const K_norm = (K0_base > 0) ? K_Y / K0_base : K_Y;
    const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
    return Y / cesKL(K_norm, L_norm, alpha, sigma_K);
}

// Solve for predicted output in one year using fixed-point iteration
// Returns the equilibrium Y where production function is satisfied
function solveYearWithPredictedOutput(row, params, A, Y_guess, trustParams, maxIter, tol) {
    maxIter = maxIter || 10;
    tol = tol || 1e-3;
    let Y = Y_guess;

    for (let iter = 0; iter < maxIter; iter++) {
        // Create row with current Y guess
        const rowWithY = { ...row, Y: Y };

        // Solve equilibrium given Y (H_trust solved endogenously inside)
        const result = solveMuOneYear(rowWithY, params, trustParams);

        // Get K_Y and L_eff from equilibrium
        const K_Y = result.K_Y;
        const L_eff = result.L_eff;

        // Use solved X_trust for production function check
        const trustParamsResolved = (trustParams && trustParams.active && result.X_trust > 0)
            ? Object.assign({}, trustParams, { X_trust: result.X_trust }) : trustParams;

        // Compute predicted Y from full production function (CES + trust)
        const Y_pred = productionFunctionFull(A, K_Y, L_eff, params.alpha, params.sigma_K, params.K0_base, params.Leff0_base, trustParamsResolved);

        // Check convergence
        const relError = Math.abs(Y_pred - Y) / Math.max(Y, TINY);
        if (relError < tol) {
            // Converged - return full result with predicted Y
            const finalRow = { ...row, Y: Y_pred };
            const finalResult = solveMuOneYear(finalRow, params, trustParams);
            finalResult.Y_predicted = Y_pred;
            finalResult.Y_forecast = row.Y;  // Original Y from spreadsheet
            return finalResult;
        }

        // Update Y with damping for stability
        Y = 0.5 * Y + 0.5 * Y_pred;
    }

    // Return last result even if not fully converged
    const finalRow = { ...row, Y: Y };
    const finalResult = solveMuOneYear(finalRow, params, trustParams);
    finalResult.Y_predicted = Y;
    finalResult.Y_forecast = row.Y;
    finalResult.Y_converged = false;
    return finalResult;
}
