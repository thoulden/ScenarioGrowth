// Production functions, TFP calibration, and predicted output solver

// ========================================
// OUTPUT PREDICTION FUNCTIONS
// ========================================

// Production function: CES or Cobb-Douglas at top level
// When σ_K ≈ 1: Cobb-Douglas Y = A × K_input^α × L_eff^(1-α)
// When σ_K ≠ 1: CES Y = A × cesKL(K_input/K₀, L_eff/L₀, α, σ_K)
// K_input is either K_Y (no K_eff nesting) or K_eff = CES(K_Y, Rp) (with nesting)
function productionFunction(A, K_input, L_eff, alpha, sigma_K, K0_base, Leff0_base) {
    sigma_K = sigma_K || 1.0;
    if (Math.abs(sigma_K - 1.0) < 1e-10) {
        return A * Math.pow(K_input, alpha) * Math.pow(L_eff, 1.0 - alpha);
    }
    // Normalize by base-year values so CES share = α at base year
    const K_norm = (K0_base > 0) ? K_input / K0_base : K_input;
    const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
    return A * cesKL(K_norm, L_norm, alpha, sigma_K);
}

// Full production function with optional trust layer
// Land is consumption-side, not in the production function.
function productionFunctionFull(A, K_input, L_eff, alpha, sigma_K, K0_base, Leff0_base, trustParams) {
    let Y = productionFunction(A, K_input, L_eff, alpha, sigma_K, K0_base, Leff0_base);
    if (trustParams && trustParams.active) {
        Y = ces2(trustParams.X_trust, Y, trustParams.s_trust, trustParams.sigma_trust);
    }
    return Y;
}

// Calibrate TFP (A) from base year data
// CES with normalization: A = Y / cesKL(K/K₀, L_eff/L₀, α, σ_K)
// Cobb-Douglas: A = Y / (K_input^α × L_eff^(1-α))
function calibrateTFP(Y, K_input, L_eff, alpha, sigma_K, K0_base, Leff0_base) {
    sigma_K = sigma_K || 1.0;
    if (Math.abs(sigma_K - 1.0) < 1e-10) {
        return Y / (Math.pow(K_input, alpha) * Math.pow(L_eff, 1.0 - alpha));
    }
    const K_norm = (K0_base > 0) ? K_input / K0_base : K_input;
    const L_norm = (Leff0_base > 0) ? L_eff / Leff0_base : L_eff;
    return Y / cesKL(K_norm, L_norm, alpha, sigma_K);
}

// Normalized CES for K_eff nesting.
// K_eff = CES(K, Rp) — robots substitute for capital.
// Inputs normalized by base-year values to handle scale mismatch (K in dollars, Rp in worker-equivalents).
// Rp_base = base-year Hp (human physical workers), so Rp_norm ≈ 0 initially, → 1 when robots match humans.
// K_eff = K_Y_base * CES(K/K_Y_base, Rp/Rp_base, nu_K, sigma_Keff)
function cesKeff(K_Y, Rp, params) {
    var K_Y_base = params.K_Y_base || K_Y;
    var Rp_base = params.Rp_base || 1.0;
    var K_norm = K_Y / Math.max(K_Y_base, TINY);
    var Rp_norm = Rp / Math.max(Rp_base, TINY);
    var Q_norm = cesTaskAgg(K_norm, Rp_norm, params.nu_K, params.sigma_Keff);
    return Q_norm * K_Y_base;
}

// Compute K_eff from K_Y and Rp_capital when nesting is enabled
// Rp_capital = capital-allocated robots only (= (1-φ)*Rp when using robot split)
// K_eff = normalized CES(K_Y, Rp_capital) or just K_Y when disabled
function computeKeff(K_Y, Rp_capital, params) {
    if (params.enable_K_eff_nest) {
        return cesKeff(K_Y, Rp_capital, params);
    }
    return K_Y;
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

        // Get K_Y and L_eff from equilibrium, compute K_eff
        // Use φ from solver result: Rp_capital = (1-φ) × Rp
        const K_Y = result.K_Y;
        const L_eff = result.L_eff;
        const phi = result.phi || 1.0;
        const Rp = row.R_phys || 0;
        const Rp_capital = (1.0 - phi) * Rp;
        const K_input = computeKeff(K_Y, Rp_capital, params);

        // Use solved X_trust for production function check
        const trustParamsResolved = (trustParams && trustParams.active && result.X_trust > 0)
            ? Object.assign({}, trustParams, { X_trust: result.X_trust }) : trustParams;

        // Compute predicted Y from full production function (CES or CD + trust)
        const Y_pred = productionFunctionFull(A, K_input, L_eff, params.alpha, params.sigma_K, params.K0_base, params.Leff0_base, trustParamsResolved);

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
