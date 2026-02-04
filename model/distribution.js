// Income distribution: household initialization, Lorenz curves, smooth distributions, Gini

// Interpolate Lorenz curve from empirical CDF breakpoints
// cdfBreaks: [L1, L25, L50, L75, L90, L99, L999]
// Piecewise-linear Lorenz interpolation from 9 knot points
function lorenzInterp(p, cdfBreaks) {
    var knots_p = [0, 0.01, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 1.0];
    var knots_L = [0, cdfBreaks[0], cdfBreaks[1], cdfBreaks[2], cdfBreaks[3], cdfBreaks[4], cdfBreaks[5], cdfBreaks[6], 1.0];
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    for (var i = 0; i < knots_p.length - 1; i++) {
        if (p <= knots_p[i + 1]) {
            var t = (p - knots_p[i]) / (knots_p[i + 1] - knots_p[i]);
            return knots_L[i] + t * (knots_L[i + 1] - knots_L[i]);
        }
    }
    return 1;
}

// Generate ownership shares from empirical CDF breakpoints for 109 households
// cdfBreaks: [L(0.50), L(0.90), L(0.99)]
function ownershipFromCDF(cdfBreaks, n) {
    n = n || N_HOUSEHOLDS;
    var shares = [];
    for (var h = 0; h < n; h++) {
        var p_lo = PERCENTILE_BOUNDARIES[h];
        var p_hi = PERCENTILE_BOUNDARIES[h + 1];
        var L_lo = lorenzInterp(p_lo, cdfBreaks);
        var L_hi = lorenzInterp(p_hi, cdfBreaks);
        shares.push(Math.max(L_hi - L_lo, 0));
    }
    return shares;
}

// Generate wage/skill distribution from CDF breakpoints
// For wages, the CDF gives income shares. We convert to per-person skill levels.
// Approach: the slope of the Lorenz curve at percentile p gives the relative wage.
// skill(p) = dL/dp at percentile midpoint, normalized so mean = 1.
function wageDistFromCDF(cdfBreaks, n) {
    n = n || N_HOUSEHOLDS;
    var skills = [];
    var sumWeighted = 0;
    for (var h = 0; h < n; h++) {
        var p_lo = PERCENTILE_BOUNDARIES[h];
        var p_hi = PERCENTILE_BOUNDARIES[h + 1];
        var dp = p_hi - p_lo;
        var L_lo = lorenzInterp(p_lo, cdfBreaks);
        var L_hi = lorenzInterp(p_hi, cdfBreaks);
        // Slope = dL/dp = relative income per unit of population
        var slope = (L_hi - L_lo) / dp;
        skills.push(slope);
        sumWeighted += slope * dp; // weighted by population fraction
    }
    // sumWeighted should be ~1.0 (integral of dL/dp over [0,1])
    // Normalize so population-weighted mean = 1
    for (var h = 0; h < n; h++) skills[h] /= sumWeighted;
    return skills;
}

// Build 109 households with ownership shares and wage skills
function initializeHouseholds(params) {
    var n = N_HOUSEHOLDS;
    var own_K = ownershipFromCDF(params.cdf_capital, n);
    var own_AI = ownershipFromCDF(params.cdf_ai, n);
    var own_R = ownershipFromCDF(params.cdf_robot, n);
    var own_Land = ownershipFromCDF(params.cdf_land, n);
    var wageSkills = wageDistFromCDF(params.cdf_wage, n);

    // Build percentile labels: 1-99, then 99.1, 99.2, ..., 100
    var labels = [];
    for (var i = 1; i <= 99; i++) labels.push(i);
    for (var i = 1; i <= 10; i++) labels.push(99 + i / 10);

    var households = [];
    for (var h = 0; h < n; h++) {
        households.push({
            percentile: labels[h],
            skill: wageSkills[h],
            own_K: own_K[h],
            own_AI: own_AI[h],
            own_R: own_R[h],
            own_Land: own_Land[h]
        });
    }
    return households;
}

// Monotone PCHIP interpolator (shape-preserving cubic Hermite)
// Returns object with evalAt(x) method for smooth interpolation
function makePchip(xs, ys) {
    var n = xs.length;
    if (n < 2) return { evalAt: function() { return 0; } };
    var h = [], d = [];
    for (var i = 0; i < n - 1; i++) {
        h.push(xs[i + 1] - xs[i]);
        if (h[i] <= 0) h[i] = 1e-12;
        d.push((ys[i + 1] - ys[i]) / h[i]);
    }
    var m = new Array(n);
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for (var i = 1; i < n - 1; i++) {
        if (d[i - 1] === 0 || d[i] === 0 || (d[i - 1] > 0) !== (d[i] > 0)) {
            m[i] = 0;
        } else {
            var w1 = 2 * h[i] + h[i - 1];
            var w2 = h[i] + 2 * h[i - 1];
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
        }
    }
    function findInterval(x) {
        if (x <= xs[0]) return 0;
        if (x >= xs[n - 1]) return n - 2;
        var lo = 0, hi = n - 2;
        while (lo <= hi) {
            var mid = (lo + hi) >> 1;
            if (xs[mid] <= x && x <= xs[mid + 1]) return mid;
            if (x < xs[mid]) hi = mid - 1; else lo = mid + 1;
        }
        return lo;
    }
    return {
        evalAt: function(x) {
            if (x <= xs[0]) return ys[0];
            if (x >= xs[n - 1]) return ys[n - 1];
            var i = findInterval(x);
            var t = (x - xs[i]) / h[i];
            var t2 = t * t, t3 = t2 * t;
            var h00 = 2 * t3 - 3 * t2 + 1;
            var h10 = t3 - 2 * t2 + t;
            var h01 = -2 * t3 + 3 * t2;
            var h11 = t3 - t2;
            return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1];
        },
        evalDeriv: function(x) {
            var i = findInterval(x);
            if (x <= xs[0]) return m[0];
            if (x >= xs[n - 1]) return m[n - 1];
            var t = (x - xs[i]) / h[i];
            var t2 = t * t;
            var dh00 = 6 * t2 - 6 * t;
            var dh10 = 3 * t2 - 4 * t + 1;
            var dh01 = -6 * t2 + 6 * t;
            var dh11 = 3 * t2 - 2 * t;
            var dydt = dh00 * ys[i] + dh10 * h[i] * m[i] + dh01 * ys[i + 1] + dh11 * h[i] * m[i + 1];
            return dydt / h[i];
        }
    };
}

// Build a smooth PCHIP Lorenz curve from CDF breakpoints (cached)
var _lorenzCache = {};
function getLorenzPchip(cdfBreaks) {
    var key = cdfBreaks.join(',');
    if (!_lorenzCache[key]) {
        var knots_p = [0, 0.01, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999, 1.0];
        var knots_L = [0, cdfBreaks[0], cdfBreaks[1], cdfBreaks[2], cdfBreaks[3], cdfBreaks[4], cdfBreaks[5], cdfBreaks[6], 1.0];
        _lorenzCache[key] = makePchip(knots_p, knots_L);
    }
    return _lorenzCache[key];
}

// Build a PCHIP interpolator in log-space from income multiplier anchors.
// For zero/near-zero multipliers, clamp to a small floor so log works.
// Returns a function: pct (0-100) -> multiplier at that percentile.
function buildMultiplierCurve(mult) {
    var pcts = mult.pcts;
    var vals = mult.vals;
    var FLOOR = 1e-6;
    var logVals = vals.map(function(v) { return Math.log(Math.max(v, FLOOR)); });
    var pchip = makePchip(pcts, logVals);
    return function(pct) {
        return Math.max(0, Math.exp(pchip.evalAt(pct)));
    };
}

// Compute smooth income/net-worth distribution at dense percentiles
// Uses income multiplier curves (PCHIP in log-space) -- guaranteed smooth.
// Returns { income: [{x,y}...], networth: [{x,y}...], components: {wage:[{x,y}...], ...} }
function computeSmoothDistribution(r, params) {
    // Build multiplier curves from params
    var mw = buildMultiplierCurve(params.mult_wage);
    var mk = buildMultiplierCurve(params.mult_capital);
    var ma = buildMultiplierCurve(params.mult_ai);
    var mr = buildMultiplierCurve(params.mult_robot);
    var ml = buildMultiplierCurve(params.mult_land);

    // Total factor incomes
    var total_wage = r.wc * (r.H_cog_allocated || r.Hc) + r.wp * (r.H_phys_allocated || r.Hp);
    var total_cap = r.r * r.K_Y;
    var total_ai = r.qc * r.AI_cog;
    var total_robot = r.qr * r.R_phys;
    var total_land = r.R_total || 0;
    var total_trust = r.trust_income || 0;

    // Use working-age population as denominator (per-person, not per-household)
    var N_pop = r.WorkingAgePop || 270e6;

    // Taxes and UBI (per working-age person -- UBI goes to everyone equally)
    var d_tau_k = params.dist_tau_k || 0, d_tau_AI = params.dist_tau_AI || 0, d_tau_R = params.dist_tau_R || 0;
    var d_ubi_start = params.dist_ubi_start_year || 9999;
    var ubi = 0;
    var net_cap, net_ai, net_robot;
    if (r.year >= d_ubi_start) {
        net_cap = total_cap * (1 - d_tau_k);
        net_ai = total_ai * (1 - d_tau_AI);
        net_robot = total_robot * (1 - d_tau_R);
        // UBI: tax revenue distributed to all working-age persons equally
        var isUS = typeof isUSMode === 'function' && isUSMode();
        var ubi_share = isUS ? (params.dist_ubi_share_us || 0) : (params.dist_ubi_share_world || 0);
        var T_total = d_tau_k * (r.r || 0) * (r.K_Y || 0) +
                      d_tau_AI * (r.qc || 0) * (r.AI_cog || 0) +
                      d_tau_R * (r.qr || 0) * (r.R_phys || 0);
        ubi = T_total * ubi_share / N_pop;
    } else { net_cap = total_cap; net_ai = total_ai; net_robot = total_robot; }

    // Asset values for net worth
    var K_total = r.K || 0, K_AI = r.K_C || 0, K_R = r.K_R || 0;
    var land_cap_rate = params.land_cap_rate || 0.04;
    var land_val = (r.R_total || 0) / land_cap_rate;

    // Mean per-person values (multiplier x mean = income at percentile)
    var mean_wage = total_wage / N_pop;
    var mean_trust = total_trust / N_pop;
    var mean_cap = net_cap / N_pop;
    var mean_ai = net_ai / N_pop;
    var mean_robot = net_robot / N_pop;
    var mean_land = total_land / N_pop;
    var mean_K = K_total / N_pop;
    var mean_KAI = K_AI / N_pop;
    var mean_KR = K_R / N_pop;
    var mean_landval = land_val / N_pop;

    var income = [], networth = [];
    var comp_wage = [], comp_cap = [], comp_ai = [], comp_robot = [], comp_land = [];
    var comp_trust = [], comp_ubi = [];

    // Dense sample from p5 to p99.9
    for (var pct = 5; pct <= 99.9 + 1e-9; pct += 0.5) {
        // Multipliers at this percentile
        var fw = mw(pct);
        var fk = mk(pct);
        var fa = ma(pct);
        var fr = mr(pct);
        var fl = ml(pct);

        // Per-person income = multiplier x mean
        var w_h = fw * mean_wage;
        var trust_h = fw * mean_trust;  // trust distributed like wages
        var k_h = fk * mean_cap;
        var a_h = fa * mean_ai;
        var r_h = fr * mean_robot;
        var l_h = fl * mean_land;
        var tot = w_h + trust_h + k_h + a_h + r_h + l_h + ubi;

        income.push({ x: pct, y: tot });
        comp_wage.push({ x: pct, y: w_h });
        comp_cap.push({ x: pct, y: k_h });
        comp_ai.push({ x: pct, y: a_h });
        comp_robot.push({ x: pct, y: r_h });
        comp_land.push({ x: pct, y: l_h });
        comp_trust.push({ x: pct, y: trust_h });
        comp_ubi.push({ x: pct, y: ubi });

        // Net worth: use capital multiplier for K, ai for K_AI, robot for K_R, land for land
        var nw = fk * mean_K + fa * mean_KAI + fr * mean_KR + fl * mean_landval;
        networth.push({ x: pct, y: nw });
    }

    return {
        income: income, networth: networth,
        components: { wage: comp_wage, capital: comp_cap, ai: comp_ai,
                      robot: comp_robot, land: comp_land, trust: comp_trust, ubi: comp_ubi }
    };
}

// Weighted Gini coefficient: values[i] = per-person value, weights[i] = population fraction
// Uses the standard formula: G = (Sigma_i Sigma_j w_i w_j |y_i - y_j|) / (2 mu)
// Optimized: sort by value, then use cumulative sums
function computeWeightedGini(values, weights) {
    var n = values.length;
    // Build sorted index array
    var idx = [];
    for (var i = 0; i < n; i++) idx.push(i);
    idx.sort(function(a, b) { return values[a] - values[b]; });

    // Compute weighted mean
    var totalWeight = 0, weightedSum = 0;
    for (var i = 0; i < n; i++) {
        totalWeight += weights[i];
        weightedSum += values[i] * weights[i];
    }
    var mean = weightedSum / Math.max(totalWeight, TINY);
    if (mean <= 0) return 0;

    // Cumulative weight approach: G = 1 - 2 * Sigma(w_i * S_i) / weightedSum
    // where S_i = cumulative income share up to and including i
    var cumWeight = 0, cumIncome = 0, sum = 0;
    for (var k = 0; k < n; k++) {
        var i = idx[k];
        var w = weights[i] / totalWeight;
        var y = values[i] * weights[i] / weightedSum;
        cumWeight += w;
        cumIncome += y;
        sum += w * (cumIncome - y / 2); // trapezoidal
    }
    return Math.max(0, Math.min(1, 1 - 2 * sum));
}

// Compute income distribution for all years
// All incomes and net worths are PER WORKING-AGE PERSON (not per-household or aggregate)
// Each of the bottom 99 buckets represents 1% of pop; top 10 each represent 0.1%
function computeIncomeDistribution(results, households, params) {
    var n = households.length;
    var distribution = [];

    // Population weight per bucket: how many real people each bucket represents
    // Bottom 99: each = 1% of pop. Top 10: each = 0.1% of pop.
    // We want per-person income, so divide bucket aggregates by (pop_weight x working_age_pop)
    // pop_fraction[h]: fraction of total population in bucket h
    var pop_fraction = [];
    for (var h = 0; h < 99; h++) pop_fraction.push(0.01);  // 1% each
    for (var h = 0; h < 10; h++) pop_fraction.push(0.001); // 0.1% each

    // Base-year median wage (individual skill x avg wage at base year)
    // Median person is at percentile 50 -> index 49
    var baseResult = results[0];
    var L_base = baseResult.L_endogenous || baseResult.H_cog || 1;
    var baseAvgWage = (baseResult.wc * (baseResult.H_cog_allocated || baseResult.Hc) +
                       baseResult.wp * (baseResult.H_phys_allocated || baseResult.Hp)) / Math.max(L_base, 1);
    var median_wage_2025 = households[49].skill * baseAvgWage;

    for (var t = 0; t < results.length; t++) {
        var r = results[t];
        var year = r.year;

        // Total factor incomes
        var total_wage_income = r.wc * (r.H_cog_allocated || r.Hc) + r.wp * (r.H_phys_allocated || r.Hp);
        var total_capital_income = r.r * r.K_Y;
        var total_ai_income = r.qc * r.AI_cog;
        var total_robot_income = r.qr * r.R_phys;
        var total_land_income = r.R_total || 0;
        var total_trust_income = r.trust_income || 0;

        // Tax revenue and UBI (uses shared helper)
        var d_tau_k = params.dist_tau_k || 0;
        var d_tau_AI = params.dist_tau_AI || 0;
        var d_tau_R = params.dist_tau_R || 0;
        var d_ubi_start = params.dist_ubi_start_year || 9999;
        var workingAgePop = r.WorkingAgePop || 270e6;
        var ubi_per_person = computeUBIPerPerson(year, r, params, workingAgePop);

        // Use the participation rate already computed by the solver (includes smoothing)
        var part_rate = r.participation_rate || 0.62;

        // Determine which households work: top (part_rate x n) by skill
        var sorted_indices = [];
        for (var h = 0; h < n; h++) sorted_indices.push(h);
        sorted_indices.sort(function(a, b) { return households[b].skill - households[a].skill; });
        var n_working = Math.round(part_rate * n);
        var working = new Array(n);
        for (var h = 0; h < n; h++) working[h] = false;
        for (var w = 0; w < Math.min(n_working, n); w++) working[sorted_indices[w]] = true;

        // Sum of skill x pop_fraction for working households (weighted by population they represent)
        var total_working_skill_weight = 0;
        for (var h = 0; h < n; h++) {
            if (working[h]) total_working_skill_weight += households[h].skill * pop_fraction[h];
        }

        // Net factor incomes (after taxes that fund UBI)
        var net_wage_income = total_wage_income; // wages not taxed
        var net_capital_income, net_ai_income, net_robot_income;
        if (year >= d_ubi_start) {
            net_capital_income = total_capital_income * (1 - d_tau_k);
            net_ai_income = total_ai_income * (1 - d_tau_AI);
            net_robot_income = total_robot_income * (1 - d_tau_R);
        } else {
            net_capital_income = total_capital_income;
            net_ai_income = total_ai_income;
            net_robot_income = total_robot_income;
        }

        // Asset values for net worth
        var K_total = r.K || 0;
        var K_AI_value = r.K_C || 0;
        var K_R_value = r.K_R || 0;
        var d_land_cap_rate = params.land_cap_rate || 0.04;
        var land_value = (r.R_total || 0) / d_land_cap_rate;

        var yearHouseholds = [];
        for (var h = 0; h < n; h++) {
            var hh = households[h];
            var pf = pop_fraction[h];
            var n_real = pf * workingAgePop; // real people in this bucket

            // Wage: bucket gets (skill x pf / total_working_skill_weight) share of total wages
            // Then divide by n_real to get per-person
            var wage_bucket = working[h] ? (hh.skill * pf / Math.max(total_working_skill_weight, TINY)) * net_wage_income : 0;
            var wage_h = wage_bucket / n_real;

            // Asset income: ownership share gives bucket's total, divide by n_real
            var cap_h = (hh.own_K * net_capital_income) / n_real;
            var ai_h = (hh.own_AI * net_ai_income) / n_real;
            var robot_h = (hh.own_R * net_robot_income) / n_real;
            var land_h = (hh.own_Land * total_land_income) / n_real;
            // Trusted labor income: distributed like wages (proportional to skill among workers)
            var trust_bucket = working[h] ? (hh.skill * pf / Math.max(total_working_skill_weight, TINY)) * total_trust_income : 0;
            var trust_h = trust_bucket / n_real;
            var ubi_h = ubi_per_person; // already per person
            var total_h = wage_h + cap_h + ai_h + robot_h + land_h + trust_h + ubi_h;

            var networth_h = (hh.own_K * K_total + hh.own_AI * K_AI_value +
                              hh.own_R * K_R_value + hh.own_Land * land_value) / n_real;

            yearHouseholds.push({
                percentile: hh.percentile,
                wage: wage_h,
                capital: cap_h,
                ai: ai_h,
                robot: robot_h,
                land: land_h,
                trust: trust_h,
                ubi: ubi_h,
                total: total_h,
                networth: networth_h
            });
        }

        // Compute Gini coefficients for this year
        var incomeValues = [], wealthValues = [];
        for (var h = 0; h < n; h++) {
            incomeValues.push(Math.max(yearHouseholds[h].total, 0));
            wealthValues.push(Math.max(yearHouseholds[h].networth, 0));
        }
        var gini_income = computeWeightedGini(incomeValues, pop_fraction);
        var gini_wealth = computeWeightedGini(wealthValues, pop_fraction);

        distribution.push({
            year: year,
            households: yearHouseholds,
            participation_rate: part_rate,
            ubi_per_person: ubi_per_person,
            gini_income: gini_income,
            gini_wealth: gini_wealth
        });
    }

    return distribution;
}
