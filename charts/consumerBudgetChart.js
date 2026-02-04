// Chart: Example Consumer Budget — what a person at a given percentile could afford
// Shows # AI copies, # robots, hectares of urban/rural land on log scale
// All budget splits are illustrative toy parameters (don't affect model prices)
// percentile: 'mean' for average, or a number (10, 50, 90, 99.5) for that percentile
function createConsumerBudgetChart(results, years, params, percentile) {
    var T = CHART_THEME.colors;

    // Read illustrative budget splits from inline chart controls
    var landShareEl = document.getElementById('cb_land_share');
    var aiShareEl = document.getElementById('cb_ai_share');
    var urbanShareEl = document.getElementById('cb_urban_share');
    var landShare = landShareEl ? parseFloat(landShareEl.value) || 0.3 : 0.3;
    var aiShare = aiShareEl ? parseFloat(aiShareEl.value) || 0.5 : 0.5;
    var robotShare = 1.0 - aiShare;
    var urbanLandShare = urbanShareEl ? parseFloat(urbanShareEl.value) || 0.5 : 0.5;
    var ruralLandShare = 1.0 - urbanLandShare;

    // Get per-person consumption for each year
    var perPersonConsumption = results.map(function(r, idx) {
        if (!r.E_t || !r.WorkingAgePop) return null;

        if (percentile === 'mean' || percentile == null) {
            // Average: total expenditure / population
            return r.E_t / r.WorkingAgePop;
        }

        // Percentile-specific: use smooth distribution
        var pct = parseFloat(percentile);
        var smooth = computeSmoothDistribution(r, params);
        if (!smooth || !smooth.income) return r.E_t / r.WorkingAgePop;

        // Find the income at the target percentile by linear interpolation
        var pts = smooth.income;
        for (var i = 0; i < pts.length - 1; i++) {
            if (pts[i].x <= pct && pts[i + 1].x >= pct) {
                var t = (pct - pts[i].x) / (pts[i + 1].x - pts[i].x);
                return pts[i].y + t * (pts[i + 1].y - pts[i].y);
            }
        }
        // Exact match at last point or out of range
        if (pts.length > 0 && Math.abs(pts[pts.length - 1].x - pct) < 0.01) {
            return pts[pts.length - 1].y;
        }
        return r.E_t / r.WorkingAgePop; // fallback
    });

    // Use user-specified land share (illustrative), not model's equilibrium land_exp_share
    var aiCopies = results.map(function(r, idx) {
        var income = perPersonConsumption[idx];
        if (!income || !r.qc || r.qc <= 0) return null;
        var nonLandBudget = income * (1 - landShare);
        return Math.max(1e-6, aiShare * nonLandBudget / r.qc);
    });

    var robots = results.map(function(r, idx) {
        var income = perPersonConsumption[idx];
        if (!income || !r.qr || r.qr <= 0) return null;
        var nonLandBudget = income * (1 - landShare);
        return Math.max(1e-6, robotShare * nonLandBudget / r.qr);
    });

    var urbanHa = results.map(function(r, idx) {
        var income = perPersonConsumption[idx];
        if (!income || !r.rent_per_ha || !r.rent_per_ha.urban || r.rent_per_ha.urban <= 0) return null;
        var landBudget = income * landShare;
        return Math.max(1e-6, urbanLandShare * landBudget / r.rent_per_ha.urban);
    });

    var ruralHa = results.map(function(r, idx) {
        var income = perPersonConsumption[idx];
        if (!income || !r.rent_per_ha || !r.rent_per_ha.rural || r.rent_per_ha.rural <= 0) return null;
        var landBudget = income * landShare;
        return Math.max(1e-6, ruralLandShare * landBudget / r.rent_per_ha.rural);
    });

    var pctLabel = (percentile === 'mean' || percentile == null)
        ? 'Average'
        : (percentile == 50 ? 'Median' : percentile + 'th %ile');

    var datasets = [
        ds('AI Copies', aiCopies, T.purple, { spanGaps: true }),
        ds('Robots', robots, T.physical, { spanGaps: true }),
        ds('Urban Land (ha)', urbanHa, T.brown, { spanGaps: true }),
        ds('Rural Land (ha)', ruralHa, T.teal, { spanGaps: true })
    ];

    var ctx = document.getElementById('consumerBudgetChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: 'Units affordable per person', yFormat: 'count' },
        { title: 'Example Consumer Budget — ' + pctLabel }
    ));
}
