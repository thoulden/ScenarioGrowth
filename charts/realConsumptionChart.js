// Chart: Real Consumption Growth — spending deflated by sector price indices
// Shows how much more goods, services, and land people can actually consume,
// accounting for the fact that AI changes relative prices over time.
//
// Method: For each year t,
//   Goods(t)    = Y(t) * (1-landSh) * (1-svcSh) / (P_G(t)/P_G(0))   [deflated by goods price]
//   Services(t) = Y(t) * (1-landSh) * svcSh     / (P_S(t)/P_S(0))   [deflated by services price]
//   Land(t)     = Σ M_i(t) × v_i(0)                                  [hectares at base-year rents]
//
// Each series indexed to its own base year = 1 (so all lines start at 1x).

function createRealConsumptionChart(results, years) {
    var T = CHART_THEME.colors;
    var hasSectors = results[0].service_share != null && results[0].P_S != null;

    // Base-year price indices (for normalization to base = 1)
    var P_S_0 = hasSectors ? (results[0].P_S || 1) : 1;
    var P_G_0 = hasSectors ? (results[0].P_G || 1) : 1;

    // Base-year land rents per hectare (for Laspeyres deflation)
    var hasLand = results[0].rent_per_ha != null && results[0].physical_land != null;
    var v_0 = hasLand ? results[0].rent_per_ha : null;   // { urban: $/ha, rural: $/ha, agricultural: $/ha }

    // Working-age population for per-capita
    var WAP = results.map(function(r) { return r.WorkingAgePop || r.H_cog + r.H_phys || 1; });
    var WAP_0 = WAP[0] || 1;

    // Compute real consumption quantities per working-age person
    var realGoods = results.map(function(r, i) {
        var Y = r.Y_predicted || r.Y || 0;
        var landSh = r.land_exp_share || 0;
        var svcSh = hasSectors ? (r.service_share || 0) : 0.77;
        var P_G = hasSectors ? (r.P_G || 1) : 1;
        // Goods spending deflated by goods price relative to base year
        var goods = Y * (1 - landSh) * (1 - svcSh);
        return (goods / (P_G / P_G_0)) / WAP[i];
    });

    var realServices = results.map(function(r, i) {
        var Y = r.Y_predicted || r.Y || 0;
        var landSh = r.land_exp_share || 0;
        var svcSh = hasSectors ? (r.service_share || 0) : 0.77;
        var P_S = hasSectors ? (r.P_S || 1) : 1;
        var svc = Y * (1 - landSh) * svcSh;
        return (svc / (P_S / P_S_0)) / WAP[i];
    });

    // Real land: value current physical hectares at base-year rents (Laspeyres quantity index)
    // If land model is active: Σ M_i(t) × v_i(0)  for endogenous categories
    // Fallback (no land model): use undeflated land spending (Y * land_exp_share)
    var realLand = results.map(function(r, i) {
        if (hasLand && r.physical_land && v_0) {
            var val = 0;
            for (var cat in r.physical_land) {
                if (v_0[cat] && v_0[cat] > 0) {
                    val += (r.physical_land[cat] || 0) * v_0[cat];
                }
            }
            return val / WAP[i];
        }
        // Fallback: nominal (no deflator available)
        var Y = r.Y_predicted || r.Y || 0;
        var landSh = r.land_exp_share || 0;
        return (Y * landSh) / WAP[i];
    });

    var realTotal = results.map(function(r, i) {
        return realGoods[i] + realServices[i] + realLand[i];
    });

    // Index each series to its own base year = 1 (all start at 1x)
    var gBase = realGoods[0] || 1;
    var sBase = realServices[0] || 1;
    var lBase = realLand[0] || 1;
    var tBase = realTotal[0] || 1;
    var gIdx = realGoods.map(function(v) { return v / gBase; });
    var sIdx = realServices.map(function(v) { return v / sBase; });
    var lIdx = realLand.map(function(v) { return v / lBase; });
    var tIdx = realTotal.map(function(v) { return v / tBase; });

    var datasets = [
        ds('Total', tIdx, T.output, { borderWidth: 3 }),
        ds('Goods', gIdx, T.physical, { borderWidth: 2 }),
        ds('Services', sIdx, T.cognitive, { borderWidth: 2 }),
        ds('Land', lIdx, T.brown, { borderWidth: 2 })
    ];

    var ctx = document.getElementById('realConsumptionChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        {
            yType: 'logarithmic',
            yTitle: 'Per-capita consumption (1 = base year)',
            yFormat: 'count'
        },
        {
            title: 'Real Consumption Growth per Working-Age Person',
            tooltipCallback: function(ctx) {
                var v = ctx.parsed.y;
                return ctx.dataset.label + ': ' + v.toFixed(v < 10 ? 2 : 0) + 'x';
            }
        }
    ));
}
