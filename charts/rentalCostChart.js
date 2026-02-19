// Chart 2: Factor marginal products over time (per actual unit)
// These are the value of one additional unit of each factor — NOT the market
// rental price, which depends on marginal cost of production/deployment.
function createRentalCostChart(results, years) {
    var T = CHART_THEME.colors;
    var mp_ai_copy = results.map(function(r) { return r.qc * (r.equivs_per_ai || 1); });
    var mp_robot = results.map(function(r) { return r.qr * (r.equivs_per_robot || 1); });
    var mp_h100e = results.map(function(r) {
        var val = r.qc * (r.equivs_per_ai || 1);
        return val / (r.h100e_per_ai_copy || 1);
    });
    var datasets = [
        ds('AI copy (yearly)', mp_ai_copy, T.green),
        ds('Robot (yearly)', mp_robot, T.red),
        ds('H100e (yearly)', mp_h100e, T.purple)
    ];
    var ctx = document.getElementById('rentalCostChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: 'Annual marginal product (log scale)', yFormat: '$' },
        { title: 'Factor Marginal Products (per actual unit)' }
    ));
}
