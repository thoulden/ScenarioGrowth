// Chart 2: Rental costs over time (per actual unit)
function createRentalCostChart(results, years) {
    var T = CHART_THEME.colors;
    var rental_ai_copy = results.map(function(r) { return r.qc * (r.equivs_per_ai || 1); });
    var rental_robot = results.map(function(r) { return r.qr * (r.equivs_per_robot || 1); });
    var rental_h100e = results.map(function(r) {
        var cost = r.qc * (r.equivs_per_ai || 1);
        return cost / (r.h100e_per_ai_copy || 1);
    });
    var datasets = [
        ds('AI copy rental (yearly)', rental_ai_copy, T.green),
        ds('Robot rental (yearly)', rental_robot, T.red),
        ds('H100e rental (yearly)', rental_h100e, T.purple)
    ];
    var ctx = document.getElementById('rentalCostChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: 'Annual rental cost (log scale)', yFormat: '$' },
        { title: 'Rental costs over time (per actual unit)' }
    ));
}
