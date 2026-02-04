// Chart 19: Gini Coefficients over time
function createGiniChart(distribution) {
    var T = CHART_THEME.colors;
    var giniYears = distribution.map(function(d) { return d.year; });
    var datasets = [
        ds('Income Gini', distribution.map(function(d) { return d.gini_income; }), T.cognitive, { pointRadius: 1 }),
        ds('Wealth Gini', distribution.map(function(d) { return d.gini_wealth; }), T.red, { pointRadius: 1 })
    ];
    var ctx = document.getElementById('giniChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, giniYears,
        { yTitle: 'Gini Coefficient', yMin: 0, yMax: 1 },
        { title: 'Gini Coefficients' }
    ));
}
