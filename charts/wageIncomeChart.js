// Chart 10: Wage Income per person (conditional)
function createWageIncomeChart(results, years) {
    var T = CHART_THEME.colors;
    var datasets = [
        ds('Wage Income', results.map(function(r) { return r.wage_per_worker || 0; }), T.green, { fill: true, fillColor: T.green + '33' })
    ];
    var ctx = document.getElementById('wageIncomeChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: '$ per Person', yMin: 10000, yFormat: '$' },
        { title: 'Wage Income', legendDisplay: false }
    ));
}
