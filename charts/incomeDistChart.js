// Chart 17: Income Distribution (per Working-Age Person)
function createIncomeDistChart(distribution, snapIndices, smoothDists) {
    var snapColors = CHART_THEME.snapshotColors;
    var datasets = snapIndices.map(function(idx, si) {
        return ds(String(distribution[idx].year), smoothDists[idx].income, snapColors[si], { tension: 0 });
    });
    var ctx = document.getElementById('incomeDistChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, null,
        { xType: 'linear', xTitle: 'Percentile', xMin: 5, xMax: 100,
          yType: 'logarithmic', yTitle: 'Annual Income ($, log scale)', yFormat: '$' },
        { title: 'Income Distribution (per Working-Age Person)' }
    ));
}
