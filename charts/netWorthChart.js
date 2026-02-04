// Chart 18: Net Worth Distribution (per Working-Age Person)
function createNetWorthChart(distribution, snapIndices, smoothDists) {
    var snapColors = CHART_THEME.snapshotColors;
    var datasets = snapIndices.map(function(idx, si) {
        return ds(String(distribution[idx].year), smoothDists[idx].networth, snapColors[si], { tension: 0 });
    });
    var ctx = document.getElementById('netWorthChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, null,
        { xType: 'linear', xTitle: 'Percentile', xMin: 5, xMax: 100,
          yType: 'logarithmic', yTitle: 'Net Worth ($, log scale)', yFormat: '$' },
        { title: 'Net Worth Distribution (per Working-Age Person)' }
    ));
}
