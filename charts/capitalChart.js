// Chart 5: Productive capital allocation shares (stacked area)
function createCapitalChart(results, years) {
    var T = CHART_THEME.colors;
    var datasets = [
        ds('Traditional', results.map(function(r) { return r.share_KY * 100; }), T.cognitive, { fill: true, fillColor: T.cognitive + 'BB', borderWidth: 1 }),
        ds('Compute', results.map(function(r) { return r.share_KC * 100; }), T.physical, { fill: true, fillColor: T.physical + 'BB', borderWidth: 1 }),
        ds('Robots', results.map(function(r) { return r.share_KR * 100; }), T.green, { fill: true, fillColor: T.green + 'BB', borderWidth: 1 })
    ];
    var ctx = document.getElementById('capitalChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yStacked: true, yTitle: '% of total productive capital', yMin: 0, yMax: 100, yFormat: '%' },
        { title: 'Productive capital allocation shares over time', legendPos: 'bottom' }
    ));
}
