// Chart: GWP Breakdown by Region (stacked area, always sums to 100%)
function createGwpBreakdownChart(gwpBreakdown, years) {
    var T = CHART_THEME.colors;

    var usOutput = gwpBreakdown.us.map(function(r) { return r.Y_predicted || 0; });
    var cnOutput = gwpBreakdown.china.map(function(r) { return r.Y_predicted || 0; });
    var rowOutput = gwpBreakdown.row.map(function(r) { return r.Y_predicted || 0; });

    var usPct = [], cnPct = [], rowPct = [];
    for (var i = 0; i < years.length; i++) {
        var total = usOutput[i] + cnOutput[i] + rowOutput[i];
        usPct.push(total > 0 ? (usOutput[i] / total) * 100 : 0);
        cnPct.push(total > 0 ? (cnOutput[i] / total) * 100 : 0);
        rowPct.push(total > 0 ? (rowOutput[i] / total) * 100 : 0);
    }

    var datasets = [
        ds('US', usPct, T.cognitive, { fill: true, fillColor: T.cognitive + 'CC', borderWidth: 1 }),
        ds('China', cnPct, T.red, { fill: true, fillColor: T.red + 'CC', borderWidth: 1 }),
        ds('Rest of World', rowPct, T.green, { fill: true, fillColor: T.green + 'CC', borderWidth: 1 })
    ];

    var ctx = document.getElementById('gwpBreakdownChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yStacked: true, yTitle: '% of GWP', yMin: 0, yMax: 100, yFormat: '%' },
        { title: 'GWP Breakdown by Region', legendPos: 'bottom' }
    ));
}
