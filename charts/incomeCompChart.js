// Chart 16: Income Composition by Percentile (stacked area, with year toggle)
// Note: this function assigns to the global incomeCompChart variable
function buildIncomeCompChart(distribution, yearIdx, percentiles, results, params) {
    if (incomeCompChart) incomeCompChart.destroy();
    var T = CHART_THEME.colors;
    var d = distribution[yearIdx];
    var r = results ? results[yearIdx] : null;
    var p = params;
    if (!r || !p) return;

    var smooth = computeSmoothDistribution(r, p);
    var fields = ['wage', 'capital', 'ai', 'robot', 'land', 'trust', 'ubi'];

    function compData(field) {
        var arr = smooth.components[field];
        var out = [];
        for (var i = 0; i < arr.length; i++) {
            var total = 0;
            for (var f = 0; f < fields.length; f++) {
                total += Math.max(smooth.components[fields[f]][i].y, 0);
            }
            out.push({ x: arr[i].x, y: total > 0 ? (Math.max(arr[i].y, 0) / total) * 100 : 0 });
        }
        return out;
    }

    var compColors = [
        { label: 'Wage', field: 'wage', color: T.cognitive },
        { label: 'Capital', field: 'capital', color: T.gray },
        { label: 'AI Cognitive', field: 'ai', color: T.purple },
        { label: 'Robot Physical', field: 'robot', color: T.physical },
        { label: 'Land', field: 'land', color: T.brown },
        { label: 'Trusted Labor', field: 'trust', color: T.darkRed },
        { label: 'UBI', field: 'ubi', color: T.green }
    ];

    var datasets = compColors.map(function(c) {
        return ds(c.label, compData(c.field), c.color, { fill: true, fillColor: c.color + 'BB', borderWidth: 1, tension: 0 });
    });

    var ctx = document.getElementById('incomeCompChart').getContext('2d');
    incomeCompChart = new Chart(ctx, makeChartConfig(datasets, null,
        { xType: 'linear', xTitle: 'Percentile', xMin: 5, xMax: 100,
          yStacked: true, yTitle: '% of Individual Income', yMin: 0, yMax: 100, yFormat: '%' },
        { title: 'Income Composition by Percentile (' + d.year + ')',
          tooltipCallback: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1) + '%'; } }
    ));
}
