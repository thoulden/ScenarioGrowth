// Chart 12: Output — Model Prediction vs GATE Benchmark
function createOutputChart(results, years) {
    var T = CHART_THEME.colors;
    var GATE_GWP = [
        1.31e14, 1.28e14, 1.57e14, 2.02e14, 2.59e14, 3.36e14, 4.45e14, 6.17e14,
        9.07e14, 1.43e15, 2.51e15, 5.19e15, 1.4e16, 5.6e16, 2.89e17, 1.45e18
    ];
    var datasets = [
        ds('Output Model Prediction', results.map(function(r) { return r.Y_predicted || r.Y; }), T.cognitive),
        ds((isUSMode() || (typeof isChinaMode === 'function' && isChinaMode())) ? 'GATE GDP' : 'GATE GWP', GATE_GWP.slice(0, years.length), T.green, { dash: [10, 5] })
    ];
    var ctx = document.getElementById('outputChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: 'Output (log scale)', yFormat: '$exp' },
        { title: 'Output: Model Prediction vs GATE Benchmark' }
    ));
}
