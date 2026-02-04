// Chart: Labor Productivity (Output per human worker, log scale)
function createLaborProductivityChart(results, years) {
    var T = CHART_THEME.colors;
    var productivity = results.map(function(r) {
        var L = r.L_endogenous || r.H_cog || r.H_data || 1;
        var Y = r.Y_predicted || r.Y || 0;
        return Y / Math.max(L, 1);
    });
    var datasets = [
        ds('Output per Worker', productivity, T.cognitive)
    ];
    var ctx = document.getElementById('laborProductivityChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: 'Output per worker (log scale)', yFormat: '$' },
        { title: 'Labor productivity over time' }
    ));
}
