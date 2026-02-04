// Chart 8: (Implied) Capital efficiency over time
function createTechChart(results, years) {
    var T = CHART_THEME.colors;
    var X_C_raw = results.map(function(r) { return r.AI_cog / r.K_C; });
    var X_P_raw = results.map(function(r) { return r.R_phys / r.K_R; });
    var X_C_0 = X_C_raw[0], X_P_0 = X_P_raw[0];
    var datasets = [
        ds('Output per AI', X_C_raw.map(function(x) { return x / X_C_0; }), T.cognitive),
        ds('Output per Robot', X_P_raw.map(function(x) { return x / X_P_0; }), T.physical)
    ];
    var ctx = document.getElementById('techChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: 'Relative efficiency (base year = 1)' },
        { title: '(Implied) Capital efficiency over time' }
    ));
}
