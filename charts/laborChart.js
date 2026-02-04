// Chart 9: Labor force and working age population (conditional)
function createLaborChart(results, years) {
    var T = CHART_THEME.colors;
    var L_trust = results.map(function(r) { return r.H_trust || 0; });
    var hasTrust = L_trust.some(function(v) { return v > 0; });
    var datasets = [
        ds('Working Age Population', results.map(function(r) { return r.WorkingAgePop; }), T.gray),
        ds('Implied Labor Force', results.map(function(r) { return r.L_endogenous || r.H_cog; }), T.cognitive)
    ];
    if (hasTrust) {
        datasets.push(ds('Trusted Workers', L_trust, T.darkRed));
    }
    var ctx = document.getElementById('laborChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yTitle: 'Population', beginAtZero: true, yFormat: 'count' },
        { title: 'Labor force and working age population over time' }
    ));
}
