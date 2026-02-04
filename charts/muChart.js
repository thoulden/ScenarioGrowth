// Chart 3: Automatable vs automated task shares
function createMuChart(results, years) {
    var T = CHART_THEME.colors;
    var datasets = [
        ds('Cognitive: automated (eq)', results.map(function(r) { return 1 - r.mu_h_c; }), T.cognitive),
        ds('Cognitive: automatable (frontier)', results.map(function(r) { return 1 - r.mu_h_c_min; }), T.cognitive, { dash: [5, 5] }),
        ds('Physical: automated (eq)', results.map(function(r) { return 1 - r.mu_h_p; }), T.physical),
        ds('Physical: automatable (frontier)', results.map(function(r) { return 1 - r.mu_h_p_min; }), T.physical, { dash: [5, 5] })
    ];
    var ctx = document.getElementById('muChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yTitle: 'Task share', yMin: 0, yMax: 1 },
        { title: 'Automatable vs automated task shares' }
    ));
}
