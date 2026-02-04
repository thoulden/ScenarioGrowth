// Chart 4: Human labor allocation over time (stacked area)
function createEllChart(results, years) {
    var T = CHART_THEME.colors;
    var H_cog = results.map(function(r) { return r.H_cog_allocated || r.Hc || 0; });
    var H_phys = results.map(function(r) { return r.H_phys_allocated || r.Hp || 0; });
    var H_trust = results.map(function(r) { return r.H_trust || 0; });
    var hasTrust = H_trust.some(function(v) { return v > 0; });

    var datasets = [
        ds('Cognitive Workers', H_cog, T.cognitive, { fill: true, fillColor: T.cognitive + 'AA', borderWidth: 1 }),
        ds('Physical Workers', H_phys, T.physical, { fill: true, fillColor: T.physical + 'AA', borderWidth: 1 })
    ];
    if (hasTrust) {
        datasets.push(ds('Trusted Workers', H_trust, T.darkRed, { fill: true, fillColor: T.darkRed + 'AA', borderWidth: 1 }));
    }
    var ctx = document.getElementById('ellChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { xStacked: true, yStacked: true, yTitle: 'Number of Workers', yMin: 0, yFormat: 'count' },
        { title: 'Human labor allocation over time' }
    ));
}
