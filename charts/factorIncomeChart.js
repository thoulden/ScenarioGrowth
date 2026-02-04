// Chart 7: Factor income shares over time (stacked area)
// Land is no longer a production factor — it's consumption-side
function createFactorIncomeChart(results, years, useLand) {
    var T = CHART_THEME.colors;
    var datasets = [
        ds('Productive Capital', results.map(function(r) { return r.capital_share * 100; }), T.gray, { fill: true, fillColor: T.gray + 'CC', borderWidth: 1 }),
        ds('Human Cognitive', results.map(function(r) { return r.human_cog_share * 100; }), T.cognitive, { fill: true, fillColor: T.cognitive + 'CC', borderWidth: 1 }),
        ds('Human Physical', results.map(function(r) { return r.human_phys_share * 100; }), T.green, { fill: true, fillColor: T.green + 'CC', borderWidth: 1 }),
        ds('AI Cognitive Labor', results.map(function(r) { return r.ai_share * 100; }), T.purple, { fill: true, fillColor: T.purple + 'CC', borderWidth: 1 }),
        ds('Robot Physical Labor', results.map(function(r) { return r.robot_share * 100; }), T.physical, { fill: true, fillColor: T.physical + 'CC', borderWidth: 1 })
    ];
    var trust_share = results.map(function(r) { return (r.trust_income_share || 0) * 100; });
    if (trust_share.some(function(v) { return v > 1e-4; })) {
        datasets.push(ds('Trusted Labor', trust_share, T.darkRed, { fill: true, fillColor: T.darkRed + 'CC', borderWidth: 1 }));
    }
    var ctx = document.getElementById('factorIncomeChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yStacked: true, yTitle: '% of ' + ((isUSMode() || (typeof isChinaMode === 'function' && isChinaMode())) ? 'GDP' : 'GWP'), yMin: 0, yMax: 100, yFormat: '%' },
        { title: 'Factor income shares over time', legendPos: 'bottom' }
    ));
}
