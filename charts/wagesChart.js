// Chart 1: Wages per human equivalent worker
function createWagesChart(results, years) {
    var T = CHART_THEME.colors;
    var wc = results.map(function(r) { return r.wc; });
    var wp = results.map(function(r) { return r.wp; });
    var qc = results.map(function(r) { return r.qc; });
    var qr = results.map(function(r) { return r.qr; });
    var trust_wage = results.map(function(r) { return r.trust_wage || null; });
    var hasTrust = trust_wage.some(function(v) { return v !== null && v > 0; });

    var datasets = [
        ds('Human Cognitive Wage', wc, T.cognitive),
        ds('AI Cognitive Wage', qc, T.cognitive, { dash: [5, 5] }),
        ds('Human Physical Wage', wp, T.physical),
        ds('Robot Physical Wage', qr, T.physical, { dash: [5, 5] })
    ];
    if (hasTrust) {
        datasets.push(ds('Trusted Labor Wage', trust_wage, T.darkRed, { spanGaps: true }));
    }
    var ctx = document.getElementById('wagesChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: 'Price / wage (per human equivalent)', yMin: 100, yFormat: '$' },
        { title: 'Wages per human equivalent worker' }
    ));
}
