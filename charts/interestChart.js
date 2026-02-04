// Chart 6: Implied return to productive capital
// Always displays as multiplier (Nx) — value is in %, so 800% → 8x
function formatReturn(value) {
    if (value == null || isNaN(value)) return '';
    var x = value / 100;
    if (Math.abs(x) >= 10) return parseFloat(x.toPrecision(3)) + 'x';
    if (Math.abs(x) >= 1) return parseFloat(x.toPrecision(2)) + 'x';
    return parseFloat(x.toPrecision(2)) + 'x';
}

function createInterestChart(results, years) {
    var T = CHART_THEME.colors;
    var delta_K = parseFloat(document.getElementById('delta_K').value) || 0.05;
    var tau_k = parseFloat(document.getElementById('tau_k').value) || 0;
    var datasets = [
        ds('r (gross rental rate)', results.map(function(r) { return r.r * 100; }), T.green),
        ds('(1−τ_k)r − δ_K (net return)', results.map(function(r) { return ((1 - tau_k) * r.r - delta_K) * 100; }), T.cognitive)
    ];
    var ctx = document.getElementById('interestChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: 'Rate (per year)', yTickCallback: formatReturn },
        { title: 'Implied return to productive capital over time',
          tooltipCallback: function(ctx) {
              return ctx.dataset.label + ': ' + formatReturn(ctx.parsed.y);
          }
        }
    ));
}
