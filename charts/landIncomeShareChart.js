// Chart: Household Expenditure Growth — Land, Services, Goods (log scale, indexed to base year)
function createLandIncomeShareChart(results, years) {
    var T = CHART_THEME.colors;
    var hasSectors = results[0].service_share != null;

    // Compute absolute spending levels for each category
    var landSpend = results.map(function(r) {
        var Y = r.Y || 0;
        var landShare = r.land_exp_share != null ? r.land_exp_share : 0;
        return Y * landShare;
    });
    var svcSpend = results.map(function(r) {
        var Y = r.Y || 0;
        var landShare = r.land_exp_share || 0;
        var svc = hasSectors ? (r.service_share || 0) : 0.77;
        return Y * (1 - landShare) * svc;
    });
    var goodsSpend = results.map(function(r) {
        var Y = r.Y || 0;
        var landShare = r.land_exp_share || 0;
        var svc = hasSectors ? (r.service_share || 0) : 0.77;
        return Y * (1 - landShare) * (1 - svc);
    });

    // Base-year total spending = normalizer (index so that 1 = total spending in year 0)
    var base = (landSpend[0] + svcSpend[0] + goodsSpend[0]) || 1;

    var landIdx = landSpend.map(function(v) { return v / base; });
    var svcIdx = svcSpend.map(function(v) { return v / base; });
    var goodsIdx = goodsSpend.map(function(v) { return v / base; });

    var datasets = [
        ds('Land Rent', landIdx, T.brown, { borderWidth: 2.5 }),
        ds('Services', svcIdx, T.cognitive, { borderWidth: 2.5 }),
        ds('Goods', goodsIdx, T.physical, { borderWidth: 2.5 })
    ];

    var ctx = document.getElementById('landIncomeShareChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        {
            yType: 'logarithmic',
            yTitle: 'Spending (1 = total base-year output)',
            yFormat: 'count'
        },
        {
            title: 'Household Expenditure Growth',
            tooltipCallback: function(ctx) {
                var v = ctx.parsed.y;
                return ctx.dataset.label + ': ' + v.toFixed(v < 10 ? 2 : 0) + 'x';
            }
        }
    ));
}
