// Chart: Household Expenditure Split — Land, Services, Goods (stacked, % of total)
function createLandIncomeShareChart(results, years) {
    var T = CHART_THEME.colors;
    var hasSectors = results[0].service_share != null;

    var datasets = [
        ds('Land Rent',
            results.map(function(r) { return r.land_exp_share != null ? r.land_exp_share * 100 : 0; }),
            T.brown,
            { fill: true, fillColor: T.brown + 'CC', borderWidth: 1 }
        ),
        ds('Services',
            results.map(function(r) {
                var land = r.land_exp_share || 0;
                var svc = hasSectors ? (r.service_share || 0) : 0.77;
                return (1 - land) * svc * 100;
            }),
            T.cognitive,
            { fill: true, fillColor: T.cognitive + 'CC', borderWidth: 1 }
        ),
        ds('Goods',
            results.map(function(r) {
                var land = r.land_exp_share || 0;
                var svc = hasSectors ? (r.service_share || 0) : 0.77;
                return (1 - land) * (1 - svc) * 100;
            }),
            T.physical,
            { fill: true, fillColor: T.physical + 'CC', borderWidth: 1 }
        )
    ];
    var ctx = document.getElementById('landIncomeShareChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yStacked: true, yTitle: '% of Household Expenditure', yMin: 0, yMax: 100, yFormat: '%' },
        { title: 'Household Expenditure Split' }
    ));
}
