// Chart 15: Expenditure Split — Land Rent vs Non-Land Consumption (stacked, % of E_t)
function createLandIncomeShareChart(results, years) {
    var T = CHART_THEME.colors;
    var datasets = [
        ds('Land Rent',
            results.map(function(r) { return r.land_exp_share != null ? r.land_exp_share * 100 : null; }),
            T.brown,
            { fill: true, fillColor: T.brown + 'CC', borderWidth: 1 }
        ),
        ds('Non-Land Consumption',
            results.map(function(r) { return r.land_exp_share != null ? (1 - r.land_exp_share) * 100 : null; }),
            T.cognitive,
            { fill: true, fillColor: T.cognitive + 'CC', borderWidth: 1 }
        )
    ];
    var ctx = document.getElementById('landIncomeShareChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yStacked: true, yTitle: '% of Household Expenditure', yMin: 0, yMax: 100, yFormat: '%' },
        { title: 'Household Expenditure Split: Land Rent vs Non-Land Consumption' }
    ));
}
