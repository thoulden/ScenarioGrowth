// Chart: Two-Sector Decomposition (Goods vs Services)
function createSectorPriceChart(results, years) {
    var T = CHART_THEME.colors;
    var pGoods = results.map(function(r) { return r.P_G; });
    var pServices = results.map(function(r) { return r.P_S; });
    var pRatio = results.map(function(r) { return r.price_ratio; });
    var sShare = results.map(function(r) { return r.service_share || 0; });

    var datasets = [
        ds('Goods Price (P_G)', pGoods, T.physical),
        ds('Services Price (P_S)', pServices, T.cognitive),
        ds('P_S / P_G Ratio', pRatio, T.purple, { dash: [5, 5] }),
        ds('Service Nominal Share', sShare, T.green, { dash: [3, 3] })
    ];
    var ctx = document.getElementById('sectorPriceChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yTitle: 'Index / Share', yFormat: '' },
        { title: 'Goods vs Services: Prices & Service Share' }
    ));
}
