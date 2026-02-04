// Chart 14: Physical Land Use Shares by Category (stacked, conditional)
function createLandUseShareChart(results, years) {
    var useShareData = results.map(function(r) {
        if (!r.physical_land) return null;
        var total = 0;
        for (var ci = 0; ci < LAND_CATEGORIES.length; ci++) total += r.physical_land[LAND_CATEGORIES[ci]] || 0;
        var shares = {};
        for (var ci = 0; ci < LAND_CATEGORIES.length; ci++) {
            var cat = LAND_CATEGORIES[ci];
            shares[cat] = total > 0 ? (r.physical_land[cat] || 0) / total : 0;
        }
        return shares;
    });
    var datasets = LAND_CATEGORIES.map(function(cat) {
        return ds(
            LAND_CATEGORY_LABELS[cat],
            useShareData.map(function(s) { return s ? s[cat] * 100 : null; }),
            LAND_COLORS[cat].border,
            { fill: true, fillColor: LAND_COLORS[cat].bg, borderWidth: 1 }
        );
    });
    var ctx = document.getElementById('landUseShareChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yStacked: true, yTitle: '% of total land area', yMin: 0, yMax: 100, yFormat: '%' },
        { title: 'Physical Land Use Shares by Category' }
    ));
}
