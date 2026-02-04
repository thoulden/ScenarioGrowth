// Chart 13: Land Rent per Hectare by Category (endogenous types only)
function createLandRentChart(results, years) {
    var datasets = LAND_ENDOGENOUS_CATEGORIES.map(function(cat) {
        return ds(
            LAND_CATEGORY_LABELS[cat] + ' ($/ha/yr)',
            results.map(function(r) { return r.rent_per_ha ? r.rent_per_ha[cat] : null; }),
            LAND_COLORS[cat].border
        );
    });
    var ctx = document.getElementById('landRentChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: '$/ha/year (log scale)', yFormat: '$' },
        { title: 'Land Rent per Hectare by Category' }
    ));
}
