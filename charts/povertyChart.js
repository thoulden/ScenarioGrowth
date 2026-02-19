// Chart: Poverty Rates over time
// Computes headcount poverty rates from the income distribution model.
//
// For each year, walks the 109 percentile buckets (99 × 1% + 10 × 0.1%)
// and sums the population fraction below each poverty line.
//
// Two types of poverty lines:
//
// 1. ABSOLUTE (World Bank, annual 2025 PPP dollars):
//    Extreme poverty:  $2.15/day = $785/year   (international poverty line)
//    Poverty:          $3.65/day = $1,332/year  (lower-middle income line)
//    Upper poverty:    $6.85/day = $2,500/year  (upper-middle income line)
//    These are most relevant for World and China modes.
//
// 2. RELATIVE (fraction of base-year median per-WAP income):
//    Below 50% of median  (OECD relative poverty definition)
//    Below 30% of median  (deep relative poverty)
//    The median is anchored to base year, so as the economy grows,
//    these become absolute lines in real terms — measuring whether
//    the bottom is keeping up with the base-year standard of living.
//
// Note: The model distributes income per working-age person (not per household),
// so these rates are not directly comparable to official household-based poverty
// statistics. The trends over time are the meaningful output.

function createPovertyChart(distribution) {
    var T = CHART_THEME.colors;
    var years = distribution.map(function(d) { return d.year; });

    // Population fractions per bucket (must match distribution.js)
    var pop_fraction = [];
    var h;
    for (h = 0; h < 99; h++) pop_fraction.push(0.01);   // 1% each
    for (h = 0; h < 10; h++) pop_fraction.push(0.001);   // 0.1% each

    // Compute base-year median income (p50 bucket = index 49)
    var baseMedian = distribution[0].households[49].total;

    // Helper: compute headcount rate (% of WAP below threshold)
    function headcount(d, threshold) {
        var below = 0;
        for (var h = 0; h < d.households.length; h++) {
            if (d.households[h].total < threshold) {
                below += pop_fraction[h];
            }
        }
        return below * 100;
    }

    // --- Build region-appropriate line sets ---
    var mode = typeof getRegionMode === 'function' ? getRegionMode() : 'global';

    // Absolute World Bank lines
    var absLines = [
        { label: 'Extreme ($2.15/day)', threshold: 2.15 * 365, color: T.red },
        { label: 'Poverty ($3.65/day)',  threshold: 3.65 * 365, color: T.darkRed },
        { label: 'Upper ($6.85/day)',    threshold: 6.85 * 365, color: T.olive }
    ];

    // Relative lines (anchored to base-year median — these are fixed real thresholds)
    var relLines = [
        { label: '<50% of base-yr median', threshold: 0.50 * baseMedian, color: T.cognitive },
        { label: '<30% of base-yr median', threshold: 0.30 * baseMedian, color: T.purple }
    ];

    var activeLines;
    if (mode === 'us') {
        // US: World Bank lines are irrelevant; use relative lines only
        activeLines = relLines;
    } else if (mode === 'china') {
        // China: World Bank lines + relative
        activeLines = absLines.concat(relLines);
    } else {
        // World: all lines
        activeLines = absLines.concat(relLines);
    }

    // Compute headcount rates for each poverty line over time
    var datasets = activeLines.map(function(line) {
        var rates = distribution.map(function(d) {
            return headcount(d, line.threshold);
        });
        return ds(line.label, rates, line.color, { borderWidth: 2 });
    });

    var ctx = document.getElementById('povertyChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yTitle: 'Population below line (%)', yMin: 0, beginAtZero: true },
        {
            title: 'Poverty Headcount Rates',
            tooltipCallback: function(ctx) {
                var v = ctx.parsed.y;
                return ctx.dataset.label + ': ' + v.toFixed(1) + '%';
            }
        }
    ));
}
