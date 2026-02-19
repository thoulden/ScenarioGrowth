// Thin dispatcher: creates/destroys all distribution charts (16-19)
function createDistributionCharts(distribution, results, params) {
    var incomeDistContainer = document.getElementById('incomeDistContainer');
    var incomeCompContainer = document.getElementById('incomeCompContainer');
    var netWorthContainer = document.getElementById('netWorthContainer');
    var giniContainer = document.getElementById('giniContainer');

    if (!distribution || distribution.length === 0) {
        incomeDistContainer.style.display = 'none';
        incomeCompContainer.style.display = 'none';
        netWorthContainer.style.display = 'none';
        giniContainer.style.display = 'none';
        _cachedDistribution = null;
        return;
    }

    incomeDistContainer.style.display = '';
    incomeCompContainer.style.display = '';
    netWorthContainer.style.display = '';
    giniContainer.style.display = '';

    var percentiles = distribution[0].households.map(function(h) { return h.percentile; });
    _cachedDistribution = distribution;
    _cachedDistLabels = percentiles;
    _cachedResults = results;
    _cachedParams = params;

    // Pick 4 snapshot years
    var baseYear = distribution[0].year;
    var snapYears = [baseYear, baseYear + 5, baseYear + 10, baseYear + 15];
    var snapIndices = snapYears.map(function(y) {
        for (var i = 0; i < distribution.length; i++) {
            if (distribution[i].year === y) return i;
        }
        return -1;
    }).filter(function(i) { return i >= 0; });

    // Pre-compute smooth distributions
    var smoothDists = {};
    for (var si = 0; si < snapIndices.length; si++) {
        var idx = snapIndices[si];
        smoothDists[idx] = computeSmoothDistribution(results[idx], params);
    }

    // Create 4 distribution charts
    incomeDistChart = createIncomeDistChart(distribution, snapIndices, smoothDists);

    // Income composition with year dropdown
    var yearSelect = document.getElementById('incomeCompYearSelect');
    yearSelect.innerHTML = '';
    var defaultSnapIdx = snapIndices.length - 1;
    for (var si = 0; si < snapIndices.length; si++) {
        var opt = document.createElement('option');
        opt.value = snapIndices[si];
        opt.textContent = String(distribution[snapIndices[si]].year);
        if (si === defaultSnapIdx) opt.selected = true;
        yearSelect.appendChild(opt);
    }
    buildIncomeCompChart(distribution, snapIndices[defaultSnapIdx], percentiles, results, params);

    netWorthChart = createNetWorthChart(distribution, snapIndices, smoothDists);
    giniChart = createGiniChart(distribution);
}
