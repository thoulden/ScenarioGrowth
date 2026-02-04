// Chart 11: UBI Transfers per person (conditional)
function createUbiTransferChart(results, years, params) {
    var T = CHART_THEME.colors;
    var wage_per_worker = results.map(function(r) { return r.wage_per_worker || 0; });
    var datasets = [];
    if (params.tau_R > 0) {
        datasets.push(ds('UBI (Robot Tax)', results.map(function(r) { return r.T_robot_per_worker || 0; }), T.physical));
    }
    if (params.tau_AI > 0) {
        datasets.push(ds('UBI (AI Tax)', results.map(function(r) { return r.T_AI_per_worker || 0; }), T.purple));
    }
    if (params.tau_k > 0) {
        datasets.push(ds('UBI (Capital Tax)', results.map(function(r) { return r.T_capital_per_worker || 0; }), T.gray));
    }
    datasets.push(ds('Wage Income', wage_per_worker, T.green));
    var ctx = document.getElementById('ubiTransferChart').getContext('2d');
    return new Chart(ctx, makeChartConfig(datasets, years,
        { yType: 'logarithmic', yTitle: '$ per Person', yMin: 1, yFormat: '$' },
        { title: 'UBI Transfers per Person' }
    ));
}
