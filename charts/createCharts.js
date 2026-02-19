// Thin dispatcher: creates/destroys all main charts
function createCharts(results, useLand, showLaborChart, showConsumptionChart, gwpBreakdown) {
    useLand = useLand || false;
    showLaborChart = showLaborChart || false;
    showConsumptionChart = showConsumptionChart || false;
    var years = results.map(function(r) { return r.year; });

    // Destroy all existing chart instances
    [outputChart, gwpBreakdownChart, factorIncomeChart, wagesChart,
     muChart, laborChart, ellChart, laborProductivityChart, rentalCostChart,
     capitalChart, interestChart, techChart, sectorPriceChart,
     landRentChart, landUseShareChart, consumerBudgetChart, landIncomeShareChart,
     realConsumptionChart, wageIncomeChart, ubiTransferChart,
     incomeDistChart, incomeCompChart, netWorthChart, giniChart
    ].forEach(function(c) { if (c) c.destroy(); });

    // 1. Output
    outputChart = createOutputChart(results, years);

    // 2. GWP Breakdown
    var gwpContainer = document.getElementById('gwpBreakdownContainer');
    if (gwpBreakdown) {
        gwpContainer.style.display = '';
        gwpBreakdownChart = createGwpBreakdownChart(gwpBreakdown, years);
    } else {
        gwpContainer.style.display = 'none';
    }

    // 3. Factor Income Shares
    factorIncomeChart = createFactorIncomeChart(results, years, useLand);
    document.getElementById('factorIncomeTitle').textContent = 'Factor Income Shares of Output';

    // 4. Wages
    wagesChart = createWagesChart(results, years);

    // 5. Automated Task Shares
    muChart = createMuChart(results, years);

    // 6. Labor Force and WAP
    var laborChartContainer = document.getElementById('laborChart').closest('.chart-container');
    if (showLaborChart) {
        laborChartContainer.style.display = '';
        laborChart = createLaborChart(results, years);
    } else {
        laborChartContainer.style.display = 'none';
    }

    // 7. Human Labor Allocation
    ellChart = createEllChart(results, years);

    // 8. Labor Productivity
    laborProductivityChart = createLaborProductivityChart(results, years);

    // 9. Factor Marginal Products
    rentalCostChart = createRentalCostChart(results, years);

    // 10. Capital Allocation
    capitalChart = createCapitalChart(results, years);

    // 11. Returns to Capital
    interestChart = createInterestChart(results, years);

    // 12. Capital Efficiency
    techChart = createTechChart(results, years);

    // 13. Two-Sector Decomposition (Goods vs Services)
    var sectorPriceContainer = document.getElementById('sectorPriceContainer');
    if (results[0].P_S != null) {
        sectorPriceContainer.style.display = '';
        sectorPriceChart = createSectorPriceChart(results, years);
    } else {
        sectorPriceContainer.style.display = 'none';
    }

    // 14. Land Rents
    var landRentContainer = document.getElementById('landRentContainer');
    if (useLand && results[0].rent_per_ha) {
        landRentContainer.style.display = '';
        landRentChart = createLandRentChart(results, years);
    } else {
        landRentContainer.style.display = 'none';
    }

    // 14. Land Use
    var landUseShareContainer = document.getElementById('landUseShareContainer');
    if (useLand && results.length > 0 && results[0].physical_land) {
        landUseShareContainer.style.display = '';
        landUseShareChart = createLandUseShareChart(results, years);
    } else {
        landUseShareContainer.style.display = 'none';
    }

    // 15. Consumer Budget
    var consumerBudgetContainer = document.getElementById('consumerBudgetContainer');
    if (useLand && results[0].E_t && results[0].qc) {
        consumerBudgetContainer.style.display = '';
        var cbParams = getParams();
        var cbPctSelect = document.getElementById('consumerBudgetPctSelect');
        var cbPct = cbPctSelect ? cbPctSelect.value : 'mean';
        consumerBudgetChart = createConsumerBudgetChart(results, years, cbParams, cbPct);
        _cachedCBResults = results;
        _cachedCBYears = years;
    } else {
        consumerBudgetContainer.style.display = 'none';
        _cachedCBResults = null;
        _cachedCBYears = null;
    }

    // Household Expenditure Split (Land / Services / Goods) — after sector prices
    landIncomeShareChart = createLandIncomeShareChart(results, years);

    // Real Consumption Growth per Working-Age Person
    var realConsContainer = document.getElementById('realConsumptionContainer');
    if (results[0].P_S != null && results[0].P_G != null) {
        realConsContainer.style.display = '';
        realConsumptionChart = createRealConsumptionChart(results, years);
    } else {
        realConsContainer.style.display = 'none';
    }

    // 17-18. Wage Income & UBI Transfers
    var wageIncomeContainer = document.getElementById('wageIncomeChartContainer');
    if (showConsumptionChart) {
        wageIncomeContainer.style.display = '';
        var params = getParams();
        wageIncomeChart = createWageIncomeChart(results, years);

        var ubiContainer = document.getElementById('ubiTransferChartContainer');
        var showAnyTax = params.tau_R > 0 || params.tau_AI > 0 || params.tau_k > 0;
        if (showAnyTax) {
            ubiContainer.style.display = '';
            ubiTransferChart = createUbiTransferChart(results, years, params);
        } else {
            ubiContainer.style.display = 'none';
        }
    } else {
        wageIncomeContainer.style.display = 'none';
        document.getElementById('ubiTransferChartContainer').style.display = 'none';
    }

    // 19-22. Income Distribution, Composition, Net Worth, Gini
    // (created by separate distribution code, just need destroy handled above)
}
