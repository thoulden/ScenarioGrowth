// Chart Theme, Chart.js defaults, background plugin, and helper functions
// All chart .js files depend on CHART_THEME, makeScales, makeChartConfig, ds

var CHART_THEME = {
    fonts: {
        family: '"Courier New", Courier, monospace',
        sizeSmall: 11,
        size: 12,
        sizeLarge: 14
    },
    colors: {
        cognitive:  '#004080',
        physical:   '#AA6600',
        green:      '#005000',
        red:        '#AA0000',
        darkRed:    '#6B0000',
        purple:     '#663399',
        gray:       '#888888',
        brown:      '#6B3A00',
        teal:       '#006060',
        olive:      '#556B2F',
        black:      '#111111',
        text:       '#333333'
    },
    grid: {
        color: 'rgba(0,0,0,0.07)',
        lineWidth: 0.8,
        borderDash: [5, 5]
    },
    background: '#FFFEF8',
    snapshotColors: ['#111111', '#AA0000', '#AA6600', '#005000'],
    landColors: {
        agricultural: { border: '#005000', bg: 'rgba(0,80,0,0.7)' },
        urban:        { border: '#6B3A00', bg: 'rgba(107,58,0,0.7)' },
        rural:        { border: '#006060', bg: 'rgba(0,96,96,0.7)' },
        commercial:   { border: '#8B008B', bg: 'rgba(139,0,139,0.7)' },
        wilderness:   { border: '#556B2F', bg: 'rgba(85,107,47,0.7)' }
    }
};

// Apply Chart.js global defaults
Chart.defaults.font.family = CHART_THEME.fonts.family;
Chart.defaults.font.size = CHART_THEME.fonts.size;
Chart.defaults.color = CHART_THEME.colors.text;
Chart.defaults.plugins.legend.labels.font = { family: CHART_THEME.fonts.family, size: CHART_THEME.fonts.sizeSmall };
Chart.defaults.plugins.title.font = { family: CHART_THEME.fonts.family, size: CHART_THEME.fonts.sizeLarge, weight: 'bold' };

// Register cream background plugin
Chart.register({
    id: 'creamBackground',
    beforeDraw: function(chart) {
        var ctx = chart.ctx;
        ctx.save();
        ctx.fillStyle = CHART_THEME.background;
        ctx.fillRect(0, 0, chart.width, chart.height);
        ctx.restore();
    }
});

// Format a number in scientific notation: $3.05e13
function formatExpValue(value, prefix, suffix) {
    if (value == null || isNaN(value)) return '';
    var abs = Math.abs(value);
    var sign = value < 0 ? '-' : '';
    if (abs === 0) return prefix + '0' + suffix;
    var exp = Math.floor(Math.log10(abs));
    var mantissa = abs / Math.pow(10, exp);
    // Round mantissa to 3 significant figures
    mantissa = parseFloat(mantissa.toPrecision(3));
    if (mantissa === 10) { mantissa = 1; exp += 1; }
    var mantissaStr = mantissa % 1 === 0 ? mantissa.toString() : mantissa.toPrecision(3);
    return sign + prefix + mantissaStr + 'e' + exp + suffix;
}

// Format a number with K/M/B/T suffixes
// prefix: '$' or '', suffix: '%' or ''
function formatTickValue(value, prefix, suffix) {
    if (value == null || isNaN(value)) return '';
    var abs = Math.abs(value);
    var sign = value < 0 ? '-' : '';
    var s;
    if (abs >= 1e15) {
        s = sign + prefix + (abs / 1e15).toPrecision(3) + 'e15' + suffix;
    } else if (abs >= 1e12) {
        s = sign + prefix + parseFloat((abs / 1e12).toPrecision(3)) + 'T' + suffix;
    } else if (abs >= 1e9) {
        s = sign + prefix + parseFloat((abs / 1e9).toPrecision(3)) + 'B' + suffix;
    } else if (abs >= 1e6) {
        s = sign + prefix + parseFloat((abs / 1e6).toPrecision(3)) + 'M' + suffix;
    } else if (abs >= 1e3) {
        s = sign + prefix + parseFloat((abs / 1e3).toPrecision(3)) + 'K' + suffix;
    } else if (abs >= 1) {
        s = sign + prefix + parseFloat(abs.toPrecision(3)) + suffix;
    } else if (abs > 0) {
        s = sign + prefix + parseFloat(abs.toPrecision(2)) + suffix;
    } else {
        s = prefix + '0' + suffix;
    }
    return s;
}

// Helper: build scales config
// yFormat: '$' for dollar amounts, '%' for percentages, 'count' for large plain numbers
function makeScales(opts) {
    var T = CHART_THEME;
    var xCfg = {
        title: { display: true, text: opts.xTitle || 'Year', font: { family: T.fonts.family, size: T.fonts.size } },
        grid: { color: T.grid.color, lineWidth: T.grid.lineWidth, borderDash: T.grid.borderDash },
        ticks: { font: { family: T.fonts.family, size: T.fonts.sizeSmall } }
    };
    if (opts.xType) xCfg.type = opts.xType;
    if (opts.xMin !== undefined) xCfg.min = opts.xMin;
    if (opts.xMax !== undefined) xCfg.max = opts.xMax;
    if (opts.xStacked) xCfg.stacked = true;

    var yCfg = {
        title: { display: true, text: opts.yTitle || '', font: { family: T.fonts.family, size: T.fonts.size } },
        grid: { color: T.grid.color, lineWidth: T.grid.lineWidth, borderDash: T.grid.borderDash },
        ticks: { font: { family: T.fonts.family, size: T.fonts.sizeSmall } }
    };
    if (opts.yType) yCfg.type = opts.yType;
    if (opts.yMin !== undefined) yCfg.min = opts.yMin;
    if (opts.yMax !== undefined) yCfg.max = opts.yMax;
    if (opts.yStacked) yCfg.stacked = true;
    if (opts.beginAtZero) yCfg.beginAtZero = true;

    // Apply tick formatting based on yFormat
    if (opts.yFormat === '$') {
        yCfg.ticks.callback = function(value) { return formatTickValue(value, '$', ''); };
    } else if (opts.yFormat === '$exp') {
        yCfg.ticks.callback = function(value) { return formatExpValue(value, '$', ''); };
    } else if (opts.yFormat === '%') {
        yCfg.ticks.callback = function(value) { return formatTickValue(value, '', '%'); };
    } else if (opts.yFormat === 'count') {
        yCfg.ticks.callback = function(value) { return formatTickValue(value, '', ''); };
    }
    if (opts.yTickCallback) {
        yCfg.ticks.callback = opts.yTickCallback;
    }

    return { x: xCfg, y: yCfg };
}

// Helper: build full Chart config
function makeChartConfig(datasets, labels, scaleOpts, extraOpts) {
    var cfg = {
        type: 'line',
        data: labels ? { labels: labels, datasets: datasets } : { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: false
                },
                legend: {
                    position: (extraOpts && extraOpts.legendPos) || 'top',
                    display: (extraOpts && extraOpts.legendDisplay !== undefined) ? extraOpts.legendDisplay : true
                }
            },
            scales: makeScales(scaleOpts)
        }
    };
    if (extraOpts && extraOpts.tooltipCallback) {
        cfg.options.plugins.tooltip = { callbacks: { label: extraOpts.tooltipCallback } };
    } else if (scaleOpts && scaleOpts.yFormat) {
        var _fmt = scaleOpts.yFormat;
        cfg.options.plugins.tooltip = { callbacks: { label: function(ctx) {
            var v = ctx.parsed.y;
            if (_fmt === '$') return ctx.dataset.label + ': ' + formatTickValue(v, '$', '');
            if (_fmt === '$exp') return ctx.dataset.label + ': ' + formatExpValue(v, '$', '');
            if (_fmt === '%') return ctx.dataset.label + ': ' + formatTickValue(v, '', '%');
            return ctx.dataset.label + ': ' + formatTickValue(v, '', '');
        }}};
    }
    return cfg;
}

// Helper: build a single dataset descriptor
function ds(label, data, color, opts) {
    opts = opts || {};
    var d = {
        label: label,
        data: data,
        borderColor: color,
        backgroundColor: color,
        borderWidth: opts.borderWidth !== undefined ? opts.borderWidth : 2,
        fill: opts.fill || false,
        tension: opts.tension !== undefined ? opts.tension : 0.1,
        pointRadius: opts.pointRadius !== undefined ? opts.pointRadius : 0
    };
    if (opts.dash) d.borderDash = opts.dash;
    if (opts.fillColor) d.backgroundColor = opts.fillColor;
    if (opts.spanGaps) d.spanGaps = true;
    return d;
}
