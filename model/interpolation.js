// Interpolation functions for scenario widget
// Supports linear, exponential, and sigmoid (logistic) interpolation between anchor points

/**
 * Linear interpolation between anchor points
 * @param {Array} anchors - [{year: number, value: number}, ...]
 * @param {Array} targetYears - [2025, 2026, ..., 2040]
 * @returns {Array} - Interpolated values for each target year
 */
function interpolateLinear(anchors, targetYears) {
    if (!anchors || anchors.length === 0) return targetYears.map(function() { return 0; });
    if (anchors.length === 1) return targetYears.map(function() { return anchors[0].value; });

    // Sort anchors by year
    anchors = anchors.slice().sort(function(a, b) { return a.year - b.year; });

    return targetYears.map(function(year) {
        // Clamp to bounds
        if (year <= anchors[0].year) return anchors[0].value;
        if (year >= anchors[anchors.length - 1].year) return anchors[anchors.length - 1].value;

        // Find surrounding anchors
        for (var i = 0; i < anchors.length - 1; i++) {
            if (year >= anchors[i].year && year <= anchors[i + 1].year) {
                var t = (year - anchors[i].year) / (anchors[i + 1].year - anchors[i].year);
                return anchors[i].value + t * (anchors[i + 1].value - anchors[i].value);
            }
        }
        return anchors[anchors.length - 1].value;
    });
}

/**
 * Exponential interpolation (constant growth rate between anchors)
 * @param {Array} anchors - [{year: number, value: number}, ...]
 * @param {Array} targetYears - [2025, 2026, ..., 2040]
 * @returns {Array} - Interpolated values for each target year
 */
function interpolateExponential(anchors, targetYears) {
    if (!anchors || anchors.length === 0) return targetYears.map(function() { return 0; });
    if (anchors.length === 1) return targetYears.map(function() { return anchors[0].value; });

    anchors = anchors.slice().sort(function(a, b) { return a.year - b.year; });

    return targetYears.map(function(year) {
        if (year <= anchors[0].year) return anchors[0].value;
        if (year >= anchors[anchors.length - 1].year) return anchors[anchors.length - 1].value;

        for (var i = 0; i < anchors.length - 1; i++) {
            if (year >= anchors[i].year && year <= anchors[i + 1].year) {
                var v0 = anchors[i].value;
                var v1 = anchors[i + 1].value;
                var y0 = anchors[i].year;
                var y1 = anchors[i + 1].year;

                // Handle zero/negative values gracefully - fall back to linear
                if (v0 <= 0 || v1 <= 0) {
                    var t = (year - y0) / (y1 - y0);
                    return v0 + t * (v1 - v0);
                }

                // Exponential: v(t) = v0 * (v1/v0)^((year-y0)/(y1-y0))
                var t = (year - y0) / (y1 - y0);
                return v0 * Math.pow(v1 / v0, t);
            }
        }
        return anchors[anchors.length - 1].value;
    });
}

/**
 * Sigmoid (logistic) interpolation between anchor points
 * S-curve transition useful for technology adoption curves
 * @param {Array} anchors - [{year: number, value: number}, ...]
 * @param {Array} targetYears - [2025, 2026, ..., 2040]
 * @param {number} midpointYear - Year at which transition is 50% complete
 * @param {number} steepness - Controls sharpness of transition (k parameter, default 1.0)
 * @returns {Array} - Interpolated values for each target year
 */
function interpolateSigmoid(anchors, targetYears, midpointYear, steepness) {
    if (!anchors || anchors.length === 0) return targetYears.map(function() { return 0; });
    if (anchors.length === 1) return targetYears.map(function() { return anchors[0].value; });

    anchors = anchors.slice().sort(function(a, b) { return a.year - b.year; });
    steepness = steepness || 1.0;

    return targetYears.map(function(year) {
        if (year <= anchors[0].year) return anchors[0].value;
        if (year >= anchors[anchors.length - 1].year) return anchors[anchors.length - 1].value;

        for (var i = 0; i < anchors.length - 1; i++) {
            if (year >= anchors[i].year && year <= anchors[i + 1].year) {
                var v0 = anchors[i].value;
                var v1 = anchors[i + 1].value;
                var y0 = anchors[i].year;
                var y1 = anchors[i + 1].year;

                // Use segment midpoint if no midpointYear specified
                var mid = midpointYear;
                if (mid === undefined || mid === null) {
                    mid = (y0 + y1) / 2;
                }

                // Logistic function: 1 / (1 + exp(-k*(year - mid)))
                // We need to scale k so the transition happens within the segment
                var segmentWidth = y1 - y0;
                var k = steepness * (4 / segmentWidth); // Scale steepness to segment

                var sigmoid = 1 / (1 + Math.exp(-k * (year - mid)));

                // Scale to [v0, v1]
                return v0 + (v1 - v0) * sigmoid;
            }
        }
        return anchors[anchors.length - 1].value;
    });
}

/**
 * Master interpolation dispatcher
 * @param {Array} anchors - Anchor points [{year, value}, ...]
 * @param {Array} targetYears - Years to interpolate
 * @param {string} method - 'linear', 'exponential', 'sigmoid'
 * @param {Object} options - Additional options (midpointYear, steepness for sigmoid)
 * @returns {Array} - Interpolated values
 */
function interpolatePath(anchors, targetYears, method, options) {
    options = options || {};
    switch (method) {
        case 'exponential':
            return interpolateExponential(anchors, targetYears);
        case 'sigmoid':
            return interpolateSigmoid(anchors, targetYears, options.midpointYear, options.steepness);
        case 'linear':
        default:
            return interpolateLinear(anchors, targetYears);
    }
}
