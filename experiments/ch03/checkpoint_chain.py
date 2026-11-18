"""Differentiate a scalar eight-layer chain using full storage or two segments.

The loss is x8 itself. Each layer is w*x + x*x/20, so its backward pass
needs only its input, its weight, and the gradient arriving from the next layer.
"""

from math import isclose


def layer(x, w):
    return w * x + x * x / 20


def chain(x, weights):
    for weight in weights:
        x = layer(x, weight)
    return x


def gradients(x, weights, stride):
    starts = {}
    for i, weight in enumerate(weights):
        if i % stride == 0:
            starts[i] = x
        x = layer(x, weight)

    result = [0.0] * len(weights)
    upstream = 1.0
    peak = len(starts)
    extra_forwards = 0
    for start in sorted(starts, reverse=True):
        end = min(start + stride, len(weights))
        inputs = [starts.pop(start)]
        for i in range(start, end - 1):
            inputs.append(layer(inputs[-1], weights[i]))
            extra_forwards += 1
        peak = max(peak, len(starts) + len(inputs))
        for i in reversed(range(start, end)):
            inp = inputs.pop()
            result[i] = upstream * inp
            upstream *= weights[i] + inp / 10
    return x, result, peak, extra_forwards


if __name__ == "__main__":
    x0 = 0.5
    weights = [0.75 + i * 0.03 for i in range(8)]
    baseline = gradients(x0, weights, 1)
    segmented = gradients(x0, weights, 4)
    assert baseline[:2] == segmented[:2]
    assert baseline[2:] == (8, 0)
    assert segmented[2:] == (5, 6)
    for i in range(len(weights)):
        plus, minus = weights.copy(), weights.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        finite_difference = (chain(x0, plus) - chain(x0, minus)) / (2e-6)
        assert isclose(finite_difference, baseline[1][i], rel_tol=1e-7)
    print("Stored input arrays at peak; additional layer forwards")
    print("All inputs:", baseline[2:])
    print("Two segments:", segmented[2:])
    print("All eight weight gradients agree with central differences.")
