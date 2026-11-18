"""第 11 章的七码投影实验。示例：python calibrate_projection.py 1 8"""
import argparse
from math import isclose


def encode_decode(values, step):
    """按照校准好的 step 舍入和饱和，返回整数码及其解码值。"""
    codes = [max(-3, min(3, round(value / step))) for value in values]
    return codes, [code * step for code in codes]


def projection(inputs, weights):
    return sum(x * w for x, w in zip(inputs, weights, strict=True))


def inspect(inputs, transform):
    calibration = (8, 1)
    weights = (1, 8)
    scaled_calibration = [x / d for x, d in zip(calibration, transform, strict=True)]
    scaled_inputs = [x / d for x, d in zip(inputs, transform, strict=True)]
    scaled_weights = [w * d for w, d in zip(weights, transform, strict=True)]
    activation_step = max(map(abs, scaled_calibration)) / 3
    weight_step = max(map(abs, scaled_weights)) / 3
    x_codes, quantized_x = encode_decode(scaled_inputs, activation_step)
    w_codes, quantized_w = encode_decode(scaled_weights, weight_step)
    reference = projection(inputs, weights)
    assert isclose(projection(scaled_inputs, scaled_weights), reference)
    result = projection(quantized_x, quantized_w)
    return reference, result, x_codes, w_codes, activation_step, weight_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=float, help="两个激活分量，默认比较章中两行")
    args = parser.parse_args()
    if args.inputs and len(args.inputs) != 2:
        parser.error("请提供两个激活分量，例如 1 8")
    assert inspect((8, 1), (1, 1))[:2] == (16, 0)
    assert inspect((8, 1), (4, 0.5))[:2] == (16, 16)
    assert inspect((1, 8), (4, 0.5))[:2] == (65, 8)
    for inputs in ([args.inputs] if args.inputs else [(8, 1), (1, 8)]):
        for transform in [(1, 1), (4, 0.5)]:
            ref, predicted, x_codes, w_codes, sx, sw = inspect(inputs, transform)
            print(f"x={tuple(inputs)}, D={transform}, scales=({sx:.4g}, {sw:.4g})")
            print(f"  activation codes={x_codes}, weight codes={w_codes}, "
                  f"quantized dot={predicted:g}, reference={ref:g}")


if __name__ == "__main__":
    main()
