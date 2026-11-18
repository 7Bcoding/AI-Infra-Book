"""Count a query-major causal attention scan and check its masked, tiled softmax.

The I/O model counts block requests, with no cross-block cache reuse.
Input/output use two bytes per element. The correctness test uses short rows
with integer scores, independent of the length requested for counting.
"""
from argparse import ArgumentParser
from decimal import Decimal, localcontext
from math import exp, isclose


def traffic(length, tile, width):
    kv_rows = scheduled = 0
    for start in range(0, length, tile):
        end = min(start + tile, length)
        kv_rows += end
        scheduled += (end - start) * end
    valid = length * (length + 1) // 2
    kv_bytes = 2 * kv_rows * width * 2
    query_and_output_bytes = 2 * length * width * 2
    return kv_rows, scheduled - valid, kv_bytes + query_and_output_bytes


def causal_row(scores, values, visible, tile):
    """Merge each nonempty tile into a running shifted numerator/denominator."""
    old_max = float("-inf")
    mass = numerator = 0.0
    for start in range(0, visible, tile):
        end = min(start + tile, visible)
        new_max = max(old_max, max(scores[start:end]))
        factor = exp(old_max - new_max)
        mass *= factor
        numerator *= factor
        for j in range(start, end):
            weight = exp(scores[j] - new_max)
            mass += weight
            numerator += weight * values[j]
        old_max = new_max
    return numerator / mass


def reference(scores, values, visible):
    with localcontext() as ctx:
        ctx.prec = 60
        weights = [Decimal(score).exp() for score in scores[:visible]]
        return float(sum(w * v for w, v in zip(weights, values)) / sum(weights))


def check_rows(length, tile):
    values = [3 * j - 7 for j in range(length)]
    for i in range(length):
        scores = [((i + 2) * (j + 5)) % 17 - 8 for j in range(length)]
        expected = reference(scores, values, i + 1)
        for offset in (-1000, 0, 1000):
            shifted = [s + offset for s in scores]
            actual = causal_row(shifted, values, i + 1, tile)
            assert isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)

    scores = [1000, 1000, 1000]
    values = [2, 4, 8]
    got = [causal_row(scores, values, i, tile) for i in (1, 2, 3)]
    assert got == [2, 3, 14 / 3]
    wrong = causal_row(scores, [2, 4, 0], 3, tile)
    assert wrong == 2 and wrong != got[1]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--tiles", type=int, nargs="+", default=[64, 128])
    args = parser.parse_args()
    if min(args.length, args.width, *args.tiles) < 1:
        parser.error("length, width and tile sizes must be positive")
    # When length is not divisible, count actual rows and columns in the edge tile.
    print("tile   K rows  diagonal masked pairs  Q/K/V/O bytes   logical MiB")
    for tile in args.tiles:
        rows, masked, total = traffic(args.length, tile, args.width)
        print(f"{tile:4d} {rows:8d} {masked:22d} {total:13d} {total / 2**20:13.3f}")
    for tile in (1, 2, 4, *args.tiles):
        check_rows(min(args.length, 19), tile)
    print("Masked prefixes and shifts of -1000 and +1000 match the decimal reference.")


if __name__ == "__main__":
    main()
