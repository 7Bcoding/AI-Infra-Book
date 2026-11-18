"""Estimate mean time per delivered token from complete speculative-verification traces."""

import argparse


def compare(survivors, round_ms):
    if not survivors or survivors[0] <= 0 or len(round_ms) != len(survivors):
        raise ValueError("provide one positive initial count and one round time per depth")
    if any(value < 0 for value in survivors) or any(b > a for a, b in zip(survivors, survivors[1:])):
        raise ValueError("survivor counts must be nonnegative and nonincreasing")
    if any(not (0 < elapsed < float("inf")) for elapsed in round_ms):
        raise ValueError("round times must be finite and positive")
    delivered = 0
    rows = []
    for depth, (count, elapsed) in enumerate(zip(survivors, round_ms)):
        delivered += count
        mean_tokens = delivered / survivors[0]
        rows.append((depth, count, mean_tokens, elapsed / mean_tokens))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survivors", type=int, nargs="+", default=[1000, 900, 720, 216],
                        help="total rounds followed by rounds surviving each depth")
    parser.add_argument("--round-ms", type=float, nargs="+", default=[8, 9, 11, 13],
                        help="total draft plus verify time at each depth, including zero")
    args = parser.parse_args()
    try:
        rows = compare(args.survivors, args.round_ms)
    except ValueError as exc:
        parser.error(str(exc))
    print("depth   survivors   tokens/round   ms/token")
    for depth, count, mean_tokens, ms_per_token in rows:
        print(f"{depth:5d}   {count:9d}   {mean_tokens:12.3f}   {ms_per_token:8.3f}")
    best = min(rows, key=lambda row: row[3])
    print(f"Lowest predicted time per token at depth {best[0]}")


if __name__ == "__main__":
    main()
