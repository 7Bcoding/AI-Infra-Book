"""Sweep physical placements for the two-expert inference example in chapter 17."""
import argparse
from math import ceil


def options(hot, other, row_us, remote_us):
    for moved in range(hot + 1):
        first, second = hot - moved, other + moved
        compute_us = max(first, second) * row_us
        # Only the exposed transfer is charged. This example assumes linear, serial cost.
        yield compute_us + moved * remote_us, moved, first, second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hot", type=int, default=60)
    parser.add_argument("--other", type=int, default=36)
    parser.add_argument("--remote-us", type=float, default=0,
                        help="extra exposed round-trip transfer cost per migrated row")
    args = parser.parse_args()
    if min(args.hot, args.other, args.remote_us) < 0:
        parser.error("choose nonnegative workloads and transfer cost")
    plan = list(options(args.hot, args.other, 10, args.remote_us))
    best = min(plan)
    for index in sorted({0, args.hot // 2, best[1]}):
        total, x, first, second = plan[index]
        print(f"move {x:3} E rows: device rows ({first}, {second}), exposed time {total:g} us")
    saving = plan[0][0] - best[0]
    migration_us = 96 / (24 * 1024) * 1e6
    if saving > 0:
        print(f"best move {best[1]}, break-even at batch {ceil(migration_us / saving)}")
    else:
        print("replication does not reduce exposed time in this window")
    assert min(options(60, 36, 10, 0)) == (480, 12, 48, 48)
    assert min(options(60, 36, 10, 5)) == (540, 12, 48, 48)
    assert min(options(60, 36, 10, 15))[1] == 0


if __name__ == "__main__":
    main()
