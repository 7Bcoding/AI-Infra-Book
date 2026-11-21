"""Replay committed snapshot generations for the two failure domains in chapter 23."""

import argparse
from fractions import Fraction


def latest_generation(failure_at, interval, commit_delay):
    """Generation zero is a durable initial state; subsequent snapshots may still be in flight."""
    if failure_at < commit_delay:
        return 0
    return int((failure_at - commit_delay) // interval) * interval


def recover(failure_at, domain, memory_delay):
    persistent = latest_generation(failure_at, 600, 0)
    if domain == "rack":
        return "storage", persistent
    memory = latest_generation(failure_at, 60, memory_delay)
    if memory > persistent:
        return "host RAM", memory
    return "storage", persistent


def mean_rollback(domain, memory_delay):
    # One midpoint per second integrates these piecewise-linear ages exactly for integer delays.
    # The second 600-second cycle also includes prior generations for failures near its start.
    ages = []
    for second in range(600, 1200):
        instant = Fraction(2 * second + 1, 2)
        _, generation = recover(instant, domain, memory_delay)
        ages.append(instant - generation)
    return sum(ages) / len(ages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-at", type=int, default=665, help="seconds since step zero")
    parser.add_argument("--memory-delay", type=int, default=0, help="seconds to commit a RAM generation")
    args = parser.parse_args()
    if min(args.failure_at, args.memory_delay) < 0:
        parser.error("times must be nonnegative")
    assert mean_rollback("host", 0) == 30
    assert mean_rollback("rack", 0) == 300
    assert recover(665, "host", 10) == ("storage", 600)
    for domain in ("host", "rack"):
        tier, generation = recover(args.failure_at, domain, args.memory_delay)
        print(f"{domain}: at {args.failure_at} s restore {tier} at {generation} s, "
              f"replay {args.failure_at - generation} s")
    average = (9 * mean_rollback("host", args.memory_delay)
               + mean_rollback("rack", args.memory_delay)) / 10
    print(f"Mean rollback for the 90:10 mix: {float(average):.2f} s")


if __name__ == "__main__":
    main()
