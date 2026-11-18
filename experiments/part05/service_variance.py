"""Compare two service-time distributions on the same Poisson arrival trace (chapter 19)."""
import argparse
import math
import random
from statistics import fmean


def replay(times, durations, warmup):
    free_at = 0.0
    waits = []
    for index, (arrival, duration) in enumerate(zip(times, durations)):
        start = max(arrival, free_at)
        if index >= warmup:
            waits.append(start - arrival)
        free_at = start + duration
    ordered = sorted(waits)
    return fmean(waits), ordered[math.ceil(0.99 * len(ordered)) - 1]


def expected_wait(rate_per_ms, second_moment):
    utilization = rate_per_ms * 10
    if utilization >= 1:
        return "no finite steady-state mean"
    return f"{rate_per_ms * second_moment / (2 * (1 - utilization)):.3f} ms"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=80, help="arrivals per second")
    parser.add_argument("--requests", type=int, default=200_000, help="measured arrivals")
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0 or args.requests < 1:
        parser.error("rate must be finite and positive; requests must be positive")
    warmup = min(10_000, args.requests)
    arrivals, classes = random.Random(args.seed), random.Random(args.seed + 1)
    rate = args.rate / 1000
    instant = 0.0
    times, mixed = [], []
    for _ in range(warmup + args.requests):
        instant += arrivals.expovariate(rate)
        times.append(instant)
        mixed.append(82 if classes.random() < 0.1 else 2)
    assert replay([0, 2, 3], [4, 1, 2], 0) == (4 / 3, 2)
    for name, service, second_moment in (("constant", [10] * len(times), 100),
                                          ("mixed", mixed, 676)):
        avg, p99 = replay(times, service, warmup)
        print(f"{name:8} mean wait {avg:8.3f} ms | p99 {p99:8.3f} ms | "
              f"analytic mean {expected_wait(rate, second_moment)}")


if __name__ == "__main__":
    main()
