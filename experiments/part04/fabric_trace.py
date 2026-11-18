"""Two full-duplex DP pairs, with one directional payload per group and rail.

Numbers are teaching traces. An exchange may start once both ranks are ready.
Concurrent exchanges sharing a rail divide its payload bandwidth equally.
All times are milliseconds since a shared origin. This program needs only Python.
"""
from argparse import ArgumentParser
from fractions import Fraction


MIB = 2**20
RATE_B_PER_MS = 40_000_000
PAIRS = ((0, 2), (1, 3))


def transfer(payload, ready, rails):
    """Integrate each independent rail at release and completion events."""
    finished = {}
    for rail in set(rails):
        releases = sorted((ready[g], g) for g in range(len(ready)) if rails[g] == rail)
        active = {}
        time = Fraction(0)
        while active or releases:
            if not active:
                time = releases[0][0]
            while releases and releases[0][0] == time:
                _, group = releases.pop(0)
                active[group] = payload
            rate = Fraction(RATE_B_PER_MS, len(active))
            interval = min(active.values()) / rate
            if releases:
                interval = min(interval, releases[0][0] - time)
            time += interval
            for group in list(active):
                active[group] -= rate * interval
                if active[group] == 0:
                    finished[group] = time
                    del active[group]
    return finished


def trace(title, size, ranks, rails):
    ready = [max(ranks[r] for r in pair) for pair in PAIRS]
    done = transfer(size * MIB, ready, rails)
    print(title)
    print("rank  group  ready_ms  group_ready_ms  done_ms")
    for group, pair in enumerate(PAIRS):
        for r in pair:
            print(f"{r:4}  {group:5}  {float(ranks[r]):8.3f}  "
                  f"{float(ready[group]):14.3f}  {float(done[group]):7.3f}")
    loads = [rails.count(i) * size for i in (0, 1)]
    print(f"TX MiB on rail 0/1: {loads[0]} / {loads[1]}; last completion: {float(max(done.values())):.3f} ms")
    print()
    return max(done.values())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-mib", type=int, default=32)
    parser.add_argument("--rails", nargs=2, type=int, choices=(0, 1), metavar=("GROUP0", "GROUP1"))
    parser.add_argument("--ready-ms", nargs=4, type=Fraction, metavar=("R0", "R1", "R2", "R3"))
    args = parser.parse_args()
    if args.bucket_mib <= 0 or any(t < 0 for t in args.ready_ms or ()):
        parser.error("Bucket size must be positive; ready times must be nonnegative.")
    if args.rails is not None or args.ready_ms is not None:
        trace("Custom replay", args.bucket_mib, args.ready_ms or [0]*4, args.rails or [0, 1])
        return
    solo = Fraction(args.bucket_mib * MIB, RATE_B_PER_MS)
    baseline = trace("Independent rails", args.bucket_mib, [0]*4, [0, 1])
    collided = trace("Shared rail", args.bucket_mib, [0]*4, [0, 0])
    straggler = trace("One late rank", args.bucket_mib, [0, 0, 0, solo], [0, 1])
    assert baseline == solo and collided == straggler == 2 * solo


if __name__ == "__main__":
    main()
