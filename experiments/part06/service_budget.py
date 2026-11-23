"""Offline capacity and service replay, using the published Splitwise conversation lengths.

Source and attribution are in the adjacent JSON. A 100-ns trace tick is exact.
Each deliberately simple worker serializes per-request processing. No batching or
overlap is modeled. These rates are inputs, not Azure or hardware measurements.
"""
import argparse
from bisect import bisect_right
from dataclasses import dataclass
import heapq
import json
from math import ceil
from pathlib import Path

TICKS = 10_000_000
GIB = 2**30
WEIGHT_BYTES = 15_231_233_024
KV_BYTES = 28 * 2 * 4 * 128 * 2


def parameters():
    width, kv_width, mlp, vocab = 3584, 512, 18944, 152064
    attention = 2 * width**2 + 2 * width * kv_width
    bias_and_norm = width + 2 * kv_width + 2 * width
    return 28 * (attention + 3 * width * mlp + bias_and_norm) + 2 * vocab * width + width


def percentile(values, fraction):
    return sorted(values)[ceil(len(values) * fraction)-1]


def capacity(rows, gib, page_tokens=16, output_limit=1024):
    budget = int((gib - 3) * GIB) - WEIGHT_BYTES
    pages = max(0, budget // (KV_BYTES * page_tokens))
    reservations = [ceil((prompt + output_limit) / page_tokens) for _, prompt, _ in rows]
    return pages, max(reservations), pages // max(reservations)


@dataclass
class Replay:
    arrivals: list[float]
    starts: list[float]
    finishes: list[float]
    first_latencies: list[float]
    end_latencies: list[float]

    def waiting(self, timestamp):
        # FCFS gives nondecreasing start times. Process all ties at timestamp.
        return bisect_right(self.arrivals, timestamp) - bisect_right(self.starts, timestamp)

    def window_metrics(self, workers):
        begin, stop = self.arrivals[0], self.arrivals[-1]
        capacity = workers * (stop - begin)
        if capacity <= 0:
            raise ValueError("Need a positive arrival span to normalize worker time")
        total = sum(f - s for s, f in zip(self.starts, self.finishes))
        in_window = sum(max(0., min(f, stop) - max(s, begin))
                        for s, f in zip(self.starts, self.finishes))
        return total / capacity, in_window / capacity, max(self.finishes) - stop

    def deadlines(self):
        first_ok = sum(t <= 1. for t in self.first_latencies)
        end_ok = sum(t <= 4. for t in self.end_latencies)
        both = sum(f <= 1. and e <= 4. for f, e in zip(self.first_latencies, self.end_latencies))
        return first_ok, end_ok, both


def replay(rows, workers, prefill_rate, decode_rate):
    free_at = [0.] * workers
    result = Replay([], [], [], [], [])
    for tick, prompt, out in rows:
        arrival = tick / TICKS
        start = max(heapq.heappop(free_at), arrival)
        first = start + prompt / prefill_rate
        finish = first + (out - 1) / decode_rate
        heapq.heappush(free_at, finish)
        result.arrivals.append(arrival)
        result.starts.append(start)
        result.finishes.append(finish)
        result.first_latencies.append(first - arrival)
        result.end_latencies.append(finish - arrival)
    return result


def half_window_counts(rows, result, prefill_rate):
    """Group by arrival time after a continuous replay, retaining cross-window work."""
    # Keep an arrival exactly at the midpoint in the second window, without rounding ticks.
    boundary = (rows[0][0] + rows[-1][0] + 1) // 2
    cut = bisect_right([row[0] for row in rows], boundary - 1)
    windows = []
    for start, stop in ((0, cut), (cut, len(rows))):
        passed = sum(first <= 1.0 and end <= 4.0 for first, end in
                     zip(result.first_latencies[start:stop], result.end_latencies[start:stop]))
        long_prompts = sum(prompt > prefill_rate for _, prompt, _ in rows[start:stop])
        windows.append((stop - start, passed, long_prompts))
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--prefill", type=float, default=4000)
    parser.add_argument("--decode", type=float, default=100)
    parser.add_argument("--split-windows", action="store_true",
                        help="Report each half of the arrival span with the queue kept intact")
    args = parser.parse_args()
    if min(args.workers) < 1 or min(args.prefill, args.decode) <= 0:
        parser.error("use positive worker counts and per-second service rates")
    path = Path(__file__).with_name("conversation_lengths.json")
    rows = json.loads(path.read_bytes())["requests"]
    assert parameters() * 2 == WEIGHT_BYTES
    assert len(rows) == 19366 and all(a[0] <= b[0] for a,b in zip(rows, rows[1:]))
    assert all(0 < o <= 1024 and p > 0 for _,p,o in rows)
    duration = rows[-1][0] / TICKS
    prompts = sum(p for _, p, _ in rows)
    outputs = sum(o for _, _, o in rows)
    print(f"span {duration:.7f}s; arrivals {len(rows)}; prompt {prompts}; output {outputs}")
    ceiling = sum(p / args.prefill <= 1. and p / args.prefill + (o-1) / args.decode <= 4.
                  for _, p, o in rows)
    print(f"with zero queue, at most {ceiling} arrivals meet both deadlines")
    for gib in (24, 48):
        pages, largest, fit = capacity(rows, gib)
        print(f"{gib} GiB: {pages} pages; largest reservation {largest}; {fit} such requests")
    for count in args.workers:
        result = replay(rows, count, args.prefill, args.decode)
        demand, busy, drain = result.window_metrics(count)
        first_ok, end_ok, both = result.deadlines()
        print(f"{count}: work_to_window_capacity={demand:.4f} window_busy_fraction={busy:.4f}"
              f" drain_s={drain:.3f} waiting_at_last_arrival={result.waiting(result.arrivals[-1])}"
              f" ttft_p95={percentile(result.first_latencies, .95):.3f}"
              f" e2e_p95={percentile(result.end_latencies, .95):.3f}"
              f" first_ok={first_ok} end_ok={end_ok} jointly_ok={both}")
        if args.split_windows:
            for i, (total, passed, oversized) in enumerate(half_window_counts(rows, result, args.prefill), 1):
                print(f"  half {i}: arrivals={total} jointly_ok={passed}"
                      f" prompt_alone_exceeds_first_deadline={oversized}")

    # One worker, two simultaneous arrivals at t=0, then one at t=2 exactly
    # when the first job finishes. The last job starts at t=3 and ends at t=4.
    tiny = [(0, 4, 2), (0, 4, 1), (2*TICKS, 4, 1)]
    result = replay(tiny, 1, 4, 1)
    assert result.starts == [0, 2, 3] and result.finishes == [2, 3, 4]
    assert [result.waiting(t) for t in [0, 2, 3, 4]] == [1, 1, 0, 0]
    assert result.window_metrics(1) == (2, 1, 2)
    assert result.deadlines() == (1, 3, 1)
    assert half_window_counts(tiny, result, 4) == [(2, 1, 0), (1, 0, 0)]
    parallel = replay(tiny, 2, 4, 1)
    assert parallel.window_metrics(2) == (1, .75, 1)
    assert parallel.deadlines() == (3, 3, 3)


if __name__ == "__main__":
    main()
