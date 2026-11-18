"""Offline capacity and service replay, using the published Splitwise conversation lengths.

Source and attribution are in the adjacent JSON. A 100-ns trace tick is exact.
Each deliberately simple worker serializes per-request processing. No batching or
overlap is modeled. These rates are inputs, not Azure or hardware measurements.
"""
import argparse
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


def replay(rows, workers, prefill_rate, decode_rate):
    # Timeline is measured from the first recorded arrival.
    free_at = [0.] * workers
    first_times, end_times, total_busy = [], [], 0.
    for tick, prompt, out in rows:
        arrival = tick / TICKS
        start = max(heapq.heappop(free_at), arrival)
        first = start + prompt / prefill_rate
        duration = prompt / prefill_rate + (out - 1) / decode_rate
        finish = start + duration
        heapq.heappush(free_at, finish)
        total_busy += duration
        first_times.append(first-arrival)
        end_times.append(finish-arrival)
    return (percentile(first_times, .95), percentile(end_times, .95),
            sum(t <= 1. for t in first_times), sum(t <= 4. for t in end_times),
            sum(f <= 1. and e <= 4. for f, e in zip(first_times, end_times)), total_busy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--prefill", type=float, default=4000)
    parser.add_argument("--decode", type=float, default=100)
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
        first, end, first_ok, end_ok, both, busy = replay(rows,count,args.prefill,args.decode)
        print(f"{count}: busy_s={busy:.2f} utilization_by_arrival_span={busy/(count*duration):.4f}"
              f" ttft_p95={first:.3f} e2e_p95={end:.3f} first_ok={first_ok} end_ok={end_ok} jointly_ok={both}")
    tiny = [(0, 4, 2), (0, 4, 1)]
    assert replay(tiny, 1, 4, 1) == (3, 3, 1, 2, 1, 3)
    assert replay(tiny, 2, 4, 1) == (1, 2, 2, 2, 2, 3)


if __name__ == "__main__":
    main()
