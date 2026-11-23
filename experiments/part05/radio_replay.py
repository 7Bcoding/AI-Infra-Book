"""Replay the recorded Verizon LTE uplink for the three placements in chapter 26.

python experiments/part05/radio_replay.py
python experiments/part05/radio_replay.py --source Verizon-LTE-short.up
All times are ms, sizes decimal bytes. Bins (left,right] serve at right.
"""
import argparse
from bisect import bisect_left
from hashlib import sha256
from itertools import accumulate
import json
from math import ceil
from pathlib import Path

DATA = Path(__file__).with_name("verizon_lte_uplink_10ms.json")
PLANS = ("local", "image", "feature")


class Radio:
    def __init__(self, counts, width):
        self.cumulative = [0, *accumulate(counts)]
        self.width = width

    def finish(self, start, size):
        if start % self.width:
            raise ValueError("Choose a release on a sampling boundary")
        needed = ceil(size / 1500)
        first = start // self.width
        target = self.cumulative[first] + needed
        last = bisect_left(self.cumulative, target, lo=first)
        if last == len(self.cumulative):
            raise ValueError("End of recording; this request is right-censored")
        return last * self.width

    def time(self, start, plan):
        if plan == "local":
            return 480
        before, size, after = (0, 240_000, 100) if plan == "image" else (70, 40_000, 80)
        return self.finish(start + before, size) + after - start

    def policy(self, start):
        # Only the delivery opportunities in the previous second are known.
        now = start // self.width
        packets = self.cumulative[now] - self.cumulative[now - 1000 // self.width]
        rate = packets * 1500 / 1_000_000  # MB/s
        if rate <= 40 / 330:
            return "local"
        return "feature" if rate <= 4 else "image"


def percentile(times, q):
    return sorted(times)[ceil(q * len(times)) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Verify original timestamps against distributed bins")
    args = parser.parse_args()
    data = json.loads(DATA.read_text(encoding="utf8"))
    counts = data["packet_opportunities"]
    if args.source:
        raw = args.source.read_bytes()
        assert sha256(raw).hexdigest() == data["source_sha256"]
        check = [0] * len(counts)
        for stamp in map(int, raw.splitlines()):
            check[(stamp - 1) // data["bin_ms"]] += 1
        assert check == counts
    radio = Radio(counts, data["bin_ms"])
    # Independent replays; no requests contend for the same capacity.
    starts = range(1000, 137001, 100)
    rows = {label: [] for label in (*PLANS, "past", "oracle")}
    for start in starts:
        times = {plan: radio.time(start, plan) for plan in PLANS}
        for plan in PLANS:
            rows[plan].append(times[plan])
        rows["past"].append(times[radio.policy(start)])
        rows["oracle"].append(min(times.values()))
    for plan, times in rows.items():
        late = sum(t > 500 for t in times)
        print(plan, "n, p50, p95, p99, >500 ms:", len(times),
              *(percentile(times, q) for q in (.50,.95,.99)), late)
    print("First ten bins:", counts[:10])
    for t in (1000, 5000, 10000, 30000, 60000, 90000, 120000):
        print("Start:", t, [radio.time(t, plan) for plan in PLANS], radio.policy(t))


if __name__ == "__main__":
    main()
