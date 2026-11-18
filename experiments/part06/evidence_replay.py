"""Replay short video events and document lookup on the same 125 synthetic jobs.

Pure Python. Times in the replay are stipulated service demands in milliseconds.
The program enumerates sample times, scores token vectors and simulates one FIFO.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from math import isclose, sqrt
from statistics import mean


@dataclass(frozen=True)
class Job:
    number: int
    onset_ms: float
    arrival_ms: float


AXES = ((1, 0), (0, 1))
DIAGONAL = (1 / sqrt(2), 1 / sqrt(2))
DOCS = {"applicable": AXES, "distractor": (DIAGONAL, DIAGONAL)}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit_mean(rows):
    totals = tuple(map(mean, zip(*rows)))
    norm = sqrt(dot(totals, totals))
    return tuple(x / norm for x in totals)


def find_document(order, late_interaction):
    if late_interaction:
        def score(doc):
            return sum(max(dot(query, row) for row in DOCS[doc]) for query in AXES)
    else:
        def score(doc):
            return dot(unit_mean(AXES), unit_mean(DOCS[doc]))
    scores = [score(doc) for doc in order]
    # The source order breaks mathematical ties, even after binary rounding.
    return next(doc for doc, val in zip(order, scores) if isclose(val, max(scores)))


def replay(jobs, fps, late_interaction, duration, deadline):
    samples = [1000 * frame / fps for frame in range(4 * fps)]
    free_at = 0
    seen = right = qualified = on_time = 0
    for job in jobs:
        observed = any(job.onset_ms <= t < job.onset_ms + duration for t in samples)
        order = ("applicable", "distractor") if job.number % 2 == 0 else ("distractor", "applicable")
        retrieved = find_document(order, late_interaction) == "applicable"
        service = 4 + len(samples) * 0.25 + (4 if late_interaction else 2)
        free_at = max(free_at, job.arrival_ms) + service
        timely = free_at - job.arrival_ms <= deadline
        seen += observed
        right += retrieved
        on_time += timely
        qualified += observed and retrieved and timely
    return seen, right, on_time, qualified


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--arrival-ms", type=float, default=32)
    parser.add_argument("--duration-ms", type=float, default=80)
    parser.add_argument("--deadline-ms", type=float, default=30)
    args = parser.parse_args()
    if min(args.arrival_ms, args.duration_ms, args.deadline_ms) <= 0 or args.duration_ms > 1800:
        parser.error("Choose positive times and an event duration no longer than 1800 ms.")
    jobs = [Job(i, 2000 + i, args.arrival_ms * i) for i in range(125)]
    print("fps  ranking   event  document  on-time  delivered (of 125)")
    for fps in (8, 16):
        for late in (False, True):
            counts = replay(jobs, fps, late, args.duration_ms, args.deadline_ms)
            print(f"{fps:>3}  {'MaxSim' if late else 'pooled':<7}", *[f"{x:>8}" for x in counts])


if __name__ == "__main__":
    main()
