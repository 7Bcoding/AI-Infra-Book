"""Replay a two-list nearest-neighbor query and an availability-time feature join."""
from itertools import chain

CENTERS = [0.0, 10.0]
PRODUCTS = {"A": 0.0, "B": 2.0, "C": 4.0, "D": 5.1, "E": 8.0, "F": 12.0}


def search(query, probes):
    lists = [[] for _ in CENTERS]
    for name, coordinate in PRODUCTS.items():
        closest = min(range(len(CENTERS)), key=lambda i: abs(coordinate - CENTERS[i]))
        lists[closest].append(name)
    selected = sorted(range(len(CENTERS)), key=lambda i: abs(query - CENTERS[i]))[:probes]
    candidates = list(chain.from_iterable(lists[i] for i in selected))
    return sorted(candidates, key=lambda name: (abs(PRODUCTS[name] - query), name)), len(candidates)


def available_at(observed, as_of):
    visible = [row for row in observed if row[0] <= as_of and row[1] <= as_of]
    return max(visible, key=lambda row: (row[0], row[1]))[2]


if __name__ == "__main__":
    for probes, expected in [(1, "C"), (2, "D")]:
        order, work = search(4.9, probes)
        assert order[0] == expected and work == 3 * probes
        print(f"lists={probes}: compared {work} products, first {order[0]}")
    # Minutes after 09:00; each row contains event time, arrival time, inventory.
    observations = [(55, 56, 7), (58, 63, 2)]
    assert available_at(observations, 60) == 7 and available_at(observations, 64) == 2
    print("Visible inventory at 10:00 and 10:04:", [available_at(observations, t) for t in (60, 64)])

