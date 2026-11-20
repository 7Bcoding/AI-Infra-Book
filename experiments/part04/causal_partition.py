"""第 20 章的双卡因果注意力模拟。无需设备或第三方库。"""
import argparse
from math import exp, isclose, sin, sqrt


def score(query, key):
    return sum(q * k for q, k in zip(query, key, strict=True)) / sqrt(len(key))


def partial(query, visible_keys, keys, values):
    """同一行在一组键上的局部最大分数、指数质量和加权分子。"""
    scores = [score(query, keys[j]) for j in visible_keys]
    origin = max(scores)
    masses = [exp(s - origin) for s in scores]
    numerator = [
        sum(mass * values[j][axis] for j, mass in zip(visible_keys, masses, strict=True))
        for axis in range(len(values[0]))
    ]
    return origin, sum(masses), numerator


def reference_row(query, last, keys, values):
    scores = [score(query, key) for key in keys[:last + 1]]
    largest = max(scores)
    weights = [exp(s - largest) for s in scores]
    total = sum(weights)
    return [sum(w * values[j][axis] for j, w in enumerate(weights)) / total
            for axis in range(len(values[0]))]


def evaluate(owners, queries, keys, values):
    groups = [[row for row, owner in enumerate(owners) if owner == rank] for rank in (0, 1)]
    work = [0, 0]
    used_remote_keys = [set(), set()]
    for row, owner in enumerate(owners):
        query = queries[row]
        parts = []
        for source_rank, group in enumerate(groups):
            visible = [j for j in group if j <= row]
            if not visible:
                continue
            work[owner] += len(visible)
            if source_rank != owner:
                used_remote_keys[owner].update(visible)
            parts.append(partial(query, visible, keys, values))
        origin = max(m for m, _, _ in parts)
        denominator = sum(exp(m - origin) * z for m, z, _ in parts)
        result = [
            sum(exp(m - origin) * u[axis] for m, _, u in parts) / denominator
            for axis in range(len(values[0]))
        ]
        expected = reference_row(query, row, keys, values)
        assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(result, expected, strict=True))
    return work, [len(indices) for indices in used_remote_keys]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=8, help="可被四整除的位置数")
    length = parser.parse_args().length
    if length < 4 or length % 4:
        parser.error("length 需为正的四的倍数")
    queries = [(sin(i), 1) for i in range(length)]
    keys = [(1, sin(2 * i)) for i in range(length)]
    values = [(sin(3 * i), sin(5 * i)) for i in range(length)]
    layouts = {
        "contiguous": [int(i >= length // 2) for i in range(length)],
        "paired": [int(length // 4 <= i < 3 * length // 4) for i in range(length)],
    }
    for name, owners in layouts.items():
        work, remote = evaluate(owners, queries, keys, values)
        assert sum(work) == length * (length + 1) // 2
        if length == 8:
            assert work == ([10, 26] if name == "contiguous" else [18, 18])
            assert remote == ([0, 4] if name == "contiguous" else [4, 2])
        print(f"{name}: row owners={owners}, valid pairs per rank={work}")
        print(f"  distinct remote KV rows needed by each receiver={remote}; "
              "all output rows match the unsharded reference")


if __name__ == "__main__":
    main()
