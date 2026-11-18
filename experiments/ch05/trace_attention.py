"""Compare independent documents, then follow a toy CSA2 candidate pool.

The integer scores are pedagogical and are not a checkpoint's outputs.
"""

from itertools import accumulate
from statistics import mean


def packed_causal_means(documents):
    values = [value for doc in documents for value in doc]
    ends = list(accumulate(map(len, documents), initial=0))
    # Equal scores imply uniform attention on each allowed prefix.
    return [mean(values[start:i + 1])
            for start, end in zip(ends, ends[1:])
            for i in range(start, end)]


def top_positions(scores, allowed, k):
    # Return one-based positions, with the smaller position as the tie-breaker.
    return sorted(allowed, key=lambda p: (-scores[p - 1], p))[:k]


def main():
    documents = [[2, 4, 6], [10, 20]]
    correct = packed_causal_means(documents)
    wrong = packed_causal_means([documents[0] + documents[1]])
    assert correct == [2, 3, 4, 10, 15]
    assert wrong == [2, 3, 4, 5.5, 8.4]
    print("B with and without a document boundary:", correct[-2:], wrong[-2:])

    first = [1, 9, 2, 3, 8, 0, 4, 5]
    second = [6, 3, 10, 1, 5, 4, 0, 2]
    positions = range(1, len(first) + 1)
    blocks = [list(positions[i:i + 2]) for i in range(0, len(first), 2)]
    maxima = [max(first[p - 1] for p in block) for block in blocks]
    chosen_blocks = top_positions(maxima, range(1, len(blocks) + 1), 2)
    candidates = sorted(p for i in chosen_blocks for p in blocks[i - 1])
    full = top_positions(first, positions, 2)
    reindex = top_positions(second, candidates, 2)
    reference = top_positions(second, positions, 2)
    assert (full, candidates, reindex, reference) == ([2, 5], [1, 2, 5, 6], [1, 5], [3, 1])
    print("Full, candidate pool, Reindex, unrestricted:", full, candidates, reindex, reference)
    print("Index scores for Full/Reindex/Reuse:", len(first), len(candidates), 0)


if __name__ == "__main__":
    main()
