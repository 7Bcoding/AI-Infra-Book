"""Compare two training orders on prose and worked answers from the book.

Python standard library only. Run from any directory within the complete
repository or after extracting the full examples ZIP. Chapters 1-6 supply
training text, chapter 7 supplies evaluation text. No external model needed.
"""
import argparse
from collections import Counter, defaultdict
from itertools import chain
from math import exp, log
from pathlib import Path
import random
import re


CHAPTERS = Path(__file__).resolve().parents[2] / "book" / "chapters"


def read_sources(numbers):
    prose, answers = [], []
    for chapter in numbers:
        text = (CHAPTERS / f"ch{chapter:02}.md").read_text(encoding="utf-8")
        body, exercises = text.split("## 练习与解答", 1)
        for lines, dest in ((body.splitlines(), prose), (exercises.splitlines(), answers)):
            for line in lines:
                if dest is answers:
                    if not line.startswith("**解答。**"):
                        continue
                    line = line.removeprefix("**解答。**").strip()
                if not re.match(r"[\u4e00-\u9fff]", line):
                    continue
                for run in re.findall(r"[\u4e00-\u9fff]{2,}", line):
                    dest.extend(zip(run, run[1:]))
    if not prose or not answers:
        raise ValueError("Both sources must have target positions")
    return prose, answers


def probabilities(row):
    peak = max(row)
    vals = [exp(x - peak) for x in row]
    total = sum(vals)
    return [v/total for v in vals]


def train(batches, size, step_size):
    # Each preceding character has its own row of next-character logits.
    weights = [[0.0]*size for _ in range(size)]
    for batch in batches:
        targets = defaultdict(Counter)
        for x, y in batch:
            targets[x][y] += 1
        for x, counts in targets.items():
            n = sum(counts.values())
            row = weights[x]
            for y, prob in enumerate(probabilities(row)):
                row[y] -= step_size*(n*prob-counts[y])/len(batch)
    return [[-log(p) for p in probabilities(row)] for row in weights]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--step-size", type=float, default=3.0)
    args = parser.parse_args()
    sources = read_sources(range(1, 7))
    heldout = read_sources([7])
    frequent = Counter(c for source in sources for pair in source for c in pair)
    vocab = {c: i for i, (c, _) in enumerate(frequent.most_common(64), 1)}
    def encode(pairs):
        return [(vocab.get(x, 0), vocab.get(y, 0)) for x, y in pairs]
    rng = random.Random(args.seed)
    docs, demos = [rng.choices(encode(pairs), k=k) for pairs, k in zip(sources, (6144, 2048))]
    def chunks(seq, width):
        return [seq[i:i+width] for i in range(0, len(seq), width)]
    schedules = {
        "staged": chunks(docs, 64) + chunks(demos, 64),
        "mixed": [a+b for a, b in zip(chunks(docs, 48), chunks(demos, 16), strict=True)],
    }
    assert Counter(chain.from_iterable(schedules["staged"])) == Counter(chain.from_iterable(schedules["mixed"]))
    for name, batches in schedules.items():
        loss = train(batches, len(vocab)+1, args.step_size)
        values = [sum(loss[x][y] for x, y in encode(pairs))/len(pairs) for pairs in heldout]
        print(f"{name}: {len(batches)} updates; prose NLL {values[0]:.5f}, answer NLL {values[1]:.5f}")
    print("Held-out target positions (prose, answers):", *(len(pairs) for pairs in heldout))


if __name__ == "__main__":
    main()
