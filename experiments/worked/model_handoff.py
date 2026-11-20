"""Resume a momentum update and independently load column-sharded weights for inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def project(inputs, weights):
    return [sum(x * row[j] for x, row in zip(inputs, weights)) for j in range(2)]


def update(state, samples):
    inputs, target = samples[state["next_sample"]]
    scores = project(inputs, state["weights"])
    # Loss is half the sum of squared output errors; samples are consumed in order.
    momentum = [
        [0.9 * state["momentum"][i][j] + x * (scores[j] - target[j]) for j in range(2)]
        for i, x in enumerate(inputs)
    ]
    return {
        "weights": [[w - 0.05 * v for w, v in zip(row, m)] for row, m in zip(state["weights"], momentum)],
        "momentum": momentum, "step": state["step"] + 1, "next_sample": state["next_sample"] + 1,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def load_model(manifest, transpose=False):
    spec = read_json(manifest)
    assert spec["shape"] == [2, 2] and spec["layout"] == "columns"
    columns = [read_json(manifest.parent / name) for name in spec["shards"]]
    assert [c["column"] for c in columns] == [0, 1]
    values = [c["values"] for c in columns]
    return values if transpose else list(zip(*values))


def demonstrate():
    samples = [([1, 0], [1, 2]), ([0, 1], [-1, 2])]
    state = {"weights": [[2, -1], [1, 3]], "momentum": [[0.2, 0.5], [0.1, -0.4]],
             "step": 4, "next_sample": 1}
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        checkpoint = folder / "training.json"
        checkpoint.write_text(json.dumps(state), encoding="utf8")
        uninterrupted = update(state, samples)
        restored = update(read_json(checkpoint), samples)
        assert restored == uninterrupted and restored["next_sample"] == 2 and restored["step"] == 5
        names = []
        for j in range(2):
            name = f"column-{j}.json"
            names.append(name)
            (folder / name).write_text(json.dumps({"column": j, "values": [r[j] for r in state["weights"]]}), encoding="utf8")
        manifest = folder / "model.json"
        manifest.write_text(json.dumps({"shape": [2, 2], "layout": "columns", "shards": names}), encoding="utf8")
        probes = [[1, 2], [1, 0], [0, 1]]
        expected = [project(x, state["weights"]) for x in probes]
        def probe(*flags):
            result = subprocess.check_output([sys.executable, __file__, "--probe", str(manifest), *flags], text=True)
            return json.loads(result)
        correct = probe()
        faulty = probe("--transpose")
        assert correct == expected and faulty != expected
        assert correct[0] == [4, 5] and faulty[0] == [0, 7]
        print("Resumed step", restored["step"], "with matching state. Independent scores:", correct, "transposed:", faulty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()
    if args.probe:
        weights = load_model(args.probe, args.transpose)
        print(json.dumps([project(x, weights) for x in [[1, 2], [1, 0], [0, 1]]]))
    else:
        demonstrate()

