"""Compare warm local reads of the same 2 MiB batch; optionally replay the recorded result."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns


def benchmark():
    block = bytes(range(256)) * 32
    batch = block * 256
    digest = hashlib.sha256(batch).digest()
    samples = {"objects": [], "shard": []}
    with TemporaryDirectory() as directory:
        base = Path(directory)
        objects = [base / str(i) for i in range(256)]
        for f in objects:
            f.write_bytes(block)
        shard = base / "batch"
        shard.write_bytes(batch)

        def timed(kind):
            start = perf_counter_ns()
            value = b"".join(f.read_bytes() for f in objects) if kind == "objects" else shard.read_bytes()
            elapsed = (perf_counter_ns() - start) / 1e6
            assert hashlib.sha256(value).digest() == digest
            return elapsed

        for kind in samples:
            timed(kind)
        for i in range(12):
            for kind in (("objects", "shard") if i % 2 == 0 else ("shard", "objects")):
                samples[kind].append(timed(kind))
    return {"system": platform.system(), "release": platform.release(), "python": platform.python_version(),
            "bytes": len(batch), "sha256": digest.hex(), "rounds_ms": samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    record = json.loads(Path(__file__).with_name("storage_local.json").read_text(encoding="utf8")) if args.replay else benchmark()
    if args.record:
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(record["system"], record["release"], record["python"], record["bytes"], "bytes")
    for k, runs in record["rounds_ms"].items():
        assert len(runs) == 12 and min(runs) > 0
        print(k, "median ms", round(median(runs), 3), "range", round(min(runs), 3), round(max(runs), 3))

