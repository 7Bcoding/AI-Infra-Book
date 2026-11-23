"""Chapter 14. Replay the published timings, or measure with --record FILE on an OpenCL GPU."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from measure_opencl_projection import SOURCE


def measure(device_index):
    import numpy as np
    import pyopencl as cl

    devices = [device for plat in cl.get_platforms() for device in plat.get_devices()
               if device.type & cl.device_type.GPU]
    if not 0 <= device_index < len(devices):
        raise SystemExit(f"GPU index {device_index} unavailable ({len(devices)} visible)")
    device = devices[device_index]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    kernels = {tile: cl.Kernel(cl.Program(ctx, SOURCE).build(options=[f"-DROWS={tile}"]), "multiply")
               for tile in (1, 4)}
    m, k, n = 128, 1024, 1024
    rng = np.random.default_rng(917)
    def matrix(shape):
        return rng.integers(-2, 3, size=shape, dtype=np.int32).astype(np.float32)
    def uploaded(data):
        return cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
    x = matrix((m, k))
    inputs = uploaded(x)
    weights = []
    outputs = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=m*n*4)
    def launch(tile, index):
        return kernels[tile](queue, (n, m//tile), (64, 1), inputs, weights[index], outputs,
                             np.int32(m), np.int32(k), np.int32(n))
    for j in range(8):
        w = matrix((k, n))
        weights.append(uploaded(w))
        expect = x.astype(np.float64) @ w.astype(np.float64)
        for tile in (1, 4):
            cl.enqueue_fill_buffer(queue, outputs, np.float32(np.nan), 0, m*n*4)
            launch(tile, j)
            actual = np.empty((m, n), dtype=np.float32)
            cl.enqueue_copy(queue, actual, outputs).wait()
            np.testing.assert_array_equal(actual, expect)

    trials = []
    cases = [(1, 1), (4, 1), (1, 8), (4, 8)]
    for round_id in range(8):
        shift = round_id % len(cases)
        order = cases[shift:] + cases[:shift]
        if round_id >= 4:
            order.reverse()
        for tile, count in order:
            indices = [(round_id + j % count) % len(weights) for j in range(8)]
            for j in indices:
                launch(tile, j).wait()
            host_us, device_us = [], []
            for j in indices:
                start = perf_counter_ns()
                event = launch(tile, j)
                event.wait()
                host_us.append((perf_counter_ns() - start)/1000)
                device_us.append((event.profile.end - event.profile.start)/1000)
            trials.append(dict(round=round_id, tile=tile, tables=count, indices=indices,
                               host_us=host_us, device_us=device_us))
    board = device.board_name_amd if "cl_amd_device_attribute_query" in device.extensions.split() else device.name
    return dict(device=device.name.strip(), board=board, driver=device.driver_version,
                opencl=device.version, os=platform.platform(), python=platform.python_version(),
                numpy=np.__version__, pyopencl=cl.VERSION_TEXT, seed=917, shape=[m,k,n],
                local_size=[64,1], source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
                verified_elements_per_tile=len(weights)*m*n, warmups_per_case_per_round=8, trials=trials)


def report(data):
    assert data["source_sha256"] == hashlib.sha256(SOURCE.encode()).hexdigest()
    print(data["board"], data["driver"], "times in us, median of 8 round medians")
    for count in (1, 8):
        for tile in (1, 4):
            records = [r for r in data["trials"] if r["tables"] == count and r["tile"] == tile]
            assert sorted(r["round"] for r in records) == list(range(8))
            assert all(len(r["host_us"]) == len(r["device_us"]) == 8 for r in records)
            host = [median(r["host_us"]) for r in records]
            device = [median(r["device_us"]) for r in records]
            print(f"{count} weight table(s), tile {tile}: host {median(host):.2f}, device {median(device):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.record:
        runs = [measure(args.device)]
        args.record.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf8")
    else:
        runs = json.loads(Path(__file__).with_name("cache_rotation_780m.json").read_bytes())["runs"]
    for session, run in enumerate(runs, 1):
        print("Session", session)
        report(run)
