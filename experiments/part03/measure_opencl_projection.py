"""Chapter 9: measure two row tiles of an FP32 projection on an OpenCL GPU.

--replay reads a measurement with only Python's standard library. Measuring
requires numpy, pyopencl and a vendor OpenCL GPU driver. --record saves both
host-clock and device-clock samples alongside launch settings and versions.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns

SOURCE = """
__kernel void multiply(__global const float *x, __global const float *w,
                       __global float *y, int m, int k, int n) {
    int col = get_global_id(0);
    int base = get_global_id(1) * ROWS;
    float acc[ROWS] = {0};
    for (int t = 0; t < k; ++t) {
        float weight = w[t*n + col];
        for (int r = 0; r < ROWS; ++r)
            if (base+r < m)
                acc[r] = fma(x[(base+r)*k + t], weight, acc[r]);
    }
    for (int r = 0; r < ROWS; ++r)
        if (base+r < m)
            y[(base+r)*n + col] = acc[r];
}
"""


def read_resources(binary):
    """Read the AMD code object only when --amd-metadata was requested."""
    from io import BytesIO
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import NoteSection
    import msgpack

    for section in ELFFile(BytesIO(binary)).iter_sections():
        if isinstance(section, NoteSection):
            for note in section.iter_notes():
                if note.n_name == "AMDGPU" and note.n_type == 32:
                    data = msgpack.unpackb(note.n_descdata)
                    for kernel in data["amdhsa.kernels"]:
                        if kernel[".name"] == "multiply":
                            keys = ("vgpr_count", "sgpr_count", "vgpr_spill_count",
                                    "sgpr_spill_count", "wavefront_size")
                            return {key: kernel["."+key] for key in keys}
    raise ValueError("No AMDGPU metadata for multiply in compiled code object")


def measure(args):
    import numpy as np
    import pyopencl as cl

    devices = [d for p in cl.get_platforms() for d in p.get_devices()
               if d.type & cl.device_type.GPU]
    if not 0 <= args.device < len(devices):
        raise SystemExit(f"GPU index {args.device} unavailable ({len(devices)} visible)")
    dev = devices[args.device]
    context = cl.Context([dev])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    kernels = {}
    resources = {}
    for rows in (1, 4):
        program = cl.Program(context, SOURCE).build(options=[f"-DROWS={rows}"])
        kernels[rows] = cl.Kernel(program, "multiply")
        if args.amd_metadata:
            resources[rows] = read_resources(program.binaries[0])

    m_large, k, n = 128, 1024, 1024
    rng = np.random.default_rng(917)
    x = rng.integers(-2, 3, size=(m_large, k), dtype=np.int32).astype(np.float32)
    w = rng.integers(-2, 3, size=(k, n), dtype=np.int32).astype(np.float32)
    golden = x.astype(np.float64) @ w.astype(np.float64)
    flags = cl.mem_flags
    a = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=x)
    b = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=w)
    y = cl.Buffer(context, flags.WRITE_ONLY, size=m_large*n*4)
    board = dev.board_name_amd if "cl_amd_device_attribute_query" in dev.extensions.split() else dev.name
    result = dict(device=dev.name.strip(), board=board, resources=resources, driver=dev.driver_version, opencl=dev.version,
                  os=platform.platform(), python=platform.python_version(),
                  numpy=np.__version__, pyopencl=cl.VERSION_TEXT,
                  source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(), seed=917,
                  dtype="FP32", local_size=[64, 1], width=k, columns=n,
                  warmups_per_variant=5, rounds=8, calls_per_round=7, results=[])
    for m in (1, m_large):
        def launch(rows):
            return kernels[rows](queue, (n, (m+rows-1)//rows), (64, 1), a, b, y,
                                 np.int32(m), np.int32(k), np.int32(n))
        for rows in (1, 4):
            cl.enqueue_fill_buffer(queue, y, np.float32(np.nan), 0, m*n*4)
            for _ in range(result["warmups_per_variant"]):
                launch(rows)
            out = np.empty((m, n), dtype=np.float32)
            cl.enqueue_copy(queue, out, y).wait()
            np.testing.assert_array_equal(out, golden[:m])
        times = {1: [], 4: []}
        spans = {1: [], 4: []}
        batch_wall = {1: [], 4: []}
        single_wall = {1: [], 4: []}
        single_event = {1: [], 4: []}
        for group in range(result["rounds"]):
            order = (1, 4) if group % 2 == 0 else (4, 1)
            for rows in order:
                begin = perf_counter_ns()
                events = [launch(rows) for _ in range(result["calls_per_round"])]
                events[-1].wait()
                batch_wall[rows].append((perf_counter_ns()-begin)/1000)
                spans[rows].append((events[-1].profile.end-events[0].profile.start)/1000)
                times[rows].append([(e.profile.end-e.profile.start)/1000 for e in events])
                host_samples, device_samples = [], []
                for _ in range(result["calls_per_round"]):
                    begin = perf_counter_ns()
                    event = launch(rows)
                    event.wait()
                    host_samples.append((perf_counter_ns()-begin)/1000)
                    device_samples.append((event.profile.end-event.profile.start)/1000)
                single_wall[rows].append(host_samples)
                single_event[rows].append(device_samples)
        for rows in (1, 4):
            group_medians = [median(samples) for samples in times[rows]]
            groups = n//64*((m+rows-1)//rows)
            result["results"].append(dict(m=m,rows=rows,workgroups=groups,
                  logical_weight_load_bytes=((m+rows-1)//rows)*k*n*4,
                  verified_elements=m*n, group_medians_us=group_medians,
                  median_us=median(group_medians), raw_us=times[rows],
                  batch_wall_us=batch_wall[rows], batch_event_span_us=spans[rows],
                  single_wall_us=single_wall[rows], single_event_us=single_event[rows]))
    return result


def report(result):
    print(result["board"], result["driver"])
    print("Microseconds: single host launch+wait | host batch/call | device batch span/call")
    calls = result["calls_per_round"]
    for row in result["results"]:
        single = median(median(group) for group in row["single_wall_us"])
        batch = median(row["batch_wall_us"]) / calls
        span = median(row["batch_event_span_us"]) / calls
        print(f"M={row['m']:3} rows={row['rows']}: {single:.2f} | {batch:.2f} | {span:.2f}")
        event_min = min(min(group) for group in row["raw_us"])
        solo_min = min(min(group) for group in row["single_event_us"])
        print(f"  shortest device event (batched/single): {event_min:.2f}/{solo_min:.2f} us")
    if result["resources"]:
        print("Compiler resources per row tile:", result["resources"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", type=Path)
    mode.add_argument("--replay", type=Path, help="Summarize saved samples without a GPU")
    parser.add_argument("--amd-metadata", action="store_true",
                        help="Extract register counts; needs pyelftools and msgpack")
    args = parser.parse_args()
    if args.replay:
        report(json.loads(args.replay.read_text(encoding="utf-8")))
        return
    result = measure(args)
    if args.record:
        args.record.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    report(result)


if __name__ == "__main__":
    main()
