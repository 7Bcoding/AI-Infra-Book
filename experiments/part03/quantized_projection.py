"""Measure FP32 and packed four-bit weights on the same OpenCL GPU.

With no arguments, summarize the supplied recording (standard library only).
Use --record FILE to measure again with NumPy, PyOpenCL and an OpenCL GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


SOURCE = """
__kernel void project(__global const float *x,
#if PACKED
                      __global const uchar *weight, __global const float *scale,
#else
                      __global const float *weight,
#endif
                      __global float *out) {
    int col = get_global_id(0), row = get_global_id(1);
    float sum = 0;
    for (int group = 0; group < 8; ++group) {
#if PACKED
        float s = scale[group*1024 + col];
#endif
        for (int k = group*128; k < (group+1)*128; k += 2) {
#if PACKED
            int pair = weight[(k/2)*1024 + col];
            float a = (((pair & 15) ^ 8) - 8)*s;
            float b = (((pair >> 4) ^ 8) - 8)*s;
#else
            float a = weight[k*1024 + col], b = weight[(k+1)*1024 + col];
#endif
            sum = fma(x[row*1024+k], a, sum);
            sum = fma(x[row*1024+k+1], b, sum);
        }
    }
    out[row*1024+col] = sum;
}
"""


def measure():
    import numpy as np
    import pyopencl as cl

    dev = next(d for p in cl.get_platforms() for d in p.get_devices()
               if d.type & cl.device_type.GPU)
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    rng = np.random.default_rng(1107)
    x = rng.integers(-2, 3, (128, 1024)).astype("float32")
    # Binary fractions keep reference accumulation exactly representable.
    w = rng.integers(-112, 113, (8, 128, 1024)).astype("float32")
    w[:, 0, :] = 112
    w *= rng.choice([1/128, 1/64, 1/32], size=(8, 1, 1024)).astype("float32")
    scale = abs(w).max(axis=1) / 7
    codes = np.rint(w / scale[:, None, :]).astype("int8")
    nibble = codes.reshape(1024, 1024).astype("uint8") & 15
    packed = nibble[::2] | (nibble[1::2] << 4)
    weights = w.reshape(1024, 1024)
    approx = (codes * scale[:, None, :]).reshape(1024, 1024)
    refs = [x.astype("float64") @ a.astype("float64") for a in (weights, approx)]
    flags = cl.mem_flags
    def upload(array):
        return cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=array)
    inputs = upload(x)
    output = cl.Buffer(ctx, flags.WRITE_ONLY, size=128*1024*4)
    kernels = []
    for variant, arrays in enumerate(([weights], [packed, scale])):
        program = cl.Program(ctx, SOURCE).build(options=[f"-DPACKED={variant}"])
        kernel = cl.Kernel(program, "project")
        # Keep buffers alive for the lifetime of the kernel.
        buffers = [upload(a) for a in arrays]
        kernel.set_args(inputs, *buffers, output)
        kernels.append((kernel, buffers))
    board = dev.board_name_amd if "cl_amd_device_attribute_query" in dev.extensions.split() else dev.name
    report = dict(board=board, device=dev.name, driver=dev.driver_version,
                  numpy=np.__version__, pyopencl=cl.VERSION_TEXT, seed=1107, group_size=128,
                  local_size=[64, 1], warmups=5, calls_per_round=5, rounds=8,
                  weight_bytes=[weights.nbytes, packed.nbytes + scale.nbytes],
                  source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(), cases=[])
    for rows in (1, 128):
        shape = (1024, rows)
        def run(v):
            return cl.enqueue_nd_range_kernel(queue, kernels[v][0], shape, (64, 1))
        for v in range(2):
            for _ in range(report["warmups"]):
                run(v)
            out = np.empty((rows, 1024), dtype="float32")
            cl.enqueue_copy(queue, out, output).wait()
            np.testing.assert_array_equal(out, refs[v][:rows])
        ref, quant = [a[:rows] for a in refs]
        case = dict(rows=rows, relative_l2=float(np.linalg.norm(ref-quant)/np.linalg.norm(ref)),
                    device_us=[])
        for r in range(report["rounds"]):
            pair = [[], []]
            for v in ((0, 1) if r % 2 == 0 else (1, 0)):
                for _ in range(report["calls_per_round"]):
                    event = run(v)
                    event.wait()
                    pair[v].append((event.profile.end-event.profile.start)/1000)
            case["device_us"].append(pair)
        report["cases"].append(case)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--replay", type=Path, default=Path(__file__).with_name("quantized_projection_780m.json"))
    args = parser.parse_args()
    if args.record:
        result = measure()
        args.record.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        result = json.loads(args.replay.read_bytes())
    print(result["board"], "weight bytes:", result["weight_bytes"])
    for case in result["cases"]:
        pairs = case["device_us"]
        times = [median(median(p[v]) for p in pairs) for v in (0, 1)]
        wins = sum(median(p[1]) < median(p[0]) for p in pairs)
        print(f'{case["rows"]} rows: FP32 {times[0]:.2f} us, W4 {times[1]:.2f} us, '
              f'W4 faster in {wins}/{len(pairs)} rounds, relative L2 {case["relative_l2"]:.4f}')
