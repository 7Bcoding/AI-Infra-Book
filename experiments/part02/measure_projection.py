"""Optional BF16 projection experiment for a CUDA-enabled PyTorch installation."""
import argparse
import json
from math import sqrt
from statistics import median

from verify import projection


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=positive, default=[1, 64, 128])
    parser.add_argument("--width", type=positive, default=4096)
    parser.add_argument("--copies", type=positive, default=1, help="Number of weight buffers visited in rotation.")
    parser.add_argument("--repeats", type=positive, default=32, help="Matrix calls per timed group.")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        parser.exit(2, "This optional measurement requires CUDA-enabled PyTorch.\n")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.exit(2, "The measurement requires an available GPU with BF16 support.\n")

    torch.manual_seed(6)
    device = torch.device("cuda")
    width = args.width
    weight = (torch.randn(width, width, device=device) / sqrt(width)).to(torch.bfloat16)
    weights = [weight] + [weight.clone() for _ in range(args.copies - 1)]
    config = dict(device=torch.cuda.get_device_name(device), torch=torch.__version__,
                  cuda=torch.version.cuda, dtype="bfloat16", width=width, weight_copies=len(weights),
                  calls_per_group=args.repeats, timed_groups=7)
    print(json.dumps(config, ensure_ascii=False))

    with torch.inference_mode():
        for rows in args.rows:
            inputs = torch.randn(rows, width, device=device).to(torch.bfloat16)
            output = torch.empty((rows, width), device=device, dtype=torch.bfloat16)
            for i in range(max(12, args.copies)):
                torch.mm(inputs, weights[i % args.copies], out=output)
            # All copies have identical values. Check a small submatrix using
            # a higher-precision reference before the timed groups begin.
            ref_rows, ref_cols = min(rows, 3), min(width, 8)
            reference = (inputs[:ref_rows].cpu().double() @ weight[:, :ref_cols].cpu().double())
            sample = output[:ref_rows, :ref_cols].cpu().double()
            torch.testing.assert_close(sample, reference, rtol=0.02, atol=0.02)

            samples = []
            for group in range(7):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for j in range(args.repeats):
                    index = (group * args.repeats + j) % args.copies
                    torch.mm(inputs, weights[index], out=output)
                stop.record()
                stop.synchronize()
                samples.append(1000 * start.elapsed_time(stop) / args.repeats)
            microseconds = median(samples)
            flops, logical_bytes, _ = projection(rows, width)
            print(json.dumps(dict(rows=rows, median_group_us_per_call=microseconds,
                                  group_means_us=samples, useful_tflops_per_s=flops/microseconds/1e6,
                                  ideal_payload_bytes=logical_bytes, sample_max_abs_error=(sample-reference).abs().max().item())))


if __name__ == "__main__":
    main()

