"""Recompute the field observations in chapter 19 with Python's standard library."""
import json
from pathlib import Path


def main():
    path = Path(__file__).with_suffix(".json")
    cases = json.loads(path.read_text(encoding="utf-8"))
    samples = cases["falcon"]["samples"]
    baseline = samples[0]["steps_per_second"]
    print("FALCON Fig. 4: approximate readings of the published curves")
    print("minute   s/step   throughput loss   NIC2 CNP/s (thousands)   mean SM %")
    for sample in samples:
        rate = sample["steps_per_second"]
        nic2 = sample["cnp_thousands_per_second_by_nic"][2]
        loss = 1 - rate / baseline
        print(f"{sample['minute']:6d}   {1 / rate:6.1f}   {loss:14.1%}"
              f"   {nic2:22.1f}   {sample['average_sm_percent']:9d}")

    reference, changed = cases["exemplar"]["reference_runs"]
    old_step, new_step = reference["iteration_ms"], changed["iteration_ms"]
    print("\nExemplar case 3: reference cluster, 1 to 4 QPs per connection")
    savings = old_step - new_step
    print(f"iteration reduction: {savings} ms ({savings / old_step:.1%})")
    print(f"steps per second gain: {old_step / new_step - 1:.1%}")
    for operation in ["allgather_ms", "reducescatter_ms"]:
        print(f"{operation} decrease: {reference[operation] - changed[operation]} ms")
    other = cases["exemplar"]["other_nccl_version_iteration_ms"]
    print(f"separate comparison, different NCCL version: {other} ms")


if __name__ == "__main__":
    main()
