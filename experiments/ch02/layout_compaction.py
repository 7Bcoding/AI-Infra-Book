"""Time a strided reduction, a reusable compact copy and the cost of making it."""

import platform
from statistics import median
from time import perf_counter_ns

import numpy as np


def main():
    source = np.arange(1024 * 4096, dtype=np.float32).reshape(1024, 4096)
    view = source[:, ::8]
    compact = np.ascontiguousarray(view)
    assert np.shares_memory(source, view) and not np.shares_memory(view, compact)
    np.testing.assert_array_equal(view, compact)
    assert view.sum(dtype=np.float64) == compact.sum(dtype=np.float64)

    def copy():
        return np.ascontiguousarray(view)

    def strided_sum():
        return view.sum(dtype=np.float64)

    def compact_sum():
        return compact.sum(dtype=np.float64)

    operations = [copy, strided_sum, compact_sum]
    samples = {op: [] for op in operations}
    # Rotate measurement order. Ignore the first round for each operation.
    for round_number in range(10):
        start = round_number % len(operations)
        for op in operations[start:] + operations[:start]:
            begin = perf_counter_ns()
            for _ in range(12):
                op()
            elapsed = (perf_counter_ns() - begin) / 12 / 1e6
            if round_number:
                samples[op].append(elapsed)

    print(platform.machine(), platform.python_version(), "NumPy", np.__version__)
    print("view shape", view.shape, "strides in bytes", view.strides,
          "extra bytes for compact array", compact.nbytes)
    for op in operations:
        print(f"{op.__name__} median {median(samples[op]):.3f} ms")
    saving = median(samples[strided_sum]) - median(samples[compact_sum])
    if saving > 0:
        print("first profitable number of uses (median-based estimate)",
              int(median(samples[copy]) / saving) + 1)
    else:
        print("no break-even point at the observed rates")

    # Time the actual decision, including making the copy once per trial.
    for uses in (1, 4, 8):
        times = {False: [], True: []}
        for trial in range(11):
            for repack in ((False, True) if trial % 2 else (True, False)):
                begin = perf_counter_ns()
                data = copy() if repack else view
                for _ in range(uses):
                    data.sum(dtype=np.float64)
                del data
                elapsed = (perf_counter_ns() - begin) / 1e6
                if trial:
                    times[repack].append(elapsed)
        print(f"{uses} uses: direct {median(times[False]):.3f} ms,"
              f" copy and reuse {median(times[True]):.3f} ms")


if __name__ == "__main__":
    main()
