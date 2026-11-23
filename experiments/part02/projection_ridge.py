"""Recompute the chapter 8 BF16 projection bounds using only the standard library."""

from math import ceil

WIDTH = 4096
# Per-chip BF16 TFLOP/s, HBM GB/s. Google Cloud TPU7x system architecture table.
CHIPS = {"v6e": (918, 1638), "TPU7x": (2307, 7380)}


def projection(rows):
    flops = 2 * rows * WIDTH * WIDTH
    hbm_bytes = 2 * (WIDTH * WIDTH + 2 * rows * WIDTH)
    return flops, hbm_bytes


def first_compute_limited_row(compute, bandwidth):
    ridge = 1000 * compute / bandwidth
    if ridge >= WIDTH / 2:
        return None
    return ceil(WIDTH * ridge / (WIDTH - 2 * ridge))


if __name__ == "__main__":
    for name, (compute, bandwidth) in CHIPS.items():
        cross = first_compute_limited_row(compute, bandwidth)
        assert cross == {"v6e": 772, "TPU7x": 369}[name]
        for rows in (128, cross - 1, cross, 512, 1024):
            flops, hbm_bytes = projection(rows)
            intensity = flops / hbm_bytes
            hbm_roof = intensity * bandwidth / 1000
            bound = min(compute, hbm_roof)
            print(f"{name:6}  rows={rows:4}  I={intensity:7.2f} FLOP/B"
                  f"  upper bound={bound:7.2f} TFLOP/s")
            if rows in (cross - 1, cross):
                assert (compute <= hbm_roof) == (rows == cross)
