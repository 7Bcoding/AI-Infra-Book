"""Reproduce the numerical examples in the eight-part reading edition. Standard Python."""
from math import ceil, isclose, log
from statistics import mean, stdev


def profiles():
    differences = [4, 1, 3, -1, 2, 3]
    gain = mean(differences)
    radius = 2.571 * stdev(differences) / len(differences) ** 0.5
    fraction = (825 - 727) / ((825 - 727) + (3926 - 2538))
    assert round(100 * fraction, 1) == 6.6
    return {"paired interval (ms)": (round(gain-radius, 2), round(gain+radius, 2)),
            "long-batch crossover": round(fraction, 4)}


def learning_and_io():
    batch_bytes = 256 * 2048 * 4
    read_ms = 1000 * batch_bytes / 2**30
    crossing = (1.56 - 1.50) / ((1.56 - 1.50) + (2.20 - 1.90))
    assert isclose(crossing, 1/6)
    assert isclose(6e17 / (6 * 2.5e8), 4e8)
    return {"industry fraction threshold": round(crossing, 4), "MiB per batch": batch_bytes/2**20,
            "serial objects (ms)": 512 + read_ms, "one shard (ms)": 2 + read_ms,
            "repair reads (s)": [4/(2-1.5), 4/(3-2.2)]}


def deployment():
    outbound_ms = 4 * 64 * 2**20 / 25e9 * 1000
    kv_gbs = 512 * 2**20 / .030 / 1e9
    n = ceil(log(.05) / log(.98))
    assert n == 149 and (1-.98**n) >= .95 and (1-.98**(n-1)) < .95
    return {"egress lower bound (ms)": round(outbound_ms, 3), "KV crossover (GB/s)": round(kv_gbs, 3),
            "no long request among 100": round(.98**100, 4), "sample for one long request at 95%": n}


def application():
    conditional = [60/80, 63/90]
    assert (100_000_000 * 128 * 2) == 25_600_000_000
    assert (64-48)*128*2 == 4096
    return {"conditional top-10 hits": conditional, "teacher answers": [100*4*.8, 100*4*.2],
            "distinct accepted questions": [round(100*(1-(1-p)**4), 2) for p in (.8,.2)]}


def facility():
    assert isclose((16+4)/60, (10+10)/60)
    return {"coolant flow estimate (kg/s)": round(40/(4.18*5), 2),
            "backlog drain (s)": 3*70/(80-70), "GPU shares": [4/12, 8/12],
            "CPU shares": [64/96, 16/96]}


if __name__ == "__main__":
    for section in [profiles, learning_and_io, deployment, application, facility]:
        print(section.__name__, section())

