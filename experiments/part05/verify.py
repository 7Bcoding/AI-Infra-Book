"""Replay the teaching examples in chapters 19–21. Python standard library only."""
import argparse
from dataclasses import dataclass, field
from fractions import Fraction as F
from math import ceil, isclose

KIB = 1024
MIB = KIB ** 2
GIB = KIB ** 3
POSITION = 2 * 32 * 8 * 128 * 2


@dataclass
class Request:
    name: str
    arrival: int
    prompt: int
    outputs: int
    consumed: int = 0
    emitted: list[int] = field(default_factory=list)


WORKLOAD = (("A", 0, 2, 5), ("B", 0, 2, 2), ("C", 2, 4, 2))


def replay(policy):
    """One prefill chunk or decode per resident per round; publish at round end."""
    slots, replace, chunk = {
        "serial": (1, False, None),
        "static": (2, False, None),
        "continuous": (2, True, None),
        "chunked": (2, True, 2),
    }[policy]
    requests = [Request(*row) for row in WORKLOAD]
    waiting = list(requests)
    active = []
    rounds = []
    now = 0
    while waiting or active:
        if not active and waiting:
            now = max(now, waiting[0].arrival)
        if replace or not active:
            while len(active) < slots and waiting and waiting[0].arrival <= now:
                active.append(waiting.pop(0))
        prefill = 0
        will_emit = []
        operations = []
        for req in active:
            remaining = req.prompt - req.consumed
            if remaining:
                take = min(chunk, remaining) if chunk else remaining
                req.consumed += take
                prefill += take
                operations.append((req.name, "P", take))
            else:
                operations.append((req.name, "D", 1))
            if req.consumed == req.prompt:
                will_emit.append(req)
        end = now + max(1, prefill)
        rounds.append((now, end, operations))
        for req in will_emit:
            req.emitted.append(end)
        active = [req for req in active if len(req.emitted) < req.outputs]
        now = end
    return requests, rounds


def service_summary(requests):
    def meets(req):
        times = req.emitted
        return (times[0] - req.arrival <= 7
                and all(b - a <= 2 for a, b in zip(times, times[1:])))
    return max(req.emitted[-1] for req in requests), sum(map(meets, requests))


def correction(target, draft):
    """Return unconditional direct mass, conditional fallback, and final law."""
    assert len(target) == len(draft) and len(target) > 0
    assert sum(target) == sum(draft) == 1 and min(*target, *draft) >= 0
    direct = [min(p, q) for p, q in zip(target, draft)]
    rejected = 1 - sum(direct)
    fallback = [(p - kept) / rejected if rejected else F(0)
                for p, kept in zip(target, direct)]
    return direct, fallback, [k + rejected * r for k, r in zip(direct, fallback)]


def two_token_law():
    # Three possible histories after the first token. The output cap is two.
    p = [F(1, 2), F(7, 20), F(3, 20)]
    q = [F(3, 5), F(1, 10), F(3, 10)]
    target_rows = [[F(1, 5), F(3, 5), F(1, 5)],
                   [F(3, 5), F(1, 5), F(1, 5)],
                   [F(1, 5), F(1, 5), F(3, 5)]]
    draft_rows = [[F(1, 2), F(1, 4), F(1, 4)],
                  [F(1, 4), F(1, 2), F(1, 4)],
                  [F(1, 4), F(1, 4), F(1, 2)]]
    joint = [[F(0) for _ in p] for _ in p]
    _, first_fallback, _ = correction(p, q)
    for x, propose_x in enumerate(q):
        accept_x = min(1, p[x] / propose_x)
        _, second_fallback, _ = correction(target_rows[x], draft_rows[x])
        for y, propose_y in enumerate(draft_rows[x]):
            pair = propose_x * propose_y
            # When x is rejected even a plausible y belongs to the wrong path.
            for first, fallback in enumerate(first_fallback):
                for second, chance in enumerate(target_rows[first]):
                    joint[first][second] += pair * (1 - accept_x) * fallback * chance
            accept_y = min(1, target_rows[x][y] / propose_y)
            joint[x][y] += pair * accept_x * accept_y
            for second, fallback in enumerate(second_fallback):
                joint[x][second] += pair * accept_x * (1 - accept_y) * fallback
    expected = [[prob * conditional for conditional in row]
                for prob, row in zip(p, target_rows)]
    assert joint == expected
    return joint


def expected_emissions(conditional):
    survival = total = F(1)
    for probability in conditional:
        survival *= probability
        total += survival
    return total


def capacity(prefill_instances, decode_instances, followup=64, link_mib_s=1024):
    return min(2 * prefill_instances, 256 * decode_instances / followup,
               link_mib_s / (1024 * POSITION / MIB))


def transfer_ms(size, bytes_per_second):
    return size / bytes_per_second * 1000


def verify():
    expected = {
        "serial": ([2, 3, 4, 5, 6], [8, 9], [13, 14]),
        "static": ([4, 5, 6, 7, 8], [4, 5], [12, 13]),
        "continuous": ([4, 5, 9, 10, 11], [4, 5], [9, 10]),
        "chunked": ([4, 5, 7, 9, 10], [4, 5], [9, 10]),
    }
    for name, delivered in zip(expected, [1, 2, 2, 3]):
        requests, rounds = replay(name)
        assert tuple(r.emitted for r in requests) == expected[name]
        end, count = service_summary(requests)
        assert end == rounds[-1][1] and count == delivered
        print(f"{name:11s}  end={end:2d} ms  good={count}/3  "
              + "  ".join(f"{r.name}{r.emitted}" for r in requests))
    assert POSITION == 128 * KIB
    assert (24 * GIB - 4 * GIB - 12_000_000_000) // (4 * POSITION) == 18071
    assert (4096 + 256) // 4 * 16 == 17408
    for page, waste_kib in [(4, 256), (8, 768)]:
        assert (ceil(10 / page) * page - 10) * POSITION == waste_kib * KIB
    assert (4096 + 2 * 32) * POSITION == 520 * MIB
    assert 2 * transfer_ms(64 * MIB, 16 * GIB) == 7.8125
    p = [F(50, 100), F(35, 100), F(15, 100)]
    q = [F(60, 100), F(10, 100), F(30, 100)]
    kept, fallback, combined = correction(p, q)
    assert sum(kept) == F(3, 4) and fallback == [0, 1, 0] and combined == p
    p2 = [F(2, 5), F(2, 5), F(1, 5)]
    q2 = [F(1, 5), F(3, 10), F(1, 2)]
    assert correction(p2, q2)[1:] == ([F(2, 3), F(1, 3), 0], p2)
    assert correction(p, p)[2] == p
    assert correction([F(1), F(0)], [F(0), F(1)])[2] == [1, 0]
    joint = two_token_law()
    assert joint[0][1] == F(3, 10) and sum(map(sum, joint)) == 1
    print('two-token verification (9 exact joint probabilities):', joint)
    emissions = expected_emissions([F(3, 4)] * 3)
    assert emissions == F(175, 64)
    varying = expected_emissions([F(9, 10), F(6, 10), F(3, 10)])
    assert varying == F(2602, 1000)
    assert expected_emissions([F(3, 10)] * 3) == F(1417, 1000)
    assert 1 - F(3, 5) ** 3 == F(784, 1000)
    assert sum(F(3, 5) ** i for i in range(3)) == F(196, 100)
    assert F(36, 48) == F(3, 4)
    print("speculation: accepted mass", kept, "fallback", fallback,
          "expected outputs", float(emissions), "ms/output", round(13 / float(emissions), 3))
    assert 10 + (1024 - 256) / 2048 * 1000 == 385
    prompt_bytes = 1024 * POSITION
    assert prompt_bytes == 128 * MIB
    assert isclose(0.2 + transfer_ms(prompt_bytes, 40e9) + 20, 23.5554432)
    assert isclose(0.2 + transfer_ms(prompt_bytes, 20e9) + 20, 26.9108864)
    assert [capacity(p, 5 - p) for p in range(1, 5)] == [2, 4, 6, 4]
    assert capacity(3, 2, link_mib_s=640) == 5
    assert max(capacity(p, 5 - p, followup=128) for p in range(1, 5)) == 4
    directions = 32 * 2
    activation = 64 * 4096 * 2
    assert directions * activation == 32 * MIB
    assert isclose(directions * 0.010 + transfer_ms(directions * activation, 40e9), 1.4788608)
    disk = transfer_ms(prompt_bytes, 2 * GIB)
    host = transfer_ms(prompt_bytes, 16 * GIB)
    assert disk + host == 70.3125
    assert disk + host / 2 == 66.40625


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandwidth-gbs", type=float, default=40,
                        help="exclusive P–D link, decimal GB/s")
    parser.add_argument("--prefill-instances", type=int, choices=range(1, 5), default=3,
                        help="number of P instances in a pool of five")
    args = parser.parse_args()
    if args.bandwidth_gbs <= 0:
        parser.error("bandwidth must be positive")
    verify()
    moved = transfer_ms(1024 * POSITION, args.bandwidth_gbs * 1e9)
    n = args.prefill_instances
    print(f"handoff: {moved + 0.2:.3f} ms; second-token gap: {moved + 20.2:.3f} ms")
    print(f"pool ({n}P, {5-n}D), 1 GiB/s shared budget: {capacity(n, 5-n):g} requests/s")
