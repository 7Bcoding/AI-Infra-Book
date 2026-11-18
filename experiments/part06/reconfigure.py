"""Count bytes by physical source and model a single stop-and-copy handoff."""
from fractions import Fraction as F


def outgoing(old_owners, new_owners, shard_gib):
    if len(old_owners) != len(new_owners):
        raise ValueError("old and new must partition the same logical heads")
    sent = {}
    for src, dst in zip(old_owners, new_owners):
        if src != dst:
            sent[src] = sent.get(src, 0) + shard_gib
    return sent


def migrate(history, growth, capacity, handoff):
    first_pass = history / capacity
    dirty = first_pass * growth
    pause = dirty / capacity + handoff
    catch_up = history / (capacity - growth) if capacity > growth else None
    return first_pass, dirty, pause, catch_up


if __name__ == "__main__":
    old = ("A", "A", "B", "B")
    for new in [("A", "B", "C", "D"), ("A", "C", "B", "D")]:
        by_port = outgoing(old, new, F(10, 4))
        print("target", new, "shared uplink GiB:", sum(by_port.values()),
              "independent port GiB:", by_port)
    assert sum(outgoing(old, ("A", "C", "B", "D"), F(10, 4)).values()) == 5
    times = migrate(F(1), F(1, 2), F(5, 2), F(1, 50))
    assert times == (F(2, 5), F(1, 5), F(1, 10), F(1, 2))
    assert migrate(F(1), F(1, 2), F(1, 2), F(1, 50))[-1] is None
    print("first pass(s), dirty GiB, pause(s), catch-up(s):", *map(float, times))
