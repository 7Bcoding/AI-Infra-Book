"""Reconstruct chapter 19's message after a gap and a repeated arrival."""
from itertools import combinations

FRAGMENTS = 8
PAYLOAD = 1500


def compare(missing):
    # The values stand for distinct fragments of a single message with a known size.
    original = [bytes([i]) * PAYLOAD for i in range(FRAGMENTS)]
    buffer = bytearray(FRAGMENTS * PAYLOAD)
    arrived = set()

    def place(index):
        if index not in arrived:
            start = index * PAYLOAD
            buffer[start:start + PAYLOAD] = original[index]
            arrived.add(index)

    for i in reversed(range(FRAGMENTS)):
        if i not in missing:
            place(i)
    if arrived:
        place(max(arrived))  # duplicate: should not fill a hole
    assert (len(arrived) == FRAGMENTS) == (not missing)
    outstanding = sorted(set(range(FRAGMENTS)) - arrived)
    restart_bytes = (FRAGMENTS - outstanding[0]) * PAYLOAD if outstanding else 0
    for i in outstanding:
        place(i)
    assert bytes(buffer) == b"".join(original)
    return restart_bytes, len(outstanding) * PAYLOAD


if __name__ == "__main__":
    for count in range(FRAGMENTS + 1):
        for missing in combinations(range(FRAGMENTS), count):
            restart, selective = compare(missing)
            assert selective <= restart
    restart, selective = compare((2,))
    assert (restart, selective) == (9000, 1500)
    print("256 loss patterns checked. For missing fragment 2:",
          restart, "vs", selective, "retransmitted payload bytes.")
