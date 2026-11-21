"""Reproduce the numerical examples in Part II. Run with standard Python 3."""
from collections import Counter
from math import ceil, gcd, isclose
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
X = ((1, 2), (3, 4))
W = ((2, -1), (1, 3))
KIB = 2**10
MIB = 2**20
GIB = 2**30


def issue_times(chain_count, chain_length, latency):
    """One issue slot, cyclic choice of the next ready chain."""
    ready = [0] * chain_count
    times = []
    tick = 0
    for _ in range(chain_length):
        for chain in range(chain_count):
            tick = max(tick, ready[chain])
            times.append((tick, chain))
            ready[chain] = tick + latency
            tick += 1
    return times, max(ready)


def block_limits(registers, shared_kib):
    return [65536 // (256 * registers), 64 // shared_kib, 2048 // 256, 16]


def sectors(word_numbers):
    return {word // 8 for word in word_numbers}


def cache_hits(sequence, slots=2):
    recent = []
    hits = []
    for item in sequence:
        hit = item in recent
        hits.append(hit)
        if hit:
            recent.remove(item)
        elif len(recent) == slots:
            recent.pop(0)
        recent.append(item)
    return hits


def tiled_product(left, right, tile_rows, tile_cols, tile_inner):
    """Explicitly load two panels per reduction segment and count input words."""
    rows, inner, cols = len(left), len(right), len(right[0])
    output = [[0] * cols for _ in range(rows)]
    reads = 0
    for i0 in range(0, rows, tile_rows):
        for j0 in range(0, cols, tile_cols):
            for k0 in range(0, inner, tile_inner):
                ii = range(i0, min(i0 + tile_rows, rows))
                jj = range(j0, min(j0 + tile_cols, cols))
                kk = range(k0, min(k0 + tile_inner, inner))
                a = {(i, k): left[i][k] for i in ii for k in kk}
                b = {(k, j): right[k][j] for k in kk for j in jj}
                reads += len(a) + len(b)
                for i in ii:
                    for j in jj:
                        output[i][j] += sum(a[i, k] * b[k, j] for k in kk)
    return output, 4 * (reads + rows * cols)


def copy_compute(segments, copy_time, compute_time):
    """Two alternating buffers, one transfer and one compute engine."""
    free_at = [0, 0]
    copy_end = compute_end = 0
    events = []
    for i in range(segments):
        slot = i % 2
        copy_start = max(copy_end, free_at[slot])
        copy_end = copy_start + copy_time
        compute_start = max(copy_end, compute_end)
        compute_end = compute_start + compute_time
        free_at[slot] = compute_end
        events.append((slot, copy_start, copy_end, compute_start, compute_end))
    return events


def systolic(left, right):
    """Move old neighbour values one hop each cycle; inject skewed boundaries."""
    rows, cols, inner = len(left), len(right[0]), len(right)
    a = [[None] * cols for _ in range(rows)]
    b = [[None] * cols for _ in range(rows)]
    sums = [[0] * cols for _ in range(rows)]
    history = []
    for tick in range(rows + cols + inner - 2):
        next_a = [[None] * cols for _ in range(rows)]
        next_b = [[None] * cols for _ in range(rows)]
        active = set()
        for i in range(rows):
            for j in range(cols):
                av = (left[i][tick-i] if 0 <= tick-i < inner else None) if j == 0 else a[i][j-1]
                bv = (right[tick-j][j] if 0 <= tick-j < inner else None) if i == 0 else b[i-1][j]
                next_a[i][j], next_b[i][j] = av, bv
                if av is not None and bv is not None:
                    sums[i][j] += av * bv
                    active.add((i, j))
        a, b = next_a, next_b
        history.append(([row.copy() for row in sums], active))
    return history


def projection(rows, width=4096):
    flops = 2 * rows * width * width
    volume = 2 * (width * width + 2 * rows * width)
    return flops, volume, flops / volume


def roof_time(flops, volume, tera_flops, tera_bytes):
    """Return microseconds, using decimal rates."""
    return max(flops / tera_flops, volume / tera_bytes) / 1e6


def check_execution():
    assert [issue_times(n, 8//n, 4)[1] for n in (1, 2, 4)] == [32, 17, 11]
    assert block_limits(64, 16) == [4, 4, 8, 16]
    assert min(block_limits(65, 16)) == 3
    assert [min(block_limits(*case)) for case in ((64,24), (32,24), (32,16))] == [2, 2, 4]
    counts = [min(32, max(0, 70 - 32*i)) for i in range(4)]
    assert counts == [32, 32, 6, 0]
    assert 8*3 + 24 == 48 and (12*2 + 20*5) == 124
    assert [next(n for n in range(500) if start + 5*n < 80*n)
            for start in (8000, 12000)] == [107, 161]
    assert [(80*n, 12000+5*n) for n in (160, 200)] == [(12800,12800), (16000,13000)]
    assert sum(range(1, 9)) == 36
    assert 16*8*16 == 2048


def check_storage():
    for offset, step, count in [(0,1,4),(1,1,5),(0,8,32),(7,1,5)]:
        assert len(sectors(offset + step*i for i in range(32))) == count
    assert len(sectors(range(7,39)) | sectors(range(39,71))) == 9
    for stride in (1, 2, 8, 31, 32, 33, 34, 35):
        loads = Counter(stride*i % 32 for i in range(32))
        assert max(loads.values()) == gcd(32, stride)
        assert len(loads) == 32 // gcd(32, stride)
    assert sum(cache_hits("ABABCDCD")) == 4
    assert sum(cache_hits("ABCDABCD")) == 0
    left = [[i*4+k-4 for k in range(4)] for i in range(4)]
    right = [[k+j*j-2 for j in range(4)] for k in range(4)]
    reference = [[sum(left[i][k]*right[k][j] for k in range(4))
                  for j in range(4)] for i in range(4)]
    for rows, cols, traffic in [(1,1,576), (2,2,320), (4,4,192), (1,2,448)]:
        result, volume = tiled_product(left, right, rows, cols, 2)
        assert result == reference and volume == traffic
    assert ceil(1.5e12 * 400e-9 / 128) == 4688
    assert copy_compute(4, 3, 5)[-1][-1] == 23
    assert copy_compute(3, 5, 2) == [(0,0,5,5,7), (1,5,10,10,12), (0,10,15,15,17)]
    assert isclose(16*GIB/32e9, 0.536870912)
    assert round(16*GIB/3e12*1000, 2) == 5.73
    assert round(16*GIB/64/32e9*1000, 2) == 8.39
    assert (32_000_000_000 * 12 // 1000) // (64*KIB) == 5859
    assert isclose(6000 * 64*KIB / 32e9 * 1000, 12.288)
    assert sum(cache_hits(list(range(12))*2, slots=8)) == 0
    assert [round(64*MIB / b * 1000, 2) for b in (24e9,12e9)] == [2.80,5.59]
    assert [value/0.002/1e12 for value in (3e9, 2e9, 1e9)] == [1.5,1,.5]


def check_architecture():
    waves = systolic(X, W)
    assert [len(active) for _, active in waves] == [1,3,3,1]
    assert waves[-1][0] == [[4,5],[10,9]]
    other = systolic([[1,2,1,2],[3,1,2,0]], [[0,1,4],[1,3,6],[2,0,1],[3,2,1]])
    assert len(other) == 7 and sum(len(active) for _, active in other) == 24
    assert other[-1][0] == [[10,11,19],[5,6,20]]
    for rows, expected in [(1,(22.38,11.19)), (64,(23.07,26.84)), (128,(35.79,53.69))]:
        flops, volume, _ = projection(rows)
        times = [roof_time(flops, volume, *device) for device in [(120,1.5), (80,3)]]
        assert [round(t,2) for t in times] == list(expected), times
    assert next(m for m in range(200) if projection(m)[2] >= 80) == 84
    assert next(m for m in range(1,200) if roof_time(*projection(m)[:2],120,1.5)
                < roof_time(*projection(m)[:2],80,3)) == 55
    assert [(v*GIB-12_000_000_000)//(128*KIB) for v in (24,22)] == [105055,88671]
    assert isclose(48e9/200e-6,240e12)
    assert [round(x,3) for x in (32/(32768+64)*100,16*GIB/7380e9*1000)] == [.097,2.328]
    assert [max(24e9/120e12,120e6/1.5e12,s/12e12)*1e6 for s in (1.8e9,4.8e9)] == [200,400]


def check_manuscripts():
    for chapter in (6, 7, 8, 9):
        path = ROOT / f"book/chapters/ch{chapter:02}.md"
        source = path.read_text(encoding="utf-8")
        snippets = re.findall(r"^```python\n(.*?)^```", source, re.M | re.S)
        for snippet in snippets:
            exec(compile(snippet, str(path), "exec"), {})
        exercises = source.split("## 练习与解答", 1)[1]
        questions = re.split(r"^### \d+\. ", exercises, flags=re.M)[1:]
        assert questions and all(q.count("**解答。**") == 1 for q in questions)
        print(f"Chapter {chapter}: {len(snippets)} Python examples and {len(questions)} worked exercises.")


if __name__ == "__main__":
    check_execution()
    check_storage()
    check_architecture()
    check_manuscripts()
    print("Scheduling, memory traffic, array timing, Roofline budgets and exercises passed.")

