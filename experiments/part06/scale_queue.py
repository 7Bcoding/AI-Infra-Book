"""Measure backlog and FIFO wait at the joins of piecewise-linear service curves."""
from fractions import Fraction as F

# Each interval gives (duration, arrival rate, available service rate).
# Fluid model, constant request sizes, empty initial queue, work-conserving service.
def curves(intervals):
    time = F(0)
    arrived = started = F(0)
    rows = [(time, arrived, started)]
    for duration, arrival, capacity in intervals:
        duration, arrival, capacity = map(F, (duration, arrival, capacity))
        if min(duration, arrival, capacity) < 0:
            raise ValueError("Negative interval, arrival rate or capacity")
        backlog = arrived - started
        if capacity > arrival and 0 < backlog < (capacity - arrival) * duration:
            delay = backlog / (capacity - arrival)
            rows.append((time + delay, arrived + delay * arrival, started + delay * capacity))
        time += duration
        arrived += arrival * duration
        started = min(arrived, started + capacity * duration)
        rows.append((time, arrived, started))
    return rows


def crossing(rows, column, rank):
    for left, right in zip(rows, rows[1:]):
        if left[column] == rank:
            return left[0]
        if left[column] < rank <= right[column]:
            fraction = (rank - left[column]) / (right[column] - left[column])
            return left[0] + fraction * (right[0] - left[0])
    raise ValueError("Rank outside observed curve")


def replay(ready):
    rows = curves([(ready, 75, 50), (36, 75, 100)])
    # Inverse curves are piecewise linear. Their difference has extrema at
    # endpoint ranks from either curve. Here arrivals and starts are strictly increasing.
    ranks = sorted({rank for _, a, s in rows for rank in (a, s)})
    worst = max((crossing(rows, 2, rank) - crossing(rows, 1, rank), rank) for rank in ranks)
    peak = max(a-s for _, a, s in rows)
    return rows, peak, worst


if __name__ == "__main__":
    rows, peak, worst = replay(12)
    assert (peak, worst) == (300, (4, 600))
    assert (24, 1800, 1800) in rows
    assert replay(3)[2] == (1, 150)
    for t, a, s in rows:
        print(f"t={t} s, arrivals={a}, started={s}, queue={a-s}")
    print(f"maximum wait={worst[0]} s at cumulative request {worst[1]}")
