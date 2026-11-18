"""Worked examples from chapters 22–25. Run with Python 3; no network or GPU."""
from contextlib import closing
from fractions import Fraction as F
from hashlib import sha256
from math import exp, floor, isclose, log, sqrt
from sqlite3 import connect


def pipeline(batches, durations):
    available = [0] * len(durations)
    schedule = []
    for _ in range(batches):
        ready = 0
        row = []
        for index, minutes in enumerate(durations):
            start = max(ready, available[index])
            ready = available[index] = start + minutes
            row.append((start, ready))
        schedule.append(row)
    return schedule


def dpo_loss(log_odds, beta):
    return log(1 + exp(-beta * log_odds))


def clipped(ratio, advantage, epsilon=0.2):
    limited = max(1-epsilon, min(1+epsilon, ratio))
    return min(ratio * advantage, limited * advantage)


def post_training():
    x, a, b = (1, 2), (1, 2), (0.1, -0.2)
    scalar = sum(left * right for left, right in zip(x, a))
    assert [v + scalar*w for v, w in zip(x, b)] == [1.5, 1]
    assert 4096**2 // (2*4096*8) == 256
    assert isclose(-log(0.4)/2, 0.4581453659370775)
    odds = log(0.75/0.5) - log(0.25/0.5)
    assert isclose(dpo_loss(odds, 1), -log(0.75))
    assert isclose(dpo_loss(-odds, 1), -log(0.25))
    h = 1e-5
    slope = (dpo_loss(odds+h, 1)-dpo_loss(odds-h, 1))/(2*h)
    assert isclose(slope, -0.25, abs_tol=1e-9)
    rewards = [1, 1, 0, 0]
    mean = sum(rewards)/len(rewards)
    sd = sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    assert [(r-mean)/sd for r in rewards] == [1, 1, -1, -1]
    assert clipped(1.4, 1) == 1.2 and clipped(0.6, -1) == -0.8
    assert pipeline(3, [8, 2, 4])[-1][-1][1] == 30
    assert pipeline(3, [8, 1, 4])[-1][-1][1] == 29
    print("DPO loss", round(dpo_loss(odds, 1), 5), "three batches", pipeline(3, [8, 2, 4]))


def visual_positions(frames=32, side=224):
    assert frames % 2 == 0 and side % 32 == 0
    return (frames//2) * (side//32)**2


def denoise(epsilon):
    clean = (1 - F(4, 5)*epsilon)/F(3, 5)
    return clean, F(4, 5)*clean + F(3, 5)*epsilon


def media():
    pixels = [1, 3, 2, 4]
    assert sum(pixels)/4 == 2.5
    assert sum(w*x for w, x in zip((-1, 1, -1, 1), pixels)) == 4
    assert 32*224**2*3 / 2**20 == 4.59375
    lengths = [visual_positions(side=side)+32 for side in [224, 448]]
    assert lengths == [816, 3168]
    assert [n*128//1024 for n in lengths] == [102, 396]
    assert (visual_positions(frames=16)+32)*128//1024 == 53
    assert 1 + floor((64000-400)/160) == 398
    assert 398*80*2 == 63680
    assert denoise(F(1, 2)) == (1, F(11, 10))
    assert denoise(F(1, 4)) == (F(4, 3), F(73, 60))
    assert 20*784**2 == 12293120
    print("media prefix lengths", lengths, "DDIM states", denoise(F(1, 2)), denoise(F(1, 4)))


def register(conn, key, body):
    digest = sha256(body.encode("utf-8")).hexdigest()
    # A unique operation key and its effect live in the same row and transaction.
    with conn:
        conn.execute("INSERT INTO ledger(op, digest, body) VALUES (?, ?, ?) "
                     "ON CONFLICT(op) DO NOTHING", (key, digest, body))
        number, previous_digest = conn.execute(
            "SELECT number, digest FROM ledger WHERE op=?", (key,)).fetchone()
        if previous_digest != digest:
            raise ValueError("operation key reused with different content")
    return number


def lost_ack():
    uri = "file:handover_book_example?mode=memory&cache=shared"
    with closing(connect(uri, uri=True)) as db:
        db.execute("CREATE TABLE ledger(number INTEGER PRIMARY KEY, op TEXT UNIQUE, "
                   "digest TEXT NOT NULL, body TEXT NOT NULL)")
        with closing(connect(uri, uri=True)) as first:
            register(first, "730/1", "2s:green")  # Response discarded after COMMIT.
        with closing(connect(uri, uri=True)) as retry:
            number = register(retry, "730/1", "2s:green")
            try:
                register(retry, "730/1", "2s:blue")
            except ValueError:
                pass
            else:
                raise AssertionError("expected a conflict")
        assert db.execute("SELECT number, body FROM ledger").fetchall() == [(number, "2s:green")]
        return number


def agent_task():
    docs = [("A", (1, 0), False, True), ("B", (.96, .28), True, False),
            ("C", (.8, .6), True, True)]
    assert all(isclose(x*x+y*y, 1) for _, (x, y), _, _ in docs)
    raw = sorted(docs, key=lambda d: d[1][0], reverse=True)[:2]
    ranked = sorted((d for d in docs if d[2]), key=lambda d: d[1][0]+.30*d[3], reverse=True)
    assert [d[0] for d in raw] == ["A", "B"] and ranked[0][0] == "C"
    tail = sum([90, 60, 40, 70, 10])
    assert max(120, 80) + tail == 390 and 120 + 80 + tail == 470
    assert 200*32 / 1024 / 2 == 3.125
    print("retrieval winner", ranked[0][0], "one committed row after lost ACK, ID", lost_ack())


def platform():
    train_a = F(1_200_000_000, 10000*3600)
    train_b = F(2_400_000_000, 16000*3600)
    assert (2+train_a+2+3)*60 == 2420 and (1+train_b+2+4)*60 == 2920
    assert 4*(train_a+2+3)*60 == 9200 and 4*(train_b+2+4)*60 == 11440
    assert (2+train_a*F(5, 4)+2+3)*60 == 2920
    met = [2000 - sum(bad) for bad in [(30, 100, 22), (10, 42, 18)]]
    assert met == [1848, 1930] and [n/40 for n in met] == [46.2, 48.25]
    assert F(1919, 40) < 48 and F(1919, 2000) > F(95, 100)
    assert (12*.5 + 2)*1.25*48 == 480
    assert 12000/(2-.5) == 8000
    print("training elapsed hours", float(2+train_a+2+3), float(1+train_b+2+4),
          "service goodput", [n/40 for n in met])


if __name__ == "__main__":
    post_training()
    media()
    agent_task()
    platform()
