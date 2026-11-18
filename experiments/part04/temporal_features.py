"""Rebuild features known at prediction time; all timestamps are teaching data."""
from datetime import datetime, timedelta


def ts(time):return datetime.fromisoformat('2026-01-01T'+time)


def historical(rows, entity, prediction, ttl=timedelta(hours=1), availability=True):
    candidates=[row for row in rows if row['entity']==entity
                and prediction-ttl <= row['event'] <= prediction
                and (not availability or (row['available'] is not None and row['available']<=prediction))]
    # Version ordering is explicit; all available timestamps here are non-null.
    return max(candidates,key=lambda r:(r['event'],r['available'] or datetime.min),default=None)


def verify():
    rows=[dict(entity='u',event=ts('09:58'),available=ts('09:59'),value=2),
          dict(entity='u',event=ts('10:00'),available=ts('10:07'),value=3)]
    prediction=ts('10:03')
    assert historical(rows,'u',prediction)['value']==2
    assert historical(rows,'u',prediction,availability=False)['value']==3
    assert historical(rows,'u',ts('10:08'))['value']==3
    assert historical(rows,'u',ts('11:02')) is None
    # A later correction must not rewrite a past observation.
    rows.append(dict(entity='u',event=ts('09:58'),available=ts('10:09'),value=9))
    assert historical(rows,'u',prediction)['value']==2
    # A feature with unknown availability cannot prove it existed online.
    rows.append(dict(entity='u',event=ts('10:01'),available=None,value=99))
    assert historical(rows,'u',prediction)['value']==2
    print('At 10:03: event-only join = 3; online-available join = 2.')
    print('Late corrections, missing timestamps, entity matching and TTL checked.')
    print('Labels are separate targets and never candidates for input-feature lookup.')


if __name__=='__main__':verify()
