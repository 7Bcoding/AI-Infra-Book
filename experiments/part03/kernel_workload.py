"""Weighted kernel timing examples; supplied teaching times, not GPU results."""
from math import isclose


def compare(counts, old, new):
    assert len(counts)==len(old)==len(new)
    assert all(n>=0 for n in counts) and sum(counts)>0
    assert all(t>0 for t in old+new)
    before=sum(n*t for n,t in zip(counts,old))
    after=sum(n*t for n,t in zip(counts,new))
    return dict(old_us=before,new_us=after,speedup=before/after,saved_fraction=1-after/before)


def verify():
    a=compare([80,15,5],[10,40,100],[12,32,60])
    b=compare([95,4,1],[10,40,100],[12,32,60])
    assert (a['old_us'],a['new_us'])==(1900,1740)
    assert (b['old_us'],b['new_us'])==(1210,1328)
    assert a['speedup']>1>b['speedup']
    assert 14<20<14+4+5
    assert isclose(160/200,.8) and isclose(160/(100*200),.008)
    print('Original workload:',a)
    print('Single-row-heavy workload:',b)
    print('Complete-operator comparison: 20 us versus 23 us.')
    print('Use measure_projection.py for optional real CUDA measurements.')


if __name__=='__main__':verify()
