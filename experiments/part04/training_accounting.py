"""MFU/HFU, distributed matrix updates and sparse table routing examples."""
from math import sqrt, isclose
from collections import defaultdict


def transpose(a):return list(map(list,zip(*a)))
def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def verify():
    model=6*500_000_000*8192
    peak=8*120e12
    metrics=[]
    for seconds,extra in [(.040,0),(.050,.5)]:
        metrics.append((model/(peak*seconds),model*(1+extra)/(peak*seconds)))
    for got,want in zip(metrics,[(.64,.64),(.512,.768)]):
        assert all(isclose(a,b) for a,b in zip(got,want))
    g=[[1.,1.],[0.,1.]]
    q=[[2/sqrt(5),1/sqrt(5)],[-1/sqrt(5),2/sqrt(5)]]
    qtq=mm(transpose(q),q)
    assert all(isclose(qtq[i][j],int(i==j),abs_tol=1e-12) for i in range(2) for j in range(2))
    h=mm(transpose(q),g)
    assert isclose(h[0][1],h[1][0])
    assert h[0][0]>0 and h[0][0]*h[1][1]-h[0][1]*h[1][0]>0
    local=[[1,1/sqrt(2)],[0,1/sqrt(2)]]
    assert isclose(mm(transpose(local),local)[0][1],1/sqrt(2))
    ids=[1,6,6,2]
    table=[[row]*128 for row in range(8)]
    remote_ids=sorted({row for row in ids if row>=4})
    replies={row:table[row] for row in remote_ids}
    recovered=[table[row] if row<4 else replies[row] for row in ids]
    assert recovered==[table[row] for row in ids]
    gradients=[(0,0),(1,2),(3,-1),(0,0)]
    summed=defaultdict(lambda:[0,0])
    for row,grad in zip(ids,gradients):
        for dim,v in enumerate(grad):summed[row][dim]+=v
    assert summed[6]==[4,1]
    assert len(remote_ids)*8==8 and len(remote_ids)*128*2==256
    assert 100_000_000*128*14==179_200_000_000
    print('MFU/HFU:',metrics)
    print('Global polar direction:',q,'Independent column update:',local)
    print('Unique remote IDs:',remote_ids,'Row 6 gradient:',summed[6])
    print('Sparse lookup and dense expert compute require different bottleneck metrics.')


if __name__=='__main__':verify()
