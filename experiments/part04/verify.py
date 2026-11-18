"""Recompute Part IV on any computer with Python 3. No other files needed."""
from math import ceil, exp, isclose, log, sqrt
from random import Random
from struct import pack, unpack

MIB = 2**20
GIB = 2**30
BW = 40e9
X = [[1,2],[3,4]]
W = [[2,-1],[1,3]]


def transpose(a):
    return [list(col) for col in zip(*a)]


def mm(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in transpose(b)] for row in a]


def close(actual, expected):
    if isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):
            close(a,b)
    else:
        assert isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9),(actual,expected)


def network():
    for k,expected in enumerate(([[2,-1],[6,-3]],[[2,6],[4,12]])):
        close(mm([[row[k]] for row in X],[W[k]]),expected)
    close(mm(X,W),[[4,5],[10,9]])
    service = lambda n: n/BW*1e6
    close([3+service(MIB),3+service(2*MIB)],[29.2144,55.4288])
    close(2*MIB/32e9*1e6+service(MIB),91.7504)
    close(16*3+service(MIB),74.2144)
    slope=(MIB-65536)/((29.2144-4.6384)*1e-6)
    close([slope/1e9,4.6384-65536/slope*1e6],[40,3])
    window = BW*10e-6
    close([window/1024,256*1024/10e-6/1e9],[390.625,26.2144])
    assert ceil(window/65536)==7 and 2*MIB>BW*30e-6
    peak = (4*BW-BW)*MIB/BW
    close([peak/MIB, service(peak), service(4*MIB)],[3,78.6432,104.8576])
    assert not (8>=10) and 7<10  # receive-ready, old consumer still active


def ring(rows):
    """p equal-sized chunks; snapshot sends before applying receives in each round."""
    p,n = len(rows),len(rows[0])
    assert n%p==0 and all(len(row)==n for row in rows)
    size=n//p
    data=[[row[i:i+size] for i in range(0,n,size)] for row in rows]
    trace=[]
    for phase in ("reduce","gather"):
        for turn in range(p-1):
            chunks=[(r-turn+(phase=="gather"))%p for r in range(p)]
            messages=[data[r][chunks[r]][:] for r in range(p)]
            for sender,(idx,msg) in enumerate(zip(chunks,messages)):
                receiver=(sender+1)%p
                if phase=="reduce":
                    data[receiver][idx]=[a+b for a,b in zip(data[receiver][idx],msg)]
                else:
                    data[receiver][idx]=msg
            trace.append((phase,chunks,[data[(r+1)%p][idx][:] for r,idx in enumerate(chunks)]))
    return [sum(chunks,[]) for chunks in data],trace


def finishes(ready,duration):
    end=0
    ends=[]
    for release in ready:
        end=max(end,release)+duration
        ends.append(end)
    return ends


def collective():
    rows=[[0,1,0,2],[3,3,4,4]]*2
    result,trace=ring(rows)
    close(result,[[6,8,8,12]]*4)
    for chunk, totals in [(0,[3,3,6]),(2,[4,4,8])]:
        assert [m[2][m[1].index(chunk)][0] for m in trace[:3]]==totals
    rng=Random(2026)
    for p in range(2,9):
        for _ in range(12):
            rows=[[rng.randrange(-5,6) for _ in range(p*3)] for _ in range(p)]
            result,_=ring(rows)
            close(result,[list(map(sum,zip(*rows)))]*p)
    assert 2*3*32/4==48 and 3*32/4==24
    ring_us = lambda n: 18+1.5*n/BW*1e6
    tree_us = lambda n: 12+4*n/BW*1e6
    close([ring_us(96000),tree_us(96000)],[21.6,21.6])
    close(ring_us(32*MIB),1276.2912)
    assert finishes([4,8,12,16],3)==[7,11,15,19]
    assert finishes([4,8,12,18],3)[-1]==21 and 16+.5+4*2.5==26.5
    close(sum(a*b for a,b in zip((2,2,4),(1,3,6)))/8,4.)
    counts=[[1,2,0,0],[0,1,1,0],[2,0,1,0],[0,0,1,1]]
    assert list(map(sum,zip(*counts)))==[3,3,3,1]
    assert sum(sum(r[2:]) for r in counts[:2])==1
    assert sum(sum(r[:2]) for r in counts[2:])==2
    narrow=lambda x: unpack("f",pack("f",x))[0]
    assert narrow(narrow(2**24+1)-2**24)==0 and narrow((2**24-2**24)+1)==1


FILL = (["F0","F1","F2","F3","--","--","B3","B2","B1","B0"],
        ["--","F0","F1","F2","F3","B3","B2","B1","B0","--"])
ALTERNATE = (["F0","F1","--","B0","F2","B1","F3","B2","--","B3"],
             ["--","F0","B0","F1","B1","F2","B2","F3","B3","--"])


def check_schedule(lanes):
    # Every dependent slot must start after the producer's slot ends.
    a,b=lanes
    for micro in range(4):
        f,rev = "F"+str(micro),"B"+str(micro)
        assert a.index(f)<b.index(f)<b.index(rev)<a.index(rev)
    active=peak=0
    for item in a:
        active+=(item[0]=="F")-(item[0]=="B")
        peak=max(peak,active)
    assert active==0
    return peak


def parallelism():
    p=32*4096**2
    close([p*n/GIB for n in [16,4+12/4,2+14/4,16/4]],[8,3.5,2.75,2])
    close((2+(4+12)/4)*p/GIB,3)
    assert 3/4*(2+1)==2.25
    h=[[0,6],[4,10]]
    close(mm(h,[[1,2],[-1,1]]),[[-6,6],[-6,18]])
    close([2*a+3*b for a,b in zip([2,1],[-1,3])],[1,11])
    assert check_schedule(FILL)==4 and check_schedule(ALTERNATE)==2
    close(8/(8+2-1),8/9)
    assert (8+2-1)*2==18 and 2*8==16
    assert 32/4+3==11 and 8*3==24
    close(2*MIB/32e9*1e6-50,15.536)
    assert 2*4*128==1024 and 16*2*4==128
    close(GIB/BW*1000,26.8435456)
    assert 2*128*2==512 and 2*32==64


def moe_loss(p):
    w,b,logits,e0,e1=p
    projected=mm(X,[w[:2],w[2:]])
    h=[[max(0,z+b[j]) for j,z in enumerate(row)] for row in projected]
    outputs=[]
    for i,row in enumerate(h):
        z=logits[2*i:2*i+2]
        raw=[exp(v-max(z)) for v in z]
        weights=[v/sum(raw) for v in raw]
        replies=[mm([row],[e[:2],e[2:]])[0] for e in (e0,e1)]
        outputs.append([sum(weights[j]*replies[j][col] for j in range(2)) for col in range(2)])
    return sum(row[0] for row in outputs),outputs


def experts():
    params=[[2.,-1.,1.,3.],[-6.,1.],[log(3),0.,0.,log(3)],[1.,0.,0.,1.],[0.,1.,1.,0.]]
    expected=[[.75,2.5,1.,3.5],[.25,1.],[-1.125,1.125]*2,[1.,0.,7.,0.],[3.,0.,9.,0.]]
    loss,outputs=moe_loss(params)
    close(loss,10); close(outputs,[[1.5,4.5],[8.5,5.5]])
    # Four available experts. Only the two selected per token are executed.
    matrices=[[[1,0],[0,1]],[[0,1],[1,0]],[[2,0],[0,2]],[[1,0],[0,3]]]
    selection=[[(0,.75),(1,.25)],[(0,.25),(1,.75)],[(0,.5),(2,.5)],[(2,.25),(3,.75)]]
    sparse=[]
    crossed=0
    for i,(row,chosen) in enumerate(zip([[0,6],[4,10],[2,4],[1,3]],selection)):
        output=[0.,0.]
        for e,p in chosen:
            crossed+= e%2 != i%2
            reply=mm([row],matrices[e])[0]
            for j,value in enumerate(reply):
                output[j]+=p*value
        sparse.append(output)
    close(sparse,[[1.5,4.5],[8.5,5.5],[3,6],[1.25,8.25]])
    assert crossed==3
    errors=[]
    for group,grads in zip(params,expected):
        for j,g in enumerate(grads):
            v=group[j]
            group[j]=v+1e-5
            plus=moe_loss(params)[0]
            group[j]=v-1e-5
            minus=moe_loss(params)[0]
            group[j]=v
            errors.append(abs((plus-minus)/2e-5-g))
    assert max(errors)<1e-7
    close([v-g/8 for v,g in zip(params[0],expected[0])],[1.90625,-1.3125,.875,2.5625])
    payload=4096*2
    close(64*payload/MIB,.5); close(2*96*payload/MIB,1.5)
    assert 4*2==8 and 4*.5==2 and 4096**2*16/MIB==256
    loads=[5,1,1,1]
    assert [sum(min(n,cap) for n in loads) for cap in (2,3,4,5)]==[5,6,7,8]
    close([sum(loads)/(4*max(loads)),sum((n/8)**2 for n in loads)*4],[.4,1.75])
    ends=[0,0,0]
    for _ in range(2):
        ends[0]+=2
        ends[1]=max(ends[:2])+3
        ends[2]=max(ends[1:])+1
    assert ends==[4,8,9]
    return max(errors)


def recovery_budget():
    triangular=lambda n:n*(n+1)//2
    assert [triangular(8)-triangular(a)-triangular(8-a) for a in (5,6)]==[15,12]
    assert max(12-2*4,0)==4
    payload=(32+2)*4096**2*12/GIB
    close([payload,payload/4,payload/3],[6.375,1.59375,2.125])
    assert sqrt(2*4*8*3600)==480
    close(4/480+480/57600,1/60)
    close(4/960+960/57600,1/48)
    close((105-5)*1024/60,1706+2/3)
    assert 105*1024/60==1792
    assert 512+32==544 and 128+8+2==138
    close(2*544*MIB/BW*1000,28.5212672)


if __name__=="__main__":
    network()
    collective()
    parallelism()
    error=experts()
    recovery_budget()
    print("PASS: transfer, 84 randomized rings, parallel schedules and memory, traffic, checkpoints.")
    print(f"All 18 MoE parameter gradients checked; maximum absolute difference {error:.3g}.")
