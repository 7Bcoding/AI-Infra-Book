"""Run with python verify.py. All numerical examples use only the standard library."""
from math import ceil, exp, isclose, log, sqrt
from random import Random
from struct import pack, unpack

X = [[1, 2], [3, 4]]
W = [[2, -1], [1, 3]]
BIAS = [-6, 1]


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b, strict=True))


def matmul(a, b):
    columns = transpose(b)
    return [[dot(row, col) for col in columns] for row in a]


def close(a, b):
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y)
    else:
        assert isclose(a, b, abs_tol=1e-9, rel_tol=1e-10), (a, b)


def tiled(a, b, m, n, k):
    """Exact valid element requests at the panel boundary, ignoring inter-block caches."""
    rows, cols, depth = len(a), len(b[0]), len(b)
    out = [[0]*cols for _ in a]
    reads = 0
    for i in range(0, rows, m):
        for j in range(0, cols, n):
            rr, cc = min(m, rows-i), min(n, cols-j)
            acc = [[0]*cc for _ in range(rr)]
            for start in range(0, depth, k):
                length = min(k, depth-start)
                left = [row[start:start+length] for row in a[i:i+rr]]
                right = [row[j:j+cc] for row in b[start:start+length]]
                reads += (rr+cc)*length
                chunk = matmul(left, right)
                for r in range(rr):
                    for c in range(cc):
                        acc[r][c] += chunk[r][c]
            for r, row in enumerate(acc):
                out[i+r][j:j+cc] = row
    return out, reads


def pipeline(stages, count, copy, compute, issue_interval=None):
    """A slot is reusable when its computation finishes; copies may be pipelined."""
    if issue_interval is None:
        issue_interval = copy
    free = [0]*stages
    next_issue = compute_end = 0
    records = []
    for i in range(count):
        buf = i % stages
        copy_start = max(next_issue, free[buf])
        next_issue = copy_start + issue_interval
        copy_end = copy_start + copy
        compute_start = max(copy_end, compute_end)
        compute_end = compute_start + compute
        free[buf] = compute_end
        records.append((copy_start, copy_end, compute_start, compute_end))
    return records


def check_gemm(rng):
    a = [[1,2,1,0],[3,4,0,1],[0,1,2,1],[2,0,1,3]]
    b = [[2,-1,1,0],[1,3,0,1],[1,2,1,-1],[-1,1,2,1]]
    expected = matmul(a,b)
    close([row[:2] for row in expected[:2]], [[5,7],[9,10]])
    close(matmul(X,W), [[4,5],[10,9]])
    for side, traffic in [(1,576),(2,320),(4,192)]:
        out, reads = tiled(a,b,side,side,2)
        close(out, expected)
        assert 4*(reads+16) == traffic
    for _ in range(50):
        rows, cols, depth = (rng.randint(1,8) for _ in range(3))
        inputs = [[rng.randint(-5,5) for _ in range(depth)] for _ in range(rows)]
        weights = [[rng.randint(-5,5) for _ in range(cols)] for _ in range(depth)]
        dims = [rng.randint(1,6) for _ in range(3)]
        close(tiled(inputs,weights,*dims)[0], matmul(inputs,weights))
    record = pipeline(2,4,5,3)
    assert record == [(0,5,5,8),(5,10,10,13),(10,15,15,18),(15,20,20,23)]
    assert pipeline(1,4,5,3)[-1][-1] == 32
    assert pipeline(3,4,5,3)[-1][-1] == 23
    runs = [pipeline(slots, 8, copy=9, compute=3, issue_interval=2)
            for slots in (1, 2, 3, 4)]
    assert [run[-1][-1] for run in runs] == [96, 51, 39, 33]
    assert [[step[0] for step in run[:4]] for run in runs] == [
        [0,12,24,36], [0,2,12,15], [0,2,4,12], [0,2,4,6]]
    for k, bytes_, flops in [(16,6144,262144),(32,12288,524288)]:
        assert 2*(64+128)*k == bytes_ and 2*64*128*k == flops
        close(flops/bytes_, 128/3)
    assert 2*(128+128)*32 == 16384 and 2*128*128*32/16384 == 64
    assert [96//(8+16*s) for s in (2,3)] == [2,1]
    assert [(r+c,r*c) for r,c in [(1,4),(2,2)]] == [(5,4),(4,4)]
    for m, useful in [(5,75),(6,90)]:
        assert m*3*5 == useful and ceil(m/2)*ceil(3/2)*ceil(5/2)*8 == 144
    assert 4*64*4 == 1024 and 2*1024 == 2048
    assert max(0,-3+5-1) == 1 and sum(max(0,x-1) for x in (-3,5)) == 4
    assert sum([1,8,3]) == 12 and 3*max([1,8,3]) == 24
    assert [dot(row,[1,-1]) for row in ([1,2],[2,3],[3,4])] == [-1]*3
    rows = 128
    side = 4096
    blocks = side//128
    panel_x, weights, output = rows*side*2, side**2*2, rows*side*2
    assert (blocks*panel_x+weights+output)//2**20 == 65
    assert (panel_x+weights+output)//2**20 == 34


def summarize(scores, values):
    if not scores:
        return None
    origin = max(scores)
    mass = [exp(s-origin) for s in scores]
    return origin, sum(mass), dot(mass, values)


def merge(a, b):
    if a is None: return b
    if b is None: return a
    m = max(a[0], b[0])
    scale_a, scale_b = exp(a[0]-m), exp(b[0]-m)
    return m, a[1]*scale_a+b[1]*scale_b, a[2]*scale_a+b[2]*scale_b


def attend(q, keys, values):
    scores = [dot(q,k)/sqrt(len(q)) for k in keys]
    m, den, num = summarize(scores, values)
    probs = [exp(s-m)/den for s in scores]
    return num/den, probs, scores


def fd(fn, array, index, eps=1e-5):
    old = array[index]
    try:
        array[index] = old + eps
        right = fn()
        array[index] = old - eps
        left = fn()
    finally:
        array[index] = old
    return (right-left)/(2*eps)


def check_attention(rng):
    q = [4.,5.]
    keys = [[0.,0.],[0.,0.],[sqrt(2)*log(4)/4,0.],[sqrt(2)*log(4)/4,0.]]
    vals = [0.,2.,8.,10.]
    out, p, scores = attend(q,keys,vals)
    close([out,*p],[7.4,.1,.1,.4,.4])
    for cut in range(5):
        m, den, num = merge(summarize(scores[:cut],vals[:cut]),summarize(scores[cut:],vals[cut:]))
        close([m,den,num,m+log(den)],[log(4),2.5,18.5,log(10)])
    first = summarize(scores[:3], vals[:3])
    close(first,[log(4),1.5,8.5])
    close(first[2]/first[1],17/3)
    assert merge(None,None) is None  # caller handles a fully masked row before division
    ds = [v_prob*(v-out) for v_prob,v in zip(p,vals)]
    close(ds,[-.74,-.54,.24,1.04]); close(sum(ds),0)
    close([2*p[2],2*p[3],2*ds[2],2*ds[3]],[.8,.8,.48,2.08])
    read = lambda: attend(q,keys,vals)[0]
    dq = [dot(ds,column)/sqrt(2) for column in transpose(keys)]
    dk = [[ds[i]*qj/sqrt(2) for qj in q] for i in range(len(keys))]
    diffs = []
    for array, grads in [(q,dq),(vals,p),*zip(keys,dk)]:
        diffs.extend(abs(fd(read,array,j)-grad) for j,grad in enumerate(grads))
    assert max(diffs) < 1e-7
    for _ in range(100):
        count = rng.randint(1,21)
        scores = [rng.uniform(-90,90)+1000 for _ in range(count)]
        values = [rng.uniform(-30,30) for _ in scores]
        ref = summarize(scores,values)
        blocks = []
        start = 0
        while start<count:
            end = start+rng.randint(1,5)
            blocks.append(summarize(scores[start:end],values[start:end]))
            start = end
        rng.shuffle(blocks)
        state = None
        for item in blocks:
            state = merge(state,item)
        close(state,ref)
    assert 8192**2*2 == 128*2**20 and 8192*4 == 32*2**10
    assert (8192**2*2)//(8192*4) == 4096
    page_sets = [{j//4 for j in selection} for selection in ((1,6,7),(0,1,2))]
    assert sum(map(len,page_sets))*4 == 12 and len(set.union(*page_sets))*4 == 8
    a = lambda s: s/2+1
    b = lambda s: s/4+2
    assert b(a(0))==2.25 and a(b(0))==2
    return max(diffs)


def half(value):
    return unpack("e",pack("e",value))[0]


def dequant(values):
    scale = max(map(abs,values))/3 or 1
    return [scale*max(-3,min(3,round(v/scale))) for v in values]


def check_precision():
    wo = [[2,-1],[1,12]]
    whole = dequant(wo[0]+wo[1])
    bycol = transpose([dequant(c) for c in transpose(wo)])
    close(matmul(X,wo),[[4,23],[10,45]])
    close(matmul(X,[whole[:2],whole[2:]]),[[0,24],[0,48]])
    close(matmul(X,bycol),[[14/3,24],[34/3,48]])
    close(dequant([0,0]),[0,0])
    close([dot([2,2],[1,4]),dot([4,0],[1,4])],[10,4])
    close(3*.4*(0-5),-6)
    assert .4*(5-5) == 0
    assert half(1+2**-11) == 1 and half(2049)==2048
    assert 2**(8-7)==2 and 256+2==258 and 2*round(257/2)==256
    direct = master = 1.
    published = []
    for _ in range(4):
        direct = half(direct-.0001)
        master -= .0001
        published.append(half(master))
    assert direct==1. and published==[1.,1.,1.-2**-11,1.-2**-11]
    n = 4096**2
    assert n//64==262144
    close([(n//2+2*(n//g))/2**20 for g in (128,64,32)],[8.25,8.5,9.])
    close(round(6+4/3.88+.6,2),7.63)
    assert n//2 + n//16 + 4 == 9*2**20+4
    samples = [1]*100+[12]
    loss = lambda step: sum((v-step*min(3,round(v/step)))**2 for v in samples)/len(samples)
    close([loss(4),loss(1)],[100/101,81/101])
    close(dot([1,1,4,8],[0,2,8,10])/14,57/7)
    close(dot([1,1,4,4],[0,2,8,11])/10,7.8)
    assert half(2**-26)==0 and half((2**-26)*(2**13))/(2**13)==2**-26


def layer(x,w,b):
    return [[max(0,p+offset) for p,offset in zip(row,b)] for row in matmul(x,w)]


def check_execution():
    y = layer(X,W,BIAS)
    close(y,[[0,6],[4,10]])
    slope = [[int(v>0) for v in row] for row in y]
    dx, dw = matmul(slope,transpose(W)), matmul(transpose(X),slope)
    db = list(map(sum,transpose(slope)))
    close(dx,[[-1,3],[1,4]]); close(dw,[[3,4],[4,6]]); close(db,[1,2])
    loss = lambda: sum(map(sum,layer(X,W,BIAS)))
    errors = []
    for array, grads in [*zip(X,dx),*zip(W,dw),(BIAS,db)]:
        errors.extend(abs(fd(loss,array,j)-grad) for j,grad in enumerate(grads))
    assert max(errors)<1e-7
    w_new = [[v-.1*g for v,g in zip(row,grad)] for row,grad in zip(W,dw)]
    updated = layer(X,w_new,[v-.1*g for v,g in zip(BIAS,db)])
    close(updated,[[0,4.2],[1.4,6.2]])
    close(sum(map(sum,updated)),11.8)
    close(layer(transpose(X),W,BIAS),[[0,9],[2,11]])
    p = matmul(X,W)
    assert sum(map(sum,p))==28 and sum(v+BIAS[j] for row in p for j,v in enumerate(row))==18
    close(matmul(transpose([[1,3],[3,4]]),slope),[[3,4],[4,7]])
    close([g+g*.5 for g in db],[1.5,3])
    assert 0*1+1*2==2 and (X[1][0]==3)
    assert not (7 >= 8) and 8 >= 6+2
    intermediate = 128*4096*2
    assert 4*intermediate==4*2**20 and 3*intermediate==3*2**20
    acc = 1+2**-11
    assert max(0,half(acc)-1)==0 and half(max(0,acc-1))==2**-11
    row = [1,2]
    view, copied = row, row.copy()
    for j in range(2): row[j]+=1
    assert sum(view)==5 and sum(copied)==3
    assert [30+8,43,32]==[38,43,32]
    compiled = lambda n: 56000+42*n
    eager = lambda n: 70*n
    assert compiled(2000)==eager(2000) and compiled(2001)<eager(2001)
    assert 3*compiled(1000)==294000 and eager(3000)==210000 and 56000+50*3000==206000
    assert compiled(2200)+eager(200)==162400 and eager(2400)==168000
    return max(errors)


if __name__ == "__main__":
    rng = Random(20261111)
    check_gemm(rng)
    attention_error = check_attention(rng)
    check_precision()
    layer_error = check_execution()
    print("PASS: 50 tiled GEMMs, 100 shuffled attention reductions, chapter calculations")
    print(f"Maximum central-difference error: attention {attention_error:.3g}, layer {layer_error:.3g}")

