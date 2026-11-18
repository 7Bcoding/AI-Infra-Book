"""Independent numerical checks for chapter 5, using only the standard library.

All vectors are rows; recurrent states have shape [key features, value features].
These are correctness experiments, not optimized kernels or performance tests.
"""
import copy
import json
import math
from pathlib import Path
import random
import re

ROOT=Path(__file__).resolve().parents[2]
D=json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))


def dot(a,b):return sum(x*y for x,y in zip(a,b))
def rowmul(x,w):return [sum(x[i]*w[i][j] for i in range(len(x))) for j in range(len(w[0]))]
def transpose(w):return [list(row) for row in zip(*w)]
def matmul(a,b):return [rowmul(row,b) for row in a]
def outer(k,v):return [[x*y for y in v] for x in k]
def add(a,b):return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]
def scale(a,c):return [[c*x for x in row] for row in a]
def eye(n):return [[float(i==j) for j in range(n)] for i in range(n)]
def zeros(m,n):return [[0.0]*n for _ in range(m)]


def softmax(s):
    e=[math.exp(x-max(s)) for x in s]
    return [x/sum(e) for x in e]


def attention(q,k,v,denominator=math.sqrt(2),position_scores=None):
    raw=[dot(q,x) for x in k]
    if position_scores is not None:raw=[a+b for a,b in zip(raw,position_scores)]
    scores=[x/denominator for x in raw]
    weights=softmax(scores)
    return dict(scores=scores,weights=weights,output=rowmul(weights,v))


def close(a,b,tol=1e-10):
    if isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y,tol)
    else:assert abs(a-b)<tol,(a,b)


def mla(c,q,uk,uv):
    # Keep the original score scale after absorption, not the latent width.
    denominator=math.sqrt(len(q))
    expanded=attention(q,matmul(c,uk),matmul(c,uv),denominator)
    absorbed_q=rowmul(q,transpose(uk))
    compressed=attention(absorbed_q,c,c,denominator)
    recovered=rowmul(compressed['output'],uv)
    close(expanded['scores'],compressed['scores'])
    close(expanded['weights'],compressed['weights'])
    close(expanded['output'],recovered)
    return dict(expanded=expanded,absorbed_query=absorbed_q,latent_output=compressed['output'],output=recovered)


def sparse_read(indices):
    s=D['sparse'];e=s['exponentials'];v=s['values']
    weights=[e[i]/sum(e[j] for j in indices) for i in indices]
    return dict(indices=indices,weights=weights,output=sum(w*v[i] for i,w in zip(indices,weights)),mass=sum(e[i] for i in indices)/sum(e))


def linear_stream(keys,values,queries):
    state=zeros(len(keys[0]),len(values[0]));normalizer=[0.0]*len(keys[0]);trace=[]
    for k,v,q in zip(keys,values,queries):
        state=add(state,outer(k,v))
        normalizer=[x+y for x,y in zip(normalizer,k)]
        denom=dot(q,normalizer)
        assert denom>0
        out=[x/denom for x in rowmul(q,state)]
        trace.append(dict(state=copy.deepcopy(state),normalizer=normalizer.copy(),output=out))
    return trace


def linear_direct(keys,values,queries):
    outputs=[]
    for t,q in enumerate(queries):
        coeffs=[dot(q,k) for k in keys[:t+1]]
        outputs.append([x/sum(coeffs) for x in rowmul(coeffs,values[:t+1])])
    return outputs


def linear_chunk(keys,values,queries,chunk_size):
    f=len(keys[0]);dv=len(values[0]);state=zeros(f,dv);z=[0.0]*f;outputs=[]
    for start in range(0,len(keys),chunk_size):
        k=keys[start:start+chunk_size];v=values[start:start+chunk_size];q=queries[start:start+chunk_size]
        local=matmul(q,transpose(k))
        local=[[a if j<=i else 0.0 for j,a in enumerate(row)] for i,row in enumerate(local)]
        numerator=add(matmul(q,state),matmul(local,v))
        denoms=[dot(row,z)+sum(local[i]) for i,row in enumerate(q)]
        outputs.extend([[x/denoms[i] for x in row] for i,row in enumerate(numerator)])
        state=add(state,matmul(transpose(k),v))
        z=[z[j]+sum(row[j] for row in k) for j in range(f)]
    return outputs


def delta(state,k,v,beta=1.0,decay=None):
    decay=[1.0]*len(k) if decay is None else decay
    decayed=[[x*decay[i] for x in row] for i,row in enumerate(state)]
    prediction=rowmul(k,decayed)
    error=[a-b for a,b in zip(v,prediction)]
    out=add(decayed,scale(outer(k,error),beta))
    return dict(decayed=decayed,prediction=prediction,error=error,state=out)


def affine(k,v,beta,decay):
    transition=add(eye(len(k)),scale(outer(k,k),-beta))
    transition=[[x*decay[j] for j,x in enumerate(row)] for row in transition]
    return transition,scale(outer(k,v),beta)


def compose(later,earlier):
    a2,b2=later;a1,b1=earlier
    return matmul(a2,a1),add(matmul(a2,b1),b2)


def budget():
    c=D['budget'];L=c['layers'];h=c['head_dim'];b=c['cache_bytes'];t=c['long_context']
    per={name:2*L*g*h*b for name,g in [('MHA',c['query_heads']),('GQA',c['kv_heads']),('MQA',1)]}
    per['MLA']=L*(c['latent_dim']+c['position_dim'])*b
    n=c['generated_choices']-1
    cumulative=per['GQA']*(n*t+n*(n+1)//2)
    state_per_layer=c['recurrent_heads']*c['state_key_dim']*c['state_value_dim']*c['state_bytes']
    recurrent=c['recurrent_layers'];att_layers=L-recurrent
    cache_per_layer_token=per['GQA']//L
    hybrid={str(T):att_layers*T*cache_per_layer_token+recurrent*state_per_layer for T in [512,4096,131072]}
    return dict(bytes_per_token=per,long_cache_bytes={k:v*t for k,v in per.items()},
                cumulative_decode_read_bytes=cumulative,decode_steps=n,
                history_read_milliseconds=per['GQA']*t/c['hypothetical_bandwidth_bytes_per_second']*1000,
                state_bytes_per_layer=state_per_layer,all_recurrent_bytes=L*state_per_layer,
                hybrid_bytes=hybrid,state_cache_crossover=state_per_layer//cache_per_layer_token,
                dense_attention_flops_per_decode=4*L*c['query_heads']*h*t,
                sparse_selected_cache_bytes=per['GQA']*2048)


def verify(markdown=True):
    base=math.sqrt(2)*math.log(2);k=D['shared_keys'];v=D['shared_values']
    qs=[[base,0],[0,base]]
    gqa=[attention(q,k,v) for q in qs]
    close(gqa[0]['weights'],[.4,.2,.4]);close(gqa[0]['output'],[1.6,2.4])
    close(gqa[1]['weights'],[.2,.4,.4]);close(gqa[1]['output'],[1.2,3.2])
    c=D['mla'];mlas=[mla(c['latents'],q,uk,uv) for q,uk,uv in zip(qs,c['key_up'],c['value_up'])]
    # Figure 5-2 holds the query and the three original records fixed.
    query = qs[0]
    reads = {
        'grouped': gqa[0]['output'],
        'latent': mlas[0]['output'],
        'last_two': attention(query, k[-2:], v[-2:])['output'],
        'linear': linear_stream(k, v, [query] * len(k))[-1]['output'],
    }
    close(list(reads.values()), [[8/5, 12/5], [8/5, 12/5], [4/3, 4], [2, 2]])
    close(mlas[0]['output'],[1.6,2.4]);close(mlas[1]['output'],[2.25,0])
    close([sum(m['output'][i] for m in mlas) for i in range(2)],[3.85,2.4])
    # Decoupled RoPE: content and position scores share the sqrt(2+2) scale.
    pos_keys=[[1,0],[0,1],[-1,0]];pos_q=[-2*math.log(2),0];cq=[2*math.log(2),0]
    pos=attention(cq,k,v,2,[dot(pos_q,p) for p in pos_keys])
    close(pos['weights'],[1/6,1/6,2/3]);close(pos['output'],[5/3,10/3])
    u=[[2,0],[0,1]];rot=[[0,-1],[1,0]]
    assert matmul(u,rot)!=matmul(rot,u)
    sparse={name:sparse_read(ids) for name,ids in [('full',list(range(8))),('window',D['sparse']['window']),('top_three',D['sparse']['top_three']),('block_and_recent',D['sparse']['selected_block_and_recent'])]}
    for name,expected in [('full',84/17),('window',.8),('top_three',7),('block_and_recent',84/13)]:close(sparse[name]['output'],expected)
    for name in ['window','top_three','block_and_recent']:
        actual=abs(sparse[name]['output']-sparse['full']['output'])
        bound=(1-sparse[name]['mass'])*10
        assert actual<=bound+1e-12
    block_ids=sorted({i//2 for i in D['sparse']['scattered_tokens']})
    loaded=[j for i in block_ids for j in range(i*2,i*2+2)]
    assert loaded==[0,1,4,5,6,7]
    queries=[[1,0],[0,1],[1,0]]
    stream=linear_stream(k,v,queries)
    close(stream[-1]['state'],[[4,4],[2,8]]);close(stream[-1]['normalizer'],[2,2]);close(stream[-1]['output'],[2,2])
    close(stream[0]['output'],[2,0])
    leaked=[x/dot(queries[0],stream[-1]['normalizer']) for x in rowmul(queries[0],stream[-1]['state'])]
    close(leaked,[2,2]);assert leaked!=stream[0]['output']
    direct=linear_direct(k,v,queries)
    close([r['output'] for r in stream],direct)
    for size in [1,2,3,4]:close(linear_chunk(k,v,queries,size),direct)
    coll={key:linear_stream(k,values,queries)[-1] for key,values in D['collision'].items()}
    close(coll['values_a']['state'],coll['values_b']['state'])
    close(coll['values_a']['normalizer'],coll['values_b']['normalizer'])
    soft_collision=[attention(qs[0],k,values)['output'] for values in D['collision'].values()]
    close(soft_collision,[[4,0],[4.2,0]])
    update=delta([[2],[4]],[1,0],[6])
    close(update['state'],[[6],[4]])
    diag=delta([[4],[8]],[1,0],[6],.5,[1,.25])
    scalar=delta([[4],[8]],[1,0],[6],.5,[.5,.5])
    close(diag['state'],[[5],[2]]);close(scalar['state'],[[4],[4]])
    diagonal_key=[1/math.sqrt(2)]*2
    interference=delta([[2],[4]],diagonal_key,[6])
    close(interference['state'],[[3*math.sqrt(2)-1],[3*math.sqrt(2)+1]])
    # Check the reconstruction-loss gradient which gives the delta update.
    state=[[.4,-.2],[.7,.9]];kt=[.6,.8];vt=[1.1,-.5]
    residual=[a-b for a,b in zip(rowmul(kt,state),vt)]
    grad=outer(kt,residual);eps=1e-6;grad_error=0
    def objective(st):return .5*sum((a-b)**2 for a,b in zip(rowmul(kt,st),vt))
    for i in range(2):
        for j in range(2):
            plus,minus=copy.deepcopy(state),copy.deepcopy(state)
            plus[i][j]+=eps;minus[i][j]-=eps
            err=abs((objective(plus)-objective(minus))/(2*eps)-grad[i][j])
            grad_error=max(grad_error,err)
    assert grad_error<1e-8
    # Nontrivial randomized shapes check absorption, causality, and chunking.
    rng=random.Random(205)
    max_absorption_error=0;max_chunk_error=0;max_affine_error=0
    for _ in range(30):
        latent=[[rng.uniform(-1,1) for _ in range(3)] for _ in range(7)]
        uk=[[rng.uniform(-1,1) for _ in range(4)] for _ in range(3)]
        uv=[[rng.uniform(-1,1) for _ in range(2)] for _ in range(3)]
        q=[rng.uniform(-1,1) for _ in range(4)]
        result=mla(latent,q,uk,uv)
        max_absorption_error=max(max_absorption_error,max(abs(x-y) for x,y in zip(result['output'],result['expanded']['output'])))
        pk=[[rng.uniform(.1,1) for _ in range(3)] for _ in range(7)]
        pv=[[rng.uniform(-1,1) for _ in range(2)] for _ in range(7)]
        pq=[[rng.uniform(.1,1) for _ in range(3)] for _ in range(7)]
        explicit=linear_direct(pk,pv,pq)
        for size in [1,2,3,7,8]:
            chunk=linear_chunk(pk,pv,pq,size);close(chunk,explicit)
            max_chunk_error=max(max_chunk_error,max(abs(x-y) for a,b in zip(chunk,explicit) for x,y in zip(a,b)))
        initial=[[rng.random() for _ in range(2)] for _ in range(3)]
        sequential=copy.deepcopy(initial);summary=(eye(3),zeros(3,2))
        for t in range(7):
            key=pk[t];norm=math.sqrt(dot(key,key));key=[x/norm for x in key]
            b=rng.random();decay=[rng.uniform(.1,1) for _ in range(3)]
            sequential=delta(sequential,key,pv[t],b,decay)['state']
            summary=compose(affine(key,pv[t],b,decay),summary)
        grouped=add(matmul(summary[0],initial),summary[1]);close(grouped,sequential)
        max_affine_error=max(max_affine_error,max(abs(x-y) for a,b in zip(grouped,sequential) for x,y in zip(a,b)))
    b=budget()
    assert b['long_cache_bytes']['GQA']==16*2**30
    assert b['long_cache_bytes']['MLA']==4.5*2**30
    assert b['hybrid_bytes']['512']==64*2**20==b['all_recurrent_bytes']
    assert b['hybrid_bytes']['4096']==176*2**20
    assert b['hybrid_bytes']['131072']==4144*2**20
    assert b['state_cache_crossover']==512
    # Exercises: fewer KV heads, appended latent, sparse error, collision,
    # delta retention, and the hybrid crossover (also checked above).
    cfg=D['budget']
    four_kv=2*cfg['layers']*4*cfg['head_dim']*cfg['cache_bytes']
    assert four_kv==64*2**10 and four_kv*cfg['long_context']==8*2**30
    close((four_kv*cfg['long_context']/cfg['hypothetical_bandwidth_bytes_per_second'])*1000,
          b['history_read_milliseconds']/2)
    extended=mla(c['latents']+[[2,0]],qs[0],c['key_up'][0],c['value_up'][0])
    close(extended['expanded']['weights'],[2/9,1/9,2/9,4/9])
    close(extended['output'],[8/3,4/3])
    close(delta([[2],[4]],[1,0],[6],.25)['state'],[[3],[4]])
    result=dict(figure_5_2=reads,gqa=gqa,mla=mlas,position_branch=pos,sparse=sparse,loaded_blocks=loaded,
                linear=stream,collision=coll,softmax_collision=soft_collision,delta=update,
                diagonal_gate=diag,scalar_gate=scalar,interference=interference,budget=b,
                max_reconstruction_gradient_error=grad_error,max_absorption_error=max_absorption_error,
                max_chunk_error=max_chunk_error,max_affine_error=max_affine_error,
                random_trials=30,exercise_mla=extended,exercises_checked=6)
    if markdown:
        text=(ROOT/'book/chapters/ch05.md').read_text(encoding='utf-8')
        snippets=re.findall(r'^```python\n(.*?)^```',text,re.M|re.S)
        assert len(snippets)==4
        ns={}
        for i,code in enumerate(snippets,1):exec(compile(code,f'ch05-example-{i}','exec'),ns)
        close(ns['latent_result'],[1.6,2.4]);close(ns['window_result'],.8)
        close(ns['linear_result'],[2,2]);close(ns['delta_result'],[6,4])
        result['executed_python_examples']=4
    return result


if __name__=='__main__':
    r=verify()
    print(json.dumps(r,ensure_ascii=False,indent=2))
    print('Chapter 5: head sharing, latent absorption, sparse reads, recurrent states, chunks and exercises passed.')
