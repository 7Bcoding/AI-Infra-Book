"""Complete scalar-checked Transformer, including exact-prefix KV-cache tests.

Standard library only. Row vectors are used throughout. Scalar reverse-mode AD
is intentionally small: its 50 parameter derivatives are checked independently
using finite differences. This is a pedagogical implementation, not a GPU kernel.
"""
import copy
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
D = json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))


class Scalar:
    def __init__(self, value, parents=()):
        self.value = float(value)
        self.parents = parents
        self.grad = 0.0

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Scalar) else Scalar(x)

    def __add__(self, other):
        other = self.wrap(other)
        return Scalar(self.value+other.value, ((self,1.0),(other,1.0)))
    __radd__ = __add__

    def __neg__(self):
        return Scalar(-self.value, ((self,-1.0),))

    def __sub__(self, other):
        return self + (-self.wrap(other))

    def __rsub__(self, other):
        return self.wrap(other) + (-self)

    def __mul__(self, other):
        other = self.wrap(other)
        return Scalar(self.value*other.value, ((self,other.value),(other,self.value)))
    __rmul__ = __mul__

    def __pow__(self, power):
        return Scalar(self.value**power, ((self,power*self.value**(power-1)),))

    def __truediv__(self, other):
        return self * self.wrap(other)**(-1)

    def __rtruediv__(self, other):
        return self.wrap(other) * self**(-1)

    def backward(self):
        ordered, seen = [], set()
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent,_ in node.parents:
                visit(parent)
            ordered.append(node)
        visit(self)
        for node in ordered:
            node.grad=0.0
        self.grad=1.0
        for node in reversed(ordered):
            for parent,derivative in node.parents:
                parent.grad += node.grad*derivative


def value(x):
    return x.value if isinstance(x,Scalar) else float(x)


def exp(x):
    result=math.exp(value(x))
    return Scalar(result,((x,result),)) if isinstance(x,Scalar) else result


def log(x):
    return Scalar(math.log(x.value),((x,1/x.value),)) if isinstance(x,Scalar) else math.log(x)


def plain(obj):
    if isinstance(obj,dict):return {k:plain(v) for k,v in obj.items()}
    if isinstance(obj,list):return [plain(v) for v in obj]
    if isinstance(obj,Scalar):return obj.value
    return obj


def parameters(ad=False):
    eye=[[1.0,0.0],[0.0,1.0]]
    c=math.log(2)/math.sqrt(2)
    theta=dict(E=copy.deepcopy(D['embedding']), Wq=[[c,0.0],[0.0,c]],
               Wk=copy.deepcopy(eye),Wv=copy.deepcopy(eye),Wo=copy.deepcopy(eye),
               Wg=copy.deepcopy(eye),Wu=copy.deepcopy(eye),Wd=[[0.5,0.0],[0.0,0.5]],
               gamma1=[1.0,1.0],gamma2=[1.0,1.0],gamma_final=[1.0,1.0],
               U=copy.deepcopy(D['vocabulary_projection']))
    def wrap(obj):
        if isinstance(obj,dict):return {k:wrap(v) for k,v in obj.items()}
        if isinstance(obj,list):return [wrap(v) for v in obj]
        return Scalar(obj)
    return wrap(theta) if ad else theta


def paths(obj,prefix=()):
    if isinstance(obj,dict):
        return [p for k,v in obj.items() for p in paths(v,prefix+(k,))]
    if isinstance(obj,list):
        return [p for i,v in enumerate(obj) for p in paths(v,prefix+(i,))]
    return [prefix]


def get(obj,path):
    for key in path:obj=obj[key]
    return obj


def put(obj,path,v):
    for key in path[:-1]:obj=obj[key]
    obj[path[-1]]=v


def project(row,weights):
    return [sum(row[i]*weights[i][j] for i in range(len(row))) for j in range(len(weights[0]))]


def norm(row,gamma,eps=0):
    scale=(sum(x*x for x in row)/len(row)+eps)**0.5
    return [x*g/scale for x,g in zip(row,gamma)]


def rotate(row,position):
    angle=position*D['rotation_radians_per_position']
    c,s=math.cos(angle),math.sin(angle)
    return [row[0]*c-row[1]*s,row[0]*s+row[1]*c]


def softmax(row):
    # The detached common shift leaves the softmax derivative unchanged.
    m=max(map(value,row))
    exponentials=[exp(x-m) for x in row]
    total=sum(exponentials)
    return [x/total for x in exponentials]


def attend(query,keys,values):
    scores=[sum(q*k for q,k in zip(query,key))/math.sqrt(2) for key in keys]
    weights=softmax(scores)
    output=[sum(a*v[j] for a,v in zip(weights,values)) for j in range(2)]
    return scores,weights,output


def finish_row(x,att,theta,eps=0):
    residual=[a+b for a,b in zip(x,project(att,theta['Wo']))]
    n=norm(residual,theta['gamma2'],eps)
    gate=project(n,theta['Wg'])
    up=project(n,theta['Wu'])
    product=[g/(1+exp(-g))*u for g,u in zip(gate,up)]
    ffn=project(product,theta['Wd'])
    hidden=[r+f for r,f in zip(residual,ffn)]
    return dict(residual=residual,ffn_input=n,gate=gate,up=up,product=product,ffn=ffn,hidden=hidden)


def forward(tokens,theta=None,eps=0,layers=1):
    theta=parameters() if theta is None else theta
    x=[theta['E'][token].copy() for token in tokens]
    traces=[]
    # The auxiliary multi-layer equivalence check repeats the same coefficients.
    # Its hidden states and KV caches still belong to distinct layer executions.
    # The chapter's 50-parameter training example has one layer only.
    for _ in range(layers):
        normalized=[norm(row,theta['gamma1'],eps) for row in x]
        qraw=[project(row,theta['Wq']) for row in normalized]
        kraw=[project(row,theta['Wk']) for row in normalized]
        q=[rotate(row,i) for i,row in enumerate(qraw)]
        k=[rotate(row,i) for i,row in enumerate(kraw)]
        v=[project(row,theta['Wv']) for row in normalized]
        rowdata=[]
        for i,row in enumerate(x):
            scores,weights,att=attend(q[i],k[:i+1],v[:i+1])
            finish=finish_row(row,att,theta,eps)
            rowdata.append(dict(scores=scores,weights=weights,attention=att,**finish))
        traces.append(dict(input=x,normalized=normalized,qraw=qraw,kraw=kraw,q=q,k=k,v=v,rows=rowdata))
        x=[row['hidden'] for row in rowdata]
    final=[norm(row,theta['gamma_final'],eps) for row in x]
    logits=[project(row,theta['U']) for row in final]
    probs=[softmax(row) for row in logits]
    return dict(layers=traces,final=final,logits=logits,probabilities=probs)


def loss(theta,eps=0):
    f=forward(D['tokens'],theta,eps)
    losses=[-log(row[target]) for row,target in zip(f['probabilities'],D['targets'])]
    return sum(losses)/len(losses)


def decode_step(token,position,caches,theta=None,eps=0,wrong_position=None):
    theta=parameters() if theta is None else theta
    x=theta['E'][token].copy()
    for cache in caches:
        n=norm(x,theta['gamma1'],eps)
        pos=position if wrong_position is None else wrong_position
        q=rotate(project(n,theta['Wq']),pos)
        k=rotate(project(n,theta['Wk']),pos)
        v=project(n,theta['Wv'])
        cache['k'].append(k)
        cache['v'].append(v)
        _,_,att=attend(q,cache['k'],cache['v'])
        x=finish_row(x,att,theta,eps)['hidden']
    final=norm(x,theta['gamma_final'],eps)
    return project(final,theta['U'])


def assert_close(a,b,tol=1e-10):
    if isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_close(x,y,tol)
    else:assert abs(value(a)-value(b))<tol,(plain(a),plain(b))


def attention_backward(q,k,v,upstream):
    _,a,o=attend(q,k,v)
    da=[sum(g*w for g,w in zip(upstream,row)) for row in v]
    mean=sum(x*y for x,y in zip(a,da))
    ds=[p*(g-mean) for p,g in zip(a,da)]
    dq=[sum(s*row[j] for s,row in zip(ds,k))/math.sqrt(2) for j in range(2)]
    dk=[[s*x/math.sqrt(2) for x in q] for s in ds]
    dv=[[p*x for x in upstream] for p in a]
    return dict(a=a,output=o,da=da,ds=ds,dq=dq,dk=dk,dv=dv)


def large_budget():
    c=D['large_model'];L,d,H,G,dh,m,V=(c[k] for k in ['layers','width','query_heads','kv_heads','head_dim','ffn_width','vocabulary'])
    dkv=G*dh
    attention_parameters=2*d*d+2*d*dkv
    ffn_parameters=3*d*m
    block_linear_parameters=attention_parameters+ffn_parameters
    total=L*(block_linear_parameters+2*d)+2*V*d+d
    T=c['context'];bytes_per_token=2*L*G*dh*c['cache_element_bytes']
    return dict(attention_parameters=attention_parameters,ffn_parameters=ffn_parameters,
                block_linear_parameters=block_linear_parameters,total_parameters=total,
                cache_bytes_per_token=bytes_per_token,cache_bytes_at_4096=bytes_per_token*T,
                score_matrix_bytes=H*T*T*2,residual_bytes=T*d*2,ffn_intermediate_bytes=T*m*2,
                per_token_block_linear_flops=2*L*block_linear_parameters,
                per_token_head_flops=2*d*V,
                prefill_block_linear_flops=2*T*L*block_linear_parameters,
                prefill_causal_attention_flops=2*L*d*T*(T+1),
                prefill_dense_attention_flops=4*L*d*T*T,
                decode_attention_flops_at_4096=4*L*d*T,
                attention_linear_crossover_length=block_linear_parameters//d-1,
                final_cache_tokens_after_256_outputs=T+256-1,
                final_cache_bytes_after_256_outputs=bytes_per_token*(T+256-1))


def verify(markdown=True):
    f=forward(D['tokens'])
    a=f['layers'][0]
    assert_close(a['k'],[[1,1],[1,1],[1,-1]])
    assert_close([r['weights'] for r in a['rows']],[[1],[.5,.5],[.25,.25,.5]])
    assert_close([r['attention'] for r in a['rows']],[[1,1],[1,0],[0,.5]])
    assert_close([r['residual'] for r in a['rows']],[[2,2],[2,-1],[-1,1.5]])
    assert [max(range(4),key=lambda j:row[j]) for row in f['probabilities']]==D['targets']
    assert len(paths(parameters()))==50
    # Same-prefix equivalence for single and multi-layer execution, both eps choices.
    max_cache_error=0.0
    for L in [1,2]:
        for eps in [0,D['epsilon_for_robustness_check']]:
            tokens=[0,1,2,0,3]
            caches=[dict(k=[],v=[]) for _ in range(L)]
            full=forward(tokens,eps=eps,layers=L)
            for i,token in enumerate(tokens):
                cached=decode_step(token,i,caches,eps=eps)
                expected=forward(tokens[:i+1],eps=eps,layers=L)['logits'][-1]
                assert_close(cached,expected)
                assert_close(cached,full['logits'][i])
                max_cache_error=max(max_cache_error,max(abs(x-y) for x,y in zip(cached,expected)))
            if L==2:
                assert max(abs(x-y) for x,y in zip(caches[0]['k'][1],caches[1]['k'][1]))>1e-4
    # Real prefill of two tokens, then one decode step (not reprocessing the prompt).
    prefill=forward(D['prompt_tokens'])
    caches=[dict(k=copy.deepcopy(prefill['layers'][0]['k']),v=copy.deepcopy(prefill['layers'][0]['v']))]
    first=max(range(4),key=lambda j:prefill['probabilities'][-1][j])
    assert first==2 and len(caches[0]['k'])==2
    next_logits=decode_step(first,2,caches)
    assert max(range(4),key=lambda j:next_logits[j])==3 and len(caches[0]['k'])==3
    assert_close(next_logits,f['logits'][2])
    # Future input changes cannot modify earlier causal outputs.
    altered=forward([0,1,3])
    assert_close(altered['logits'][:2],f['logits'][:2])
    # Wrong positions are detected on the second input (the third alone has a symmetry).
    wrong=[dict(k=[],v=[])]
    decode_step(0,0,wrong)
    incorrect=decode_step(1,1,wrong,wrong_position=0)
    assert max(abs(x-y) for x,y in zip(incorrect,f['logits'][1]))>0.01
    # Check all 50 parameter gradients through the complete Transformer and LM loss.
    theta=parameters(ad=True);objective=loss(theta);objective.backward()
    base=parameters();eps=D['finite_difference_step'];max_grad_error=0.0
    for p in paths(base):
        plus,minus=copy.deepcopy(base),copy.deepcopy(base)
        put(plus,p,get(base,p)+eps);put(minus,p,get(base,p)-eps)
        numeric=(loss(plus)-loss(minus))/(2*eps)
        error=abs(numeric-get(theta,p).grad)
        assert error<1e-7, (p,numeric,get(theta,p).grad,error)
        max_grad_error=max(max_grad_error,error)
    updated=copy.deepcopy(base)
    for p in paths(base):put(updated,p,get(base,p)-.1*get(theta,p).grad)
    assert loss(updated)<loss(base)
    # The local attention VJP and exercise use different upstream directions.
    g=attention_backward(a['q'][2],a['k'],a['v'],[1,0])
    assert_close(g['ds'],[.25,.25,-.5])
    exercise=attention_backward(a['q'][2],a['k'],a['v'],[0,1])
    assert_close(exercise['ds'],[.125,-.375,.25])
    assert_close(exercise['dv'],[[0,.25],[0,.25],[0,.5]])
    # Independently check the displayed local Q/K/V VJP in both directions.
    for upstream,analytic in [([1,0],g),([0,1],exercise)]:
        inputs=dict(q=copy.deepcopy(a['q'][2]),k=copy.deepcopy(a['k']),v=copy.deepcopy(a['v']))
        def local_objective(data):
            output=attend(data['q'],data['k'],data['v'])[2]
            return sum(x*y for x,y in zip(output,upstream))
        for p in paths(inputs):
            plus,minus=copy.deepcopy(inputs),copy.deepcopy(inputs)
            put(plus,p,get(inputs,p)+eps);put(minus,p,get(inputs,p)-eps)
            numeric=(local_objective(plus)-local_objective(minus))/(2*eps)
            expected=get(analytic,('d'+p[0],)+p[1:])
            assert abs(numeric-expected)<1e-8
    # Masking after softmax without renormalizing changes the function.
    unmasked=softmax([math.log(2),math.log(2),0])
    assert_close(unmasked,[.4,.4,.2])
    assert_close([unmasked[0]+unmasked[1],unmasked[0]-unmasked[1]],[.8,0])
    visibility=[[int(j<=2+i) for j in range(4)] for i in range(2)]
    assert visibility==[[1,1,1,0],[1,1,1,1]]
    budget=large_budget()
    assert budget['total_parameters']==5933109248
    assert budget['cache_bytes_per_token']==131072
    assert budget['cache_bytes_at_4096']==512*2**20
    assert budget['score_matrix_bytes']==2**30
    assert (4*2**30)//budget['cache_bytes_at_4096']==8
    assert (4*2**30)//(2*budget['cache_bytes_at_4096'])==4
    # Exercise 4: zero direct loss on position 0 still permits indirect gradients.
    masked_theta=parameters(ad=True)
    masked_forward=forward(D['tokens'],masked_theta)
    masked_loss=sum(-log(masked_forward['probabilities'][i][D['targets'][i]]) for i in [1,2])/2
    masked_loss.backward()
    assert all(c.grad==0 for c in masked_forward['logits'][0])
    assert max(abs(x.grad) for x in masked_theta['E'][0])>1e-5
    assert abs(value(masked_loss)-0.371589400224453)<1e-12
    result=dict(forward=plain(f),token_losses=[-math.log(row[t]) for row,t in zip(f['probabilities'],D['targets'])],
                training_loss=loss(base),loss_after_one_step=loss(updated),parameters=50,
                max_parameter_gradient_error=max_grad_error,max_cache_error=max_cache_error,
                attention_gradient=g,exercise_attention_gradient=exercise,budget=budget,
                masked_loss=value(masked_loss),masked_first_embedding_gradient=[x.grad for x in masked_theta['E'][0]],
                exercises_checked=5)
    if markdown:
        text=(ROOT/'book/chapters/ch04.md').read_text(encoding='utf-8')
        snippets=re.findall(r'^```python\n(.*?)^```',text,re.M|re.S)
        ns={}
        for i,code in enumerate(snippets,1):exec(compile(code,f'ch04-example-{i}','exec'),ns)
        assert len(snippets)==3
        assert_close(ns['weighted_value'],[0,.5])
        assert_close(ns['rotated_key'],[1,1])
        assert_close(ns['new_output'],[0,.5])
        result['executed_python_examples']=3
    return result


if __name__=='__main__':
    r=verify()
    print(json.dumps(r,ensure_ascii=False,indent=2))
    print('Chapter 4: complete forward, all 50 parameter gradients, cached prefixes and exercises passed.')
