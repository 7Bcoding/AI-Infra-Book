"""Executable chapter 2 examples, using the Python standard library only."""
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import struct

ROOT=Path(__file__).resolve().parents[2]
D=json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))


@dataclass
class View:
    """Explicit teaching model: strides and offsets are measured in elements."""
    storage: list
    shape: tuple
    strides: tuple
    offset: int=0
    base: int=1000
    itemsize: int=4

    def index(self,*coords):
        if len(coords)!=len(self.shape) or any(i<0 or i>=n for i,n in zip(coords,self.shape)):
            raise IndexError(coords)
        return self.offset+sum(i*s for i,s in zip(coords,self.strides))

    def read(self,*coords):return self.storage[self.index(*coords)]
    def address(self,*coords):return self.base+self.itemsize*self.index(*coords)
    def write(self,coords,value):self.storage[self.index(*coords)]=value
    def flatten(self):return [self.read(*c) for c in itertools.product(*(range(n) for n in self.shape))]
    def addresses(self):return [self.address(*c) for c in itertools.product(*(range(n) for n in self.shape))]


def float32(v):return struct.unpack('f',struct.pack('f',v))[0]


def calculate():
    flat=sum(D['A'],[])
    a=View(flat,(2,3),(3,1))
    t=View(flat,(3,2),(1,3))
    reshape=View(flat,(3,2),(2,1))
    sliced=View(flat,(2,2),(3,1),1)
    broadcast=View(D['bias'].copy(),(2,3),(1,0),base=3000)
    z=[sum(v*w for v,w in zip(row,D['x'])) for row in D['A']]
    u=[p+q for p,q in zip(z,D['bias'])]
    y=[max(v,0) for v in u]
    trace=[0]
    for p,q in zip(D['A'][0],D['x']):trace.append(trace[-1]+p*q)
    # Same operation graph for both rows, with one unit per node, including comparison.
    graph={}
    for i in range(2):
        for k in range(3):graph[f'm{i}{k}']=[]
        graph[f's{i}']=[f'm{i}0',f'm{i}1']
        graph[f'z{i}']=[f's{i}',f'm{i}2']
        graph[f'u{i}']=[f'z{i}']
        graph[f'y{i}']=[f'u{i}']
    depth={}
    for node,deps in graph.items():depth[node]=1+max([depth[v] for v in deps] or [0])
    schedule=[[f'm{i}0',f'm{i}1',f's{i}',f'm{i}2',f'z{i}',f'u{i}',f'y{i}'] for i in range(2)]
    when={node:slot for row in schedule for slot,node in enumerate(row)}
    assert all(when[dep]+1<=when[node] for node,deps in graph.items() for dep in deps)
    fa,fb,fc=map(float32,D['float_example'])
    return dict(z=z,u=u,y=y,accumulator=trace,
                original_addresses=a.addresses(),transpose=t.flatten(),transpose_addresses=t.addresses(),
                reshape=reshape.flatten(),slice=sliced.flatten(),slice_addresses=sliced.addresses(),
                broadcast=broadcast.flatten(),broadcast_addresses=broadcast.addresses(),
                graph=graph,graph_depth=depth,work=len(graph),critical_path=max(depth.values()),schedule=schedule,
                separate_stage_bytes=[(6+3+2)*4,(2+2+2)*4,(2+2)*4],
                fused_bytes=(6+3+2+2)*4,
                correct_energy=sum(v*v for v in z),overwritten_energy=sum(v*v for v in y),
                float_left=float32(float32(fa+fb)+fc),float_right=float32(fa+float32(fb+fc)))


def verify(markdown=True):
    r=calculate()
    assert (r['z'],r['u'],r['y'])==([3,9],[-1,10],[0,10])
    assert r['accumulator']==[0,2,0,3]
    assert r['transpose']==[1,4,2,5,3,6] and r['reshape']==[1,2,3,4,5,6]
    assert r['slice']==[2,3,5,6] and r['slice_addresses']==[1004,1008,1016,1020]
    assert r['transpose_addresses']==[1000,1012,1004,1016,1008,1020]
    # A transpose is not a copy; a materialized copy is independent.
    flat=sum(D['A'],[])
    a=View(flat,(2,3),(3,1))
    t=View(flat,(3,2),(1,3))
    copied=View(t.flatten(),(3,2),(2,1),base=2000)
    t.write((0,1),40)
    assert a.read(1,0)==40 and copied.read(0,1)==4
    # Writing along a broadcast axis aliases the same two backing elements.
    v=View(D['bias'].copy(),(2,3),(1,0),base=3000)
    assert v.address(0,0)==v.address(0,2)==3000
    assert v.address(1,0)==v.address(1,2)==3004
    assert r['work']==14 and r['critical_path']==5
    # Exercise 3: a feasible schedule reaches the five-step lower bound with four workers.
    steps=[['m00','m01','m10','m11'],['m02','m12','s0','s1'],['z0','z1'],['u0','u1'],['y0','y1']]
    done=set()
    for step in steps:
        assert len(step)<=4 and not done.intersection(step)
        assert all(set(r['graph'][node])<=done for node in step)
        done.update(step)
    assert done==set(r['graph']) and len(steps)==5
    assert r['separate_stage_bytes']==[44,24,16] and sum(r['separate_stage_bytes'])==84
    assert r['fused_bytes']==52
    assert r['correct_energy']==90 and r['overwritten_energy']==100
    assert (r['float_left'],r['float_right'])==(1,0)
    # Deliberately specified read/modify/write interleaving, not a language race test.
    shared=0
    local0=shared;local1=shared
    shared=local0+2;shared=local1-2;shared+=3
    assert shared==1
    # Exercise 1: modifying the slice changes its original storage.
    flat=sum(D['A'],[]);s=View(flat,(2,2),(3,1),1)
    assert s.address(1,0)==1016
    s.write((1,0),50)
    assert flat==[1,2,3,4,50,6]
    # Exercise 2: a large square permutation must read AND write each element.
    assert 4096*4096*4==67108864
    assert 2*4096*4096*4/1e12*1000==0.134217728
    if markdown:
        text=(ROOT/'book/chapters/ch02.md').read_text(encoding='utf-8')
        snippets=re.findall(r'^```python\n(.*?)^```',text,re.M|re.S)
        ns={}
        for index,code in enumerate(snippets,1):exec(compile(code,f'chapter2-snippet-{index}','exec'),ns)
        assert ns['bad_energy']==100 and ns['good_energy']==90
        assert ns['fused_output']==[0,10]
        assert len(snippets)==5
        r['executed_python_examples']=len(snippets)
    return r


if __name__=='__main__':
    r=verify()
    print(json.dumps({k:v for k,v in r.items() if k not in ('graph','graph_depth','schedule')},ensure_ascii=False,indent=2))
    print('All chapter 2 numeric, address, dependency, aliasing and code checks passed.')
