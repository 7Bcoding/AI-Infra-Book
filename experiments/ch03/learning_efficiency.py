"""Reproduce the book's batch-noise, matrix-update and tokenizer examples.

These are mathematical teaching cases, not training benchmark measurements.
Run with the Python standard library only.
"""
from fractions import Fraction as F
from itertools import product
from math import exp, log, sqrt, isclose


def expected_loss(batch, eta=F(1, 2), distance=1, variance=4):
    return ((1-eta)**2*distance**2 + eta**2*F(variance, batch))/2


def verify():
    for batch in (1, 2, 4):
        # Exhaust the independent, equiprobable +/-2 sample errors.
        outcomes = []
        for errors in product((-2, 2), repeat=batch):
            gradient = 1 + F(sum(errors), batch)
            outcomes.append((1-gradient/2)**2/2)
        assert sum(outcomes)/len(outcomes) == expected_loss(batch)
    assert [expected_loss(b) for b in (1,4,16)] == [F(5,8),F(1,4),F(5,32)]
    eta = F(1, 1+4)
    assert expected_loss(1, eta) < expected_loss(1, eta-F(1,100))
    assert expected_loss(1, eta) < expected_loss(1, eta+F(1,100))
    assert F(4,4)*(1+3*F(1,2)) == F(5,2)
    assert 2000*.004 < 700*.015
    x=F(1,3)
    x=(3*x-x**3)/2
    assert x == F(13,27)
    path=[float(x)]
    # Convert to float before iterating to avoid giant exact fractions.
    x=float(x)
    for _ in range(6):
        x=(3*x-x**3)/2
        path.append(x)
    assert abs(x-1)<1e-6
    cases=[dict(tokens=10,nll=40),dict(tokens=25,nll=50)]
    for c in cases:
        c['perplexity']=exp(c['nll']/c['tokens'])
        c['bits_per_byte']=c['nll']/(100*log(2))
    assert cases[0]['perplexity'] > cases[1]['perplexity']
    assert cases[0]['bits_per_byte'] < cases[1]['bits_per_byte']
    print('Expected losses:',[float(expected_loss(b)) for b in (1,4,16)])
    print('Scalar singular-value iteration:',path)
    print('Same 100-byte text, different canonical tokenizations:',cases)
    print('All learning-efficiency cases passed; no hardware was measured.')


if __name__=='__main__':verify()
