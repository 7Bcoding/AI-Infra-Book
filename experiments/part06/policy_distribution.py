"""Compute the sampled probability ratio and the full categorical forward KL."""
from math import isclose, log
from itertools import accumulate

old = (.5, .49, .01)
new = (.5, .01, .49)
assert isclose(sum(old), 1) and isclose(sum(new), 1)
ratios = [q/p for p, q in zip(old, new)]
terms = [p * log(p/q) for p, q in zip(old, new)]
kl = sum(terms)
assert ratios[0] == 1 and isclose(kl, .48 * log(49))
print("Ratios:", ratios, "KL(old || new):", round(kl, 6))
# Exactly stratified sample from the old distribution, 100 observations.
endpoints = [round(100 * x) for x in accumulate(old)]
observed = []
for n in range(100):
    action = next(i for i, end in enumerate(endpoints) if n < end)
    observed.append(-log(ratios[action]))
assert isclose(sum(observed)/len(observed), kl)
