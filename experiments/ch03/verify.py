"""Chapter 3 numerical and derivative checks; Python standard library only.

No learning framework is used: analytical derivatives are independently checked
against central differences. The same data drive the original SVG figures.
"""
import copy
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
D = json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def forward(theta, x, y):
    a = [sum(w * t for w, t in zip(row, x)) + b for row, b in zip(theta['W'], theta['b'])]
    h = [max(t, 0) for t in a]
    z = sum(v * t for v, t in zip(theta['v'], h)) + theta['c']
    loss = max(z, 0) - y * z + math.log1p(math.exp(-abs(z)))
    return dict(a=a, h=h, z=z, p=sigmoid(z), loss=loss)


def backward(theta, x, y):
    f = forward(theta, x, y)
    delta = f['p'] - y
    gh = [delta * v for v in theta['v']]
    ga = [g if a > 0 else 0.0 for g, a in zip(gh, f['a'])]
    gradients = dict(W=[[g * t for t in x] for g in ga], b=ga,
                     v=[delta * h for h in f['h']], c=delta)
    gx = [sum(theta['W'][i][j] * ga[i] for i in range(2)) for j in range(2)]
    return dict(delta=delta, gh=gh, ga=ga, gx=gx, gradients=gradients)


PATHS = [('W', i, j) for i in range(2) for j in range(2)] + [('b', i) for i in range(2)] + [('v', i) for i in range(2)] + [('c',)]


def get(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def put(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def update(theta, gradients, eta):
    changed = copy.deepcopy(theta)
    for path in PATHS:
        put(changed, path, get(theta, path) - eta * get(gradients, path))
    return changed


def central_gradient(theta, sample, path, eps):
    plus, minus = copy.deepcopy(theta), copy.deepcopy(theta)
    put(plus, path, get(theta, path) + eps)
    put(minus, path, get(theta, path) - eps)
    return (forward(plus, **sample)['loss'] - forward(minus, **sample)['loss']) / (2 * eps)


def normalized(a, eps=0.0):
    mean = sum(a) / len(a)
    var = sum((t - mean) ** 2 for t in a) / len(a)
    scale = math.sqrt(var + eps)
    return [(t - mean) / scale for t in a], mean, var, scale


def norm_backward(a, d, eps=0.0):
    h, _, _, scale = normalized(a, eps)
    md = sum(d) / len(d)
    mdh = sum(v * t for v, t in zip(d, h)) / len(d)
    return [(v - md - t * mdh) / scale for v, t in zip(d, h)]


def calculate():
    theta = D['parameters']
    sample = D['sample']
    f = forward(theta, **sample)
    b = backward(theta, **sample)
    theta_new = update(theta, b['gradients'], D['learning_rate'])
    f_new = forward(theta_new, **sample)
    s2 = D['second_training_sample']
    g2 = backward(theta, **s2)['gradients']
    batch_g = copy.deepcopy(g2)
    for path in PATHS:
        put(batch_g, path, (get(b['gradients'], path) + get(g2, path)) / 2)
    diffs = [dict(parameter='.'.join(map(str, p)), analytic=get(b['gradients'], p),
                  numeric=central_gradient(theta, sample, p, D['finite_difference_step'])) for p in PATHS]
    adam = D['adam']
    m = v = 0.0
    scalar = 1.0
    steps = []
    for t, g in enumerate(adam['gradients'], 1):
        m = adam['beta1'] * m + (1 - adam['beta1']) * g
        v = adam['beta2'] * v + (1 - adam['beta2']) * g * g
        mh = m / (1 - adam['beta1'] ** t)
        vh = v / (1 - adam['beta2'] ** t)
        scalar -= D['learning_rate'] * mh / (math.sqrt(vh) + adam['epsilon'])
        steps.append(dict(t=t, g=g, m=m, v=v, mhat=mh, vhat=vh, parameter=scalar))
    size = D['large_layer']
    work = 2 * size['batch'] * size['input_width'] * size['output_width']
    return dict(forward=f, backward=b, updated_parameters=theta_new, updated_forward=f_new,
                validation_before=forward(theta, **D['validation_sample']),
                validation_after=forward(theta_new, **D['validation_sample']),
                second_gradient=g2, batch_gradient=batch_g, finite_differences=diffs,
                inactive=backward(theta, **D['inactive_sample']),
                normalization=normalized(D['normalization']['a']),
                normalization_gradient=norm_backward(**dict(a=D['normalization']['a'], d=D['normalization']['upstream'])),
                adam_steps=steps, forward_flops=work, three_matmuls_flops=3*work,
                activation_bytes=size['batch']*size['output_width']*size['element_bytes'],
                tiny_parameter_count=len(PATHS), tiny_adam_state_bytes=4*4*len(PATHS))


def verify(markdown=True):
    r = calculate()
    assert r['forward']['a'] == r['forward']['h'] == [1, 2]
    assert r['forward']['z'] == 0 and r['forward']['p'] == 0.5
    assert math.isclose(r['forward']['loss'], math.log(2))
    expected = dict(W=[[-0.5, -1.0], [0.5, 1.0]], b=[-0.5, 0.5], v=[-0.5, -1.0], c=-0.5)
    assert r['backward']['gradients'] == expected
    assert r['backward']['gx'] == [-0.5, 0.5]
    for p, v in zip(PATHS, [1.05, 0.1, -0.05, 0.9, 0.05, -0.05, 1.05, -0.9, 1.05]):
        assert math.isclose(get(r['updated_parameters'], p), v)
    assert math.isclose(r['updated_forward']['z'], 0.885)
    assert r['updated_forward']['loss'] < r['forward']['loss']
    assert math.isclose(r['validation_after']['z'], 0.18)
    assert r['validation_after']['loss'] > r['validation_before']['loss']
    expected_batch = dict(W=[[0.25, 0.25], [-0.25, -0.25]], b=[0.0, 0.0], v=[0.25, 0.25], c=0.0)
    assert r['batch_gradient'] == expected_batch
    # Independent central differences for all nine coordinates, at both parameter versions.
    max_error = 0.0
    for theta in [D['parameters'], r['updated_parameters']]:
        g = backward(theta, **D['sample'])['gradients']
        for path in PATHS:
            error = abs(central_gradient(theta, D['sample'], path, 1e-5) - get(g, path))
            max_error = max(max_error, error)
            assert error < 1e-8
    # An inactive ReLU exercises the zero derivative branch, absent in the main sample.
    q = sigmoid(2)
    inactive_expected = dict(W=[[q,-2*q],[0.0,0.0]], b=[q,0.0], v=[q,0.0], c=q)
    assert r['inactive']['gradients'] == inactive_expected
    for path in PATHS:
        assert abs(central_gradient(D['parameters'],D['inactive_sample'],path,1e-5)-get(inactive_expected,path))<1e-8
    # Exercise 2 deliberately mixes an old loss derivative with updated output weights.
    wrong = [[r['backward']['delta']*v*x for x in D['sample']['x']] for v in r['updated_parameters']['v']]
    for row, expected_row in zip(wrong,[[-0.525,-1.05],[0.45,0.9]]):
        assert all(math.isclose(x,y) for x,y in zip(row,expected_row))
    # Averaging per-example gradients agrees with differentiating the two-sample mean.
    for path in PATHS:
        mean_numeric=sum(central_gradient(D['parameters'],sample,path,1e-5) for sample in [D['sample'],D['second_training_sample']])/2
        assert abs(mean_numeric-get(r['batch_gradient'],path))<1e-8
    # A single shared parameter appears in W[0][0] and v[0]; perturb both uses together.
    def tied_loss(w):
        theta = copy.deepcopy(D['parameters'])
        theta['W'][0][0] = theta['v'][0] = w
        return forward(theta, **D['sample'])['loss']
    assert abs((tied_loss(1+1e-5)-tied_loss(1-1e-5))/(2e-5) + 1) < 1e-8
    # Check that an intentionally omitted x[1] factor would fail the numerical test.
    assert abs(central_gradient(D['parameters'], D['sample'], ('W',0,1), 1e-5) - (-0.5)) > 0.49
    # Layer normalization: check the full dependency through both mean and variance.
    a, d = D['normalization']['a'], D['normalization']['upstream']
    for stabilization in [0.0, 1e-5]:
        g = norm_backward(a, d, stabilization)
        assert abs(sum(g)) < 1e-12
        for i in range(3):
            plus, minus = a.copy(), a.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            fp = sum(v*t for v,t in zip(d,normalized(plus,stabilization)[0]))
            fm = sum(v*t for v,t in zip(d,normalized(minus,stabilization)[0]))
            assert abs((fp-fm)/2e-5-g[i]) < 1e-8
    assert abs(r['normalization_gradient'][1] - math.sqrt(3/8)) < 1e-12
    # Extreme logits remain finite with the stable BCE expression.
    for z,y,expected in [(1000,1,0),(-1000,0,0),(1000,0,1000),(-1000,1,1000)]:
        loss = max(z,0)-y*z+math.log1p(math.exp(-abs(z)))
        assert loss == expected
    assert r['tiny_parameter_count'] == 9 and r['tiny_adam_state_bytes'] == 144
    assert r['forward_flops'] == 134217728 and r['activation_bytes'] == 262144
    assert r['three_matmuls_flops'] == 402653184
    assert math.isclose(r['adam_steps'][1]['mhat'], 1/38)
    assert math.isclose(r['adam_steps'][1]['vhat'], 0.25)
    assert (2*1+1*4)/(2+1) == 2
    if markdown:
        text = (ROOT/'book/chapters/ch03.md').read_text(encoding='utf-8')
        snippets = re.findall(r'^```python\n(.*?)^```', text, re.M|re.S)
        ns = {}
        for index, code in enumerate(snippets, 1):
            exec(compile(code, f'ch03-snippet-{index}', 'exec'), ns)
        assert len(snippets) == 4
        assert math.isclose(ns['new_loss'], r['updated_forward']['loss'])
        assert abs(ns['numeric_gradient'] + 1) < 1e-8
        r['executed_python_examples'] = len(snippets)
    r['max_gradient_check_absolute_error'] = max_error
    return r


if __name__ == '__main__':
    r = verify()
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print('Chapter 3 numerical examples, analytical derivatives, answers and Python excerpts passed.')
