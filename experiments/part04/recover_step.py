"""Checkpoint/replay for the Chapter 17 projection and experts, standard Python only.
Optionally pass --directory PATH to keep artifacts in a new run folder.
"""
import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from uuid import uuid4

VERSION = "six-rows-v1"
SEED = 1701
SAMPLES = ((1, 2), (3, 4), (2, 1), (1, 3), (2, 4), (3, 2))


def initial_state():
    params = {"W": [2., -1., 1., 3.], "bias": [-6., 1.],
              "logits": [math.log(3), 0., 0., math.log(3)],
              "expert0": [1., 0., 0., 1.], "expert1": [0., 1., 1., 0.]}
    return {"params": params, "velocity": {k: [0.]*len(v) for k,v in params.items()},
            "step": 0, "cursor": 0, "version": VERSION, "seed": SEED}


def sample_id(cursor, seed):
    epoch, offset = divmod(cursor, len(SAMPLES))
    order = list(range(len(SAMPLES)))
    if epoch:
        random.Random(seed + epoch).shuffle(order)
    return order[offset]


def project(row, matrix):
    return [row[0]*matrix[j]+row[1]*matrix[2+j] for j in range(2)]


def sample_loss_gradient(params, idx):
    x = SAMPLES[idx]
    grad = {key: [0.]*len(v) for key,v in params.items()}
    h = [max(0., v+b) for v,b in zip(project(x,params["W"]),params["bias"])]
    lo = 2*(idx % 2)
    logits = params["logits"][lo:lo+2]
    weights = [math.exp(z-max(logits)) for z in logits]
    p = [w/sum(weights) for w in weights]
    values = [project(h,params[key])[0] for key in ("expert0","expert1")]
    loss = sum(weight*value for weight,value in zip(p,values))
    dh = [0.,0.]
    for e,key in enumerate(("expert0","expert1")):
        grad["logits"][lo+e] = p[e]*(values[e]-loss)
        for k in range(2):
            grad[key][2*k] = h[k]*p[e]
            dh[k] += params[key][2*k]*p[e]
    for j in range(2):
        g = dh[j] if h[j]>0 else 0.
        grad["bias"][j] = g
        for k in range(2):
            grad["W"][2*k+j] = x[k]*g
    return loss, grad


def train_step(state):
    ids = [sample_id(state["cursor"]+i,state["seed"]) for i in range(2)]
    rows = [sample_loss_gradient(state["params"],idx) for idx in ids]
    for k,param in state["params"].items():
        speed = state["velocity"][k]
        for j in range(len(param)):
            speed[j] = .5*speed[j]+sum(grad[k][j] for _,grad in rows)
            param[j] -= speed[j]/8
    state["cursor"] += 2
    state["step"] += 1
    return ids, sum(loss for loss,_ in rows)


def train_to(state, last):
    return [train_step(state) for _ in range(state["step"],last)]


def encoded(obj):
    return json.dumps(obj,sort_keys=True,allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_file(path,data):
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def save(directory,state,interrupt=False):
    # Each attempt uses unique shard names, even if a previous attempt was interrupted.
    generation = f'{state["step"]:06}-{uuid4().hex}'
    manifest = {key:v for key,v in state.items() if key not in ("params","velocity")}
    manifest["shards"] = []
    for part,keys in enumerate((("W","bias","logits"),("expert0","expert1"))):
        data = encoded({field:{k:state[field][k] for k in keys}
                        for field in ("params","velocity")})
        name = f"{generation}-{part}.json"
        write_file(directory/name,data)
        manifest["shards"].append({"name":name,"size":len(data),"sha256":digest(data)})
        if interrupt and part==0:
            return
    temp = directory/(generation+".pending")
    write_file(temp,encoded(manifest))
    os.replace(temp,directory/f'commit-{state["step"]:06}.json')


def restore(directory):
    expected = initial_state()["params"]
    for path in sorted(directory.glob("commit-*.json"), reverse=True):
        try:
            meta = json.loads(path.read_bytes())
            matches_boundary = meta["version"]==VERSION and meta["cursor"]==2*meta["step"]
            if not matches_boundary or len(meta["shards"])!=2:
                continue
            state = {k:meta[k] for k in ("version","seed","cursor","step")}
            state.update(params={},velocity={})
            for shard in meta["shards"]:
                raw = (directory/shard["name"]).read_bytes()
                if len(raw)!=shard["size"] or digest(raw)!=shard["sha256"]:
                    raise ValueError("incomplete or damaged shard")
                fields = json.loads(raw)
                for name in ("params","velocity"):
                    if state[name].keys() & fields[name].keys():
                        raise ValueError("duplicate parameter")
                    state[name].update(fields[name])
            if all({k:len(v) for k,v in state[f].items()} ==
                   {k:len(v) for k,v in expected.items()} for f in ("params","velocity")):
                return state
        except (OSError,ValueError,KeyError,TypeError):
            continue
    raise ValueError("No complete checkpoint")


def exercise(directory):
    reference = initial_state()
    trace = train_to(reference,6)
    assert math.isclose(trace[0][1],10) and trace[0][0]==[0,1]
    assert trace[2][0]==trace[3][0]==[4,5]
    assert math.isclose(reference["params"]["W"][0],1.5525918534,abs_tol=1e-9)
    live = initial_state()
    train_to(live,2)
    save(directory,live)
    train_to(live,4)
    save(directory,live,interrupt=True)

    def compare(expected_step):
        state = restore(directory)
        assert state["step"] == expected_step
        assert train_to(state,6) == trace[expected_step:]
        assert state == reference
        return expected_step

    partial = compare(2)
    save(directory,live)
    complete = compare(4)
    manifest = json.loads((directory/"commit-000004.json").read_bytes())
    broken = directory/manifest["shards"][1]["name"]
    with broken.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0]^1]))
    damaged = compare(2)
    print(f"Restored steps: partial={partial}, committed={complete}, damaged={damaged}.")
    print("All replayed sample IDs, losses, weights and momentum match uninterrupted step 6.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path)
    args = parser.parse_args()
    if args.directory:
        run = args.directory/("replay-"+uuid4().hex)
        run.mkdir(parents=True)
        exercise(run)
        print("Artifacts:",run)
    else:
        with tempfile.TemporaryDirectory() as folder:
            exercise(Path(folder))
