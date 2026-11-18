"""Allocate three scalar registers for two legal schedules of the same dot product.

Inputs are single-use SSA values. Instructions may overwrite a dying operand.
When all registers are busy, spill the unpinned value used farthest in the future.
The virtual and emitted programs are checked on all inputs in {-1, 0, 1}^4.
"""
from itertools import product


def generate(eager):
    loads = [("load", k, ()) for k in "abcd"]
    multiply = ("mul", "p", ("a", "b"))
    return [*(loads if eager else loads[:2]), multiply,
            *([] if eager else loads[2:]), ("fma", "y", ("c", "d", "p"))]


def allocate(program):
    code = []
    live = {}  # live SSA value -> physical register
    uses = {v: i for i, (_, _, args) in enumerate(program) for v in args}

    def acquire(pinned):
        for reg in range(3):
            if reg not in live.values():
                return reg
        victim = max((v for v in live if v not in pinned), key=uses.__getitem__)
        reg = live.pop(victim)
        code.append(("spill", victim, reg))
        return reg

    for op, dest, args in program:
        for v in args:
            if v not in live:
                reg = acquire(args)
                code.append(("reload", reg, v))
                live[v] = reg
        operands = [live[v] for v in args]
        for v in args:  # all uses in this teaching IR are last uses
            del live[v]
        reg = acquire(())
        code.append((op, reg, dest) if op == "load" else (op, reg, *operands))
        live[dest] = reg
    code.append(("return", live["y"]))
    return code


def execute(code, inputs):
    reg, spill = [None] * 3, {}
    for op, *a in code:
        if op == "return":
            return reg[a[0]]
        if op == "spill":
            spill[a[0]] = reg[a[1]]
        elif op == "reload":
            reg[a[0]] = spill[a[1]]
        elif op == "load":
            reg[a[0]] = inputs[a[1]]
        elif op == "mul":
            reg[a[0]] = reg[a[1]] * reg[a[2]]
        elif op == "fma":
            reg[a[0]] = reg[a[1]] * reg[a[2]] + reg[a[3]]
        else:
            raise ValueError(op)
    raise ValueError("Missing return")


if __name__ == "__main__":
    for eager in (True, False):
        code = allocate(generate(eager))
        for a, b, c, d in product((-1, 0, 1), repeat=4):
            assert execute(code, dict(zip("abcd", (a, b, c, d)))) == a*b+c*d
        assert sum(op[0] == "spill" for op in code) == int(eager)
        assert sum(op[0] == "reload" for op in code) == int(eager)
        print("Load all first" if eager else "Multiply early", *code, sep="\n")
