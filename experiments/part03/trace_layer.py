"""Inspect the real layer from chapters 13–14. Reference run: PyTorch 2.8 CPU."""
import json
import torch
from torch.profiler import ProfilerActivity, profile


def layer(x, w, b):
    return (x @ w + b).relu()


def inputs():
    return [torch.tensor(values, requires_grad=True) for values in
            ([[1., 2.], [3., 4.]], [[2., -1.], [1., 3.]], [-6., 1.])]


def capture_and_differentiate():
    x, w, b = inputs()
    saved = []
    restored = []

    def pack(tensor):
        saved.append(list(tensor.shape))
        return tensor.detach()

    def unpack(tensor):
        restored.append(list(tensor.shape))
        return tensor

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = layer(x, w, b)
            out.sum().backward()

    assert out.tolist() == [[0., 6.], [4., 10.]]
    assert x.grad.tolist() == [[-1., 3.], [1., 4.]]
    assert w.grad.tolist() == [[3., 4.], [4., 6.]]
    assert b.grad.tolist() == [1., 2.]
    with torch.no_grad():
        w.add_(w.grad, alpha=-0.1)
        b.add_(b.grad, alpha=-0.1)
    updated = layer(x, w, b).detach()
    torch.testing.assert_close(updated, torch.tensor([[0., 4.2], [1.4, 6.2]]))
    interesting = {"aten::matmul", "aten::mm", "aten::add", "aten::relu",
                   "aten::sum", "aten::threshold_backward"}
    calls = {event.key: event.count for event in prof.key_averages() if event.key in interesting}
    assert {"aten::mm", "aten::relu", "aten::sum"} <= calls.keys()
    return {"cpu_operator_counts": calls, "packed_shapes": saved, "unpacked_shapes": restored}


def alias_example():
    x, w, b = inputs()
    view = x.T
    assert x.stride() == (2, 1) and view.stride() == (1, 2)
    assert view.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    loss = layer(x, w, b).sum()
    with torch.no_grad():
        view[1, 0] += 1
    try:
        loss.backward()
    except RuntimeError as error:
        if "inplace operation" not in str(error):
            raise
        return "backward detected the changed saved tensor"
    raise AssertionError("Expected an in-place version error")


def view_storage():
    large = torch.empty((4096, 4096), dtype=torch.float16)
    large[0, :2] = torch.tensor([1., 2.])
    small = large[0, :2]
    del large
    independent = small.clone()
    assert small.tolist() == independent.tolist() == [1., 2.]
    assert small.contiguous().data_ptr() == small.data_ptr()
    storage_sizes = [t.untyped_storage().nbytes() for t in (small, independent)]
    assert storage_sizes == [33554432, 4]
    return {"view_and_clone_storage_bytes": storage_sizes}


def compile_example():
    graphs = []

    def observe(graph, example_inputs):
        graphs.append([str(node.target) for node in graph.graph.nodes if node.op == "call_function"
                       or node.op == "call_method"])
        return graph.forward

    x, w, b = [t.detach() for t in inputs()]
    captured = torch.compile(layer, backend=observe, dynamic=False, fullgraph=True)
    for rows in (2, 2, 3, 2):
        arg = x if rows == 2 else torch.cat((x, x[:1]))
        torch.testing.assert_close(captured(arg, w, b), layer(arg, w, b))
    assert len(graphs) == 2, graphs
    shape_graphs = graphs.copy()

    layout_graphs = []

    def observe_layout(graph, example_inputs):
        layout_graphs.append([list(t.stride()) for t in example_inputs])
        return graph.forward

    by_layout = torch.compile(layer, backend=observe_layout, dynamic=False, fullgraph=True)
    for arg in (x, x.T, x):
        torch.testing.assert_close(by_layout(arg, w, b), layer(arg, w, b))
    assert by_layout(x.T, w, b).tolist() == [[0., 9.], [2., 11.]]
    assert len(layout_graphs) == 2, layout_graphs

    differentiable = torch.compile(layer, backend="aot_eager", fullgraph=True)
    compiled_args, eager_args = inputs(), inputs()
    differentiable(*compiled_args).sum().backward()
    layer(*eager_args).sum().backward()
    for compiled, eager in zip(compiled_args, eager_args):
        torch.testing.assert_close(compiled.grad, eager.grad)
    return {"static_graphs_for_rows_2_2_3_2": shape_graphs,
            "static_graphs_for_two_layouts": len(layout_graphs), "aot_eager_gradients": "match"}


if __name__ == "__main__":
    print(json.dumps({"torch": str(torch.__version__), "device": "cpu",
                      "eager": capture_and_differentiate(), "alias": alias_example(),
                      "storage": view_storage(), "capture": compile_example()}, indent=2))

