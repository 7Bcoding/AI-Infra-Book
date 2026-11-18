"""Two CPU ranks, one shared projection, two experts, one optimizer step.
Run as a regular file with CPU PyTorch; rank-to-rank transport is a local pipe.
"""
import math
from multiprocessing import get_context

import torch



def tensor(data, grad=False):
    return torch.tensor(data, dtype=torch.float64, requires_grad=grad)


def transfer(peer, rank, outgoing):
    # Ordered to work even if both payloads exceed the pipe buffer.
    if rank == 0:
        peer.send(outgoing)
        return peer.recv()
    incoming = peer.recv()
    peer.send(outgoing)
    return incoming


def worker(rank, peer):
    torch.set_num_threads(1)

    def exchange(rows):
        # Destination slots on input become source slots on output.
        result = rows.detach().clone()
        result[1-rank] = tensor(transfer(peer, rank, rows[1-rank].tolist()))
        return result

    def all_reduce_sum(value):
        value.add_(tensor(transfer(peer, rank, value.tolist())))

    try:
        x = tensor(([1, 2], [3, 4])[rank])
        w = tensor([[2, -1], [1, 3]], True)
        b = tensor([-6, 1], True)
        expert = tensor(([[1, 0], [0, 1]], [[0, 1], [1, 0]])[rank], True)
        logits = tensor([[math.log(3), 0], [0, math.log(3)]], True)
        h = (x @ w + b).relu()
        gate = logits[rank].softmax(dim=0)

        # The process boundary cuts the autograd graph at both message transfers.
        batch = exchange(h.detach().repeat(2, 1)).requires_grad_()
        transformed = batch @ expert
        replies = exchange(transformed.detach()).requires_grad_()
        y = (gate[:, None] * replies).sum(dim=0)
        torch.testing.assert_close(y, tensor(([1.5, 4.5], [8.5, 5.5])[rank]))
        y[0].backward()
        transformed.backward(exchange(replies.grad))
        h.backward(exchange(batch.grad).sum(dim=0))
        for shared in (w, b, logits):
            all_reduce_sum(shared.grad)
        torch.testing.assert_close(w.grad, tensor([[.75, 2.5], [1, 3.5]]))
        torch.testing.assert_close(b.grad, tensor([.25, 1]))
        torch.testing.assert_close(logits.grad, tensor([[-1.125, 1.125]] * 2))
        torch.testing.assert_close(expert.grad, tensor(([[1, 0], [7, 0]], [[3, 0], [9, 0]])[rank]))
        loss = y[0].detach().clone()
        all_reduce_sum(loss)
        torch.testing.assert_close(loss, tensor(10))
        with torch.no_grad():
            for parameter in (w, b, logits, expert):
                parameter.add_(parameter.grad, alpha=-.125)
        torch.testing.assert_close(w, tensor([[1.90625, -1.3125], [.875, 2.5625]]))
        if rank == 0:
            print(f"torch {torch.__version__}; two CPU ranks on a local pipe; global sum loss {loss.item()}")
            print("Dispatch, combine and both reverse exchanges passed. W after SGD:", w.tolist())
    finally:
        peer.close()


if __name__ == "__main__":
    context = get_context("spawn")
    ends = context.Pipe()
    workers = [context.Process(target=worker, args=(i, end)) for i, end in enumerate(ends)]
    for process in workers:
        process.start()
    for end in ends:
        end.close()
    try:
        for process in workers:
            process.join(timeout=60)
            if process.exitcode != 0:
                raise RuntimeError(f"Training worker failed or timed out: {process.exitcode}")
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join()
