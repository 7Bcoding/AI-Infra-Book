"""Chapter 9. On a Triton-supported CUDA GPU, run: python triton_projection.py"""
import torch
import triton
import triton.language as tl


@triton.jit
def projection(X, W, Y, M, N, K, sxm, sxk, swk, swn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    k0 = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(triton.cdiv(K, BK)):
        k = block * BK + k0
        left = tl.load(X + rows[:, None]*sxm + k[None, :]*sxk,
                       mask=(rows[:, None] < M) & (k[None, :] < K), other=0.)
        right = tl.load(W + k[:, None]*swk + cols[None, :]*swn,
                        mask=(k[:, None] < K) & (cols[None, :] < N), other=0.)
        acc = tl.dot(left, right, acc)
    tl.store(Y + rows[:, None]*N + cols[None, :], acc,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def run(x, w):
    m, k = x.shape
    depth, n = w.shape
    assert depth == k and x.dtype == w.dtype == torch.float16
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    tile = 16
    grid = (triton.cdiv(m, tile), triton.cdiv(n, tile))
    projection[grid](x, w, out, m, n, k, *x.stride(), *w.stride(), tile, tile, tile)
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("Use CUDA-enabled PyTorch and a GPU supported by Triton.")
    x = torch.tensor([[1,2],[3,4]], device="cuda", dtype=torch.float16)
    w = torch.tensor([[2,-1],[1,3]], device="cuda", dtype=torch.float16)
    assert run(x,w).tolist() == [[4.,5.],[10.,9.]]
    torch.manual_seed(9)
    for m,n,k in [(5,3,5),(16,16,32),(35,53,67)]:
        # Small integers keep every product and sum exactly representable in FP32.
        x = torch.randint(-3,4,(m,k),device="cuda").half()
        weight_rows = torch.randint(-3,4,(n,k),device="cuda").half()
        for w in (weight_rows.T, weight_rows.T.contiguous()):
            ref = x.cpu().double() @ w.cpu().double()
            torch.testing.assert_close(run(x,w).cpu().double(), ref, rtol=0, atol=0)
    print("PASS: chapter matrix, reduction and output fringes, transposed weight view")

