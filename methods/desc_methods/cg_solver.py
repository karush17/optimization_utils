import torch

def cg_solve(func, b, max_iter=10):
    if b.ndim==2:
        b = b.mean(dim=0)
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    for i in range(max_iter):
        val = func(p)
        alp = torch.matmul(r, r) / torch.matmul(p, val)
        x += alp*p

        if i==max_iter-1:
            return x
        
        r_new = r - alp*val
        beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
        r = r_new
        p = r + beta*p
