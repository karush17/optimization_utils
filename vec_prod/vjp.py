import torch
from utils import flat_grad

def Vjp(functional, inputs, v):
    grad_f = flat_grad(functional, inputs, create_graph=True)
    return torch.matmul(v, grad_f)
