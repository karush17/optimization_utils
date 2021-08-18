import torch
from utils import flat_grad

def Hvp(functional, inputs, v):
    grad_f = flat_grad(functional, inputs, create_graph=True)
    jvp = torch.matmul(grad_f, v)
    hvp = flat_grad(jvp, inputs, retain_graph=True)
    return hvp
