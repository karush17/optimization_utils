import torch
from utils import flat_grad

def Vhp(functional, inputs, v):
    grad_f = flat_grad(functional, inputs, create_graph=True)
    vjp = torch.matmul(v, grad_f)
    vhp = flat_grad(vjp, input, retain_graph=True)
    return vhp
