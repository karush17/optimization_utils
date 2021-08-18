import torch
from torch.autograd import grad

def flatten(vecs):
    flattened = torch.cat([v.view(-1) for v in vecs])
    return flattened

def get_params(model):
    params = model.parameters()
    return params

def revert_params(model, new_params):
    new_params = flatten([param.view(-1) for param in new_params])
    n = 0
    for param in model.parameters():
        numel = param.numel()
        new_param = new_params[n:n+numel].view(param.size())
        param.data = new_param
        n += numel

def flat_grad(functional, input, retain_graph=False, create_graph=False):
    if create_graph==True:
        retain_graph = True    
    grads = grad(functional, input, retain_graph=retain_graph, create_graph=create_graph)
    return grads

def normalize(x):
    mean = torch.mean(x)
    std = torch.std(x)
    x_norm = (x - mean) / std
    return x_norm

def log(args, epoch, loss, step_rate):
    print('Epoch: {} | Loss: {}'.format(epoch, loss))
    if args.line_search==True:
        print('Line Search Step Length: {}'.format(step_rate))
