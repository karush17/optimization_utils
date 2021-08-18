import torch
from torch.autograd import grad
from utils import get_params, flat_grad


class PrimalDualConstrainedDescent(object):
    def __init__(self, args, model, loss):
        self.args = args
        self.model = model
        self.params = self.model.parameters()
        self.loss = loss
        self.lamb = torch.FloatTensor([1])
    
    def optimize(self, inputs, targets, alpha, epoch):
        outputs = self.model(inputs)
        self.params = self.model.parameters()
        loss_arr = self.loss(outputs, targets)
        loss = loss_arr + self.lamb*(torch.exp(loss_arr) - 1)
        loss = loss.mean()
        grads = flat_grad(loss, self.params, retain_graph=True)
        for idx, p in enumerate(self.model.parameters()):
            p.data = p.data - alpha*grads[idx]
        self.lamb  = self.lamb - alpha*(torch.exp(loss_arr).detach() - 1)
        return self.model, loss.item()

