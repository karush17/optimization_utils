import torch
from torch.autograd import grad
from utils import get_params, flat_grad


class NewtDescent(object):
    def __init__(self, args, model, loss):
        self.args = args
        self.model = model
        self.params = self.model.parameters()
        self.loss = loss
    
    def optimize(self, inputs, targets, alpha, epoch):
        outputs = self.model(inputs)
        self.params = self.model.parameters()
        loss = self.loss(outputs, targets).mean()
        grads = flat_grad(loss, self.params, create_graph=True)
        for idx, p in enumerate(self.model.parameters()):
            p.data = p.data - alpha*(loss / grads[idx]).data
        return self.model, loss.item()

