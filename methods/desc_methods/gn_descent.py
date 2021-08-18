import torch
from torch.autograd import grad
from utils import get_params, flat_grad

class NaiveGaussNewtDescent(object):
    def __init__(self, args, model, loss):
        self.args = args
        self.model = model
        self.params = self.model.parameters()
        self.loss = loss
    
    def optimize(self, inputs, targets, alpha, epoch):
        outputs = self.model(inputs)
        self.params = self.model.parameters()
        loss = self.loss(outputs, targets).mean()
        grads = list(flat_grad(loss, self.params))
        for idx, p in enumerate(self.model.parameters()):
            p.data = p.data - alpha*grads[idx]*grads[idx]
        return self.model, loss.item()

