import torch
from torch.autograd import grad, Variable
from utils import get_params, flat_grad


class ConstrainedLagrangianDescent(object):
    def __init__(self, args, model, loss):
        self.args = args
        self.model = model
        self.params = self.model.parameters()
        self.loss = loss
        self.log_lamb = torch.zeros(1, requires_grad=True)
        self.gamma = torch.Tensor([1])
    
    def optimize(self, inputs, targets, alpha, epoch):
        outputs = self.model(inputs)
        self.params = self.model.parameters()
        loss_arr = self.loss(outputs, targets)
        lamb = self.log_lamb.exp()
        loss = loss_arr + lamb*(torch.exp(loss_arr) - 1) + self.gamma*(lamb - 1)
        loss = loss.mean()
        lamb_loss = - loss
        lamb_grad = flat_grad(lamb_loss, self.log_lamb, retain_graph=True)[0]
        self.log_lamb = self.log_lamb - alpha*lamb_grad
        grads = flat_grad(loss, self.params, retain_graph=True)
        for idx, p in enumerate(self.model.parameters()):
            p.data = p.data - alpha*grads[idx]
        self.gamma = self.gamma - alpha*(lamb.detach() - 1)
        return self.model, loss.item()

