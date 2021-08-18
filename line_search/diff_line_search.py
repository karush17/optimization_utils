import torch
from torch.autograd import grad
from utils import revert_params, get_params, flat_grad


class DiffLineSearch(object):
    def __init__(self, args, model, loss):
        self.args = args
        self.model = model
        self.params = self.model.parameters()
        self.loss = loss

    def criterion(self, inputs, outputs, search_dir, step_len):
        for grads in search_dir:
            grads *= step_len
        self.params = self.model.parameters()
        outs = self.model(inputs)
        prev_loss = self.loss(outs, outputs)

        for idx, p in enumerate(self.model.parameters()):
            p.data = p.data - step_len*search_dir[idx]

        with torch.no_grad():
            test_outs = self.model(inputs)
            loss_val = self.loss(test_outs, outputs)
            loss_improv_cond = loss_val <= prev_loss
            loss_diff = torch.abs(loss_val - prev_loss)
        
        revert_params(self.model, self.params)

        return loss_improv_cond, loss_diff

    def direction(self, inputs, outputs):
        outs = self.model(inputs)
        params = self.model.parameters()
        loss = self.loss(outs, outputs)
        dir = flat_grad(loss, params)
        return dir
    
    def optimize(self, inputs, outputs, search_dir, max_step_len, line_search_coef=0.8, max_iter=20):

        step_len = max_step_len / line_search_coef

        for i in range(max_iter):
            loss_improv, loss_diff = self.criterion(inputs, outputs, search_dir, step_len)

            if loss_improv:
                return step_len
            
            step_len  *= (loss_diff/max_step_len)
        
        return step_len
