import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, device, lr_mul = 1., n_warmup_steps = 2000, quantile = 0.7, is_quantile = False, warmup_epoch = 0):
        self.model = TESTAM(device, num_nodes, dropout, in_dim=in_dim, out_dim=seq_length, hidden_size=nhid)
        self.model.to(device)
        # The learning rate setting below will not affct initial learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3, betas = (0.9, 0.98), eps = 1e-9)
        self.schedule = util.CosineWarmupScheduler(self.optimizer, d_model = nhid, n_warmup_steps = n_warmup_steps, lr_mul = lr_mul)
        self.loss = util.masked_mae
        
        self.scaler = scaler
        self.clip = 5
        self.n_warmup_steps = n_warmup_steps
        self.flag = is_quantile
        self.quantile = quantile
        self.cur_epoch = 0
        self.warmup_epoch = warmup_epoch

    def train(self, input, real_val, cur_epoch):
        self.model.train()
        self.schedule.zero_grad()
        
        output, gate, res = self.model(input)
        
        predict = self.scaler.inverse_transform(output)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)

        ind_loss = self.loss(self.scaler.inverse_transform(res), real.permute(0,2,3,1), 0.0, reduce = None)
        if self.flag:
            gated_loss = self.loss(predict, real, reduce = None).permute(0,2,3,1)
            l_worst_avoidance, l_best_choice = self.get_quantile_label(gated_loss, gate, real)
        else:
            l_worst_avoidance, l_best_choice = self.get_label(ind_loss, gate, real)

        worst_avoidance = -.5 * l_worst_avoidance * torch.log(gate)
        best_choice = -.5 * l_best_choice * torch.log(gate)

        if cur_epoch <= self.warmup_epoch:
            loss = self.loss(self.scaler.inverse_transform(res), real.permute(0,2,3,1), 0.0)
        else:
            loss = ind_loss.mean() + worst_avoidance.mean() + best_choice.mean()
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.schedule.step_and_update_lr()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def get_quantile_label(self, gated_loss, gate, real):
        max_quantile = gated_loss.quantile(self.quantile)
        min_quantile = gated_loss.quantile(1 - self.quantile)
        incorrect = (gated_loss > max_quantile).expand_as(gate)
        correct = ((gated_loss < min_quantile) & (real.permute(0,2,3,1) != 0)).expand_as(gate)
        cur_expert = gate.argmax(dim = -1, keepdim = True)
        not_chosen = gate.topk(dim = -1, k = 2, largest = False).indices
        selected = torch.zeros_like(gate).scatter_(-1, cur_expert, 1.0)
        scaling = torch.zeros_like(gate).scatter_(-1, not_chosen, 0.5)
        selected[incorrect] = scaling[incorrect]
        l_worst_avoidance = selected.detach()
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0) * correct
        l_best_choice = selected.detach()
        return l_worst_avoidance, l_best_choice

    def get_label(self, ind_loss, gate, real):
        empty_val = (real.permute(0,2,3,1).expand_as(gate)) == 0
        max_error = ind_loss.argmax(dim = -1, keepdim = True)
        cur_expert = gate.argmax(dim = -1, keepdim = True)
        incorrect = max_error == cur_expert
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0)
        scaling = torch.ones_like(gate)
        scaling[max_error] = 0.
        scaling = scaling / 2 * (1 - selected)
        l_worst_avoidance = torch.where(incorrect, scaling, selected)
        l_worst_avoidance = torch.where(empty_val, torch.zeros_like(gate), l_worst_avoidance)
        l_worst_avoidance = l_worst_avoidance.detach()
        min_error = ind_loss.argmin(dim = -1, keepdim = True)
        correct = min_error == cur_expert
        scaling = torch.ones_like(gate)
        scaling[min_error] = 2.
        scaling = scaling / scaling.sum(dim = -1, keepdim = True)
        l_best_choice = torch.where(correct, selected, scaling)
        l_best_choice = torch.where(empty_val, torch.zeros_like(gate), l_best_choice)
        l_best_choice = l_best_choice.detach()
        return l_worst_avoidance, l_best_choice
