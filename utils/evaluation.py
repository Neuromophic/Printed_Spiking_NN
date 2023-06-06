import torch
import numpy as np

class Evaluator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sensing_margin = 0.01
        self.performance = self.nominal
        self.args = args
        
    def nominal(self, prediction, label):
        act, idx = torch.max(prediction, dim=1)
        corrects = (label.view(-1) == idx)
        return corrects.float().sum().item() / label.numel()
    
    def maa(self, prediction, label):
        act, idx = torch.topk(prediction, k=2, dim=1)
        corrects = (act[:,0] >= self.sensing_margin) & (act[:,1]<=0) & (label.view(-1)==idx[:,0])
        return corrects.float().sum().item() / label.numel()
        
    def forward(self, nn, x, label):
        if self.args.metric == 'acc':
            self.performance = self.nominal
        elif self.args.metric == 'maa':
            self.performance = self.maa
        return self.performance(nn(x), label), nn.Power()
    
  