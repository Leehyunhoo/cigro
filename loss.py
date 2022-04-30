import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    def __init__(self, args):
        super(ClassificationLoss, self).__init__()
        self.gamma = args.gamma
        self.weight = args.weight
        
    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, targets, self.weight)
        return loss
    
