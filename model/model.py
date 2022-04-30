import timm

import torch
import torch.nn as nn
from transformers import AutoModel, ElectraModel



'''
class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(args.model) 
        self.drop = nn.Dropout(args.drop_rate)
        self.fc = nn.Linear(args.feature_dim, args.num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight) 
        
        
    def forward(self, x):
        x = self.model(x[0], attention_mask=x[1],)[1]
        x = self.drop(x)
        return self.fc(x)

'''

class Model(nn.Module):


    def __init__(self, args):
        super(Model, self).__init__()
        self.model = ElectraModel.from_pretrained(args.model) 
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.fc = nn.Linear(args.feature_dim, args.num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight) 
        
    def forward(self, x):
        x = self.model(x[0], attention_mask=x[1],)[0][:,0,:]
        x = self.dropout(x)
        return self.fc(x)
