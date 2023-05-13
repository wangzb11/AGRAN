import torch
import torch.nn.functional as F
import torch.nn as nn
from utils_ag import sparse_dropout
import math

device = 'cuda'


class AGCN_anchor(nn.Module):
    def __init__(self,input_dim,output_dim,
                 dropout=0.2,
                 layer = 2,
                 is_sparse_inputs=False,
                 bias=False):

        super(AGCN_anchor,self).__init__()
        self.dropout = dropout
        self.layer_num = layer
        self.is_sparse_inputs = is_sparse_inputs
        self.output_dim = output_dim
        self.anchor = torch.nn.init.xavier_uniform_(torch.empty(100,self.output_dim))
        self.weight = nn.ParameterList()
        self.cos_weight = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(input_dim,output_dim)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        self.bn = nn.BatchNorm1d(output_dim)
    
    def cosine_matrix_div(self,emb,anchor):
        node_norm = emb.div(torch.norm(emb, p=2, dim=-1, keepdim=True))
        anchor_norm = anchor.div(torch.norm(anchor,p=2,dim=-1,keepdim=True))
        cos_adj = torch.mm(node_norm, anchor_norm.transpose(-1, -2))
        return cos_adj
    
    def get_neighbor_hard_threshold(self,adj, epsilon=0, mask_value=0):
        mask = (adj > epsilon).detach().float()
        update_adj = adj * mask + (1 - mask) * mask_value
        return update_adj

    def forward(self,inputs,anchor_idx):
        x = inputs.weight[1:,:]
        anchor = inputs(anchor_idx)
        anchor_adj = self.cosine_matrix_div(x,anchor)
        anchor_adj = self.get_neighbor_hard_threshold(anchor_adj)

        x_fin = [x]
        layer = x
        for f in range(self.layer_num):
            node_norm = anchor_adj/torch.clamp(torch.sum(anchor_adj,dim=-2,keepdim=True),min=1e-12) #mp1
            anchor_norm = anchor_adj/torch.clamp(torch.sum(anchor_adj,dim=-1,keepdim=True),min=1e-12) #mp2
            layer = torch.matmul(anchor_norm,torch.matmul(node_norm.transpose(-1,-2),layer))+layer
            x_fin+=[layer]
        x_fin = torch.stack(x_fin,dim=1)
        output = torch.sum(x_fin,dim=1)

        mp2 = torch.cat([inputs.weight[0, :].unsqueeze(dim=0),output], dim=0)
        return mp2,anchor_adj
