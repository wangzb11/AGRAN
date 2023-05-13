import torch
import torch.nn.functional as F
import torch.nn as nn
import math

device = 'cuda'

class AGCN(nn.Module):

    def __init__(self,input_dim,output_dim, layer = 3,
                 dropout=0.2,
                 bias=False):

        super(AGCN,self).__init__()

        self.dropout = dropout
        self.layer_num = layer
        self.cos_weight = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(input_dim,output_dim)))

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def get_neighbor_hard_threshold(self,adj, epsilon=0, mask_value=0):

        mask = (adj > epsilon).detach().float()
        raw_adj = adj * mask + (1 - mask) * mask_value
        adj_dig = torch.clamp(torch.pow(torch.sum(raw_adj, dim=-1, keepdim=True),0.5), min=1e-12)
        update_adj = raw_adj/adj_dig/adj_dig.transpose(-1,-2)
        return update_adj,raw_adj

    def get_neighbor_soft_row_threshold(self,adj, epsilon=100, device=None):
        top_k = min(epsilon, adj.size(-1))
        _, index = torch.topk(adj, top_k, dim=-1)
        update_adj = torch.zeros_like(adj).scatter_(-1, index, 1)
        return update_adj

    def get_neighbor_soft_threshold(self,adj, epsilon, device=None):
        top_k = math.ceil(epsilon * adj.size(-1) ** 2)
        adj_float = adj.flatten()
        _, index = torch.topk(adj_float, top_k, dim=-1)
        update_adj = torch.zeros_like(adj_float).scatter_(-1, index, 1)
        update_adj = update_adj.reshape(adj.size(0), adj.size(1))
        return update_adj

    def cosine_matrix_div(self,emb):
        node_norm = emb.div(torch.norm(emb, p=2, dim=-1, keepdim=True))
        cos_adj = torch.mm(node_norm, node_norm.transpose(-1, -2))
        return cos_adj
    
    def weight_cosine_matrix_div(self,emb):
        emb = torch.matmul(emb,self.cos_weight)
        node_norm = F.normalize(emb,p=2,dim=-1)
        cos_adj = torch.mm(node_norm, node_norm.transpose(-1, -2))
        return cos_adj

    def cos_matirx_sim(self,emb):
        cos_adj = torch.cosine_similarity(emb.unsqueeze(0), emb.unsqueeze(1), dim=-1).detach()
        return cos_adj

    def forward(self,inputs):
        x = inputs.weight[1:,:]
        support = self.weight_cosine_matrix_div(x)
        support,support_loss = self.get_neighbor_hard_threshold(support)

        if self.training:
            support = F.dropout(support,self.dropout)

        x_fin = [x]
        layer = x
        for f in range(self.layer_num):
            layer = torch.matmul(support,layer)
            layer = torch.tanh(layer)
            x_fin += [layer]
        x_fin = torch.stack(x_fin,dim=1)
        out = torch.sum(x_fin,dim=1)

        if self.bias is not None:
            out += self.bias

        fin_out = torch.cat([inputs.weight[0, :].unsqueeze(dim=0),out], dim=0)

        return fin_out,support_loss
