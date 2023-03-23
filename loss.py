import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from nasa.utils import pdist 
import numpy as np

class NASA_loss(nn.Module):
    def __init__(self, t= 3,alpha=10):
        super().__init__()
        self.margin=t
        self.scale=alpha
    def forward(self, embeddings,embeddings_strc):
        dist_mat_ori = pdist(embeddings).view(-1)
        mean = dist_mat_ori.mean().detach()
        std = dist_mat_ori.std().detach()
        self.eta =  torch.tensor([mean, torch.max(0.2*mean,mean-self.margin*std)]) 
        
        mm,nn=embeddings.shape[0],embeddings.shape[1]
        adapt_dist_mat = torch.zeros([mm,mm ])
        for i in range(0, embeddings.shape[0]):
            dis =  (embeddings[i] - embeddings).pow(2).clamp(min=1e-12)
            tmp = torch.mul(embeddings_strc[i].expand(mm, nn),dis)
            adapt_dist_mat[i] = torch.sqrt(torch.sum(tmp,1)) 
            
        dis_mat_noanchor = adapt_dist_mat[~torch.eye(adapt_dist_mat.shape[0], dtype=torch.bool, device=adapt_dist_mat.device,)]
        pos_group=(dis_mat_noanchor[None].cuda() - self.eta[:, None].cuda()).abs()[0]
        neg_group=(dis_mat_noanchor[None].cuda() - self.eta[:, None].cuda()).abs()[1]
        c=torch.exp(-self.scale*pos_group)/(torch.exp(-self.scale*pos_group)+torch.exp(-self.scale*neg_group))
        self_ranking_loss = (c*pos_group+(1-c)*neg_group).mean() 
        
        return self_ranking_loss    

class Triplet(nn.Module):
    def __init__(self, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.margin = margin
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p == 2))
        self.reduce = reduce
        self.size_average = size_average
    def forward(self, x, y):
        a_id, p_id, n_id = self.sampler(x, y)
        loss_metric = F.triplet_margin_loss(x[a_id],x[p_id],x[n_id],margin=self.margin,reduction="none")
        return loss_metric.mean()
