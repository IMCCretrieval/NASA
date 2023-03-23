import torch
import torch.nn as nn
import torch.nn.functional as F
from nasa.utils import pdist

class Twin_Embeddings(nn.Module):
    def __init__(self, base, feature_size=512, embedding_size=512):
        super(Twin_Embeddings, self).__init__()
        self.base = base
        self.linear = nn.Linear(feature_size, embedding_size)
        self.linear_strc = nn.Sequential(nn.Linear(feature_size, 512), 
                                      nn.BatchNorm1d(512),
                                      nn.Linear(512, embedding_size), 
                                      nn.BatchNorm1d(embedding_size),
                                      nn.Sigmoid() )
    def forward(self, x):
        feat = self.base(x)
        feat = feat.view(x.size(0), -1)
        embedding = self.linear(feat) 
        embedding_strc = 1+0.5*(2*self.linear_strc(feat)-1)
        if self.training:
            dist_mat = pdist(embedding)
            mean_d = dist_mat[~torch.eye(dist_mat.shape[0], dtype=torch.bool, device=dist_mat.device)].mean()
            return embedding / mean_d, embedding_strc

        return embedding,embedding_strc
