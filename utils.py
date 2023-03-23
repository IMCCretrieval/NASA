import torch
from tqdm import tqdm
import math
import random
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F

BIG_NUMBER = 1e12
__all__ = [
    "AllPairs",
    "DistanceWeighted",
    "pdist"
]


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def pos_neg_mask(labels):
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) * (
        1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device)
    )
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) * (
        1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device)
    )

    return pos_mask, neg_mask


class _Sampler(nn.Module):
    def __init__(self, dist_func=pdist):
        self.dist_func = dist_func
        super().__init__()

    def forward(self, embeddings, labels):
        raise NotImplementedError

class DistanceWeighted(_Sampler):
    cut_off = 0.5
    nonzero_loss_cutoff = 1.4
    #Distance Weighted loss.

    def forward(self, embeddings, labels):
        with torch.no_grad():
            embeddings = F.normalize(embeddings, dim=1, p=2)
            pos_mask, neg_mask = pos_neg_mask(labels)       
            pos_pair_idx = pos_mask.nonzero()
            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            d = embeddings.size(1)
            dist = (
                pdist(embeddings, squared=True)
                + torch.eye(
                    embeddings.size(0), device=embeddings.device, dtype=torch.float32
                )
            ).sqrt()
            dist = dist.clamp(min=self.cut_off)

            log_weight = (2.0 - d) * dist.log() - ((d - 3.0) / 2.0) * (
                1.0 - 0.25 * (dist * dist)
            ).log()
            weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
            weight = weight * (neg_mask.cuda() * (dist < self.nonzero_loss_cutoff)).float()

            weight = (
                weight + ((weight.sum(dim=1, keepdim=True) == 0) * neg_mask.cuda()).float()
            )
            weight = weight / (weight.sum(dim=1, keepdim=True))
            weight = weight[anchor_idx]
            neg_idx = torch.multinomial(weight, 1).squeeze(1)

        return anchor_idx, pos_idx, neg_idx

def index_dataset(dataset: ImageFolder):
    kv = [(cls_ind, idx) for idx, (_, cls_ind) in enumerate(dataset.imgs)]
    cls_to_ind = {}

    for k, v in kv:
        if k in cls_to_ind:
            cls_to_ind[k].append(v)
        else:
            cls_to_ind[k] = [v]

    return cls_to_ind


class PKSampler:
    def __init__(self, data_source: ImageFolder, batch_size, m=5, iter_per_epoch=100):
        self.m = m
        self.batch_size = batch_size
        self.n_batch = iter_per_epoch
        self.class_idx = list(data_source.class_to_idx.values())
        self.images_by_class = index_dataset(data_source)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            selected_class = random.sample(self.class_idx, k=len(self.class_idx))
            example_indices = []

            for c in selected_class:
                img_ind_of_cls = self.images_by_class[c]
                new_ind = random.sample(
                    img_ind_of_cls, k=min(self.m, len(img_ind_of_cls))
                )
                example_indices += new_ind

                if len(example_indices) >= self.batch_size:
                    break

            yield example_indices[: self.batch_size]

