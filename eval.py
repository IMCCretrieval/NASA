import torch
import torchvision.transforms as transforms
import network.backbone as backbone

from tqdm import tqdm
from PIL import Image


def recall_strc (embeddings,embeddings_strc, labels, K=[]):
    knn_inds = []
    knn_inds_w = []
    evaluation_iter = tqdm(embeddings, ncols=80)
    evaluation_iter.set_description("test:")
    for i, e in enumerate(evaluation_iter):
        d = (e.unsqueeze(0) - embeddings).pow(2).sum(dim=1).clamp(min=1e-12)

        d[i] = 0
        knn_ind = d.topk(1 + max(K), dim=0, largest=False, sorted=True)[1][1:]
        knn_inds.append(knn_ind)
        d_w=torch.mul(embeddings_strc[i],(e.unsqueeze(0) - embeddings[knn_ind]).pow(2)).sum(dim=1).clamp(min=1e-12)
        knn_ind_w = d_w.topk( max(K), dim=0, largest=False, sorted=True)[1] 
        knn_inds_w.append(knn_ind[knn_ind_w])
    knn_inds = torch.stack(knn_inds, dim=0)
    knn_inds_w = torch.stack(knn_inds_w, dim=0)

    assert (
        knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)
    ).sum().item() == 0
    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels
    recall_k = []
    selected_labels_w = labels[knn_inds_w.contiguous().view(-1)].view_as(knn_inds_w)
    correct_labels_w = labels.unsqueeze(1) == selected_labels_w
    recall_k_strc = []
    for k in K:
        correct_k = 100 * (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
        correct_k_w = 100 * (correct_labels_w[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k_strc.append(correct_k_w)
    return recall_k,recall_k_strc

def fix_batchnorm(net):
    for m in net.modules():
        if (
            isinstance(m, torch.nn.BatchNorm1d)
            or isinstance(m, torch.nn.BatchNorm2d)
            or isinstance(m, torch.nn.BatchNorm3d)
        ):
            m.eval()


def build_transform(model):
    if isinstance(model, backbone.BNInception):
        normalize = transforms.Compose(
            [
                transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255.0),
                transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
            ]
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            transforms.RandomResizedCrop(
                scale=(0.16, 1),
                ratio=(0.75, 1.33),
                size=224,
                interpolation=Image.LANCZOS,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, test_transform
