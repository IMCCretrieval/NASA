import os
import argparse
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import dataset
import network.backbone as backbone
import nasa.loss as loss
import nasa.utils as utils
from tqdm import tqdm
from torch.utils.data import DataLoader
from network.twin_heads import Twin_Embeddings
from utils.eval import *


def nasa_train(model, loader, optimizer, criterion, reg_criterion,lambda_1, args,ep=0):
    model.train()
    fix_batchnorm(model)
    train_iter = tqdm(loader, ncols=80)
    loss_all = []
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()
        embedding,embedding_strc = model(images)
        loss_metric = criterion(embedding, labels)
        loss_strc = lambda_1* reg_criterion(embedding,embedding_strc)
        optimizer.zero_grad()
        (loss_metric + loss_strc).backward()
        optimizer.step()
        train_iter.set_description("epoch %d: loss_metric: %.5f loss_strc: %.5f"% (ep, loss_metric.item(), loss_strc.item()))
    
def nasa_eval(model, loader, K=[1], ep=0):
    model.eval()
    test_iter = tqdm(loader, ncols=80)
    embeddings_all, embeddings_all_strc,labels_all = [],[], []

    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding, embedding_strc = model(images)
            embeddings_all.append(embedding.data)
            embeddings_all_strc.append(embedding_strc.data)
            labels_all.append(labels.data)

        embeddings_all = torch.cat(embeddings_all)
        embeddings_all_strc = torch.cat(embeddings_all_strc)
        labels_all = torch.cat(labels_all)
        rec,rec_strc = recall_strc(embeddings_all,embeddings_all_strc, labels_all, K=K)
        for k,r_w,r in zip(K,rec_strc,rec):
            print("epoch %d: NASA+ R@%d: [%.4f] NASA R@%d: [%.4f]\n" % (ep, k,r_w,k,r))

    return rec[0], K, rec, rec_strc[0],rec_strc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    LookupChoices = type("",(argparse.Action,),dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--dataset",choices=dict(cars=dataset.Cars196Metric,), default=dataset.Cars196Metric, action=LookupChoices,)
    parser.add_argument("--backbone",default=backbone.BNInception,)
    parser.add_argument("--lambda_1", type=float, default=2.0)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--lr-decay-epochs", type=int, default=[40, 60, 80], nargs="+")
    parser.add_argument("--lr-decay-gamma", default=0.2, type=float)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--IPC", default=4, type=int)
    parser.add_argument("--iter-per-epoch", default=100, type=int)
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("--recall", default=[1, 2, 4, 8, 32], type=int, nargs="+")
    parser.add_argument("--scale", default=10, type=int)
    parser.add_argument("--margin", default=3, type=int)
    parser.add_argument("--seed", default=random.randint(1, 1000), type=int)
    parser.add_argument("--data", default="../MyDataset")
    parser.add_argument("--save-dir", default="./log/result")
    opts = parser.parse_args()

    counter = 1
    checkfolder = opts.save_dir
    while os.path.exists(checkfolder):
        checkfolder = opts.save_dir + '_' +str(opts.dataset.base_folder[0:3]) + '_'+str(opts.dim)+'_' + str(counter)
        counter += 1
    opts.save_dir = checkfolder
    os.makedirs(opts.save_dir)
    
    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
        set_random_seed(opts.seed)

    base_model = opts.backbone(pretrained=True)
    model = Twin_Embeddings(base_model,feature_size=base_model.output_size,embedding_size=opts.dim,).cuda()
    
    train_transform, test_transform = build_transform(base_model)
    dataset_train = opts.dataset(
        opts.data, train=True, transform=train_transform, download=True
    )
    dataset_train_eval = opts.dataset(
        opts.data, train=True, transform=test_transform, download=True
    )
    dataset_eval = opts.dataset(
        opts.data, train=False, transform=test_transform, download=True
    )

    print("Training images: %d" % len(dataset_train))
    print("Test images: %d" % len(dataset_eval))

    loader_train = DataLoader(
        dataset_train,
        batch_sampler=utils.PKSampler(
            dataset_train, opts.batch, m=opts.IPC, iter_per_epoch=opts.iter_per_epoch
        ),
        pin_memory=True,
        num_workers=8,
    )
    loader_eval = DataLoader(
        dataset_eval,
        shuffle=False,
        batch_size=64,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
    )

    criterion =loss.Triplet(sampler=utils.DistanceWeighted(), margin=0.2).cuda()
    reg_criterion = loss.NASA_loss(t=opts.margin,alpha=opts.scale).cuda()

    optimizer = optim.Adam(
        [
            {"lr": opts.lr, "params": model.parameters()},
            {"lr": opts.lr, "params": criterion.parameters()},
            {"lr": opts.lr, "params": reg_criterion.parameters()},
        ],
        weight_decay=1e-5,
    )

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma
    )

    best_rec=0.0
    best_rec_strc=0.0  

    for epoch in range(1, opts.epochs + 1):
        nasa_train(
            model,
            loader_train,
            optimizer,
            criterion,
            reg_criterion,
            lambda_1=opts.lambda_1,
            args=opts,
            ep=epoch,
        )
        
        lr_scheduler.step()

        _, _, _, nasa_r_strc,nasa_r_nasa_all = nasa_eval(model, loader_eval, opts.recall, epoch)
        if best_rec_strc < nasa_r_strc:
            best_rec_strc = nasa_r_strc
            best_rec_all_strc=nasa_r_nasa_all

            
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
#                 torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "triplet.pth"))

        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            with open("%s/result.txt" % opts.save_dir, "a+") as f:
                f.write("Best nasa+ Recall_@1: %.4f\n" % best_rec_strc)
                f.write("Best nasa+ Recall@2: %.4f\n" % best_rec_all_strc[1])
                f.write("Best nasa+ Recall@4: %.4f\n" % best_rec_all_strc[2])
                f.write("Best nasa+ Recall@8: %.4f\n" % best_rec_all_strc[3])

        print("Best nasa+ Recall_@1: %.4f" % best_rec_strc)