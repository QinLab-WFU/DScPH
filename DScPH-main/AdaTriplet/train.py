import json
import os
import time
from copy import deepcopy

import torch
from loguru import logger
# pip install pytorch-metric-learning
from pytorch_metric_learning import distances, reducers
from torch.optim import Adam

from AdaTriplet.config import get_config
from AdaTriplet.losses import TripletCustomMarginLoss, LowerBoundLoss
from AdaTriplet.methods import MetricLearningMethods
from AdaTriplet.miners.triplet_automargin_miner import TripletAutoParamsMiner
from AdaTriplet.networks import BackboneModel


def build_model(args, pretrained=True):
    net = BackboneModel(args, pretrained)
    return net.cuda()


def train_epoch(args, model, optimizer, train_loader, epoch_id):
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)

    # mining_func = mining_func["AutoParams"]
    # loss_matching_func = loss_func["Triplet"]
    # loss_id_func = loss_id_selected
    mining_func = TripletAutoParamsMiner(
        distance=distance,
        margin_init=args.margin_m_loss,
        beta_init=args.margin_beta,
        type_of_triplets=args.type_of_triplets,
        k=args.k_param_automargin,
        k_n=args.k_n_param_autobeta,
        k_p=args.k_p_param_autobeta,
        mode=args.automargin_mode,
    )
    loss_matching_func = TripletCustomMarginLoss(margin=args.margin_m_loss, distance=distance, reducer=reducer)

    loss_id_func = LowerBoundLoss()

    model.train()
    n_batches = len(train_loader)

    sum_loss = 0
    sum_triplets = 0
    sum_neg_pairs = 0
    sum_pos_pairs = 0
    for batch_id, batch in enumerate(train_loader):
        data = batch[0].cuda()
        labels = batch[1].cuda()
        optimizer.zero_grad()
        embeddings = model(data)

        method = MetricLearningMethods(args, mining_func, loss_matching=loss_matching_func, loss_identity=loss_id_func)
        loss = method.calculate_total_loss(embeddings, labels, epoch_id=epoch_id, batch_id=batch_id)
        no_triplets_batch, no_neg_pairs_batch, no_pos_pairs_batch = method.get_no_triplets()
        if ~torch.isnan(loss):
            sum_loss += loss
        sum_triplets += no_triplets_batch
        sum_neg_pairs += no_neg_pairs_batch
        sum_pos_pairs += no_pos_pairs_batch

        loss.backward()
        optimizer.step()
    mean_loss = sum_loss / n_batches
    mean_triplets = sum_triplets / n_batches
    mean_neg_pairs = sum_neg_pairs / n_batches
    mean_pos_pairs = sum_pos_pairs / n_batches
    print(f"Average Loss: {mean_loss}")
    print(f"Average numbers of triplets: {mean_triplets}")
    print(f"Average numbers of negative pairs: {mean_neg_pairs}")
    print(f"Average numbers of positive pairs: {mean_pos_pairs}")
    return mean_loss


def train_val(args, train_loader, query_loader, dbase_loader, logger):
    model = build_model(args)

    # distance = distances.CosineSimilarity()

    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_map = 0.0
    best_epoch = 0
    best_checkpoint = None
    count = 0
    for epoch in range(args.n_epochs):
        epoch_loss = train_epoch(args, model, optimizer, train_loader, epoch)



def prepare_loaders(args, bl_func):
    train_loader, query_loader, dbase_loader = bl_func(
        args.data_dir,
        args.dataset,
        None,
        None,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    return train_loader, query_loader, dbase_loader


if __name__ == "__main__":
    init("0")

    args = get_config()
    # args.automargin_mode = "adaptive"

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args, build_loaders)

        for hash_bit in [16, 32, 48, 64, 128]:
            print(f"processing hash-bit: {hash_bit}")
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", rotation="500 MB", level="INFO")

            with open(f"{args.save_dir}/config.json", "w+") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            best_epoch, best_map = train_val(args, train_loader, query_loader, dbase_loader, logger)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})
    for x in rst:
        print(
            f"[dataset:{x['dataset']}][bits:{x['hash_bit']}][best-epoch:{x['best_epoch']}][best-mAP:{x['best_map']:.3f}]"
        )
