from argparse import Namespace

import torch


class MetricLearningMethods(torch.nn.Module):
    def __init__(self, args: Namespace, mining_func, loss_matching, loss_identity=None):
        super(MetricLearningMethods, self).__init__()
        self.args = args
        self.mining_function = mining_func
        self.loss_matching = loss_matching
        self.loss_identity = loss_identity
        self.list_dist = {}

    def get_no_triplets(self):
        num_triplets = self.mining_function.num_triplets
        if self.args.method in ["SCT", "AdaTriplet", "AdaTriplet-AM"]:
            num_negative_pairs = self.mining_function.num_negative_pairs
        else:
            num_negative_pairs = 0
        if self.args.automargin_mode == "ap":
            num_positive_pairs = self.mining_function.num_positive_pairs
        else:
            num_positive_pairs = 0
        return num_triplets, num_negative_pairs, num_positive_pairs

    def distance(self, f_a, f_p, f_n):
        if self.args.distance_loss == "cosine":
            no_triplets = f_a.shape[0]
            no_features = f_a.shape[1]

            ap = torch.matmul(f_a.view(no_triplets, 1, no_features), f_p.view(no_triplets, no_features, 1))
            an = torch.matmul(f_a.view(no_triplets, 1, no_features), f_n.view(no_triplets, no_features, 1))
            d = an - ap + self.args.margin_m_loss

        elif self.args.distance_loss == "l2":
            d_ap = torch.nn.functional.pairwise_distance(f_p, f_a, p=2)
            d_an = torch.nn.functional.pairwise_distance(f_n, f_a, p=2)
            d = d_ap - d_an + self.args.margin_m_loss
        else:
            raise ValueError(f"Not support distance type {self.args.distance_loss}")
        d = torch.squeeze(d)
        return d

    def distance_an(self, f_a, f_n):
        if self.args.distance_loss == "cosine":
            no_tripets = f_a.shape[0]
            no_features = f_a.shape[1]
            an = torch.matmul(f_a.view(no_tripets, 1, no_features), f_n.view(no_tripets, no_features, 1))
            d_an = torch.squeeze(an)
        elif self.args.distance_loss == "l2":
            d_an = torch.nn.functional.pairwise_distance(f_n, f_a, p=2)
        else:
            raise ValueError(f"Not support distance type {self.args.distance_loss}")
        return d_an

    def extract_regu_features(self, f_a, f_i, pair_type=None):
        no_tripets = f_a.shape[0]
        no_features = f_a.shape[1]
        if pair_type == "negative":
            if self.args.method == "AdaTriplet":
                beta_n = float(self.args.margin_beta)
            else:
                beta_n = self.auto_beta_n
            an = torch.matmul(f_a.view(no_tripets, 1, no_features), f_i.view(no_tripets, no_features, 1))
            regu = an - beta_n
            embeddings_regu = torch.squeeze(regu).cuda(1)
        elif pair_type == "positive":
            if self.args.method == "ap":
                beta_p = float(self.args.margin_beta)
            else:
                beta_p = self.auto_beta_p
            ap = torch.matmul(f_a.view(no_tripets, 1, no_features), f_i.view(no_tripets, no_features, 1))
            regu = beta_p - ap
            embeddings_regu = torch.squeeze(regu).cuda(1)
        else:
            raise NotImplementedError

        return embeddings_regu

    def calculate_total_loss(self, embeddings, labels, epoch_id=-1, batch_id=-1):
        self.mining_function.set_epoch_id_batch_id(epoch_id, batch_id)
        indices = self.mining_function(embeddings, labels)
        indices_tuple = indices[0]
        indices_negative_pairs = indices[1]
        indices_positive_pairs = indices[2]
        auto_margin = self.mining_function.get_margin()
        self.auto_beta_n = self.mining_function.get_beta_n()
        self.auto_beta_p = self.mining_function.get_beta_p()
        self.loss_matching.set_margin(auto_margin)

        if self.loss_identity is not None:
            if len(indices_negative_pairs[0]) > 0 and len(indices_positive_pairs[0]) == 0:
                f_anchor_neg_pairs = embeddings[indices_negative_pairs[0]]
                f_neg_pairs = embeddings[indices_negative_pairs[1]]
                embeddings_regu = self.extract_regu_features(f_anchor_neg_pairs, f_neg_pairs, pair_type="negative")
                loss_neg = self.loss_identity(embeddings_regu)
                loss_pos = 0
            elif len(indices_negative_pairs[0]) == 0 and len(indices_positive_pairs[0]) > 0:
                f_anchor_pos_pairs = embeddings[indices_positive_pairs[0]]
                f_pos_pairs = embeddings[indices_positive_pairs[1]]
                embeddings_regu = self.extract_regu_features(f_anchor_pos_pairs, f_pos_pairs, pair_type="positive")
                loss_pos = self.loss_identity(embeddings_regu)
                loss_neg = 0
            elif len(indices_negative_pairs[0]) > 0 and len(indices_positive_pairs[0]) > 0:
                f_anchor_neg_pairs = embeddings[indices_negative_pairs[0]]
                f_neg_pairs = embeddings[indices_negative_pairs[1]]
                neg_embeddings_regu = self.extract_regu_features(f_anchor_neg_pairs, f_neg_pairs, pair_type="negative")
                loss_neg = 2 * self.loss_identity(neg_embeddings_regu)

                f_anchor_pos_pairs = embeddings[indices_positive_pairs[0]]
                f_pos_pairs = embeddings[indices_positive_pairs[1]]
                pos_embeddings_regu = self.extract_regu_features(f_anchor_pos_pairs, f_pos_pairs, pair_type="positive")
                loss_pos = self.loss_identity(pos_embeddings_regu)
            else:
                loss_neg = 0
                loss_pos = 0
            loss_id = self.args.loss_w_neg * loss_neg + loss_pos
        else:
            loss_id = 0

        loss_matching = self.loss_matching(embeddings, labels, indices_tuple)

        loss = loss_matching + self.args.loss_w_lambda * loss_id

        return loss
