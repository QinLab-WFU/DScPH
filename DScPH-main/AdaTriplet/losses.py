import torch
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils import common_functions as c_f


def bit_var_loss():
    def F(x):
        return 1 / (1+torch.exp(-x))
    def loss(z):
        return torch.mean(F(z) * (1-F(z)))
    return loss

class LowerBoundLoss(torch.nn.Module):
    def __init__(self):
        super(LowerBoundLoss, self).__init__()

    def forward(self, output):
        max_loss = torch.clamp(output, min=0, max=None)
        mean_max_loss = torch.mean(max_loss)
        return mean_max_loss


class TripletCustomMarginLoss(TripletMarginLoss):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
    """

    def __init__(self, margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all", **kwargs):
        super().__init__(
            margin=margin, swap=swap, smooth_loss=smooth_loss, triplets_per_anchor=triplets_per_anchor, **kwargs
        )

    def set_ref_emb(self, embeddings, labels, ref_emb, ref_labels):
        if ref_emb is not None:
            ref_labels = c_f.to_device(ref_labels, ref_emb)
        else:
            ref_emb, ref_labels = embeddings, labels
        # c_f.check_shapes(ref_emb, ref_labels)
        return ref_emb, ref_labels

    def get_matches_and_diffs_onehot(self, labels, ref_labels=None):
        if ref_labels is None:
            ref_labels = labels
        matches = (labels @ ref_labels.T > 0).byte()
        diffs = matches ^ 1
        if ref_labels is labels:
            matches.fill_diagonal_(0)
        return matches, diffs

    def get_all_triplets_indices_onehot(self, labels, ref_labels=None):
        matches, diffs = self.get_matches_and_diffs_onehot(labels, ref_labels)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        return torch.where(triplets)

    def convert_to_triplets(self, indices_tuple, labels, ref_labels=None, t_per_anchor=100):
        """
        This returns anchor-positive-negative triplets
        regardless of what the input indices_tuple is
        """
        if indices_tuple is None:
            if t_per_anchor == "all":
                return self.get_all_triplets_indices_onehot(labels, ref_labels)
            else:
                raise NotImplementedError
        elif len(indices_tuple) == 3:
            return indices_tuple
        else:
            a1, p, a2, n = indices_tuple
            p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
            return a1[p_idx], p[p_idx], n[n_idx]

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = self.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, an_dists)
        violation = current_margins + self.margin
        if self.smooth_loss:
            loss = torch.nn.functional.softplus(violation)
        else:
            loss = torch.nn.functional.relu(violation)

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def forward(self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        # c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = self.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def set_margin(self, margin):
        self.margin = margin
