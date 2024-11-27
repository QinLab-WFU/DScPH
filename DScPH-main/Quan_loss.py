import torch




class QUA(torch.nn.Module):
    def __init__(self, embed_dim, n_classes, device):
        super(QUA, self).__init__()
        self.device = device

        self.in_features = embed_dim
        self.out_features = n_classes

        S_inter = calc_neighbor(label, self.train_L)
        inter_loss_img = calc_inter_loss(hash_layers1, S_inter, G,
                                         self.parameters['alpha'])
        inter_loss_txt = calc_inter_loss(hash_layers2, S_inter, F,
                                         self.parameters['alpha'])
        inter_loss = 0.5 * (inter_loss_img + inter_loss_txt)
        # intra_loss of img and txt
        intra_loss_1 = calc_inter_loss(hash_layers1, S_inter, F,
                                       self.parameters['alpha'])
        intra_loss_2 = calc_inter_loss(hash_layers2, S_inter, G,
                                       self.parameters['alpha'])

        intra_loss = (intra_loss_1 + intra_loss_2) * self.parameters['lambda']

        intra = intra_loss + inter_loss