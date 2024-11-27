import argparse


def get_config():
    parser = argparse.ArgumentParser(description="AdaTriplet")

    # model
    parser.add_argument("--backbone", type=str, default="resnet18", help="the type of base model")
    parser.add_argument("--backbone_out_features", type=int, default=512, help="backbone_out_features")
    # parser.add_argument("--img_out_features", type=int, default=128, help="img_out_features")
    parser.add_argument("--normalize_img_features", type=bool, default=True, help="normalize_img_features")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay")
    parser.add_argument("--n_epochs", type=int, default=100, help="training epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="the batch size for training")
    parser.add_argument("--eval_frequency", type=int, default=5, help="the evaluate frequency for testing")
    parser.add_argument("--data_dir", type=str, default="../_datasets", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="nus", help="database name")
    parser.add_argument("--n_workers", type=int, default=4, help="number of data loader workers")

    # Hashing
    parser.add_argument("--n_bits", type=int, default="16", help="hash bit length")

    # Testing
    parser.add_argument("--topk", type=int, default=1000, help="mAP@topk")

    # Others
    parser.add_argument("--method", type=str, default="AdaTriplet-AM", help="method")
    parser.add_argument("--distance_loss", type=str, default="cosine", help="distance_loss")

    parser.add_argument("--margin_beta", type=float, default=0, help="β in paper")
    parser.add_argument("--margin_m_loss", type=float, default=0.25, help="ε in paper")

    parser.add_argument("--type_of_triplets", type=str, default="semihard", help="all, hard, semihard")
    parser.add_argument(
        "--automargin_mode", type=str, default="normal", help="normal or adaptive, and exp, linear, add-ap, Q1, Q2"
    )
    parser.add_argument("--k_param_automargin", type=float, default=2, help="K_Δ of Eq. (7) in paper")
    parser.add_argument("--k_n_param_autobeta", type=float, default=2, help="K_an of Eq. (8) in paper")
    parser.add_argument("--k_p_param_autobeta", type=float, default=2, help="k_p_param_autobeta")

    parser.add_argument("--loss_w_lambda", type=float, default=1, help="loss_w_lambda")
    parser.add_argument("--loss_w_neg", type=float, default=1, help="loss_w_neg")

    return parser.parse_args()
