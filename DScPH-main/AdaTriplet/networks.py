import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BackboneModel(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.args = args
        self.features_extractor = arch_backbone_model(args, pretrained=pretrained)

    def forward(self, x):
        img_features = self.features_extractor(x)
        if self.args.normalize_img_features:
            img_features = F.normalize(img_features, p=2)
        return img_features


def arch_backbone_model(args, pretrained=True):
    # arch_name = args.backbone
    # model = torch.hub.load("pytorch/vision:v0.9.0", arch_name, pretrained=pretrained)
    if args.backbone != "resnet18":
        raise NotImplementedError(f"not support: {args.backbone}")
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(args.backbone_out_features, args.n_bits)
    return model
