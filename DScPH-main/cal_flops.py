from thop import profile
import torch
import random
import numpy as np
import scipy.io as scio
import torchvision.models as models
import my_vit
from my_vit import Text,VisionTransformer

# from MLP import MLP
# from MULTI_CAL import TextNet

# model = models.resnet50()

# model = MLP()
# model1 = TextNet()
#
# model_T = VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512)
# model = Text(vocab_size=49408, transformer_width=512, context_length=77, transformer_heads=8, transformer_layers=12, embed_dim=512)
model_T = models.resnet18()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_T .to(device)
input = torch.zeros((1, 3, 224, 224)).to(device)

total_params = sum(p.numel() for p in model_T.parameters() if p.requires_grad)
# flops, params = profile(model_T.to(device), inputs=(input,))
print("totals paters:",total_params / 1e6)
# print("totals paters:",params / 1e6)
# print("Total FLOPs:",flops / 1e6)


import torchprofile
with torch.no_grad():
    flops = torchprofile.profile_macs(model_T, input)
print("Total FLOPs: ", flops / 1e6)


# def _load_text(self, index: int):
#     captions = self.captions[index]
#     use_cap = captions[random.randint(0, len(captions) - 1)]
#
#     words = self.tokenizer.tokenize(use_cap)
#     words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
#     total_length_with_CLS = self.maxWords - 1
#     if len(words) > total_length_with_CLS:
#         words = words[:total_length_with_CLS]
#
#     words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
#     caption = self.tokenizer.convert_tokens_to_ids(words)
#
#     while len(caption) < self.maxWords:
#         caption.append(0)
#     caption = torch.tensor(caption)
#
#     return caption
#
# def dataloader(captionFile: str,
#                 indexFile: str,
#                 labelFile: str,
#                 maxWords=32,
#                 imageResolution=224,
#                 query_num=5000,
#                 train_num=10000,
#                 seed=None,
#                 npy=False):
#     if captionFile.endswith("mat"):
#         captions = scio.loadmat(captionFile)["caption"]
#         captions = captions[0] if captions.shape[0] == 1 else captions
#     elif captionFile.endswith("txt"):
#         with open(captionFile,'r',encoding='UTF-8') as f:
#             captions = f.readlines()
#         captions = np.asarray([[item.strip()] for item in captions])
#         # with open(captionFile,'r') as f:
#         #     captions = f.read().split()
#     else:
#         raise ValueError("the format of 'captionFile' doesn't support, only support [txt, mat] format.")
#     if not npy:
#         indexs = scio.loadmat(indexFile)["index"]
#     else:
#         indexs = np.load(indexFile, allow_pickle=True)
#     labels = scio.loadmat(labelFile)["category"]