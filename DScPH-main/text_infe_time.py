import torch
import torchvision.models

from my_vit import Text, VisionTransformer
import numpy as np
import random
from simple_tokenizer import SimpleTokenizer
import torch.nn as nn
import scipy.io as scio
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import time


# from MLP import MLP
# from MULTI_SCAL import TextNet
#

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


class LinearHash(nn.Module):

    def __init__(self, inputDim=512, outputDim=32):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        # self.drop_out = nn.Dropout(p=0.2)

    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(result)


model_T = Text(vocab_size=49408, transformer_width=512, context_length=77, transformer_heads=8, transformer_layers=12, embed_dim=512)

text_hash = LinearHash()

# model1 = MLP()
# model2 = TextNet()

captions = 'lastfm:event=292183 music concert tegansara 112407 tegan sara highres fullsize lisnerauditorium lisner auditorium saraquin quin teganandsara teganquin dc washingtondc washington'
use_cap = captions[random.randint(0, len(captions) - 1)]
tokenizer = SimpleTokenizer()
words = tokenizer.tokenize(use_cap)
SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
total_length_with_CLS = 32 - 1
if len(words) > total_length_with_CLS:
    words = words[:total_length_with_CLS]

words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]
caption = tokenizer.convert_tokens_to_ids(words)

while len (caption) < 32:
    caption.append(0)
caption = torch.tensor(caption)
caption = caption.unsqueeze(0)
###parameters

import scipy
txt = scipy.io.loadmat('/home/abc/wlproject/HNH_demo-main/HNH_demo/Datasets/mirflickr/mirflickr25k-yall.mat')['YAll'][:1]
txt = torch.from_numpy(txt).float()

total_params = sum(p.numel() for p in model_T.parameters() if p.requires_grad)
print("Total parameters: ", total_params / 1e6)

###flops

# import torchprofile
# with torch.no_grad():
#     flops = torchprofile.profile_macs(model_T, caption)
# print("Total FLOPs: ", flops / 1e6)

model_I = VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512)
database_code = scio.loadmat('/home/abc/wulei/CPF（复件）/DCHMT-main/result/11/32-ours-flickr25k-i2t.mat')['r_img']
print(database_code)
start_time = time.time()
output_txt = model_T(caption)
tanh_I = text_hash(output_txt)
hash_I = torch.sign(tanh_I)
hash_I = hash_I.cpu().detach().numpy()
similarity = calculate_hamming(hash_I, database_code)
sim_ord = np.argsort(similarity)
index = sim_ord[:10]
end_time = time.time() - start_time
print(end_time)