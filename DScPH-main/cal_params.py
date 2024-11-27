import torch
import torchvision
import scipy
from thop import profile
from my_vit import Text,VisionTransformer


print('==BUilding modal')

model = MLP()
model1 = TextNet()
txt = scipy.io.loadmat('')
txt = torch.from_numpy(txt).float()


total_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
print("totals paters:",total_params / 1e6)

import torchprofile
with torch.no_grad():
    flops = torchprofile.profile_macs(model1, txt)
print("Total FLOPs:",flops / 1e6)
