import torch

from model.overfeat_accurate_base import OverFeat_accurate_base

if __name__ == "__main__":
    model = OverFeat_accurate_base(1000).to('cuda')
    x = torch.randn(128,3,256,256).to("cuda")
    output = model.feature_extractor(x)
    print(output.size())


