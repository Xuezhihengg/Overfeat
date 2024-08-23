import torch
import torch.nn as nn

class OverFeat_accurate_base(nn.Module):
    '''
        This model is only used to train the feature extractor --Overfeat.
        The model of multi-scale classification task, positioning task or
        detection task share the feature_extractor part of the model.
    '''
    def __init__(self,num_classes = 1000):
        super().__init__()
        # input size can be arbitrary,here take (b x 3 x 221 x 221) as an example
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=7,stride=2),  # (b x 96 x 108 x 108)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=3),  # (b x 96 x 36 x 36)
            # Layer 2
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=7,stride=1),  # (b x 256 x 30 x 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),  # (b x 256 x 15 x 15)
            # Layer 3
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  # (b x 512 x 15 x 15)
            nn.ReLU(),
            # Layer 4
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  # (b x 512 x 15 x 15)
            nn.ReLU(),
            # Layer 5
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1),  # (b x 1024 x 15 x 15)
            nn.ReLU(),
            # Layer 6
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),  # (b x 1024 x 15 x 15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=3)  # (b x 1024 x 5 x 5)
        )
        self.base_classifier = nn.Sequential(
            # Layer 7
            nn.Dropout(p=0.5,inplace=False),
            nn.Conv2d(in_channels=1024,out_channels=4096,kernel_size=5,stride=1),  # (b x 4096 x 1 x 1)
            nn.ReLU(),
            # Layer 8
            nn.Dropout(p=0.5,inplace=False),
            nn.Conv2d(in_channels=4096,out_channels=4096,kernel_size=1,stride=1),  # (b x 4096 x 1 x 1)
            nn.ReLU(),
            # Layer 9
            nn.Conv2d(in_channels=4096,out_channels=num_classes,kernel_size=1,stride=1),  # (b x num_classes x 1 x 1)
        )
        self.init_weight()

    def init_weight(self):
        for layer in self.feature_extractor:
            if isinstance(layer,nn.Conv2d):
                nn.init.normal_(layer.weight,mean=0,std=0.01)
        for layer in self.base_classifier:
            if isinstance(layer,nn.Conv2d):
                nn.init.normal_(layer.weight,mean=0,std=0.01)

    def forward(self,x):
        x = self.feature_extractor(x)  # (b x 1024 x 5 x 5)
        return self.base_classifier(x).squeeze()  # (b x num_classes x 1 x 1) -> (b x num_classes)



