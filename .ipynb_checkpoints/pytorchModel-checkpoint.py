import torch
import torch.nn.functional as F
import config

# Regular PyTorch Module
class PyTorchModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        
        # Calculate "same" padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        self.features = torch.nn.Sequential(
            # 224x224x3 => 224x224x64
            torch.nn.Conv2d(
                in_channels=config.IN_CHANNELS,
                out_channels=config.OUT_CHANNELS,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1),  # (1(28-1) - 28 + 3) / 2 = 1
            torch.nn.ReLU(),
            
            # 224x224x64 => 112x112x64
            torch.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=0),  # (2(14-1) - 28 + 2) = 0
            
            # 112x112x64 => 112x112x128
            torch.nn.Conv2d(
                in_channels=config.OUT_CHANNELS,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1),  # (1(14-1) - 14 + 3) / 2 = 1 
            torch.nn.ReLU(),
            
            # 112x112x128 => 56x56x128                            
            torch.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=0)  # (2(7-1) - 14 + 2) = 0
        )

        self.output_layer = torch.nn.Linear(56*56*128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output_layer(x)
        return x   # x are the model's logits