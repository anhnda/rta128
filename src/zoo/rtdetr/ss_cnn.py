import torch
import torch.nn as nn
import torch.nn.init as init 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._reset_params()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def _reset_params(self):
        init.constant_(self.fc1.weight,0)
        init.constant_(self.fc1.bias,0)

        init.constant_(self.fc2.weight,0)
        init.constant_(self.fc2.bias,0)
class SameShapeCNN(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(SameShapeCNN, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                              kernel_size=kernel_size, stride=1, padding=padding)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.mlp = Mlp(channels, 4*channels, channels)
    def forward(self, x):
        # x: [B, H, W, C] -> [B, C, H, W]
        B, L, C = x.shape
        shortcut = x
        H = 84
        W = 100
        assert H * W == L
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        # Back to [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H*W, C)
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x

