import torch
from torch import nn
from torch.nn import functional as F


class MNP(nn.Module):
    NAME: str

    def __init__(self, s_feature: torch.Tensor, t_feature: torch.Tensor) -> None:
        super(MNP, self).__init__()
        assert s_feature.shape == t_feature.shape
        assert s_feature.shape[1] % 2 == 0
        self.shape = t_feature.shape

class MNPSeparable(MNP):
    NAME = 'dwseparable'

    def __init__(self, s_feature: torch.Tensor, t_feature: torch.Tensor) -> None:
        super(MNPSeparable, self).__init__(s_feature, t_feature)
        self.c_in = s_feature.shape[1] * 2 
        self.c_out = s_feature.shape[1]
        self.depthwise = nn.Conv2d(self.c_in, self.c_in, kernel_size=3, padding=1, groups=self.c_in)
        self.pointwise = nn.Conv2d(self.c_in, self.c_out, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.c_out)

    def forward(self, s_feature: torch.Tensor, t_feature: torch.Tensor):
        out = self.depthwise(torch.cat([s_feature, t_feature], dim=1))
        out = self.pointwise(out)
        out = self.bn(out)
        return F.relu(out)

    
class MNPAggregate(MNP):
    NAME = 'aggregate'

    def __init__(self, s_feature: torch.Tensor, t_feature: torch.Tensor) -> None:
        super(MNPAggregate, self).__init__(s_feature, t_feature)
        self.c_in = 2 * s_feature.shape[1]
        self.c_out = s_feature.shape[1]
        self.conv = nn.Conv2d(self.c_in, self.c_out, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(self.c_out)

    def forward(self, s_feature: torch.Tensor, t_feature: torch.Tensor):
        out = self.conv(torch.cat([s_feature, t_feature], dim=1))
        out = self.bn(out)
        return F.relu(out)

class MNPMultiply(MNP):
    NAME = 'multiply'

    def __init__(self, s_feature: torch.Tensor, t_feature: torch.Tensor) -> None:
        super(MNPMultiply, self).__init__(s_feature, t_feature)
    
    def forward(self, s_feature: torch.Tensor, t_feature: torch.Tensor):
        out = s_feature * t_feature
        return F.relu(out)