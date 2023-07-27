import torch
import torch.nn as nn
class CBL(nn.Module):
    def__init__(self, in_ch, out_ch, k=3, s=1, p=0, batch_norm=False):
    super().__init__()
    self.batch_norm = batch_norm
    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size =k, stride = s, padding =p)
    self.bn= nn.BatchNorm2d(out_ch)

def forward(self,x):
    x = self.conv(x)
    if(self.batch_norm):
        return self.bn(x)
    else:
        return x

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, activation="ReLU"):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                act_layers(activation),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

def GsBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1= CBL(in_ch, in_ch//2, 3, 1,True)
        self.depthwise = nn.Conv2d(in_ch, out_channels = in_ch, kernel_size = 3, stride =1, groups = in_ch)
        self.shuffle = ShuffleV2Block(in_ch, out_ch, stride = 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.depthwise(x1)
        x = self.shuffle(torch.concat((x1,x2), dim =0))
        return x



