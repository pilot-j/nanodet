# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from ..module.conv import ConvModule
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        activation="ReLU",
        gate_fn=hard_sigmoid,
        divisor=4,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layers(activation)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
class InvertedRes(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.act = nn.ReLU()
        self.conv_pw = nn.Sequential(
            nn.Conv2d(in_ch//2, mid_ch//2,kernel_size =1, stride =1, padding =0, groups =1),
            BatchNorm2d(mid_ch//2)
        )
        self.conv_dw = nn.Sequential(
            nn.Conv2d(mid_ch//2, mid_ch//2,kernel_size =3, stride =stride, padding =1, groups =mid_ch//2),
            BatchNorm2d(mid_ch//2)
        )
        self.SE = SqueezeExcite(mid_ch)
        self.conv_linear = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch//2, kernel_size = 1, stride =1, padding =0, groups =1)
            nn.BatchNorm(out_ch//2)
        )
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.act(self.conv_pw(x2))
        x3 = self.act(self.conv_dw(x2))
        x3 = torch.concat((x2,x3), dim=1)
        x3 = self.act(self.conv_linear(self.SE(x3)))
        out = torch.cat((x1,x3), dim=1)
        out = channel_shuffle(out,2)
        return out
        
        
class ESBlock(nn.Module):
    def __init__(self, make = True):
        if make:
            super().__init__()
            self.ratio = [0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625, 0.625, 0.5, 0.625, 1.0, 0.625, 0.75]
            self.stage_repeats = [3, 7, 3]
            stage_out_channels = [-1, 32, make_divisible(128, divisor=16), make_divisible(256, divisor=16),make_divisible(512, divisor=16), 1024]
        
        self.num_res = num_res
        res_block = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(self.stage_repeats):
            for i in range(num_repeat):
                channels_scales = self.ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                res_conv = InvertedRes(in_ch = stage_out_channels[stage_id+1], mid_ch = mid_c, out_ch=stage_out_channels[stage_id+2], stride = 1)
                res_block.append(res_conv)
                arch_idx +=1
        self.res_block = nn.Sequential(*res_block)
    def forward(self,x):
        return self.res_block(x)

class ESNet(nn.Module):
    def __init__(
        self,
        net_cfg,
        out_stages,
        norm_cfg=dict(type="BN", requires_grad=True),
        activation="LeakyReLU",
    ):
        super(ESNet, self).__init__()
        assert isinstance(net_cfg, list)
        assert set(out_stages).issubset(i for i in range(len(net_cfg)))
        self.out_stages = out_stages
        self.activation = activation
        self.stages = nn.ModuleList()
        for stage_cfg in net_cfg:
            if stage_cfg[0] == "Conv":
                in_channels, out_channels, kernel_size, stride = stage_cfg[1:]
                stage = ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=(kernel_size - 1) // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            elif stage_cfg[0] == "ESBlock":
                make = stage_cfg[1:]
                stage = ESBlock(make)
            elif stage_cfg[0] == "MaxPool":
                kernel_size, stride = stage_cfg[1:]
                stage = nn.MaxPool2d(
                    kernel_size, stride, padding=(kernel_size - 1) // 2
                )
            else:
                raise ModuleNotFoundError
            self.stages.append(stage)
        self._init_weight()

    def forward(self, x):
        output = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _init_weight(self):
        for m in self.modules():
            if self.activation == "LeakyReLU":
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=nonlinearity
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
