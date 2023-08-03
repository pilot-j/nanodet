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
import torch.nn.functional as F
from ..module.conv import ConvModule
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()
        c_ = int(in_channels//2)
        self.c1 = nn.Conv2d(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = nn.Conv2d(c_ * 4, out_channels, 1, 1, 0)
        self.batch_norm1 = nn.BatchNorm2d(c_)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.batch_norm1(self.c1(x))
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)
        return self.batch_norm2(self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1)))
class TinyResBlock(nn.Module):
    def __init__(
        self, in_channels, kernel_size, expand = 2 norm_cfg, activation, res_type="add"
    ):
        super(TinyResBlock, self).__init__()
        assert in_channels % 2 == 0
        self.mid_ch = expand * in_channels
        assert res_type in ["concat", "add"]
        self.res_type = res_type
        self.in_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            norm_cfg=norm_cfg,
            activation=activation,
        )
        self.mid_conv = ConvModule(
            in_channels,
            expand * in_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            norm_cfg=norm_cfg,
            activation=activation,
        )
        if res_type == "add":
            self.out_conv = ConvModule(
                expand* in_channels,
                in_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                activation=activation,
            )

    def forward(self, x):
        x1 = self.out_conv(self.mid_conv(self.in_conv(x)))
        if self.res_type == "add":
            return x+x1
        else:
            return torch.cat((x1, x), dim=1)
            
class TinyResBlock_attn(TinyResBlock):
    def __init__(
        self, in_channels, kernel_size, norm_cfg, activation, res_type="add"
    ):
        super(TinyResBlock_attn, self).__init__()
        
        self.short_conv = nn.Sequential(
                nn.Conv2d(in_channels ,in_channels,kernel_size= 1, stride = 1, bias = False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels,in_channels, kernel_size=(1,5), stride=1, padding=(0,2), groups=in_channels,bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels,in_channels, kernel_size=(5,1), stride=1, padding=(2,0), groups=in_channels,bias=False),
                nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):
        x1 = self.out_conv(self.mid_conv(self.in_conv(x)))
        DFC=self.fn(self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2)))
        x = F.interpolate(DFC, (x.shape[-2], x.shape[-1]), mode ='nearest')
        return x1+x


class CspBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_res,
        kernel_size=3,
        stride=0,
        norm_cfg=dict(type="BN", requires_grad=True),
        activation="LeakyReLU",
        dfc=False
    ):
        super(CspBlock, self).__init__()
        assert in_channels % 2 == 0
        self.dfc = dfc
        self.in_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            norm_cfg=norm_cfg,
            activation=activation,
        )
        res_blocks = []
        for i in range(num_res):
            if(self.dfc == True):
                res_block = TinyResBlock_attn(in_channels, kernel_size, norm_cfg, activation)
            else:
                res_block = TinyResBlock(in_channels, kernel_size, norm_cfg, activation)
            res_blocks.append(res_block)
        self.res_blocks = nn.Sequential(*res_blocks)
        self.res_out_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            norm_cfg=norm_cfg,
            activation=activation,
        )

    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.res_blocks(x)
        x1 = self.res_out_conv(x1)
        out = torch.cat((x1, x), dim=1)
        return out


class CustomCspNet(nn.Module):
    def __init__(
        self,
        net_cfg,
        out_stages,
        norm_cfg=dict(type="BN", requires_grad=True),
        activation="LeakyReLU",
    ):
        super(CustomCspNet, self).__init__()
        assert isinstance(net_cfg, list)
        assert set(out_stages).issubset(i for i in range(len(net_cfg)))
        self.out_stages = out_stages
        #self.activation = activation
        self.stages = nn.ModuleList()
        for stage_cfg in net_cfg:
            if stage_cfg[0] == "Conv":
                in_channels, out_channels, kernel_size, stride, activation = stage_cfg[1:]
                stage = ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=(kernel_size - 1) // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            elif stage_cfg[0] == "CspBlock":
                in_channels, num_res, kernel_size, stride, activation, dfc = stage_cfg[1:]
                stage = CspBlock(
                    in_channels, num_res, kernel_size, stride, norm_cfg, activation, dfc 
                )
            elif stage_cfg[0] == "MaxPool":
                kernel_size, stride = stage_cfg[1:]
                stage = nn.MaxPool2d(
                    kernel_size, stride, padding=(kernel_size - 1) // 2
                )
            elif stage_cfg[0] =="SPPF":
                in_channels, out_channels = stage_cfg[1:]
                stage = SPPF(int_channels, out_channels)
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
