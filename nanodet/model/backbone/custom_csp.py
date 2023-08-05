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
import math 
import torch.nn.functional as F
from ..module.conv import ConvModule

class Ghostblock(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(Ghostblock, self).__init__()
        self.gate_fn=nn.Sigmoid()
        self.act = nn.ReLU()
        self.oup = oup
        init_channels = math.ceil(oup / ratio) 
        new_channels = init_channels*(ratio-1)
        self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels)
            )
        self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels)
        )
    def forward(self, x):
        x1 = self.act(self.primary_conv(x))
        x2 = self.act(self.cheap_operation(x1))
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class Ghost_attn(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(Ghost_attn, self).__init__()
        self.act = nn.ReLU()
        self.oup = oup
        init_channels = math.ceil(oup / ratio) 
        new_channels = init_channels*(ratio-1)
        self.primary_conv = nn.Sequential(  
                nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(init_channels)
            )
        self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels)
            ) 
        self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            ) 
    def forward(self,x):
         res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))  
         x1 = self.act(self.primary_conv(x))
         x2 = self.cheap_operation(x1)
         out = torch.cat([x1,x2], dim=1)
         return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res),size=(out.shape[-2],out.shape[-1]),mode='nearest')
class TinyResBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride):
        super().__init__()
        self.attn = Ghost_attn(inp,oup, kernel_size = kernel_size)
        self.ghost = Ghostblock(oup, oup, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(oup)
    
    def forward(self,x):
        x1= self.bn(self.ghost(self.attn(x)))
        return x1+x


class CspBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_res,
        kernel_size=3,
        stride=0,
        norm_cfg=dict(type="BN", requires_grad=True),
        activation="LeakyReLU",
    ):
        super(CspBlock, self).__init__()
        assert in_channels % 2 == 0
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
            res_block = TinyResBlock(in_channels,out_channels, kernel_size, stride)
            res_blocks.append(res_block)
        self.res_blocks = nn.Sequential(*res_blocks)
        self.res_out_conv = ConvModule(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            norm_cfg=norm_cfg,
            activation=activation,
        )

    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.res_blocks(x)
        x1 = self.res_out_conv(x1)
        out = x+x1
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
            elif stage_cfg[0] == "CspBlock":
                in_channels, out_channels, num_res, kernel_size, stride = stage_cfg[1:]
                stage = CspBlock(
                    in_channels, num_res, kernel_size, stride, norm_cfg, activation
                )
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
