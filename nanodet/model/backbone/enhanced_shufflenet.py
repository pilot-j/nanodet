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
import math
from ..module.activation import act_layers
from ..module.conv import ConvModule
import torch.utils.model_zoo as model_zoo

model_urls = {
    "esnet_m": "https://drive.google.com/file/d/1d1aW8dhaKiL1-44M7saBkP0_RVAoioK8/view?usp=drive_link"
}

def make_divisible(v, divisor, min_value=None):
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
    divisor=4,
    **_
    ):
    super(SqueezeExcite, self).__init__()
    self.gate_fn = nn.Sigmoid()
    reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
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
      nn.BatchNorm2d(mid_ch//2)
      )
    self.conv_dw = nn.Sequential(
      nn.Conv2d(mid_ch//2, mid_ch//2,kernel_size =3, stride =stride, padding =1, groups =mid_ch//2),
      nn.BatchNorm2d(mid_ch//2)
      )
    self.SE = SqueezeExcite(mid_ch)
    self.conv_linear = nn.Sequential(
      nn.Conv2d(mid_ch, out_ch//2, kernel_size = 1, stride =1, padding =0, groups =1),
      nn.BatchNorm2d(out_ch//2)
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
        
class ESNet(nn.Module):
  def __init__(self, model_name, out_stages =(2,9,12),activation="ReLU6", pretrain = False):
    super().__init__()
    self.model_name = model_name
    self.out_stages = out_stages
    es_block_settings = [[0.875, 32, 128],  [0.5, 128, 128], [1, 128,128], [0.625, 128, 256],[0.5, 256,256],[0.75, 256, 256],[0.625, 256, 256],[0.625, 256, 256],
            [0.5, 256, 256],
            [0.625,256,256],
            [1, 256, 512],
            [0.625, 512, 512],
            [0.625, 512,512],]
      
    self.stem = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(32),
      act_layers(activation)
      )
    self.blocks = nn.ModuleList([])
    for i, stage_setting in enumerate(es_block_settings):
      (ratio,in_ch,out_ch) = stage_setting
      mid_ch = make_divisible(int(out_ch * ratio),divisor=8)
      self.blocks.append(InvertedRes(in_ch, mid_ch, out_ch, stride = 1))

  def forward(self,x):
    x = self.stem(x)
    output = []
    for j, stage in enumerate(self.blocks):
      x = stage(x)
      if j in self.out_stages:
        output.append(x)

    return output
  def _initialize_weights(self, pretrain=False):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()
    if pretrain:
      url = model_urls[self.model_name]
      if url is not None:
        pretrained_state_dict = model_zoo.load_url(url)
        print("=> loading pretrained model {}".format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)
  def load_pretrain(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict, strict=True)
