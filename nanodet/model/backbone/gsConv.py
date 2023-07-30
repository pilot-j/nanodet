import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)




class GSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        self.gs = GSConv(c2,c2)
        self.c2 = c2
        c_ = int(c2*e)
        self.fn = nn.Mish()
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False)
            )
        
        self.shortcut1 = DWConv(c1, c2, 1, 1, act=False)
        self.shortcut2 = Conv(c1, c2, k=3, s=1,p=1,  g=math.gcd(c1, c2), act=False)
        self.short_conv = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=(1,5), stride=1, padding=(0,2), groups=c2,bias=False),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2,c2, kernel_size=(5,1), stride=1, padding=(2,0), groups=c2,bias=False),
                nn.BatchNorm2d(c2),
            ) 
    
    def forward(self, x):
        x1 = self.conv_lighting(x)
        DFC=self.fn(self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))) # Downsample --> sequential layer
        y = F.interpolate(DFC, (x1.shape[-2], x1.shape[-1]), mode ='nearest')
        out1 = x1*y
        out2 = torch.cat((self.shortcut1(x), self.shortcut2(x)), dim =1)
        out2 = out2[:, :self.c2,:,:]
        
        return self.gs(out1) + out2

class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, 3, 1, act=False)





