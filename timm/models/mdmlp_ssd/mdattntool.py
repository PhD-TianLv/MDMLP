import torch
import torch.nn as nn
from einops import rearrange, reduce


class MLPLayer(nn.Module):
    def __init__(self, dim, init_b=None):
        super(MLPLayer, self).__init__()
        self.dense1 = nn.Linear(dim, 8)
        self.dense2 = nn.Linear(8, dim)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.init_b = init_b

        if init_b:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0)
                m.bias.data.normal_(self.init_b, 0)

    def forward(self, x):
        y = x
        y = self.norm(y)
        y = self.dense1(y)
        y = self.gelu(y)
        y = self.dense2(y)
        y = self.gelu(y)
        return y + x


class IMGAttnTool(nn.Module):
    def __init__(self, img_size):
        super(IMGAttnTool, self).__init__()
        self.hlayer = MLPLayer(img_size, init_b=1.)
        self.wlayer = MLPLayer(img_size, init_b=1.)

    def forward(self, v):
        # img: (b c h w)
        v = rearrange(v, 'b c h w -> b c w h')
        v = self.hlayer(v)
        v = rearrange(v, 'b c w h -> b c h w')
        v = self.wlayer(v)
        v = reduce(v, 'b c h w -> b 1 h w', 'mean')
        return v


class MDAttnTool(nn.Module):
    def __init__(self, img_size):
        super(MDAttnTool, self).__init__()
        self.img_attn_tool = IMGAttnTool(img_size)

    def forward(self, x):
        v = self.img_attn_tool(x)  # output v here for visualization
        y = v * x
        return y
