import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from einops.layers.torch import Rearrange, Reduce

from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'mdmlp_patch4_lap2_dim64_depth8_32': _cfg(),
    'mdmlp_patch14_lap7_dim64_depth8_224': _cfg(),
}


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size, overlap=0):
        super(OverlapPatchEmbed, self).__init__()
        self.patch_size, self.overlap = patch_size, overlap

    def forward(self, x):
        batch_size, in_channs, img_size, _ = x.shape
        patch_size, overlap = self.patch_size, self.overlap

        stride = patch_size - overlap
        patch_num = (img_size - patch_size) // stride + 1
        assert (img_size - patch_size) % stride == 0, 'adjust the patch_size and the overlap_size!'

        patches = torch.zeros([batch_size, patch_num, patch_num, in_channs, patch_size, patch_size], device=x.device)
        for i in range(patch_num):
            for j in range(patch_num):
                patch = x[:, :, i * stride:i * stride + patch_size, j * stride:j * stride + patch_size]
                patches[:, i, j, :, :, :] = patch

        patches = rearrange(patches, 'b h w c d1 d2 -> b (h w) c (d1 d2)')
        return patches


class MLPLayer(nn.Module):
    def __init__(self, base_dim, dim, factor, forw_perm, **axes_lengths):
        super(MLPLayer, self).__init__()
        back_perm = forw_perm.split("->")[1] + " -> " + forw_perm.split("->")[0]

        self.norm = nn.LayerNorm(base_dim)

        self.forw_rearr = Rearrange(forw_perm, **axes_lengths)
        self.dense1 = nn.Linear(dim, int(dim * factor))
        self.dense2 = nn.Linear(int(dim * factor), dim)
        self.gelu = nn.GELU()
        self.back_rearr = Rearrange(back_perm, **axes_lengths)

    def forward(self, x):
        y = x

        y = self.norm(y)

        y = self.forw_rearr(y)
        y = self.dense1(y)
        y = self.gelu(y)
        y = self.dense2(y)
        y = self.gelu(y)
        y = self.back_rearr(y)

        y = y + x
        return y


class MDBlock(nn.Module):
    def __init__(self, patch_num, channs, base_dim, factor):
        super(MDBlock, self).__init__()
        h = w = patch_num
        self.hlayer = MLPLayer(base_dim, patch_num, factor, "b (h w) c d -> b w c d h", h=h, w=w)
        self.wlayer = MLPLayer(base_dim, patch_num, factor, 'b (h w) c d -> b c h d w', h=h, w=w)
        self.clayer = MLPLayer(base_dim, channs, factor, 'b (h w) c d -> b h w d c', h=h, w=w)
        self.dlayer = MLPLayer(base_dim, base_dim, factor, 'b (h w) c d -> b h w c d', h=h, w=w)

    def forward(self, x):
        y = self.hlayer(x)
        y = self.wlayer(y)
        y = self.clayer(y)
        y = self.dlayer(y)
        return y


class MDMLP(nn.Module):
    def __init__(self,
                 img_size,
                 in_chans,
                 num_classes,
                 base_dim,
                 depth,
                 patch_size,
                 overlap,
                 factor=4,
                 **kwargs):
        super(MDMLP, self).__init__()
        self.num_classes = num_classes
        patch_num = (img_size - patch_size) // (patch_size - overlap) + 1
        channs = in_chans

        self.stem = nn.Sequential(OrderedDict([
            ('OverlapPatchEmbed', OverlapPatchEmbed(patch_size, overlap)),
            ('Linear', nn.Linear((patch_size ** 2), base_dim)),
        ]))

        self.blocks = nn.Sequential(OrderedDict([
            *[(f'MDLayer{d}', MDBlock(patch_num, channs, base_dim, factor))
              for d in range(depth)]
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('Reduce', Reduce('b p c d -> b d', 'mean')),
            ('Linear', nn.Linear(base_dim, num_classes)),
        ]))

    def forward(self, x):
        if len(x.shape) == 3: x = torch.unsqueeze(x, dim=0)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x


def _create_mdmlp(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(MDMLP, variant, pretrained, default_cfg=default_cfgs[variant], **kwargs)


@register_model
def mdmlp_patch4_lap2_dim64_depth8_32(pretrained=False, **kwargs):
    model_args = dict(img_size=32, base_dim=64, depth=8, patch_size=4, overlap=2, **kwargs)
    return _create_mdmlp('mdmlp_patch4_lap2_dim64_depth8_32', pretrained=pretrained, **model_args)


@register_model
def mdmlp_patch14_lap7_dim64_depth8_224(pretrained=False, **kwargs):
    model_args = dict(img_size=224, base_dim=64, depth=8, patch_size=14, overlap=7, **kwargs)
    return _create_mdmlp('mdmlp_patch14_lap7_dim64_depth8_224', pretrained=pretrained, **model_args)


