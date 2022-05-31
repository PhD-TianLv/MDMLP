# The code of the paper [MDMLP](https://arxiv.org/abs/2205.14477)

## Thanks to [TIMM](https://github.com/rwightman/pytorch-image-models) library!

## How to use

It's used in the same way as timm.

We use [fvcore](https://github.com/facebookresearch/fvcore) to measure params and flops.

We changed `timm/data/dataset_factory.py` to be able to train on Flowers102 and Food101.

```
# For cifar10
python3 train.py /path-to-cifar10 -c ymls/cifar10_sgd.yml --model mdmlp_ssd_patch4_lap2_dim64_depth8_32
# For cifar100
python3 train.py /path-to-cifar100 -c ymls/cifar100_sgd.yml --model mdmlp_ssd_patch4_lap2_dim64_depth8_32

# For Flowers102
python3 train.py /path-to-flowers102 -c ymls/flowers102_sgd.yml --model mdmlp_ssd_patch14_lap7_dim64_depth8_224
# For Food101
python3 train.py /path-to-food101 -c ymls/food101_sgd.yml --model mdmlp_ssd_patch14_lap7_dim64_depth8_224

# For MDAttnTool
python3 train.py /path-to-cifar10 -c ymls/cifar10_sgd.yml --model mdmlp_attn_32
```

