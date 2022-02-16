# Efficient Feature Map of Neural Tangent Kernels via Sketching and Random Features

Implementations developed in [[1]](#1-scaling-neural-tangent-kernels-via-sketching-and-random-features). The library is written for users familar with [JAX](https://github.com/google/jax) and [Neural Tangents](https://github.com/google/neural-tangents) library.

[PyTorch](https://pytorch.org/) Implementations can be found in [here](https://github.com/insuhan/ntk-sketch-rf).

#### [1] [Scaling Neural Tangent Kernels via Sketching and Random Features](https://arxiv.org/pdf/2106.07880.pdf)


## Examples
### Fully-connected NTK approximation via random features:
```python
from jax import random
from features import _inputs_to_features, DenseFeatures, ReluFeatures, serial

relufeat_arg = {
    'feature_dim0': 128,
    'feature_dim1': 128,
    'sketch_dim': 256,
    'method': 'rf',
}

init_fn, _, features_fn = serial(
    DenseFeatures(512), ReluFeatures(**relufeat_arg),
    DenseFeatures(512), ReluFeatures(**relufeat_arg),
    DenseFeatures(1)
)

key1, key2 = random.split(random.PRNGKey(1))
x = random.normal(key1, (5, 4))

_, feat_fn_inputs = init_fn(key2, (x.shape, (-1,0)))
feats = features_fn(_inputs_to_features(x), feat_fn_inputs)
# feats.nngp_feat is a feature map of NNGP kernel
# feats.ntk_feat is a feature map of NTK
```
