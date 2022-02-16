from jax import random
from jax import numpy as np

import jax.example_libraries.stax as ostax
from neural_tangents.utils.typing import Callable, Tuple
from neural_tangents.utils import utils, dataclasses

from sketching import TensorSRHT2, PolyTensorSRHT
"""Implementation for NTK Sketching and Random Features

"""


# Arc-cosine kernel functions is for debugging.
def _arccos(x):
  return np.arccos(np.clip(x, -1, 1))


def _sqrt(x):
  return np.sqrt(np.maximum(x, 1e-20))


def kappa0(x):
  xxt = x @ x.T
  prod = np.outer(np.linalg.norm(x, axis=-1)**2, np.linalg.norm(x, axis=-1)**2)
  return (1 - _arccos(xxt / _sqrt(prod)) / np.pi) / 2


def kappa1(x):
  xxt = x @ x.T
  prod = np.outer(np.linalg.norm(x, axis=-1)**2, np.linalg.norm(x, axis=-1)**2)
  return (_sqrt(prod - xxt**2) +
          (np.pi - _arccos(xxt / _sqrt(prod))) * xxt) / np.pi / 2


@dataclasses.dataclass
class Features:
  nngp_feat: np.ndarray
  ntk_feat: np.ndarray

  shape: Tuple[int, ...] = dataclasses.field(pytree_node=False)

  batch_axis: int = dataclasses.field(pytree_node=False)
  channel_axis: int = dataclasses.field(pytree_node=False)

  replace = ...  # type: Callable[..., 'Features']


def _inputs_to_features(x: np.ndarray,
                        batch_axis: int = 0,
                        channel_axis: int = -1,
                        eps: float = 1e-12,
                        **kwargs) -> Features:
  """Transforms (batches of) inputs to a `Features`."""

  # Followed the same initialization of Neural Tangents library.
  nngp_feat = x / x.shape[channel_axis]**0.5
  ntk_feat = np.zeros((), nngp_feat.dtype)

  return Features(nngp_feat=nngp_feat,
                  ntk_feat=ntk_feat,
                  shape=x.shape,
                  batch_axis=batch_axis,
                  channel_axis=channel_axis)


# Modified the serial process of feature map blocks.
# Followed https://github.com/google/neural-tangents/blob/main/neural_tangents/stax.py
def serial(*layers):
  init_fns, apply_fns, features_fns = zip(*layers)
  init_fn, apply_fn = ostax.serial(*zip(init_fns, apply_fns))

  def features_fn(k, inputs, **kwargs):
    for f, input_ in zip(features_fns, inputs):
      k = f(k, input_, **kwargs)
    return k

  return init_fn, apply_fn, features_fn


def DenseFeatures(out_dim: int,
                  W_std: float = 1.,
                  b_std: float = None,
                  parameterization: str = 'ntk',
                  batch_axis: int = 0,
                  channel_axis: int = -1):

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_ntk_feat_shape = nngp_feat_shape[:-1] + (nngp_feat_shape[-1] +
                                                 ntk_feat_shape[-1],)
    return (nngp_feat_shape, new_ntk_feat_shape), ()

  def apply_fn(**kwargs):
    return None

  def kernel_fn(f: Features, input, **kwargs):
    nngp_feat, ntk_feat = f.nngp_feat, f.ntk_feat

    if np.all(ntk_feat == 0.0):
      ntk_feat = nngp_feat
    else:
      ntk_feat = np.concatenate((ntk_feat, nngp_feat), axis=channel_axis)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, apply_fn, kernel_fn


def ReluFeatures(
    feature_dim0: int = 1,
    feature_dim1: int = 1,
    sketch_dim: int = 1,
    poly_degree0: int = 4,
    poly_degree1: int = 4,
    method: str = 'rf',
    debug: bool = False,
):

  method = method.lower()

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_nngp_feat_shape = nngp_feat_shape[:-1] + (feature_dim1,)
    new_ntk_feat_shape = ntk_feat_shape[:-1] + (sketch_dim,)

    if not debug and method == 'rf':
      rng1, rng2, rng3 = random.split(rng, 3)
      W0 = random.normal(rng1, (nngp_feat_shape[-1], feature_dim0))
      W1 = random.normal(rng2, (nngp_feat_shape[-1], feature_dim1))
      ts2 = TensorSRHT2(rng3, ntk_feat_shape[-1], feature_dim0, sketch_dim)
      return (new_nngp_feat_shape, new_ntk_feat_shape), (W0, W1, ts2)
    elif not debug and method == 'pts':
      rng1, rng2 = random.split(rng)
      raise NotImplementedError
    else:
      return (new_nngp_feat_shape, new_ntk_feat_shape), ()

  def apply_fn(**kwargs):
    return None

  def random_features_fn(f: Features, input, **kwargs) -> Features:

    W0: np.ndarray = input[0]
    W1: np.ndarray = input[1]
    ts2: TensorSRHT2 = input[2]
    kappa0_feat = (f.nngp_feat @ W0 > 0) / np.sqrt(W0.shape[-1])
    nngp_feat = np.maximum(f.nngp_feat @ W1, 0) / np.sqrt(W1.shape[-1])
    ntk_feat = ts2.sketch(f.ntk_feat, kappa0_feat)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  def polysketch_features_fn(f: Features, input, **kwargs) -> Features:
    pass

  def features_fn_debug(f: Features, input=None, **kwargs) -> Features:

    # Exact feature maps of arc-cosine kernels.
    n = f.nngp_feat.shape[0]
    nngp_feat = np.linalg.cholesky(kappa1(f.nngp_feat))
    kappa0_feat = np.linalg.cholesky(kappa0(f.nngp_feat))

    # Exact tensor product without sketching.
    ntk_feat = np.einsum('ij, ik->ijk', f.ntk_feat, kappa0_feat).reshape(n, -1)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  if debug:
    features_fn = features_fn_debug
  elif method == 'rf':
    features_fn = random_features_fn
  else:
    raise NotImplementedError

  return init_fn, apply_fn, features_fn
