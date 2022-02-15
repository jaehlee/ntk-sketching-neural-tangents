from jax import random
from neural_tangents import stax

from features import _inputs_to_features, DenseFeatures, ReluFeatures, serial

seed = 1
n = 6
d = 4

key1, key2 = random.split(random.PRNGKey(seed))
x1 = random.normal(key1, (n, d))

width = 512  # this does not matter the output


print("================= Result of Neural Tangent Library =================")

init_fn, apply_fn, kernel_fn = stax.serial(
  stax.Dense(width), stax.Relu(),
  stax.Dense(width), stax.Relu(),
  stax.Dense(1))

nt_kernel = kernel_fn(x1, None)


print("NNGP :")
print(nt_kernel.nngp)
print()

print("NTK :")
print(nt_kernel.ntk)
print()

print("================= Result of NTK Random Features =================")

kappa0_feat_dim = 10000
kappa1_feat_dim = 10000
sketch_dim = 20000

f0 = _inputs_to_features(x1)

relufeat_arg = {
    'feature_dim0':kappa0_feat_dim,
    'feature_dim1':kappa1_feat_dim,
    'sketch_dim': sketch_dim,
    'debug': False # if debug is True, it returns the exact feature map
}

init_fn, _, features_fn = serial(
    DenseFeatures(width), ReluFeatures(**relufeat_arg),
    DenseFeatures(width), ReluFeatures(**relufeat_arg),
    DenseFeatures(1)
)

# Initialize random vectors and sketching algorithms
init_nngp_feat_shape = x1.shape
init_ntk_feat_shape = (-1,0)
init_feat_shape = (init_nngp_feat_shape, init_ntk_feat_shape)
_, feat_fn_inputs = init_fn(key2, init_feat_shape)

# Transform input vectors to NNGP/NTK feature map
f0 = _inputs_to_features(x1)
feats = features_fn(f0, feat_fn_inputs)


print("Result of NTK Random Features")

print("NNGP :")
print(feats.nngp_feat @ feats.nngp_feat.T)
print()

print("NTK :")
print(feats.ntk_feat @ feats.ntk_feat.T)
print()

print("================= (Debug) NTK Random Features =================")
relufeat_arg = {
    'feature_dim0':kappa0_feat_dim,
    'feature_dim1':kappa1_feat_dim,
    'sketch_dim': sketch_dim,
    'debug': True # if debug is True, it returns the exact feature map
}

init_fn, _, features_fn = serial(
    DenseFeatures(width), ReluFeatures(**relufeat_arg),
    DenseFeatures(width), ReluFeatures(**relufeat_arg),
    DenseFeatures(1)
)
f0 = _inputs_to_features(x1)
feats = features_fn(f0, feat_fn_inputs)


print("Result of NTK Random Features")

print("NNGP :")
print(feats.nngp_feat @ feats.nngp_feat.T)
print()

print("NTK :")
print(feats.ntk_feat @ feats.ntk_feat.T)
print()
