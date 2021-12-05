import tensorflow as tf
from training import misc
import dnnlib


dnnlib.tflib.init_tf()
inception = misc.load_pkl('cache/inception_v3_features.pkl') # inception_v3_features.pkl
print(inception.output_shape)