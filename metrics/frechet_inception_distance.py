# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Frechet Inception Distance (FID)."""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc
import tensorflow_probability as tfp


# def convert_to_numpy(tensor):
#     print(tensor)
#     return tensor.numpy()

# def tensor_function(tensor):
#     result = tf.numpy_function(
#         convert_to_numpy, [tensor], tf.float32
#     )
#     return result

# def return_numpy(tensor):
#     return tensor_function(tensor)

def return_cov(tensor):
    return tfp.stats.covariance(tensor)

#----------------------------------------------------------------------------

class FID(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu
    
    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        # inception = misc.load_pkl('cache/inception_v3_features.pkl') # inception_v3_features.pkl
        inception = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
        )
        inception.run_eagerly = True
        # print(inception.summary())
        activations = np.empty([self.num_images, inception.get_layer(name='mixed10').output.shape[-1]], dtype=np.float32)

        # Calculate statistics for reals.
        cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
        print(f"CACHE FILE: {cache_file}\n")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # try:
        #     if os.path.isfile(cache_file):
        #         print(f"LOADING CACHE FILE...")
        #         mu_real, sigma_real = misc.load_pkl(cache_file)
        # except:
        print('TOTAL NUM IMAGES', self.num_images, '\n')
        for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
            if idx == 3:
                break
            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images)
            
            # Change channels of `images` according Keras Inception H5 model.
            images_channels_last = np.transpose(images, (0, 2, 3, 1))
            print('NUM IMAGE:', idx, images_channels_last.shape)

            # activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)
            activations = inception(images_channels_last)
            if end == self.num_images:
                break
        mu_real = tf.reduce_mean(activations, axis=0)
        # sigma_real = np.cov(activations, rowvar=False)
        sigma_real = return_cov(activations)
        misc.save_pkl((mu_real, sigma_real), cache_file)

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = tf.keras.models.clone_model(inception)
                latents = tf.compat.v1.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                images = Gs_clone.get_output_for(latents, None, is_validation=True, randomize_noise=True)
                images = tflib.convert_images_to_uint8(images)
                result_expr.append(inception(images_channels_last))

        # Calculate statistics for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            end = min(begin + minibatch_size, self.num_images)
            activations = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]
        mu_fake = tf.reduce_mean(activations, axis=0)
        sigma_fake = return_cov(activations)

        # Calculate FID.
        m = tf.math.reduce_sum(tf.square(mu_fake - mu_real))
        # print(sigma_fake.shape)
        # print(tf.transpose(sigma_real, perm=[3, 2, 0, 1]).shape)
        dot_p = tf.experimental.numpy.dot(sigma_fake, sigma_real)
        print(f"DOT P: {dot_p}")
        print(f"PERM DOT P: {tf.transpose(dot_p, perm=[0, 1, 3, 4, 2, 5])}")
        try:
            s, _ = tf.linalg.sqrtm(tf.transpose(dot_p, perm=[0, 1, 3, 4, 2, 5])) # pylint: disable=no-member
            print(f"S: {s}")
            dist = m + tf.experimental.numpy.trace(sigma_fake + sigma_real - 2*s)
            self._report_result(tf.experimental.numpy.real(dist))
        except:
            pass

#----------------------------------------------------------------------------
