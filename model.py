"""
Multimodal Autoregressive Image Generator from Scratch in JAX

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - generate_toy_images
# import jax.numpy as jnp


def generate_toy_images(key, num_images, image_size):
    # TODO: return num_images grayscale images, each a bright square on a zero background
    keys = jax.random.split(key, num_images)
    bright_patch = jnp.ones((image_size//2, image_size//2))

    def generate_one_image(single_key):
        bg = jnp.zeros((image_size, image_size))
        inds = jax.random.randint(single_key, shape=(2,), minval=0, maxval=image_size- image_size//2)
        return jax.lax.dynamic_update_slice(bg, bright_patch, inds)

    return jax.vmap(generate_one_image)(keys)

# Step 2 - assign_image_labels
def assign_image_labels(images):
    # TODO: label each image 'left' or 'right' by comparing left vs right pixel mass
    image_size = images.shape[-1]
    row_sums = images.sum(axis=1)
    labels = row_sums[:, :image_size//2].sum(axis=-1) <= row_sums[:, image_size//2:]
    print(labels)

    return ['left' if label==1 else 'right' for label in labels.tolist()]

