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
    labels = row_sums[:, :image_size//2].sum(axis=-1) >= row_sums[:, image_size//2:].sum(axis=-1)

    return ['left' if label else 'right' for label in labels.tolist()]

# Step 3 - normalize_image_batch
def normalize_image_batch(images):
    # TODO: rescale images from [0, 1] into the symmetric [-1, 1] range
    # B, H, W = images.shape
    # means = images.reshape((B, H*W)).mean(axis=0, keepdims=True)
    # sigma = images.reshape((B, H*W)).var(axis=0, keepdims=True)

    # means = means.reshape((B, 1, 1))
    # sigma = sigma.reshape((B, 1, 1))

    return 2*images - 1

# Step 4 - split_image_into_patches
def split_image_into_patches(image, patch_size):
    # TODO: Split a single (H, W) image into a grid of non-overlapping square patches.
    H, W = image.shape
    num_patches_h = H//patch_size
    num_patches_w = W//patch_size

    return image.reshape((num_patches_h, patch_size, num_patches_w, patch_size)).transpose(0,2,1,3)

# Step 5 - flatten_patches
def flatten_patches(patches):
    # TODO: flatten each (p, p) patch in a (gh, gw, p, p) grid into a 1D pixel vector
    gh, gw, ph, pw = patches.shape
    return patches.reshape(gh*gw, ph*pw)

# Step 6 - init_patch_encoder
def init_patch_encoder(key, patch_dim, latent_dim):
    # TODO: return a (patch_dim, latent_dim) scaled Gaussian weight matrix from key
    return jax.random.normal(key, shape=(patch_dim, latent_dim))/(patch_dim**0.5)

# Step 7 - encode_patches
def encode_patches(flat_patches, encoder_weight):
    # TODO: Project each flattened patch to a latent vector with the linear encoder.
    return flat_patches @ encoder_weight

# Step 8 - init_patch_decoder
def init_patch_decoder(key, latent_dim, patch_dim):
    # TODO: sample a (latent_dim, patch_dim) weight from key, scaled by 1/sqrt(latent_dim)
    return jax.random.normal(key, shape=(latent_dim, patch_dim))/(latent_dim**0.5)

# Step 9 - decode_latents
def decode_latents(latents, decoder_weight):
    # TODO: project each latent vector back to flat patch pixels with the linear decoder
    return latents @ decoder_weight

# Step 10 - reassemble_patches_into_image
def reassemble_patches_into_image(flat_patches, grid_h, grid_w, patch_size):
    # TODO: reshape each flat patch to a square and tile them into the full image
    return flat_patches.reshape(grid_h, grid_w, patch_size, patch_size).transpose(0,2,1,3).reshape((grid_h*patch_size, grid_w*patch_size))

# Step 11 - init_codebook
def init_codebook(key, num_codes, latent_dim):
    # TODO: draw a (num_codes, latent_dim) table of small random values from key
    return jax.random.normal(key, shape=(num_codes, latent_dim)) * 0.1

# Step 12 - squared_distance_to_codebook
def squared_distance_to_codebook(latent, codebook):
    # TODO: squared Euclidean distance from one latent vector to every codebook vector
    return ((codebook - latent[None,:])**2).sum(axis=-1)

# Step 13 - grid_distances_to_codebook
def grid_distances_to_codebook(latents, codebook):
    # TODO: squared distance from each latent (P, D) to each code (K, D) -> (P, K)
    dist = lambda x: squared_distance_to_codebook(x, codebook)
    return jax.vmap(dist)(latents)

