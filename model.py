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
    # dist = lambda x: squared_distance_to_codebook(x, codebook)
    # return jax.vmap(dist)(latents)

    return ((latents[:, None, :] - codebook[None, :, :])**2).sum(axis=-1)

# Step 14 - assign_nearest_codes
import jax

def assign_nearest_codes(distances):
    # TODO: return the nearest codebook index for each latent via argmin over codes
    return jnp.argmin(distances, axis=-1)

# Step 15 - lookup_codebook_vectors
def lookup_codebook_vectors(indices, codebook):
    # TODO: return the codebook row for each token index, shape (num_patches, latent_dim)
    return codebook[indices]

# Step 16 - straight_through_quantize
def straight_through_quantize(latents, quantized):
    # TODO: return quantized in value but with latents' gradient (straight-through)
    return latents + jax.lax.stop_gradient(quantized - latents)

# Step 17 - codebook_loss
def codebook_loss(latents, quantized):
    # TODO: mean squared error between stop_gradient(latents) and quantized
    return ((jax.lax.stop_gradient(latents) - quantized)**2).mean()

# Step 18 - commitment_loss
def commitment_loss(latents, quantized):
    # TODO: mean squared error between latents and stop_gradient(quantized)
    return ((latents - jax.lax.stop_gradient(quantized))**2).mean()

# Step 19 - reconstruction_loss
def reconstruction_loss(image, reconstruction):
    # TODO: return the mean squared error between image and reconstruction
    return ((image - reconstruction)**2).mean()

# Step 20 - total_vqvae_loss
def total_vqvae_loss(recon_loss, cb_loss, commit_loss, commitment_weight):
    # TODO: return recon_loss + cb_loss + commitment_weight * commit_loss as a scalar
    return recon_loss + cb_loss + commitment_weight * commit_loss

# Step 21 - vqvae_loss_and_grads
def vqvae_loss_and_grads(params, image_batch, patch_size, commitment_weight):
    # TODO: Compute the VQ-VAE total loss and gradients wrt encoder/decoder/codebook over a batch of images.
    imsplit = lambda x: split_image_into_patches(x, patch_size)
    imrecon = lambda x, gh, gw: reassemble_patches_into_image(x, gh, gw, patch_size)
    total_loss = lambda rc, cb, cl: total_vqvae_loss(rc, cb, cl, commitment_weight)

    def forward(params, image_batch):
        patches = jax.vmap(imsplit)(image_batch)
        b, gh, gw, ph, pw = patches.shape
        patches_flat = patches.reshape((b, gh*gw, ph*pw))

        latents = encode_patches(patches_flat, params['encoder'])
        dists = jax.vmap(grid_distances_to_codebook, in_axes=(0,None))(latents, params['codebook'])
        inds = assign_nearest_codes(dists)

        quantized = lookup_codebook_vectors(inds, params['codebook'])
        latents_st = straight_through_quantize(latents, quantized)

        decoded = decode_latents(latents_st, params['decoder'])
        recon_batch = jax.vmap(imrecon, in_axes=(0,None,None))(decoded, gh, gw)

        cb_loss = codebook_loss(latents, quantized)
        commit_loss = commitment_loss(latents, quantized)
        recon_loss = reconstruction_loss(image_batch, recon_batch)

        loss = total_loss(recon_loss, cb_loss, commit_loss)

        return loss

    loss_fn = lambda p: forward(p, image_batch)
    grad_fn = jax.value_and_grad(loss_fn)

    return grad_fn(params)

# Step 22 - apply_vqvae_update
def apply_vqvae_update(params, grads, opt_state, optimizer):
    # TODO: Apply one optax update to the VQ-VAE params and return new params + opt state.
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state

# Step 23 - encode_image_to_tokens
def encode_image_to_tokens(image, params, patch_size):
    # TODO: split, encode, quantize, and reshape patch codes into a token grid
    patches = split_image_into_patches(image, patch_size)
    gh, gw, _, _ = patches.shape
    flat_patches = flatten_patches(patches)

    latents = encode_patches(flat_patches, params['encoder'])
    dists = grid_distances_to_codebook(latents, params['codebook'])
    inds = assign_nearest_codes(dists)

    return inds.reshape((gh, gw))

# Step 24 - flatten_token_grid
def flatten_token_grid(token_grid):
    # TODO: Flatten a (grid_h, grid_w) token grid into a 1D sequence in row-major order.
    return token_grid.reshape(-1)

# Step 25 - reshape_tokens_to_grid
def reshape_tokens_to_grid(token_sequence, grid_h, grid_w):
    # TODO: reshape a 1D token sequence back into a 2D (grid_h, grid_w) grid
    return token_sequence.reshape((grid_h, grid_w))

# Step 26 - build_char_vocab
def build_char_vocab(labels):
    # TODO: map each unique character across all labels to a deterministic integer id
    vocab = set()
    for label in set(labels):
        vocab.update(list(label))

    return {ch:i for i, ch in enumerate(sorted(vocab))}

