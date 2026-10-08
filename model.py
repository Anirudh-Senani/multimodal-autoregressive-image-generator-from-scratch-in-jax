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

# Step 27 - encode_label_to_ids
def encode_label_to_ids(label, char_vocab):
    # TODO: map each character of label to its id and return a jnp int array
    return jnp.asarray([char_vocab[ch] for ch in label])

# Step 28 - form_multimodal_sequence
def form_multimodal_sequence(text_ids, image_tokens, image_token_offset):
    # TODO: prepend text_ids before image_tokens shifted by image_token_offset
    return jnp.concatenate([text_ids, image_tokens+image_token_offset])

# Step 29 - init_token_embedding
def init_token_embedding(key, vocab_size, embed_dim):
    # TODO: build a small randomly initialized (vocab_size, embed_dim) embedding table
    return jax.random.normal(key, shape=(vocab_size, embed_dim)) * 0.02

# Step 30 - init_positional_embedding
def init_positional_embedding(key, max_seq_len, embed_dim):
    # TODO: sample a small-magnitude (max_seq_len, embed_dim) table from the PRNG key
    return jax.random.normal(key, shape=(max_seq_len, embed_dim)) * 0.02

# Step 31 - lookup_token_embeddings
def lookup_token_embeddings(token_embedding, token_ids):
    # TODO: Select the embedding row for each id to get (seq_len, d_model).
    return token_embedding[token_ids]

# Step 32 - add_positional_embeddings
def add_positional_embeddings(token_embeds, positional_embedding):
    # TODO: add the first seq_len positional rows to the token embeddings
    seq_len, _ = token_embeds.shape
    max_len, _ = positional_embedding.shape
    return token_embeds + positional_embedding[:min(seq_len, max_len)]

# Step 33 - build_causal_mask
def build_causal_mask(seq_len):
    # TODO: return (seq_len, seq_len) additive mask: 0.0 where j<=i else -1e9
    return jnp.triu(jnp.full((seq_len, seq_len), -1e9, dtype=jnp.float32), k=1)

# Step 34 - layer_norm
def layer_norm(x, scale, shift, eps=1e-5):
    # TODO: standardize x over its last axis, then apply learned scale and shift
    norm = (x - x.mean(axis=-1, keepdims=True))/jnp.sqrt(x.var(axis=-1, keepdims=True) + eps)
    return scale * norm + shift

# Step 35 - init_attention_params
def init_attention_params(key, d_model):
    # TODO: return dict with 'wq','wk','wv','wo', each (d_model, d_model) small random
    keys = jax.random.split(key, 4)
    init_w = lambda x: jax.random.normal(x, shape=(d_model, d_model)) * 0.02

    weights = jax.vmap(init_w)(keys)

    return dict(
        wq=weights[0],
        wk=weights[1],
        wv=weights[2],
        wo=weights[3]
    )

# Step 36 - project_qkv
def project_qkv(x, attn_params):
    # TODO: Project x into query, key, and value matrices with wq, wk, wv.
    return x @ attn_params['wq'], x @ attn_params['wk'], x @ attn_params['wv']

# Step 37 - reshape_to_heads
def reshape_to_heads(matrix, num_heads):
    # TODO: split the (seq_len, d_model) projection into num_heads attention heads
    seq_len, d_model = matrix.shape
    return matrix.reshape((seq_len, num_heads, d_model//num_heads)).transpose(1,0,2)

# Step 38 - scaled_dot_product_scores
def scaled_dot_product_scores(q_heads, k_heads):
    # TODO: compute scaled dot-product attention scores between query and key heads
    d_model = q_heads.shape[-1]
    scale = 1.0/(d_model**0.5)

    return (q_heads @ k_heads.transpose(0,2,1)) * scale

# Step 39 - add_causal_mask_to_scores
def add_causal_mask_to_scores(scores, causal_mask):
    # TODO: broadcast-add the (seq_len, seq_len) mask onto (num_heads, seq_len, seq_len) scores
    return scores + causal_mask[None,:,:]

# Step 40 - attention_weights_softmax
def attention_weights_softmax(masked_scores):
    # TODO: numerically stable softmax over the last (key) axis of masked_scores
    shifted = jnp.exp(masked_scores - masked_scores.max(axis=-1, keepdims=True))
    return shifted/jnp.sum(shifted, axis=-1, keepdims=True)

# Step 41 - weighted_sum_of_values
def weighted_sum_of_values(attn_weights, v_heads):
    # TODO: per head, combine value vectors using the attention weights...
    return attn_weights @ v_heads

# Step 42 - merge_heads_and_project
def merge_heads_and_project(head_outputs, attn_params):
    # TODO: concatenate per-head outputs into d_model and apply the wo projection
    return head_outputs.transpose(1, 0, 2).reshape(head_outputs.shape[1], -1) @ attn_params['wo']

# Step 43 - init_feedforward_params
def init_feedforward_params(key, d_model, d_ff):
    # TODO: return dict with 'w1' (d_model, d_ff) and 'w2' (d_ff, d_model), small random
    keys = jax.random.split(key, 2)

    return dict(
        w1=jax.random.normal(keys[0], shape=(d_model, d_ff))*0.02,
        w2=jax.random.normal(keys[1], shape=(d_ff, d_model))*0.02
    )

# Step 44 - feedforward_mlp
def feedforward_mlp(x, ff_params):
    # TODO: expand with w1, apply GELU, then project back with w2
    h1 = x @ ff_params['w1']
    a1 = h1/2 * (1 + jnp.tanh((2/jnp.pi)**0.5 * (h1 + 0.044715 * h1**3)))

    return a1 @ ff_params['w2']

# Step 45 - transformer_block
def transformer_block(x, block_params, causal_mask, num_heads):
    # TODO: pre-norm attention with a residual, then pre-norm MLP with a residual
    xin = layer_norm(x, block_params['ln1_scale'], block_params['ln1_shift'])
    q, k, v = project_qkv(xin, block_params['attn'])

    q_heads = reshape_to_heads(q, num_heads)
    k_heads = reshape_to_heads(k, num_heads)
    v_heads = reshape_to_heads(v, num_heads)

    scores = scaled_dot_product_scores(q_heads, k_heads)
    scores = add_causal_mask_to_scores(scores, causal_mask)
    attn = attention_weights_softmax(scores)

    attn = weighted_sum_of_values(attn, v_heads)
    out = merge_heads_and_project(attn, block_params['attn'])

    x = x + out

    xff = layer_norm(x, block_params['ln2_scale'], block_params['ln2_shift'])
    ff = feedforward_mlp(xff, block_params['ff'])

    return x + ff

# Step 46 - transformer_backbone
def transformer_backbone(x, blocks_params, causal_mask, num_heads):
    # TODO: Apply each transformer block in sequence to the hidden states.
    for block_params in blocks_params:
        x = transformer_block(x, block_params, causal_mask, num_heads)

    return x

# Step 47 - init_output_projection
def init_output_projection(key, d_model, vocab_size):
    # TODO: build dict with 'w_out' (d_model, vocab_size) and 'b_out' (vocab_size,)
    return dict(
        w_out=jax.random.normal(key, shape=(d_model, vocab_size))*0.02,
        b_out=jnp.zeros((vocab_size,))
    )

# Step 48 - project_to_logits
def project_to_logits(hidden_states, output_params):
    # TODO: map each (d_model,) hidden vector to (vocab_size,) logits via a linear layer
    return hidden_states @ output_params['w_out'] + output_params['b_out']

# Step 49 - image_token_cross_entropy
def image_token_cross_entropy(logits, target_ids, image_start_index):
    # TODO: mean next-token cross entropy over image-token positions only
    shifted = logits[image_start_index:] - logits[image_start_index:].max(axis=-1, keepdims=True)
    logsumexp = jnp.log(jnp.exp(shifted).sum(axis=-1, keepdims=True))

    logprobs = shifted - logsumexp
    return (-logprobs[jnp.arange(logprobs.shape[0]), target_ids[image_start_index:]]).mean()

# Step 50 - transformer_loss_and_grads
def transformer_loss_and_grads(params, batch_sequences, causal_mask, num_heads, image_start_index):
    # TODO: average per-sequence cross entropy then take value_and_grad over params
    # tok_embed = lambda x, p: lookup_token_embeddings(p, x)
    # pos_embed = lambda x, p: add_positional_embeddings(x, p)
    # transformer_forward = lambda x, p: transformer_backbone(x, p, causal_mask, num_heads)
    # batch_loss = lambda log, seq: image_token_cross_entropy(log, seq, image_start_index)

    # def forward(params, batch_sequences):
    #     x = jax.vmap(tok_embed, in_axes=(0,None))(batch_sequences, params['token_embedding'])
    #     x = jax.vmap(pos_embed, in_axes=(0,None))(x, params['positional_embedding'])
    #     hidden = jax.vmap(transformer_forward, in_axes=(0,None))(x, params['blocks'])
    #     logits = project_to_logits(hidden, params['output'])

    #     loss = jax.vmap(batch_loss, in_axes=(0,0))(logits, batch_sequences)

    #     return loss.mean()

    # loss_fn = lambda p: forward(p, batch_sequences)
    # grad_fn = jax.value_and_grad(loss_fn)

    # return grad_fn(params)

    def forward_single(seq, params):
        x = lookup_token_embeddings(params['token_embedding'], seq)
        x = add_positional_embeddings(x, params['positional_embedding'])

        hidden = transformer_backbone(x, params['blocks'], causal_mask, num_heads)
        logits = project_to_logits(hidden, params['output'])

        loss = image_token_cross_entropy(logits, seq, image_start_index)

        return loss

    loss_fn = lambda p: jax.vmap(forward_single, in_axes=(0, None))(batch_sequences, p).mean()
    grad_fn = jax.value_and_grad(loss_fn)

    return grad_fn(params)

# Step 51 - apply_transformer_update
def apply_transformer_update(params, grads, opt_state, optimizer):
    # TODO: apply one optax update and return (new_params, new_opt_state)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state

# Step 52 - drop_text_prefix
def drop_text_prefix(sequence, key, image_start_index, drop_prob, null_token_id):
    # TODO: with prob drop_prob, replace text-prefix positions with null_token_id
    if jax.random.uniform(key, shape=()) <= drop_prob:
        out = sequence.at[:image_start_index].set(null_token_id)
    else:
        out = sequence

    return out

# Step 53 - combine_guided_logits
def combine_guided_logits(cond_logits, uncond_logits, guidance_scale):
    # TODO: return uncond + guidance_scale * (cond - uncond) for classifier-free guidance
    return uncond_logits + guidance_scale * (cond_logits - uncond_logits)

# Step 54 - logits_to_probabilities
def logits_to_probabilities(logits, temperature):
    # TODO: scale logits by temperature, then apply a stable softmax
    if temperature <= 0.0:
        temperature = 1.0

    logits = logits.astype(jnp.float32)
    shifted = logits/temperature
    shifted = jnp.exp(shifted - shifted.max(axis=-1, keepdims=True))
    probs = shifted/(shifted.sum(axis=-1, keepdims=True))
    return np.array(probs.tolist())

