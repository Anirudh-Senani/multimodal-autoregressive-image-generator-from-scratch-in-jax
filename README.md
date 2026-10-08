# Multimodal Autoregressive Image Generator from Scratch in JAX

Build a text-conditioned image generator end to end in JAX by training a VQ-VAE to turn tiny images into discrete tokens, then an autoregressive transformer that models text-prefixed token sequences. You implement every piece from patch encoding and vector quantization to multi-head attention and classifier-free guided sampling.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** generate_toy_images
- [x] **2.** assign_image_labels
- [x] **3.** normalize_image_batch
- [x] **4.** split_image_into_patches
- [x] **5.** flatten_patches
- [x] **6.** init_patch_encoder
- [x] **7.** encode_patches
- [x] **8.** init_patch_decoder
- [x] **9.** decode_latents
- [x] **10.** reassemble_patches_into_image
- [x] **11.** init_codebook
- [x] **12.** squared_distance_to_codebook
- [x] **13.** grid_distances_to_codebook
- [x] **14.** assign_nearest_codes
- [x] **15.** lookup_codebook_vectors
- [x] **16.** straight_through_quantize
- [x] **17.** codebook_loss
- [x] **18.** commitment_loss
- [x] **19.** reconstruction_loss
- [x] **20.** total_vqvae_loss
- [x] **21.** vqvae_loss_and_grads
- [x] **22.** apply_vqvae_update
- [x] **23.** encode_image_to_tokens
- [x] **24.** flatten_token_grid
- [x] **25.** reshape_tokens_to_grid

---

Built on Deep-ML.
