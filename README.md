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

---

Built on Deep-ML.
