# Multimodal Autoregressive Image Generator from Scratch in JAX

Build a text-conditioned image generator end to end in JAX by training a VQ-VAE to turn tiny images into discrete tokens, then an autoregressive transformer that models text-prefixed token sequences. You implement every piece from patch encoding and vector quantization to multi-head attention and classifier-free guided sampling.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** generate_toy_images
- [x] **2.** assign_image_labels

---

Built on Deep-ML.
