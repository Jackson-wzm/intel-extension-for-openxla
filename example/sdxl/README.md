# Quick Start for Stable Diffusion XL Inference

[Stable Diffusion XL](https://arxiv.org/abs/2307.01952)(SDXL) is a powerful text-to-image generation model that iterates on the previous Stable Diffusion models in three key ways:

1. the UNet is 3x larger and SDXL combines a second text encoder (OpenCLIP ViT-bigG/14) with the original text encoder to significantly increase the number of parameters
2. introduces size and crop-conditioning to preserve training data from being discarded and gain more control over how a generated image should be cropped
3. introduces a two-stage model process; the base model (can also be run as a standalone model) generates an image as an input to the refiner model which adds additional high-quality details

More details could be found in [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)

## Requirements

### 1. Install intel-extension-for-openxla

please got the [main page](https://github.com/intel/intel-extension-for-openxla/blob/main/README.md#build-and-install), and follow the instructions to build and install intel-extension-for-openxla.

### 2. Install jax
```bash
pip install jax==0.4.24 jaxlib==0.4.24 flax==0.8.1
```
### 3. Install huggingface transformers

```bash
pip install transformers==4.37 diffusers==0.26.3 datasets==2.12.0 msgpack==1.0.7
```
## Run

### 1. Environmental Variables

| **ENV** | **Description** | **PVC Platform** | **ATSM/DG2 Platform** | 
| :---: | :---: | :---: |:---: |
| ZE_AFFINITY_MASK | Run this model on single GPU tile |export ZE_AFFINITY_MASK=0 | export ZE_AFFINITY_MASK=0 |
| XLA_FLAGS | Customize xla debug options | export XLA_FLAGS="--xla_gpu_force_conv_nhwc" | export XLA_FLAGS="--xla_gpu_force_conv_nhwc" |

### 2. Options

```
--dtype: support bfloat16 and float16, default is bfloat16.
--num-iter: The number of times to run generation, default is 10.
--num-inference-steps: The inference steps for each generated image, default is 25.
```

### 3. Inference Command Example

```
python inference.py --dtype=bfloat16
```

## Expected Output

```
Average Latency per image is: x.xxxs
```
