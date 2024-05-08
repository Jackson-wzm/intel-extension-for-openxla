# GPT-J-6B Jax Example

Script jax_gptj.py for [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6b).

## Prerequisites

```bash
pip install jax==0.4.25 jaxlib==0.4.25 flax==0.8.2 transformers==4.38 datasets==2.12.0
```

## Options

| Option | Default Value | Description|
| :-- | :--: | :--: |
| *```--dtype```*| *```float16```*| Data type, support *```float16```*, *```bfloat16```*, and *```float32```*. |
| *```--batch-size```*| *```1```*| Batch size |
| *```--prompt```*| *```None```*| Customized prompt, not supported  when *```--accuracy-only```* is on. |
| *```--input-tokens```*| *```32```*| Input tokens. |
| *```--max-new-tokens```*| *```32```*| Output max new tokens. |
| *```--greedy```*| *```False```*| Enable greedy search or beam search. |
| *```--num-iter```*| *```10```*| Number of iterations. |
| *```--num-warmup```*| *```3```*| Number of warmup iterations. |
| *```--accuracy-only```*| *```False```*| Run for performance or accuracy only. |

## Accuracy Example

```bash
export ZE_AFFINITY_MASK=0
python jax_gptj.py --accuracy-only --dtype "float16"
```

## Performance Example

To fully utilize the hardware capabilities and achieve the best performance, you may consider setting the below ENV variables to enable our customized optimization strategies.

| **ENV** | **Description** | **PVC Platform** | **ATSM/DG2 Platform** | 
| :---: | :---: | :---: |:---: |
| ZE_AFFINITY_MASK | Run this model on single GPU tile |export ZE_AFFINITY_MASK=0 | export ZE_AFFINITY_MASK=0 |
| XETLA_GEMM | Call the [XETLA](https://github.com/intel/xetla) library to run GEMMs, instead of using oneDNN.|export XETLA_GEMM=1 | NA |

### Greedy Search

```bash
export ZE_AFFINITY_MASK=0
python jax_gptj.py --greedy
```

### Beam Search = 4

```bash
export ZE_AFFINITY_MASK=0
python jax_gptj.py --input-tokens 1024 --max-new-tokens 128
```

### Performance Output
```
Inference latency: x.xxx sec.
Inference throughput: x.xxx samples/sec.
```
