# GLM-5.3 Flash NVFP4 on NVIDIA GB10

[简体中文](README.zh-CN.md)

This repository packages the minimal FlashInfer/vLLM compatibility layer used
to run the GLM-5.3 Flash NVFP4 architecture on NVIDIA GB10 (SM121).

Status: **validated with the faithful tiny-random fixture on one GB10**. The
181 GiB full checkpoint does not fit on one GB10 and still requires an end-to-end
two-GB10/TP=2 validation before this image should be described as production
ready for the full model.

- Runtime image: `ghcr.io/cyijun/glm-5.3-flash-nvfp4-gb10:vllm-glm53-sm121`
- Mock model: [`cyijun2k/glm-5.3-flash-tiny-random-nvfp4`](https://huggingface.co/cyijun2k/glm-5.3-flash-tiny-random-nvfp4)
- Upstream checkpoint: [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)

The mock preserves every dimension found to select a loader branch, CUDA
specialization, packed NVFP4 layout, cache ABI, router, or MTP path. Its random
text is deliberately meaningless.

## What the adapter changes

The model is logically NoPE, but FlashInfer's SM120/SM121 `fp8_ds_mla` GLM_NSA
kernel has a fixed physical ABI:

| Contract | Logical GLM-5.3 | Physical kernel ABI |
| --- | ---: | ---: |
| absorbed query | 512 | 512 + 64 zero padding |
| KV latent | 512 FP8 | 512 FP8 |
| scale metadata | — | four FP32 values |
| positional payload | none | 64 BF16 zeros |
| cache bytes/token | — | 656 |
| architecture sparse top-k | 2048 | — |
| aligned sparse buffer capacity | — | 2176 |

Appending zero dimensions leaves the NoPE dot product unchanged. No checkpoint
tensor or Hugging Face configuration field is rewritten.

The image makes three scoped changes:

1. write a 64-BF16 zero tail when the logical positional key is empty;
2. pad the absorbed query to the physical 576-wide GLM_NSA ABI and preserve
   per-row valid lengths;
3. add and rebuild the exact FlashInfer AOT decode and prefill specializations
   for `num_heads=64, topk=2176` on `sm_121a`.

The CLI must use `--block-size 256` for GLM's index-cache divisibility. Inside
the compressed sparse-attention path, FlashInfer receives 64-token physical
pages; these are two different cache levels.

## Run the published mock

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  ghcr.io/cyijun/glm-5.3-flash-nvfp4-gb10:vllm-glm53-sm121 \
  cyijun2k/glm-5.3-flash-tiny-random-nvfp4 \
  --served-model-name glm53-tiny \
  --max-model-len 512 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.10 \
  --block-size 256 \
  --moe-backend marlin \
  --enforce-eager
```

For an HF mirror, add `-e HF_ENDPOINT=https://hf-mirror.com` before the image
name. Optional one-token MTP validation adds:

```text
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Build and smoke test

```bash
./scripts/build-image.sh
./scripts/smoke-mock.sh
MTP=1 ./scripts/smoke-mock.sh
```

Override `IMAGE`, `MODEL`, `PORT`, or `HF_ENDPOINT` as needed. `MODEL` may be a
public Hugging Face repo ID or an absolute local checkpoint path.

The base image is digest-pinned in `config/versions.env`; its vLLM,
FlashInfer, and Transformers versions are checked during every build. The base
ships the sparse module as an AOT `.so`, so changing only Python/JIT source is
not sufficient—the Docker build deliberately recompiles and replaces that
module.

## Validation boundary

On one GB10 the fixture passed model load, `/health`, the OpenAI models endpoint,
short continuous decode, a 491-token prefill plus 16-token decode, and one-token
MTP. It loaded with about 0.64 GiB of model memory (0.99 GiB with MTP).

The full `LibertAIDAI` checkpoint is about 181 GiB. Use two GB10 systems and
tensor parallelism 2; this repository does not claim that distributed full-model
gate has passed. The topology and container orchestration in
[`cyijun/deepseek-v4-flash-dgx-spark-tp2`](https://github.com/cyijun/deepseek-v4-flash-dgx-spark-tp2)
are the intended deployment reference, but GLM-5.3 needs the different 656-byte
cache ABI and 2176 specialization documented here.

See [`docs/adaptation.md`](docs/adaptation.md) for the rationale and
[`docs/test-matrix.md`](docs/test-matrix.md) for exact pass/pending status.
