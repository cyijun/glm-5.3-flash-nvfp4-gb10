# Validation matrix

Validated on 2026-08-27 with one NVIDIA GB10 (SM121). The source fixture is
[`cyijun2k/glm-5.3-flash-tiny-random-nvfp4`](https://huggingface.co/cyijun2k/glm-5.3-flash-tiny-random-nvfp4).

| Gate | Status | Evidence/result |
| --- | --- | --- |
| pinned base contract | pass | vLLM `0.1.dev20051+g487ecf187`, FlashInfer `0.6.17`, Transformers `5.15.1`, PyTorch `2.13.0+cu130` |
| target/mock config schema | pass | 21/21 critical fields; 5,342/5,342 fixture names map to target tensors |
| NVFP4 schema | pass | 1,728 projections; `U8` / `F8_E4M3` / `F32` storage |
| AOT source and binary build contract | pass | H=64/top-k=2176 decode and GLM_NSA prefill present; rebuilt for `sm_121a` |
| faithful mock load | pass | ~0.64 GiB model memory; sparse SM120 backend and `fp8_ds_mla` selected |
| health and models APIs | pass | HTTP 200 |
| short chat/decode | pass | 19 prompt + 8 generated tokens; HTTP 200 |
| long prefill/decode | pass | 491 prompt + 16 generated tokens; HTTP 200; exercises >64-token prefill |
| MTP load and request | pass | ~0.99 GiB model memory; 22 prompt + 16 generated tokens; HTTP 200 |
| full 181 GiB checkpoint load on TP2 | pending | requires two GB10 systems |
| full-checkpoint deterministic serve | pending | depends on TP2 load gate |
| long-context capacity/soak | pending | depends on full-checkpoint TP2 deployment |
| CUDA graph mode | pending | initial release intentionally uses eager mode |
| multimodal image request | pending | text/cache/kernel compatibility was the first release gate |

Random weights make MTP acceptance rate and generated text quality meaningless.
The MTP pass means proposal, verification, and response generation completed
without loader, shape, dispatch, or CUDA errors.

The conservative runtime flags used for validation were:

```text
--attention-backend FLASHINFER_MLA_SPARSE_SM120
--kv-cache-dtype fp8_ds_mla
--block-size 256
--moe-backend marlin
--enforce-eager
```

Alternative MoE backends, CUDA graphs, full context, and higher concurrency
should each get a separate correctness and memory gate on the full TP2 setup.
