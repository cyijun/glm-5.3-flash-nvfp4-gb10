# 在 NVIDIA GB10 上运行 GLM-5.3 Flash NVFP4

本仓库提供 GLM-5.3 Flash NVFP4 在 NVIDIA GB10（SM121）上运行所需的
最小 vLLM/FlashInfer 兼容层。

当前状态：已用保留关键架构尺寸和 NVFP4 布局的
[`cyijun2k/glm-5.3-flash-tiny-random-nvfp4`](https://huggingface.co/cyijun2k/glm-5.3-flash-tiny-random-nvfp4)
在单台 GB10 上验证加载、短请求、491-token prefill、连续 decode 和一层
MTP。完整模型权重约 181 GiB，单台 GB10 放不下，完整模型仍需两台 GB10
做 TP=2 端到端验证。

镜像：

```text
ghcr.io/cyijun/glm-5.3-flash-nvfp4-gb10:vllm-glm53-sm121
```

## 关键修改

GLM-5.3 在逻辑上是 NoPE（NoPE/RoPE 为 256/0），但当前 FlashInfer
GLM_NSA kernel 的物理 ABI 要求：

```text
query = [吸收后的 NoPE 512 | 64 个零]
KV    = [FP8 latent 512 | 4 个 FP32 scale | 64 个 BF16 零]
```

补零不改变 NoPE 点积。与此同时，模型的 `index_topk=2048` 经过 k-pool
尾项和 `BLOCK_N=128` 对齐后，传给 kernel 的实际 buffer 宽度是 2176。
因此镜像会：

1. 给空的逻辑位置键补 64 个 BF16 零；
2. 把吸收后的 512 维 query 补到物理 576 维；
3. 增加 H=64/top-k=2176 的 decode 和 prefill 实例，并为 `sm_121a`
   重新编译 FlashInfer AOT 模块。

命令行的 `--block-size 256` 是 vLLM 上层 index cache 的整除约束；压缩
稀疏注意力内部使用的物理 page 是 64 token，两者不是同一层。

## 运行 mock

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  ghcr.io/cyijun/glm-5.3-flash-nvfp4-gb10:vllm-glm53-sm121 \
  cyijun2k/glm-5.3-flash-tiny-random-nvfp4 \
  --served-model-name glm53-tiny \
  --max-model-len 512 --max-num-seqs 2 \
  --gpu-memory-utilization 0.10 \
  --block-size 256 --moe-backend marlin --enforce-eager
```

使用 HF 镜像时，在镜像名之前增加
`-e HF_ENDPOINT=https://hf-mirror.com`。

完整技术说明和验证边界见 [README.md](README.md)、
[docs/adaptation.md](docs/adaptation.md) 和
[docs/test-matrix.md](docs/test-matrix.md)。双机容器编排可参考已有的
[`deepseek-v4-flash-dgx-spark-tp2`](https://github.com/cyijun/deepseek-v4-flash-dgx-spark-tp2)，
但不要复用它的 584-byte cache kernel；GLM-5.3 的物理 ABI 是 656-byte。
