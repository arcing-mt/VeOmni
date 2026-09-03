# VeOmni MUSA 迁移说明

本文记录把旧版 `./veomni` 中已经验证过的 MUSA 运行时适配，迁移到
最新版 `./VeOmni` 的结果。目标是让最新版 VeOmni 的 Qwen3.5
训练入口可以在 Moore Threads MUSA 上使用 MCCL、MUSA fused AdamW、MUSA 版
FlashAttention 3，以及 `flash-linear-attention` 的 GatedDeltaNet 算子。

## 结论

最新版 VeOmni 已经包含 Qwen3.5 的 Transformers 5.x patchgen 模型、统一 trainer、
OpSlot kernel registry 和 `tasks/train_text.py` 入口。旧仓库中的 GLM-Image/MoE 专用
MUSA 代码不能直接 cherry-pick，因为模型、参数系统和 kernel registry 已经重构。
本次迁移按最新版架构重新接入了 MUSA，而不是覆盖最新版的 Qwen3.5 实现。

在本机环境中已验证：

- `torch_musa` 可用，MUSA 设备数为 8，设备名为 `MTT S5000`；
- VeOmni 设备层返回 `musa`，分布式 backend 返回 `mccl`；
- `flash_attn_interface.flash_attn_func` 的 MUSA forward 正常；
- VeOmni 的 FA3 SP wrapper（普通和 varlen 输入）正常；
- MUSA `FusedAdamW` 一步更新正常；
- 单进程 MCCL `all_reduce` 正常；
- Qwen3.5-4B 的 VeOmni patched model 可以构造并加载 checkpoint；
- Qwen3.5 的 FLA GatedDeltaNet（gated RMSNorm、causal conv1d、chunk gated delta rule）
  可以完成 forward/backward；当前环境实测没有复现先前的卡住问题。

当前仍需注意：目标仓库声明 Python `>=3.11,<3.13`，而本次 shell 是 Python 3.10；
当前解释器已经有 `torchdata 0.11.0+cpu`，但正式训练仍建议切换到仓库声明的
Python 3.11/3.12 环境。本文没有自动安装 `torchdata`，也没有自动覆盖任何
MUSA/torch 依赖。

## 迁移的文件

| 文件 | 作用 |
|---|---|
| `veomni/utils/device.py` | 识别 `torch.musa`；返回 `musa` 设备；选择 `mccl`；MUSA stream、设备事件和 compute unit 支持 |
| `veomni/utils/import_utils.py` | 增加 `torch_musa`、FA3 可用性检测；保留 CUDA/MLU fused-MoE 的硬件 gate |
| `veomni/utils/helper.py` | MUSA seed、cache、BF16 精度开关、profiler 和显存统计 |
| `veomni/distributed/async_offload.py` | 把 MUSA tensor 纳入 accelerator offload 判断 |
| `veomni/distributed/torch_parallelize.py` | MUSA + ExtraParallel 时安装 MCCL PREMUL_SUM wrapper；兼容两种 grad divide API |
| `veomni/optim/optimizer.py` | `fused=True` 时使用 `torch_musa.optim.FusedAdamW`，避免调用 CUDA-only fused AdamW |
| `veomni/ops/kernel_registry.py` | 增加 `device_type="musa"` 硬件 gate |
| `veomni/ops/kernels/gated_delta_rule/__init__.py` | 保持仓库原始 FLA GDN 注册，仅让 FLA 的 GPU-family gate 识别 MUSA |
| `veomni/ops/platform/musa/` | MCCL PREMUL_SUM 和 Transformers FA3 availability runtime shim |
| `veomni/models/auto.py` | 构造模型前安装 MUSA FA3 availability shim |
| `veomni/arguments/arguments_types.py` | 默认 FA2 在有本地 FA3 时显式切到 FA3；GDN 三项恢复仓库原始默认 `fla` |
| `train.sh` | 自动检测 `mthreads-gmi`、设置 MUSA/MCCL 环境、使用当前 Python 的 torch launcher |

## 算子配置（完整保留仓库原始字段）

下面的字段没有被删除或合并。除 attention 为 MUSA 上的 FA3 显式适配外，其余
字段仍按 `OpsImplementationConfig` 的原始值解析；MUSA 不会因为某个 kernel
不可用就偷偷替换为 eager/native fallback，而是按正常 registry 错误暴露问题。

| 配置字段 | 仓库原始默认 | MUSA 说明 |
|---|---|---|
| `attn_implementation` | `flash_attention_2` | 本机有 `flash_attn_interface` 时显式使用 FA3，最终为 `veomni_flash_attention_3_with_sp` |
| `moe_implementation` | `fused_triton` | Qwen3.5-4B 是 dense，不走 MoE；MoE backend 仍需显式验证 |
| `cross_entropy_loss_implementation` | `liger_kernel` | 保持原值，不自动改成 `chunk_loss` |
| `rms_norm_implementation` | `liger_kernel` | 保持原值 |
| `swiglu_mlp_implementation` | `liger_kernel` | 保持原值 |
| `rotary_pos_emb_implementation` | `liger_kernel` | 保持原值 |
| `rotary_pos_emb_vision_implementation` | `eager` | 保持原值 |
| `load_balancing_loss_implementation` | `triton` | dense Qwen3.5 不使用；保持原值 |
| `rms_norm_gated_implementation` | `fla` | 回归仓库原始默认 FLA |
| `causal_conv1d_implementation` | `fla` | 回归仓库原始默认 FLA |
| `chunk_gated_delta_rule_implementation` | `fla` | 回归仓库原始默认 FLA |
| `dsa_indexer_implementation` | `eager` | 保持原值 |
| `dsa_attention_implementation` | `eager` | 保持原值 |
| `mhc_implementation` | `eager` | 保持原值 |

三个 GDN OpSlot 在 MUSA 上直接使用仓库原始的
`flash-linear-attention`：`FusedRMSNormGated`、`fla.modules.convolution.causal_conv1d`
和 `fla.ops.gated_delta_rule.chunk_gated_delta_rule`。如果 FLA 不可用或 kernel
不支持当前输入，VeOmni 按正常错误路径抛出异常，不替换成其它实现。

Qwen3.5-4B 是 dense 模型，不会走 MoE expert kernel；如果以后跑 Qwen3.5-MoE，
需要显式选择并验证 MUSA group-GEMM，不能把 CUDA Quack 误标为 MUSA。

## FlashAttention 3

本机提供：

```python
import flash_attn_interface
flash_attn_interface.flash_attn_func(...)
```

并且该函数已经在 MUSA tensor 上完成测试。因此不需要为了兼容性主动绕开 Flash
Attention。推荐使用：

```yaml
model:
  ops_implementation:
    attn_implementation: flash_attention_3
```

或者命令行覆盖：

```bash
--model.ops_implementation.attn_implementation flash_attention_3
```

VeOmni 在 `MODELING_BACKEND=veomni` 下会把它改写为
`veomni_flash_attention_3_with_sp`，再由 `veomni/ops/kernels/attention/flash.py`
执行 Ulysses gather/scatter 和 Transformers FA3 wrapper。

之所以加入 `veomni/ops/platform/musa/flash_attn.py`，不是替换 FA3 kernel，而是修复
Transformers 的设备可用性判断：Transformers 5.9 的通用 `is_flash_attn_3_available()`
只检查 `torch.cuda`，而 MUSA 使用 `torch.musa`。这个 shim 只修改运行时 gate 和
compatibility matrix，不修改已安装的 Transformers 包。

当前机器没有可导入的 `flash_attn` FA2 包。因此 MUSA 上检测到
`flash_attn_interface` 时，默认的 FA2 配置会明确改成 FA3；如果 FA3 也不可用，
不会静默改成 eager，需由用户显式处理缺失的 attention backend。

## 启动入口

最新版普通文本训练入口是：

```bash
cd VeOmni

bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
  --model.model_path /data/share/models/Qwen3.5-4B \
  --data.train_path /path/to/text_dataset.parquet \
  --data.data_type plaintext \
  --data.text_keys text \
  --model.ops_implementation.attn_implementation flash_attention_3 \
  --model.ops_implementation.rms_norm_gated_implementation fla \
  --model.ops_implementation.causal_conv1d_implementation fla \
  --model.ops_implementation.chunk_gated_delta_rule_implementation fla \
  --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
  --train.init_device meta \
  --train.global_batch_size 8 \
  --train.max_steps 20 \
  --train.checkpoint.output_dir /path/to/qwen35_output \
  --train.wandb.enable false
```

`train.sh` 的 MUSA 分支会：

1. 用 `mthreads-gmi --list-gpus` 计算 `NPROC_PER_NODE`；
2. 设置 `PYTORCH_MUSA_ALLOC_CONF`、`MUSA_EXECUTION_TIMEOUT` 和 MCCL 环境；
3. 调用 `${TORCHRUN_PYTHON:-${PYTHON:-python}} -m torch.distributed.run`。

最后一点很重要：镜像里的 `/usr/local/bin/torchrun` 可能绑定到 `/usr/bin/python3`，
会绕过当前 venv。迁移后的 launcher 使用当前 Python，避免“依赖装在 venv、launcher
却使用系统 Python”的隐蔽问题。

上面的命令面向 8 卡 FSDP2，因此使用 `init_device: meta`。如果只是单卡 smoke，
FSDP2 在 world size=1 时不会建立 FSDP shard，应改用 DDP 并让模型直接放到 MUSA：

```bash
NPROC_PER_NODE=1 bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
  --model.model_path /data/share/models/Qwen3.5-4B \
  --data.train_path /data/share/datasets/no_robots_test/data/train-00000-of-00001.parquet \
  --data.data_type conversation --data.text_keys messages \
  --data.max_seq_len 128 --train.max_steps 1 \
  --train.accelerator.fsdp_config.fsdp_mode ddp \
  --train.accelerator.fsdp_config.mixed_precision.enable false \
  --train.init_device musa --train.wandb.enable false
```

单卡 DDP 下关闭 `mixed_precision.enable` 是必要的：该模式没有 FSDP forward cast，
而本机 FA3 接口要求 fp16/bf16；关闭后 VeOmni 会按 bf16 构造模型。

## 数据格式

`tasks/train_text.py` 的 `plaintext` transform 要求每一行包含文本字段，例如：

```text
{"text": "A training document ..."}
```

训练数据可以是 parquet、json/jsonl、csv 或 arrow。对于对话 SFT，使用：

```text
data_type: conversation
text_keys: messages
chat_template: chatml
```

`/data/share/datasets/no_robots_test` 的 parquet 字段是 `prompt`、`messages`、
`category`，没有 `text`。把它用于启动 smoke test 时，可以显式把 `prompt` 当纯文本：

```bash
--data.train_path /data/share/datasets/no_robots_test/data/train-00000-of-00001.parquet \
--data.data_type plaintext \
--data.text_keys prompt
```

这不是正规的 conversation SFT，只是验证数据读取和训练链路。

## 已执行的验证

所有验证均为只读/临时进程，不安装新的依赖：

```text
device type                  musa
MUSA device count            8
distributed backend          mccl
FA3 ordinary forward         pass (MUSA tensor)
FA3 varlen VeOmni wrapper    pass (cu_seq_lens_q/k)
FusedAdamW one-step          pass
MCCL one-process all_reduce  pass
MCCL PREMUL_SUM all_reduce   pass (wrapper)
Qwen3.5 config/tokenizer     pass
Qwen3.5 VeOmni model build   pass (meta construction)
Qwen3.5 checkpoint load      pass (main model weights)
Qwen3.5 full forward/loss     pass (MUSA, FA3 + FLA GDN)
Qwen3.5 full backward         pass (MUSA, one short sequence + FLA GDN)
`tasks/train_text.py` smoke   pass (single-card DDP, 3 short iterations)

FLA GDN benchmark (B=1,S=128,H=32,D=128，warm cache 后)：FLA forward 约 1--2 ms、
backward 约 2--3 ms；此前加入的 native fallback forward 约 8--9 ms、backward
约 16--17 ms，性能明显不如 FLA。因此 native fallback 已删除，不再保留或自动启用。
```

Qwen3.5 checkpoint 中的 `mtp.*` 参数会被当前 VeOmni patched model 按设计忽略，
因为生成模型声明了 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`。`lm_head` 与
输入 embedding 的 tied-weight 关系由模型构造后恢复。也就是说，当前入口验证的是
主语言模型训练，不是包含 MTP auxiliary objective 的完整 Qwen3.5 原生训练。

## 环境阻塞和已知限制

### 必须由环境提供

- Python `>=3.11,<3.13`（目标仓库 `pyproject.toml` 的声明；当前 shell 为 3.10）；
- `torchdata>=0.8,<1.0`（`veomni.data.data_loader` 的硬依赖；当前环境为 0.11.0+cpu）；
- `torch_musa` 与 torch 的匹配版本；
- `flash-linear-attention`（MUSA GDN 默认且唯一 backend）；
- `flash_attn_interface`（如果使用 FA3；本机已存在）。

本次迁移只显式升级了用户要求的 `transformers==5.9.0`，没有自动安装
`torchdata`、FA3、FLA 或其它依赖。

### 当前环境的非致命警告

本机 `flash-linear-attention` 会提示 Triton 3.2 低于其建议的 3.3，且 Python 3.10
低于其建议的 3.11；当前 FLA forward/backward 复测正常。这些提示不影响已完成的
小算子调用，但正式训练应使用目标仓库声明的 Python 3.11+ 环境并确认
Triton/torch_musa 组合经过验证。

当前 shell 里还存在若干“已安装但超出最新版 pyproject 范围”的包：`datasets 5.0.1`
（仓库声明 `<=2.21.0`）、`packaging 26.3`（仓库声明 `<26.0`）。本次按你的要求
没有降级或替换它们；现有 parquet 数据读取和单步训练 smoke 已正常。如果正式任务
遇到数据层/API 回归，再单独决定是否调整版本。

### 未迁移的旧 GLM 专用功能

旧仓库中的 GLM-Image 专用 MUSA MATE group-GEMM、MUSA token-permute、Muon NS epilogue
和 d16 launcher 没有直接搬进最新版。最新版已经有不同的 `veomni/ops/kernels/moe`
和统一 trainer；把旧文件原样覆盖会破坏新 OpSlot/patchgen 结构。当前迁移范围是
最新版 Qwen3.5 普通文本训练所需的运行时与 kernel 适配。
