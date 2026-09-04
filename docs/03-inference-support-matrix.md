# Qwen3.8-27B 推理框架支持矩阵与实战情报

> 调研日期：2026-09-04。框架适配变动很快，动手前请复核对应 issue/PR 状态。
> 本文覆盖官方大框架 + 民间/个人项目，是回家实测前的作战地图。

## 0. 一句话结论

第一站 **llama.cpp + unsloth UD-Q4_K_M（选择性 FFN 卸载）或 UD-IQ4_XS（全进显存）**；
第二站 **SGLang**（正在为 sm_120 做专属优化）；Transformers 当正确性裁判；
MTP 在消费级 Blackwell（sm_120）上暂为负优化，**先关**。

## 1. 目标硬件与预算对照

| 项目 | 数值 | 来源 |
|---|---:|---|
| 本机 GPU | RTX 5060 Ti 16 GB（sm_120） | 家里机器 |
| INT4 权重预算（本项目 CLI 估算） | 12.94 GiB | `cargo run -- plan` |
| UD-IQ4_XS（全进显存候选） | 14.3 GB | unsloth GGUF |
| UD-Q4_K_M（需 FFN 卸载） | 16.5 GB | unsloth GGUF |
| KV cache @4K ctx（BF16） | 0.25 GiB | CLI 估算 |
| KV cache @64K ctx（Q4_0 量化后） | ~1 GiB | 按 2,048 B/tok/layer 折算 |

显存 16 GB 决定：Q4 档可跑，长上下文必须量化 KV cache，FP8/BF16 免谈。

## 2. 官方框架支持矩阵

Qwen 官方模型卡点名兼容：Transformers / vLLM / SGLang / TokenSpeed，
且均发布官方部署教程。

| 框架 | Qwen3.8 支持 | sm_120 状态 | 已知坑（调研时点） | 建议 |
|---|---|---|---|---|
| **llama.cpp** | 适配中，GGUF 生态成熟 | CUDA 构建可用 | ① MTP 在 RTX 5090(sm_120) 仅 ~1.06x（4090 为 1.8x）（issue #28196）② 混合 layer_types 加载 bug（#28207）③ 双 50 系卡张量并行输出乱码（#26257，单卡不受影响） | **主力**。单卡 + 单 GPU 模式避开 ②③ |
| **SGLang** | 官方 Cookbook；GDN/MTP/VL 活跃开发 | **正在专门做 sm_120 优化**（decode GEMM、内存层级调优 PR 持续合并） | 早期版本波动大 | **第二站**。等版本稳定或直接实测最新版 |
| **vLLM** | 官方 Recipe | 可用 | ① Qwen3.5 家族 KV cache scale 未加载，fp8 KV 静默按 1.0 跑（#54623）② GDN 长预填 1.9x 回归 + 1.15 GiB 额外显存（#53787） | 对比项。**勿信 fp8 KV 输出**，正确性对拍需绕开 |
| **Transformers** | 官方定义架构，最先支持 | 慢，不适合生产 | 无（就是基准） | **裁判**：greedy 对拍验证其他框架 |
| **TokenSpeed** | 官方 Recipe | 未验证 | 资料少 | 支持矩阵里留一行，暂不投入 |

## 3. 民间/个人项目情报（重点：16 GB 同档实测）

### 3.1 直接可抄的作业

| 项目 | 硬件 | 关键结果 | 可抄配置 |
|---|---|---|---|
| [johnconnor2020/qwen38-27b-rtx5080-16gb](https://github.com/johnconnor2020/qwen38-27b-rtx5080-16gb) | RTX 5080 16 GB | UD-Q4_K_M + **64K ctx**，13.2 tok/s @49K，召回 4/4 | 选择性 FFN 卸载：全层进 GPU，手动把 16 个最大 FFN tensor（2.76 GiB）挪 CPU：`63\|25\|55\|56\|57\|58\|59\|60\|61\|62\|24\|26\|50\|52\|53\|54`；KV Q4_0；FA on；**MTP off**（8.6 < 13.2 tok/s，负优化） |
| [7269827-rgb/qwen38-256k-on-16gb](https://github.com/7269827-rgb/qwen38-256k-on-16gb) | RX 9070 XT 16 GB | **256K ctx**（llama.cpp Vulkan） | 量化 KV + 磁盘配合；证明 16 GB 摸到长上下文门槛 |
| [adrienbrault/qwen3.8-27b-rtx5090](https://github.com/adrienbrault/qwen3.8-27b-rtx5090) | RTX 5090 | NVFP4 + vLLM，~300 tok/s，262K ctx | 天花板参考，非本机可达 |

### 3.2 量化生态（unsloth/Qwen3.8-27B-GGUF 为例）

| 量化 | 大小 | 16 GB 卡可行性 |
|---|---:|---|
| UD-IQ4_XS | 14.3 GB | 全进显存，留 ~1.7 GB 给 KV/开销，4K-8K ctx 起步 |
| UD-Q4_K_M | 16.5 GB | 需选择性 FFN 卸载 ~2.8 GiB（5080 实测方案） |
| UD-Q3_K_XL | 13.1 GB | 更宽松，精度换空间备选 |
| UD-Q2_K_XL / IQ2 | 8-10 GB | 精度损失大，仅作极端对照 |
| Q8_0 / BF16 | 29 / 54.7 GB | 超纲，不在 16 GB 讨论 |
| mmproj (vision) | 另计 | 纯文本跑法不加载，省 0.85 GiB（CLI vision 估算） |

### 3.3 消费级应用与极客方向

- **Ollama**：Qwen3.8 支持尚未稳（下载损坏 issue、Windows 支持请求仍开着）。观望。
- **LM Studio / MLX**：MTP draft-only 仓库会被误索引致崩溃；MLX 忽略内建 MTP 头。可用但别开 MTP。
- **Intel Arc**：vLLM + KVarN 4-bit/2-bit KV 量化冲 550K ctx（da3dsoul/Qwen3.8-vLLM-KVarN-MTP-Arc-Experiments）。
- **ROCm/RDNA4**：shape 特化 kernel（Dyluhn/R9V）；Tenstorrent Blackhole 移植中。
- **DGX Spark 阵营**：多个个人项目用 vLLM + NVMe streaming/n-gram 跑 176B-Flash-Next。

## 4. 回家实测作战方案（RTX 5060 Ti 16 GB）

1. 环境：llama.cpp 官方 CUDA 构建（最新版，CUDA 12.8+，驱动装最新）。
2. 实验组 A：UD-IQ4_XS 全进显存，ctx 4K/8K，KV fp16 → 记 tok/s、显存峰值。
3. 实验组 B：UD-Q4_K_M + FFN 卸载（抄 §3.1 profile，5060 Ti 带宽减半预计多卸 2-4 个 tensor）。
4. 实验组 C：B 基础上开 `-ctk q4_0 -ctv q4_0`，ctx 拉到 32K/64K。
5. 全程 **MTP off**（§2/§3.1 双重证据：sm_120 负优化）。
6. 记录字段：GPU/驱动/CUDA 版本、llama.cpp commit、量化文件 SHA、ctx、TTFT、tok/s、VRAM 峰值。
7. 预期：5060 Ti 带宽 448 GB/s（5080 的一半），长上下文 decode 预估 6-9 tok/s，
   短上下文更好——**预测 vs 实测偏差本身就是 benchmark 报告第一章**。
8. 正确性对拍：同一批 prompt 用 Transformers greedy 跑基准 token，与 llama.cpp 输出比对首差异位置。

## 5. 证据边界

- §2 框架坑均注明 issue 号，时效以调研日为准。
- §3.1 民间数据为个人实测，环境各异，只能当参考点不能当结论。
- 本文所有显存数字可与本项目 CLI 输出交叉验证：`cargo run -- plan`。
- 速度预估未经本机验证，属推测，标注为推测。
