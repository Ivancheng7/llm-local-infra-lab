# Handoff：llm-local-infra-lab 项目状态

> 最后更新：2026-09-04
> 仓库：`C:\Users\27236\Desktop\llm-local-infra-lab`
> 远端：https://github.com/Ivancheng7/llm-local-infra-lab（main 分支，CI 绿灯）
>
> 注：本文件是随 git 走的项目级 handoff。工作区还有多线程 handoff 体系
> （`C:\Users\27236\handoffs\`，含 INDEX.md 总览和各主题线程文件），
> 两者内容保持同步；恢复本主题前读其一即可。

## 项目背景（一句话）

用户是**纯小白**（自述“啥都不知道”），正在做一个 AI Infra 简历项目：不下载权重本体，
只解析 Qwen3.8-27B 的 `config.json` + `model.safetensors.index.json`（110 KB），
用 Rust CLI 生成精度感知的显存规划报告。这是既定路线的 **Phase 1**（metadata planner），
Phase 2 是回家用 16GB 卡实测 benchmark，Phase 3 是 MoE expert-cache 分析。

起因是用户在 localMoE 学习仓库（`C:\Users\27236\Desktop\localmoe`）学 AI Infra，
前一轮会话决定不 fork localMoE，另建独立仓库。用户偏好的称呼/语气：随意、
中文、大白话解释概念（用户明确说过“纯小白”，解释时要用类比，别堆术语）。

## 当前状态（全部已完成，无未竟事项）

### 代码（5 个 commit，最后一个 `34e53cb`）

- **Rust CLI**（`src/`）：`inspect-config` / `inspect-index` / `plan` 三个子命令；
  plan 支持 `--precision bf16,fp16,int8,int4`、`--format markdown|json`、
  `--output <file>`（自动建目录）。
- **模块**：`model_config.rs`（含 linear-attention 头几何 + vision config 解析）、
  `safetensors_index.rs`（兼容 HF 把 total_size 写成浮点的坑）、
  `tensor_classifier.rs`（按名字分类）、`memory_planner.rs`（精度缩放、KV 估算、
  **几何参数建模**）、`report.rs`（render_* 返回 String）。
- **质量门**：5 个测试 / clippy -D warnings / fmt 全绿；GitHub Actions CI 已跑绿。

### 关键数字（已验证，别再重算）

- 1199 tensors / 18 shards / total 55,562,855,904 bytes ≈ 51.75 GiB BF16 / ~27.78B params
- 64 层 = 48 linear attention（gated delta net，qkvz+ba 融合投影）+ 16 full attention（间隔 4）
- 几何模型解释 27.24B，**residual 0.54B ≈ 2%**（从 6.95B 缩下来的，这是本次会话的增强成果）；
  有回归测试断言 gap < 5%
- MLP 是大头：17.11B（31.88 GiB）；vision 0.46B；MTP 0.39B
- KV cache：2,048 bytes/token/layer × 16 full layers（BF16）→ 4K ctx 0.25 GiB / 262K ctx 16 GiB

### 文档

- `README.md`：命令用法、数据获取（hf-mirror.com，直连 HF 会超时）、证据边界
- `docs/03-inference-support-matrix.md`：**Phase 2 作战地图**（调研于 2026-09-04，有失效风险）——
  官方框架矩阵（llama.cpp/SGLang/vLLM/Transformers/TokenSpeed + 具体 issue 编号）、
  民间 16GB 卡实测仓库、量化选择表、回家实测 8 条清单
- `metadata/Qwen3.8-27B/`：两份官方 metadata 已下载入库
- `artifacts/`（不进 git）：`qwen3.8-27b-plan.md` 报告样例

## 用户硬件与网络环境

- 家里：**RTX 5060 Ti 16GB（sm_120，Blackwell）**——用户在公司，**还没实测过**
- 公司/当前：CPU only
- 网络：直连 HF/GitHub 会超时；本机 **127.0.0.1:7890 有代理**（Clash）。
  git push 用 `git -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 push`；
  HF 下载走 `https://hf-mirror.com`。**没改全局 git 代理配置**（用户没要求）。
- `gh` CLI 已登录（Ivancheng7，有 repo+workflow 权限）；cargo/rustc 1.96.0

## 教学线（会话中给用户讲过的概念）

用户已能复述/理解的：Phase 1 = 只读“说明书”做静态分析，不推理不下载权重；
MLP 占 60% 是因为 3×5120×17408×64；KV cache 是“水电费”（随上下文涨），权重是“房租”（固定）；
48 线性层用固定大小状态换掉了无限增长的 KV cache。
用户还问过“要不要自己写推理引擎”——答复：Phase 3 再说，先从小模型练手。

## 下一步（按优先级）

1. **回家实测（Phase 2 开工）**：按 `docs/03-inference-support-matrix.md` 第 4 节执行——
   llama.cpp + UD-IQ4_XS（14.3GB 全进显存）或 UD-Q4_K_M（16.5GB + 选择性 FFN 卸载，
   抄 5080 那位的 profile）；MTP 先关（sm_120 负优化实锤）；KV 量化 Q4_0 拉长上下文。
   **预设速度预测：6-9 tok/s 长上下文（写明是推测）——等实测来验证/打脸**。
2. 实测数据回来后：做“预测 vs 实测”偏差分析（benchmark 报告第一章素材）；
   Transformers greedy 对拍验正确性（注意 vLLM 那个 KV scale bug 类的静默错误）。
3. 可选 Phase 1 余温：vision/MTP 字节更精细建模、benchmark 记录模板脚本。

## 环境备忘

- Shell 是 Git Bash（Windows）；`cargo init` 默认分支是 master，已改名 main
- CRLF 警告是正常的（core.autocrlf），提交时用 `-c core.safecrlf=false` 可静音
- localMoE 仓库（`C:\Users\27236\Desktop\localmoe`）是教学材料，**别动它的工作树**
