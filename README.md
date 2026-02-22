# MiniMind — 小型 LLM 训练复现与调优（Pretrain + SFT + 数据诊断 + 代码能力对齐）

作者：张亦朕

本仓库记录我在 **MiniMind** 项目上的个人复现与改进：跑通 **Pretrain → Full SFT → Eval** 全流程，并围绕 **SFT label mask 正确性**、**数据分布（截断/代码占比）**、**推理解码策略（greedy vs sampling）** 做了系统诊断、对照实验与迭代训练。最终在代码任务上验证：同一模型使用 **greedy（do_sample=0）** 可稳定输出可运行 Python 代码（以 Fibonacci 为例）。

> 面向目标：把项目做成“可复现、可解释、可量化”的作品集，用于大模型实习面试展示。

---

## 目录
- [1. 项目亮点](#1-项目亮点)
- [2. 仓库结构](#2-仓库结构)
- [3. 环境安装（一遍成功）](#3-环境安装一遍成功)
- [4. 快速验证（1 分钟）](#4-快速验证1-分钟)
- [5. 训练流程（Pretrain / Full SFT）](#5-训练流程pretrain--full-sft)
- [6. 关键诊断：SFT mask 检查](#6-关键诊断sft-mask-检查)
- [7. 关键诊断：数据审计（截断/代码占比）](#7-关键诊断数据审计截断代码占比)
- [8. 数据构建：合成代码 SFT + 过采样混合](#8-数据构建合成代码-sft--过采样混合)
- [9. 评测与推理（Chat vs Code 解码策略）](#9-评测与推理chat-vs-code-解码策略)
- [10. 数据与权重（GitHub 上传建议）](#10-数据与权重github-上传建议)
- [11. 常见问题](#11-常见问题)
- [12. License](#12-license)

---

## 1. 项目亮点
- ✅ **训练闭环**：Pretrain + Full SFT（全参）+ Eval（推理生成），训练脚本参数可控（LR/seq_len/batch/accum 等）  
- ✅ **SFT 监督信号验证**：用 `scripts/check_sft_mask.py` 统计 `zero_labeled_samples≈0%`，确认只训练 assistant 段（排除“labels 全 -100 导致训练无效”）  
- ✅ **数据驱动定位根因**：用 `scripts/audit_sft.py` 审计发现：代码样本占比过低、SFT 截断比例偏高（max_seq_len 太短）  
- ✅ **迭代训练**：提高 `max_seq_len`（如 340→512），并通过合成/过采样补足代码类 SFT 数据  
- ✅ **推理侧对照实验（关键）**：代码任务下 sampling 容易碎片化，greedy（do_sample=0）更稳定，能输出可运行代码

---

## 2. 仓库结构

推荐结构（本仓库已按此整理）：

```
minimind/
  model/                 # 模型定义
  trainer/               # 训练脚本
  dataset/
    sample/              # 小样本数据（演示/跑通用）
  scripts/               # 评测/诊断/数据构建脚本（本项目新增/整理）
  docs/                  # 实验记录（可选）
  reports/               # 报告（可选）
  README.md
  .gitignore
  LICENSE
  environment.yml
  requirements.txt
  requirements-torch-cu126.txt
  requirements-torch-cpu.txt
  requirements.lock.txt  # 可选：你的本机 freeze 备份
```

---

## 3. 环境安装（一遍成功）

> 关键点：**PyTorch（torch/torchvision/torchaudio）单独安装**，因为不同机器（CPU/CUDA 版本）需要不同的 wheel 源与组合。  
> 本仓库提供两份文件：
> - `requirements-torch-cu126.txt`：CUDA 12.6
> - `requirements-torch-cpu.txt`：CPU 版（通用）

### 3.1 创建 conda 环境
```bash
conda env create -f environment.yml
conda activate minimind
```

### 3.2 安装 PyTorch（选择其一）
如果你有 NVIDIA GPU 且 CUDA 版本匹配（cu126）：
```bash
pip install -r requirements-torch-cu126.txt
```

如果你只想 CPU 运行（更通用）：
```bash
pip install -r requirements-torch-cpu.txt
```

### 3.3 安装其余依赖
```bash
pip install -r requirements.txt
```

---

## 4. 快速验证（1 分钟）

### 4.1 验证基础依赖
```bash
python -c "import torch, transformers; print('ok', torch.__version__, transformers.__version__)"
```

### 4.2 验证代码生成（推荐 greedy）
> 建议在 `trainer/` 目录运行，避免相对路径问题。Windows 下可直接复制。

**Windows（cmd）**
```bat
cd trainer
python ..\scripts\eval_sampling.py --weight full_sft_seq512_code --preset code --mode single --do_sample 0 --max_new_tokens 220 --system "你是严谨的编程助手。请只输出一个 Python 代码块，不要解释，不要反问。" --query "写一个Python函数fib(n)返回第n个斐波那契数（n>=0），用迭代实现，并给出简单测试。"
```

**macOS / Linux（bash）**
```bash
cd trainer
python ../scripts/eval_sampling.py --weight full_sft_seq512_code --preset code --mode single --do_sample 0 --max_new_tokens 220 \
  --system "你是严谨的编程助手。请只输出一个 Python 代码块，不要解释，不要反问。" \
  --query "写一个Python函数fib(n)返回第n个斐波那契数（n>=0），用迭代实现，并给出简单测试。"
```

---

## 5. 训练流程（Pretrain / Full SFT）

> **强烈建议从 `trainer/` 目录运行训练脚本**，避免 tokenizer / 权重路径解析成错误的 HuggingFace repo id（如 `../model`）。

### 5.1 Pretrain（示例）
**Windows（cmd）**
```bat
cd trainer
python train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl --save_weight pretrain --epochs 1 --batch_size 32 --learning_rate 5e-4 --max_seq_len 340
```

**bash**
```bash
cd trainer
python train_pretrain.py --data_path ../dataset/pretrain_hq.jsonl --save_weight pretrain --epochs 1 --batch_size 32 --learning_rate 5e-4 --max_seq_len 340
```

### 5.2 Full SFT（基础版：seq 340）
```bash
cd trainer
python train_full_sft.py --data_path ../dataset/sft_mini_512.jsonl --from_weight pretrain --save_weight full_sft --epochs 1 --learning_rate 1e-6 --max_seq_len 340
```

### 5.3 继续训练（示例：seq 512 + 代码混合）
```bash
cd trainer
python train_full_sft.py --from_weight full_sft --save_weight full_sft_seq512_code --epochs 1 --learning_rate 5e-6 --batch_size 8 --accumulation_steps 2 --max_seq_len 512 --data_path ../dataset/sft_mix_75k.jsonl
```

---

## 6. 关键诊断：SFT mask 检查

验证 SFT labels 是否只覆盖 assistant 段（排除“训练没学到”的常见坑）：

```bash
cd trainer
python ../scripts/check_sft_mask.py --weight full_sft --data_path ../dataset/sft_mini_512.jsonl --max_len 340 --scan_samples 2000 --show_index 0
```

关注输出：
- `zero_labeled_samples` 是否接近 0%
- token-level view 中 assistant 段是否出现大量 `L` 标记

---

## 7. 关键诊断：数据审计（截断/代码占比）

审计 SFT 数据分布，定位“不会写代码/容易截断/复读”的数据侧根因：

```bash
cd trainer
python ../scripts/audit_sft.py --weight full_sft --data_path ../dataset/sft_mini_512.jsonl --max_len 340 --scan_samples 20000
```

常见结论与对策：
- code-like 占比极低 → 补充代码 SFT 数据并过采样
- pad<=5 比例高（接近满长） → 提升 `max_seq_len`（如 512/768）减少截断

---

## 8. 数据构建：合成代码 SFT + 过采样混合

### 8.1 生成合成代码 SFT（示例：5000 条）
```bash
cd trainer
python ../scripts/generate_code_sft.py --out ../dataset/sft_code_5k.jsonl --n 5000 --seed 42
```

### 8.2 混合数据（示例：抽 50k 通用 + 代码 5k×5 过采样 → 75k）
```bash
cd trainer
python ../scripts/build_sft_mix.py --base_path ../dataset/sft_mini_512.jsonl --base_n 50000 --code_path ../dataset/sft_code_5k.jsonl --code_repeat 5 --out_path ../dataset/sft_mix_75k.jsonl --seed 42
```

---

## 9. 评测与推理（Chat vs Code 解码策略）

### 9.1 Chat（sampling 更自然）
```bash
cd trainer
python ../scripts/eval_sampling.py --weight full_sft_seq512_code --preset chat --mode auto --temperature 0.5 --top_p 0.95 --repetition_penalty 1.08 --no_repeat_ngram_size 0 --max_new_tokens 256
```

### 9.2 Code（✅推荐 greedy 更稳定）
```bash
cd trainer
python ../scripts/eval_sampling.py --weight full_sft_seq512_code --preset code --mode single --do_sample 0 --max_new_tokens 220 --system "你是严谨的编程助手。请只输出一个 Python 代码块，不要解释，不要反问。" --query "写一个Python函数fib(n)返回第n个斐波那契数（n>=0），用迭代实现，并给出简单测试。"
```

---

## 10. 数据与权重（GitHub 上传建议）

- ✅ 建议上传：代码、脚本、文档、小样本数据（`dataset/sample/`）  
- ❌ 不建议上传：全量大数据（如 121 万条 jsonl）、训练权重（`*.pth/*.safetensors`）、训练输出目录（`out/`、`checkpoints/`）  
- 推荐做法：
  - 权重上传 HuggingFace Hub / 网盘 / GitHub Release（小模型）
  - 数据只上传处理脚本和小样本，避免体积与版权风险

---

## 11. 常见问题

### 11.1 `ModuleNotFoundError: No module named 'model'`
已通过在 `scripts/*.py` 中加入项目根目录到 `sys.path` 修复；仍建议从 `trainer/` 目录运行命令，减少相对路径坑。

### 11.2 `pynvml package is deprecated`
这是 torch 在某些环境下的提示。想“更清爽”可以：
```bash
pip uninstall -y pynvml
pip install -U nvidia-ml-py
```

### 11.3 `pip install -r requirements.txt` 报 torch 版本找不到
仓库已将 torch 拆分到 `requirements-torch-*.txt`，请按 [环境安装](#3-环境安装一遍成功) 先装 torch，再装其它依赖。

---

## 12. License
MIT License（若你基于原项目二次开发，请同时遵守原项目 License）。
