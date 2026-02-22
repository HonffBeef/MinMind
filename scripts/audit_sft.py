import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import re
import argparse
import statistics
import torch

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset          # ✅ 很多审计逻辑也会用到它
from trainer.trainer_utils import init_model


CODE_PATTERNS = [
    r"```", r"\bdef\s+\w+\(", r"\bclass\s+\w+\(", r"\bimport\s+\w+", r"\bfrom\s+\w+\s+import\b",
    r"public\s+static\s+void\s+main",  # Java
    r"#include\s+<",                   # C/C++
]


def has_code(text: str) -> bool:
    t = text.lower()
    for p in CODE_PATTERNS:
        if re.search(p, t):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=str, default="full_sft")
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data_path", type=str, default=r"dataset/sft_mini_512.jsonl")
    ap.add_argument("--max_len", type=int, default=340)
    ap.add_argument("--scan_samples", type=int, default=20000)  # 扫 2w 条基本够看趋势
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # load tokenizer
    lm_config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
    _, tokenizer = init_model(lm_config, args.weight, device=args.device)

    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_len)

    n = min(len(ds), args.scan_samples)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    labeled_counts = []
    pad_counts = []
    full_len = 0
    near_full = 0
    code_cnt = 0
    python_cnt = 0

    for i in range(n):
        # 取原始文本判断 code 覆盖
        sample = ds.samples[i]
        cs = sample.get("conversations", [])
        all_text = "\n".join([x.get("content", "") for x in cs])

        if has_code(all_text):
            code_cnt += 1
        if "python" in all_text.lower():
            python_cnt += 1

        input_ids, labels = ds[i]
        pad_c = int((input_ids == pad_id).sum().item())
        lab_c = int((labels != -100).sum().item())

        pad_counts.append(pad_c)
        labeled_counts.append(lab_c)

        if pad_c == 0:
            full_len += 1
        if pad_c <= 5:  # 几乎装满
            near_full += 1

    print(f"\n[Audit] data={args.data_path} max_len={args.max_len} scanned={n}/{len(ds)}")
    print(f"[Audit] code-like samples: {code_cnt} ({code_cnt/n*100:.2f}%)")
    print(f"[Audit] mention 'python': {python_cnt} ({python_cnt/n*100:.2f}%)")

    print(f"[Audit] pad_count mean={statistics.mean(pad_counts):.1f} median={statistics.median(pad_counts):.1f} min={min(pad_counts)} max={max(pad_counts)}")
    print(f"[Audit] sequences fully packed (pad=0): {full_len} ({full_len/n*100:.2f}%)")
    print(f"[Audit] sequences near full (pad<=5): {near_full} ({near_full/n*100:.2f}%)")

    print(f"[Audit] labeled_tokens mean={statistics.mean(labeled_counts):.1f} median={statistics.median(labeled_counts):.1f} min={min(labeled_counts)} max={max(labeled_counts)}")

    # 简单判断建议
    print("\n[Suggestion]")
    if near_full / n > 0.20:
        print("⚠️  截断比例看起来不低（pad<=5 超过 20%），建议把 SFT max_seq_len 提到 512 或 768。")
    else:
        print("✅  截断比例不算高，max_seq_len=340 可能还能接受（但代码任务仍可能受限）。")

    if code_cnt / n < 0.01:
        print("⚠️  代码样本占比 <1%，模型不会写代码很正常：建议补充代码类 SFT 数据。")
    elif code_cnt / n < 0.05:
        print("⚠️  代码样本占比偏低（<5%）：建议补充一些代码类 SFT 数据并用更长 seq 训一轮。")
    else:
        print("✅  代码样本占比不算太低：不会写代码更可能是 seq_len 截断或训练策略/LR 问题。")


if __name__ == "__main__":
    main()