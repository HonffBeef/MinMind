import sys
from pathlib import Path

# ✅ 让脚本能 import 到项目根目录下的 model/trainer/dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../minimind
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import statistics
import torch

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset          # ✅ 关键：缺了这行就会 NameError
from trainer.trainer_utils import init_model

def find_subseq_positions(seq, pattern):
    if not pattern:
        return []
    out = []
    for i in range(0, len(seq) - len(pattern) + 1):
        if seq[i:i + len(pattern)] == pattern:
            out.append(i)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="full_sft")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_path", type=str, default="dataset/sft_mini_512.jsonl")
    parser.add_argument("--max_len", type=int, default=340)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])

    parser.add_argument("--scan_samples", type=int, default=2000, help="统计多少条样本（越大越准）")
    parser.add_argument("--show_index", type=int, default=0, help="打印哪条样本的细节")
    parser.add_argument("--show_tokens", type=int, default=220, help="打印前多少个 token 对齐信息")

    args = parser.parse_args()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model, tokenizer = init_model(lm_config, args.weight, device=args.device)
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_len)

    # ===== 1) 全局统计：每条样本有多少 token 被训练（labels != -100）=====
    n = min(len(ds), args.scan_samples)
    counts = []
    zeros = 0
    for i in range(n):
        _, labels = ds[i]
        c = int((labels != -100).sum().item())
        counts.append(c)
        if c == 0:
            zeros += 1

    print(f"\n[Stats] dataset={args.data_path}, max_len={args.max_len}, scanned={n}/{len(ds)}")
    print(f"[Stats] labeled_tokens: mean={statistics.mean(counts):.1f}, "
          f"median={statistics.median(counts):.1f}, min={min(counts)}, max={max(counts)}")
    print(f"[Stats] zero_labeled_samples: {zeros} ({zeros / n * 100:.2f}%)")

    # ===== 2) 打印一条样本细节：prompt、bos/eos marker、labels 对齐 =====
    idx = args.show_index % len(ds)
    sample = ds.samples[idx]
    prompt = ds.create_chat_prompt(sample["conversations"])

    input_ids, labels = ds[idx]
    input_list = input_ids.tolist()
    label_list = labels.tolist()

    print("\n" + "=" * 80)
    print(f"[Sample {idx}] raw prompt (first 800 chars):")
    print(prompt[:800])

    print("\n[Marker Tokens]")
    print("bos_id:", ds.bos_id, "decoded:", repr(tokenizer.decode(ds.bos_id, skip_special_tokens=False)))
    print("eos_id:", ds.eos_id, "decoded:", repr(tokenizer.decode(ds.eos_id, skip_special_tokens=False)))

    bos_pos = find_subseq_positions(input_list, ds.bos_id)
    eos_pos = find_subseq_positions(input_list, ds.eos_id)
    print("\n[Marker Positions]")
    print("bos positions:", bos_pos[:10], ("..." if len(bos_pos) > 10 else ""))
    print("eos positions:", eos_pos[:10], ("..." if len(eos_pos) > 10 else ""))

    labeled_pos = [i for i, y in enumerate(label_list) if y != -100]
    print(f"\n[Labeled Tokens] count={len(labeled_pos)}")
    if labeled_pos:
        print(f"first labeled idx={labeled_pos[0]}, last labeled idx={labeled_pos[-1]}")

    print("\n[Token-level view] (L = participates in loss)")
    show_n = min(len(input_list), args.show_tokens)
    for i in range(show_n):
        tok = tokenizer.decode([input_list[i]], skip_special_tokens=False)
        mark = "L" if label_list[i] != -100 else " "
        print(f"{i:4d} {mark} id={input_list[i]:5d} tok={tok!r}")

    print("\n[Conclusion]")
    if zeros / n > 0.05:
        print("⚠️  有较多样本 labels 全为 -100：很可能 bos_id/eos_id 与 chat_template 不匹配，需要修。")
    else:
        print("✅  labels 覆盖看起来正常：SFT 至少在训练 assistant 段落。")


if __name__ == "__main__":
    main()