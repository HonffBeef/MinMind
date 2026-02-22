# build_sft_mix.py
import argparse
import random

def reservoir_sample_jsonl(in_path: str, k: int, seed: int):
    random.seed(seed)
    reservoir = []
    seen = 0
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(line)
            else:
                j = random.randint(1, seen)
                if j <= k:
                    reservoir[j - 1] = line
    return reservoir, seen

def read_all_lines(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", required=True, help="原始大SFT jsonl")
    ap.add_argument("--base_n", type=int, default=50000, help="从base里抽多少条")
    ap.add_argument("--code_path", required=True, help="代码SFT jsonl")
    ap.add_argument("--code_repeat", type=int, default=5, help="代码数据重复次数（过采样）")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_lines, seen = reservoir_sample_jsonl(args.base_path, args.base_n, args.seed)
    code_lines = read_all_lines(args.code_path)

    mixed = []
    mixed.extend(base_lines)
    mixed.extend(code_lines * args.code_repeat)

    random.seed(args.seed)
    random.shuffle(mixed)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for line in mixed:
            f.write(line + "\n")

    print(f"Base: sampled {len(base_lines)} from {seen}")
    print(f"Code: {len(code_lines)} * {args.code_repeat} = {len(code_lines)*args.code_repeat}")
    print(f"Total mixed: {len(mixed)} -> {args.out_path}")

if __name__ == "__main__":
    main()