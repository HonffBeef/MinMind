import sys

DROP_CONTAINS = [
    " @ file:///",  # conda/pip freeze 常见
    " @ file:",     # 有些环境是这种格式
    "@ file:///",
    "@ file:",
]

DROP_PREFIX = (
    "pip==",
    "setuptools==",
    "wheel==",
)

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "requirements.cleaned.txt"

    kept, dropped = [], []
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            if raw.startswith(DROP_PREFIX):
                dropped.append(raw)
                continue

            if any(s in raw for s in DROP_CONTAINS):
                dropped.append(raw)
                continue

            kept.append(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        for x in kept:
            f.write(x + "\n")

    print(f"[clean_requirements] input:  {in_path}")
    print(f"[clean_requirements] output: {out_path}")
    print(f"[clean_requirements] kept={len(kept)}  dropped={len(dropped)}")
    if dropped:
        print("[clean_requirements] example dropped lines:")
        for x in dropped[:5]:
            print("  -", x)

if __name__ == "__main__":
    main()