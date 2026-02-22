# eval_sampling.py
import sys
from pathlib import Path

# ✅ 关键：把项目根目录加入 sys.path（必须在 import model/trainer/dataset 之前）
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../minimind
sys.path.insert(0, str(PROJECT_ROOT))

# 下面再 import 你的项目模块
import os
import time
import argparse
from dataclasses import dataclass

import torch

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import init_model, setup_seed

@dataclass
class GenCfg:
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.10
    no_repeat_ngram_size: int = 0  # 0 表示关闭；想强力防复读可设 3~6


def build_prompt(tokenizer, user_text: str, system_text: str = "你是一个乐于助人的助手。") -> str:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    # add_generation_prompt=True: 在末尾加上 assistant 开始标记，让模型续写
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, device: str, cfg: GenCfg):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 计时更准（CUDA 同步）
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()

    gen_kwargs = dict(
    input_ids=input_ids,
    max_new_tokens=cfg.max_new_tokens,
    do_sample=cfg.do_sample,
    repetition_penalty=cfg.repetition_penalty,
    no_repeat_ngram_size=cfg.no_repeat_ngram_size,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=pad_id,
    use_cache=True,
    )

    # ✅ 只有 sampling 才需要这些
    if cfg.do_sample:
        gen_kwargs.update(dict(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        ))

    out = model.generate(**gen_kwargs)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.time()

    gen_ids = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    speed = gen_ids.numel() / max(t1 - t0, 1e-9)
    return text.strip(), speed


def preset_to_cfg(preset: str) -> GenCfg:
    preset = preset.lower().strip()
    if preset == "chat":
        return GenCfg(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.10,
            no_repeat_ngram_size=0,
        )
    if preset == "strong_anti_repeat":
        return GenCfg(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
        )
    if preset == "code":
        # 代码任务一般更适合“更低温度 + 更高top_p + 较轻重复惩罚”
        return GenCfg(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            no_repeat_ngram_size=0,
        )
    raise ValueError(f"Unknown preset: {preset}. Use chat / strong_anti_repeat / code")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="full_sft", help="权重名（和你 eval_llm.py 用法一致）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--preset", type=str, default="chat", help="chat / strong_anti_repeat / code")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "interactive", "single"])
    parser.add_argument("--query", type=str, default="你有什么特长？")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system", type=str, default="你是一个乐于助人的助手。")
    parser.add_argument("--do_sample", type=int, default=None, choices=[0, 1], help="0=greedy, 1=sampling")

    # 模型结构（按你当前 25.83M 的默认：hidden=512, layers=8）
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])

    # 允许 CLI 覆盖部分生成参数
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)

    
    args = parser.parse_args()

    setup_seed(args.seed)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    model, tokenizer = init_model(lm_config, args.weight, device=args.device)
    model.eval()

    # dtype（推理）
    if args.device.startswith("cuda"):
        if args.dtype == "float16":
            model = model.half()
        elif args.dtype == "bfloat16":
            model = model.to(dtype=torch.bfloat16)
        # float32 就不动

    cfg = preset_to_cfg(args.preset)

# ✅ 这里才可以用 args
    if args.do_sample is not None:
        cfg.do_sample = bool(args.do_sample)
    # CLI 覆盖
    for k in ["max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "no_repeat_ngram_size"]:
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    auto_tests = [
        "你有什么特长？",
        "为什么天空是蓝色的？",
        "请用Python写一个计算斐波那契数列的函数，直接给出代码。",
        "解释一下“光合作用”的基本过程。",
        "如果明天下雨，我应该如何出门？请给出简洁的要点。",
        "比较一下猫和狗作为宠物的优缺点（用条目列出，避免重复）。",
        "解释什么是机器学习，用通俗语言。",
        "推荐一些中国的美食，尽量多样化，不要重复。",
    ]

    def run_one(q: str):
        prompt = build_prompt(tokenizer, q, system_text=args.system)
        ans, speed = generate_one(model, tokenizer, prompt, args.device, cfg)
        print(f"\n💬: {q}\n🤖: {ans}\n[Speed]: {speed:.2f} tokens/s")

    if args.mode == "single":
        run_one(args.query)
        return

    if args.mode == "auto":
        for q in auto_tests:
            run_one(q)
        return

    # interactive
    while True:
        q = input("\n💬(输入 'exit' 退出): ").strip()
        if not q or q.lower() == "exit":
            break
        run_one(q)


if __name__ == "__main__":
    main()