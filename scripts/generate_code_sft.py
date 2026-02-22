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

SYSTEM = "你是严谨的编程助手。请直接输出可运行的 Python 代码块，包含函数定义与简单测试。不要输出解释文字。"

def wrap(code: str) -> str:
    return "```python\n" + code.strip() + "\n```"

TASKS = []

# 1) Fibonacci
TASKS.append((
    [
        "请用Python写一个计算斐波那契数列第n项的函数，支持n=0,n=1，时间复杂度O(n)，并给出测试用例。",
        "写一个Python函数fib(n)返回第n个斐波那契数（n>=0），用迭代实现，并给出简单测试。",
        "用Python实现斐波那契：输入n输出F(n)，要求处理边界n=0/1，并给出2个测试。"
    ],
    wrap(r"""
from typing import *

def fib(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

if __name__ == "__main__":
    print(fib(0))   # 0
    print(fib(10))  # 55
""")
))

# 2) Factorial
TASKS.append((
    [
        "请用Python写一个计算阶乘的函数factorial(n)，要求n>=0，迭代实现，并给出测试。",
        "实现factorial(n)：返回n!，处理n=0，给出2个测试用例。",
    ],
    wrap(r"""
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    ans = 1
    for i in range(2, n + 1):
        ans *= i
    return ans

if __name__ == "__main__":
    print(factorial(0))  # 1
    print(factorial(5))  # 120
""")
))

# 3) GCD
TASKS.append((
    [
        "写一个Python函数gcd(a,b)计算最大公约数（欧几里得算法），并给出测试。",
        "请实现gcd(a, b)返回最大公约数，要求支持负数输入，并测试。"
    ],
    wrap(r"""
def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b != 0:
        a, b = b, a % b
    return a

if __name__ == "__main__":
    print(gcd(54, 24))   # 6
    print(gcd(-12, 18))  # 6
""")
))

# 4) Prime check
TASKS.append((
    [
        "写一个Python函数is_prime(n)判断n是否为质数，n<=1返回False，并给出测试。",
        "实现质数判断：is_prime(n)，要求效率O(sqrt(n))，并给出测试用例。"
    ],
    wrap(r"""
import math

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    print(is_prime(1))   # False
    print(is_prime(2))   # True
    print(is_prime(49))  # False
""")
))

# 5) Two sum
TASKS.append((
    [
        "请用Python实现two_sum(nums, target)返回两数之和的下标（任意一组即可），并给出测试。",
        "写函数two_sum：给定数组和target，返回一对下标使得nums[i]+nums[j]=target，用哈希表实现，并测试。"
    ],
    wrap(r"""
from typing import List, Tuple

def two_sum(nums: List[int], target: int) -> Tuple[int, int]:
    seen = {}
    for i, x in enumerate(nums):
        y = target - x
        if y in seen:
            return (seen[y], i)
        seen[x] = i
    raise ValueError("no solution")

if __name__ == "__main__":
    print(two_sum([2, 7, 11, 15], 9))  # (0, 1)
""")
))

# 6) Binary search
TASKS.append((
    [
        "写一个Python函数binary_search(nums, x)在有序数组中查找x，找到返回下标，否则返回-1，并测试。",
        "实现二分查找：binary_search，输入升序列表和目标值，返回索引或-1，并给出测试。"
    ],
    wrap(r"""
from typing import List

def binary_search(nums: List[int], x: int) -> int:
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == x:
            return mid
        if nums[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

if __name__ == "__main__":
    print(binary_search([1, 3, 5, 7, 9], 7))  # 3
    print(binary_search([1, 3, 5, 7, 9], 2))  # -1
""")
))

# 7) Dedup keep order
TASKS.append((
    [
        "写一个Python函数dedup_keep_order(lst)对列表去重但保留原顺序，并测试。",
        "实现列表去重：保留第一次出现的顺序，写函数并给出测试用例。"
    ],
    wrap(r"""
from typing import List, TypeVar

T = TypeVar("T")

def dedup_keep_order(lst: List[T]) -> List[T]:
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

if __name__ == "__main__":
    print(dedup_keep_order([1, 2, 1, 3, 2, 4]))  # [1,2,3,4]
""")
))

# 8) Word frequency
TASKS.append((
    [
        "请用Python写一个函数word_freq(text)统计文本中每个单词出现次数（忽略大小写），并测试。",
        "实现词频统计：输入字符串，按空格切分，忽略大小写，返回dict，并给出测试。"
    ],
    wrap(r"""
from typing import Dict

def word_freq(text: str) -> Dict[str, int]:
    freq = {}
    for w in text.lower().split():
        freq[w] = freq.get(w, 0) + 1
    return freq

if __name__ == "__main__":
    print(word_freq("Hello hello world"))  # {'hello':2,'world':1}
""")
))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            prompts, answer = random.choice(TASKS)
            prompt = random.choice(prompts)
            item = {
                "conversations": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {args.n} code SFT samples -> {args.out}")

if __name__ == "__main__":
    main()