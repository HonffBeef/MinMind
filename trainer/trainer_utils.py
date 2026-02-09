import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


# =========================
# 通用工具
# =========================

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def Logger(content: str):
    if is_main_process():
        print(content)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    # 防止 total_steps=0
    total_steps = max(int(total_steps), 1)
    current_step = min(max(int(current_step), 0), total_steps)
    # 余弦退火 + 最小 lr = lr/10
    return lr / 10.0 + 0.5 * lr * (1.0 + math.cos(math.pi * current_step / total_steps))


# =========================
# 分布式初始化（Windows 兼容）
# =========================

def init_distributed_mode():
    """
    仅当通过 torchrun / mpirun 等方式启动并设置了环境变量 RANK/WORLD_SIZE/LOCAL_RANK 时进入 DDP。
    Windows 下 NCCL 不可用，自动切到 gloo。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Windows: gloo；Linux: 优先 nccl（有 CUDA 时）
    if os.name == "nt":
        backend = "gloo"
    else:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Checkpoint
# =========================

def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="checkpoints",
    **kwargs,
):
    """
    保存模式：传入 model!=None
      - 保存 half 权重到 {save_dir}/{weight}_{hidden}{_moe}.pth
      - 保存断点到 {save_dir}/{weight}_{hidden}{_moe}_resume.pth

    加载模式：model=None
      - 若存在 resume 文件则返回字典，否则 None
    """
    os.makedirs(save_dir, exist_ok=True)

    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel

        # 取真实模型 state_dict
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        # 1) 保存 half 权重（原子替换）
        ckp_tmp = ckp_path + ".tmp"
        torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 2) 记录 wandb/swanlab run id（如果有）
        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        # 3) 断点数据（模型用 full precision state_dict，便于继续训练）
        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }

        # 额外状态（如 scaler）
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(value, "state_dict"):
                if isinstance(value, DistributedDataParallel):
                    resume_data[key] = value.module.state_dict()
                else:
                    resume_data[key] = value.state_dict()
            else:
                resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:
        # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")

            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            # world_size 变化时粗略换算 step（避免继续训练进度错位太多）
            if "step" in ckp_data and saved_ws != current_ws and saved_ws > 0:
                ckp_data["step"] = int(ckp_data["step"]) * saved_ws // current_ws
                Logger(f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}")

            return ckp_data

        return None


# =========================
# 模型/Tokenizer 初始化（修复你当前的 tokenizer_path 逻辑）
# =========================

_TOKENIZER_CORE_FILES = (
    "tokenizer.json",      # fast tokenizer
    "tokenizer.model",     # sentencepiece
    "spiece.model",        # sentencepiece 常见命名
    "vocab.json",          # BPE
    "merges.txt",          # BPE
)


def _looks_like_tokenizer_dir(path: str) -> bool:
    if not path or (not os.path.isdir(path)):
        return False
    files = set(os.listdir(path))
    # 满足任意一种核心文件存在即可
    if "tokenizer.json" in files:
        return True
    if ("tokenizer.model" in files) or ("spiece.model" in files):
        return True
    if ("vocab.json" in files) and ("merges.txt" in files):
        return True
    return False


def _resolve_project_root() -> str:
    # trainer_utils.py 所在目录是 trainer/，项目根是它的上一级
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def _resolve_tokenizer_target(tokenizer_path: str | None) -> str:
    """
    返回传给 AutoTokenizer.from_pretrained 的 target：
    - 若 tokenizer_path 是 HF 模型 id（如 "gpt2" / "Qwen/Qwen2-0.5B"），直接返回该字符串
    - 若 tokenizer_path 为本地目录，则必须是“真 tokenizer 目录”
    - 若未传 tokenizer_path，则按候选目录自动寻找：{project_root}/tokenizer 等
    """
    # 1) 优先使用环境变量（便于你不改代码直接切 tokenizer）
    env_tok = os.environ.get("MINIMIND_TOKENIZER") or os.environ.get("TOKENIZER_PATH")
    if tokenizer_path is None and env_tok:
        tokenizer_path = env_tok

    # 2) 如果用户明确传了 tokenizer_path
    if tokenizer_path is not None:
        # a) 本地目录且像 tokenizer 目录
        if os.path.isdir(tokenizer_path):
            if _looks_like_tokenizer_dir(tokenizer_path):
                return tokenizer_path
            # 目录存在但不像 tokenizer，继续给出更明确的报错
            raise FileNotFoundError(
                f"tokenizer_path 指向的目录不存在可用 tokenizer 文件：{tokenizer_path}\n"
                f"需要至少包含以下之一：{_TOKENIZER_CORE_FILES}\n"
                f"你现在的目录文件：{os.listdir(tokenizer_path)}"
            )
        # b) 不存在的本地路径：当作 HF 模型 id 交给 transformers 处理
        return tokenizer_path

    # 3) 未传：按项目默认目录找（修复你原来误指向 model/ 源码目录的问题）
    project_root = _resolve_project_root()
    candidates = [
        os.path.join(project_root, "tokenizer"),
        os.path.join(project_root, "tokenizers"),
        os.path.join(project_root, "assets", "tokenizer"),
        os.path.join(project_root, "assets", "tokenizers"),
        # 兼容一些人把 tokenizer 放在 model/tokenizer 里
        os.path.join(project_root, "model", "tokenizer"),
    ]

    for cand in candidates:
        if _looks_like_tokenizer_dir(cand):
            return cand

    # 4) 找不到：给出可执行的解决方案
    raise FileNotFoundError(
        "未找到可用的 tokenizer 目录。\n"
        "请在项目根目录创建 tokenizer/ 并放入 tokenizer 文件（tokenizer.json 或 tokenizer.model/spiece.model 或 vocab.json+merges.txt）。\n"
        "最小可跑通方案（会下载 gpt2 tokenizer 并保存到 ./tokenizer）：\n"
        "  python -c \"from transformers import AutoTokenizer; tok=AutoTokenizer.from_pretrained('gpt2'); tok.save_pretrained('./tokenizer')\"\n"
        "或者你也可以通过环境变量指定：\n"
        "  set MINIMIND_TOKENIZER=gpt2\n"
        "  (Linux/Mac: export MINIMIND_TOKENIZER=gpt2)\n"
    )


def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="out",
    device="cuda",
    trust_remote_code: bool = False,
):
    """
    关键修复：
    1) 默认 tokenizer_path 不再指向项目根的 model/（那是源码目录），而是自动寻找项目根 tokenizer/。
    2) 强制 use_fast=False，避免 fast backend 依赖/转换路径导致的报错。
    3) 自动补 pad_token（若缺）。
    4) 尝试同步 lm_config.vocab_size = len(tokenizer)（若存在该字段）。
    """
    from transformers import AutoTokenizer
    from model.model import MokioMindForCausalLM

    target = _resolve_tokenizer_target(tokenizer_path)

    # 强制 slow tokenizer：稳定、少坑
    tokenizer = AutoTokenizer.from_pretrained(
        target,
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )

    # pad_token 兜底：不少 tokenizer 默认无 pad（训练 padding 会出问题）
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 同步 vocab_size（如果你的 config 有这个字段）
    if hasattr(lm_config, "vocab_size"):
        try:
            lm_config.vocab_size = len(tokenizer)
        except Exception:
            pass

    model = MokioMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = "_moe" if getattr(lm_config, "use_moe", False) else ""
        weight_name = f"{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

        # 兼容相对路径：先按传入 save_dir 找，再按项目根找
        weight_path_1 = os.path.join(save_dir, weight_name)
        weight_path_2 = os.path.join(_resolve_project_root(), save_dir, weight_name)

        if os.path.exists(weight_path_1):
            weight_path = weight_path_1
        elif os.path.exists(weight_path_2):
            weight_path = weight_path_2
        else:
            raise FileNotFoundError(
                f"未找到权重文件：\n- {weight_path_1}\n- {weight_path_2}\n"
                f"你设置的 from_weight={from_weight}, save_dir={save_dir}"
            )

        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    return model.to(device), tokenizer


# =========================
# 断点续训：跳 batch 采样器
# =========================

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.skip_batches = int(skip_batches)

    def __iter__(self):
        batch = []
        skipped = 0

        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue

                yield batch
                batch = []

        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
