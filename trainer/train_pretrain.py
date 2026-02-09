import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model import MokioMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    epoch: 当前 epoch（从0开始）
    loader: DataLoader（可能是 SkipBatchSampler 之后的剩余数据）
    iters: 本 epoch 的总 step 数（完整 epoch 的 steps，不是剩余 steps）
    start_step: 本 epoch 已完成的 step 数（断点续训用）
    """
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    is_ddp = isinstance(model, DistributedDataParallel)

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)

        # 全局 step：用于学习率调度（保持和断点续训一致）
        global_step = epoch * iters + step
        total_steps = iters * args.epochs
        lr = get_lr(global_step, total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 梯度累积：用“本次恢复后”本地计数，避免断点时残留累积不可恢复带来的错位
        local_step = step - start_step  # 从 1 开始
        do_step = (local_step % args.accumulation_steps == 0) or (step == iters)

        # DDP 梯度累积时，非 do_step 的 micro-step 用 no_sync 减少通信
        sync_ctx = nullcontext() if (not is_ddp or do_step) else model.no_sync()

        with sync_ctx:
            with autocast_ctx:
                res = model(X)
                logits = res.logits  # [B, T, V]

                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                ).view(Y.size())

                # mask 平均，防止极端情况下除 0
                denom = loss_mask.sum().clamp_min(1).to(loss.dtype)
                loss = (loss * loss_mask).sum() / denom

                # 梯度累积缩放
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

        if do_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志（注意：loss 这里是缩放后的，打印要乘回 accumulation_steps）
        if (step % args.log_interval == 0) or (step == iters):
            elapsed = time.time() - start_time
            done = max(step - start_step, 1)
            avg = elapsed / done
            remaining = max(iters - step, 0)
            eta_min = int((avg * remaining) // 60)

            current_loss = loss.detach().float().item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.6f} lr:{current_lr:.12f} eta:{eta_min}min"
            )

            if wandb and is_main_process():
                wandb.log({"loss": current_loss, "lr": current_lr, "eta_min": eta_min})

        # 保存（只在主进程）
        if ((step % args.save_interval == 0) or (step == iters)) and is_main_process():
            model.eval()

            moe_suffix = "_moe" if getattr(lm_config, "use_moe", False) else ""
            os.makedirs(args.save_dir, exist_ok=True)
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            if isinstance(model, DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存 half 权重到 CPU，减少显存占用，文件也更小
            state_dict_half = {k: v.detach().cpu().half() for k, v in state_dict.items()}
            torch.save(state_dict_half, ckp)

            # 保存完整训练状态
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="checkpoints",
            )

            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # 基础训练参数
    parser.add_argument("--save_dir", type=str, default="out", help="模型保存目录")
    parser.add_argument("--save_weight", default="pretrain", type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # 硬件和性能参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型：bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    # 训练策略参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")

    # 模型架构参数
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=512, type=int, help="训练的最大截断长度")
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1], help="是否使用MoE（0/1）")

    # 数据和恢复参数
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument("--from_weight", default="none", type=str, help="基于哪个权重训练，none=从头")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0/1）")

    # 实验跟踪参数
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb/swanlab")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")

    args = parser.parse_args()

    # 1) 初始化分布式
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    # 2) 随机种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 3) 配置目录、模型参数、检查点
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="checkpoints")
        if args.from_resume == 1
        else None
    )

    # 4) 混合精度
    use_cuda = torch.cuda.is_available() and ("cuda" in str(args.device))
    device_type = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and args.dtype == "float16"))

    # 5) 实验跟踪（主进程）
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 6) 初始化模型和 tokenizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # ✅ 修正：PretrainDataset 参数名是 max_len
    train_ds = PretrainDataset(args.data_path, tokenizer, max_len=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 7) 断点恢复
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        if ckp_data.get("optimizer") is not None:
            optimizer.load_state_dict(ckp_data["optimizer"])
        if ckp_data.get("scaler") is not None:
            scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = int(ckp_data.get("epoch", 0))
        start_step = int(ckp_data.get("step", 0))

    # 8) DDP 包装
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 9) 训练
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # 断点续训：同一 epoch 内跳过 start_step
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)),
                args.batch_size,
                skip_batches=start_step,   # ✅ 修正：跳过 start_step，不是 start_step+1
            )

            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=use_cuda,
            )

            # ✅ iters = 本 epoch 总 steps = 剩余 steps + 已完成 steps
            total_iters = len(loader) + start_step

            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
            train_epoch(epoch, loader, total_iters, start_step=start_step, wandb=wandb)

        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=use_cuda,
            )

            train_epoch(epoch, loader, len(loader), start_step=0, wandb=wandb)
