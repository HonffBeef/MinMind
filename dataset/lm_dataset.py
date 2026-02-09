import json
import os
import torch
from torch.utils.data import Dataset

# 建议用 setdefault，避免外部已经设置时被覆盖
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512, text_key="text"):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.text_key = text_key
        self.samples = self._load_data(data_path)

        # 必须：pad_token_id 检查（很多 tokenizer 默认没有 pad）
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id is None。你使用了 padding='max_length'，必须先设置 pad_token。\n"
                "例如：tokenizer.pad_token = tokenizer.eos_token"
            )

    def _load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON 解析失败，文件 {path} 第 {line_no} 行：{e}") from e
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = sample.get(self.text_key, "")
        if text is None:
            text = ""

        encoding = self.tokenizer(
            str(text),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # (1, max_len) -> (max_len,)
        input_ids = encoding["input_ids"].squeeze(0).long()

        # True 表示有效 token（非 pad）
        valid = input_ids.ne(self.tokenizer.pad_token_id)

        # 自回归：X 预测 Y
        X = input_ids[:-1]
        Y = input_ids[1:]

        # mask 对齐到 Y（预测目标位置）
        loss_mask = valid[1:]

        return X, Y, loss_mask
