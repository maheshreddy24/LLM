import os
import random
import re
import unicodedata
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
from icecream import ic
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    text = re.sub(r"\[\[.*?\]\]", "", text)   # wiki markup
    text = re.sub(r"\{\{.*?\}\}", "", text)   # templates
    text = re.sub(r"<[^>]+>", "", text)       # leftover tags

    return text


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.encode("utf-8", "ignore").decode("utf-8")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    return text.strip()


class Pretraining_Dataset(Dataset):
    def __init__(self, context_length, data_dir, split="train"):
        super().__init__()

        self.context_length = context_length

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # self.tokenizer.add_special_tokens(
        #     {
        #         "pad_token": "[PAD]",
        #     }
        # )
        self.tokenizer.padding_side = "left"
        paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".txt")
        ]

        random.shuffle(paths)
        split_idx = int(0.8 * len(paths))

        if split == "train":
            paths = paths[:split_idx]
        elif split == "test":
            paths = paths[split_idx:]
        else:
            raise ValueError("split must be either 'train' or 'test'")

        print(f"{split} files: {len(paths)}")
        tokenized_seq = []

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            text = normalize_text(remove_html(text))

            tokenized_seq.extend(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )
            tokenized_seq.append(self.tokenizer.eos_token_id)
            

        print(f"Total number of {split} tokens: {len(tokenized_seq):,}")
        stride = context_length # lets not take any duplicates

        self.dataset = []

        for start in range(
            0,
            len(tokenized_seq) - context_length,
            stride,
        ):
            self.dataset.append(
                tokenized_seq[start : start + context_length]
            )

        print(f"{split} samples: {len(self.dataset):,}")
        ic(max(tokenized_seq))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = torch.tensor(self.dataset[index], dtype=torch.long)

        # Labels are next-token prediction
        return {
            "input_ids": x,
            "labels": x.clone(),
            "attention_mask": torch.ones_like(x)
        }

if __name__ == "__main__":
    ds = Pretraining_Dataset(context_length = 1024, data_dir='data/arxiv_papers/txts', split='test')
    test_loader = DataLoader(ds, batch_size = 4, shuffle = True)
    batch = next(iter(test_loader))
    ic(batch['input_ids'].shape)
    ic(batch['labels'].shape)
    ic(batch['attention_mask'].shape)

    