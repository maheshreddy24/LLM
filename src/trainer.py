from torch.utils.data import DataLoader
from datasets import Pretraining_Dataset
from decoder import Decoder
from optimisation import Optimisation
import yaml
import torch
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def main():
    config_path = "configs/v1.yaml"
    data_dir = "data/arxiv_papers/txts"

    model = Decoder(config_path=config_path)

    with open(config_path, "r") as file:
        data = yaml.safe_load(file)

    print(data)

    train_ds = Pretraining_Dataset(
        context_length=data["context_length"],
        data_dir=data_dir,
        split="train",
    )

    test_ds = Pretraining_Dataset(
        context_length=data["context_length"],
        data_dir=data_dir,
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=data["batch_size"],
        num_workers=8,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=data["batch_size"],
        num_workers=8,
    )

    optim = Optimisation(
        model=model,
        device="cuda",
        epochs=data["epochs"],
        train_loader=train_loader,
        test_loader=test_loader,
        config_path=config_path,
    )

    optim.train()


if __name__ == "__main__":
    main()