import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import yaml
from datetime import datetime
import math
import wandb


class Optimisation():
    def __init__(self, model, device, epochs, train_loader, test_loader, config_path):
        with open(config_path, "r") as file:
            self.data = yaml.safe_load(file)

        # print(self.data)
        # print(type(self.data["lr"]), self.data["lr"])
        # print(type(self.data["weight_decay"]), self.data["weight_decay"])

        self.device = device
        self.model = model.to(device)
        self.lr = float(self.data["lr"])
        warmup_ratio = float(self.data["warmup_ratio"])
        weight_decay = float(self.data["weight_decay"])
        self.grad_accum = float(self.data["grad_accum"])
        self.save_every = 1_000/self.grad_accum

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs

        self.output_dir = os.path.join(
            "checkpoints",
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay,
        )

        steps_per_epoch = math.ceil(
            len(train_loader) / self.grad_accum
        )

        num_training_steps = epochs * steps_per_epoch

        num_warmup_steps = int(
            warmup_ratio * num_training_steps
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=max(1, num_warmup_steps),
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, num_training_steps - num_warmup_steps),
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[num_warmup_steps],
        )

        wandb.init(
            project="lm_f",
            mode="online",
            config={
                "lr": self.lr,
                "warmup_ratio": warmup_ratio,
                "epochs": self.epochs,
                "grad_accum": self.grad_accum,
                "dataset": "pretraining_data",
            },
        )

    def train(self):
        global_step = 0

        for epoch in tqdm(
            range(self.epochs),
            total=self.epochs,
            leave=False,
            desc="Epochs",
        ):
            self.model.train()

            total_loss = 0.0
            step_loss = 0.0

            self.optimizer.zero_grad()

            for step, batch in tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                leave=True,
                desc="Train dataloader",
            ):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                # print(labels.min(), labels.max())
                # print(input_ids.min(), input_ids.max())
                # print(self.model.lm_head.out_features)   # vocab size


                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    labels=labels.to(self.device),
                )

                loss = outputs.loss

                total_loss += loss.item()
                step_loss += loss.item()

                (loss / self.grad_accum).backward()

                if (
                    (step + 1) % self.grad_accum == 0
                    or (step + 1) == len(self.train_loader)
                ):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                    wandb.log(
                        {
                            "step_loss": step_loss / self.grad_accum,
                            "lr": self.scheduler.get_last_lr()[0],
                            "global_step": global_step,
                        }
                    )

                    if global_step % self.save_every == 0:
                        self.save_checkpoint(epoch + 1, global_step)

                    step_loss = 0.0

            train_loss = total_loss / len(self.train_loader)
            val_loss = self.eval()

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

        self.save_checkpoint(self.epochs, global_step)

    def eval(self):
        self.model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                self.test_loader,
                total=len(self.test_loader),
                leave=False,
                desc="Validation",
            ):
                # input_ids, attention_mask, labels = batch
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]

                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    labels=labels.to(self.device),
                )

                total_loss += outputs.loss.item()

        self.model.train()

        return total_loss / len(self.test_loader)

    def save_checkpoint(self, epoch, global_step):
        ckpt_dir = os.path.join(
            self.output_dir,
            f"epoch-{epoch}-step-{global_step}",
        )

        os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            os.path.join(ckpt_dir, "optim.pt"),
        )

        print(f"Checkpoint saved -> {ckpt_dir}")

    def resume(self, checkpoint_path):
        state = torch.load(
            os.path.join(checkpoint_path, "optim.pt"),
            map_location=self.device,
        )

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

        return state["epoch"], state["step"]