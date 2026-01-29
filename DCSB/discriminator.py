"""
DCSB discriminator (rewritten for readability & robustness) + Weights & Biases logging
"""

import os
from typing import Tuple

import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
from tqdm import trange

# NEW: wandb
import wandb


# ======================
# CONFIG (match original)
# ======================
INPUT_DIM: int = 20
BATCH_SIZE: int = 128
EPOCHS: int = 30

LR: float = 0.009
BETAS = (0.9, 0.999)
EPS: float = 1e-8
WEIGHT_DECAY: float = 0.0

THRESHOLD: float = 0.5  # for classification in evaluation

MODEL_DIR: str = "./Model_path"
MODEL_SUBDIR: str = "model_name"  # original uses "model_name/"
BEST_CKPT_NAME: str = "best_accuracy.pth"

NUM_WORKERS: int = 8  # keep original

# NEW: wandb config
WANDB_PROJECT: str = "DCSB"
WANDB_RUN_NAME: str | None = None  # e.g., "disc_coco_dim20"
WANDB_MODE: str | None = None      # set "offline" if no internet, else None
# ======================


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

        self.fc_1 = nn.Sequential(
            nn.Linear(input_dim, 10, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(10, input_dim, bias=False),
            nn.Sigmoid(),
        )

        # keep for compatibility (unused)
        self.fc_2 = nn.Sequential(
            nn.Linear(10, 5, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(5, 10, bias=False),
            nn.Sigmoid(),
        )

        self.hidden1 = nn.Linear(input_dim, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.relu = nn.ReLU()

        self.hidden2 = nn.Linear(300, 150)
        self.bn2 = nn.BatchNorm1d(150)

        self.hidden3 = nn.Linear(150, 50)
        self.bn3 = nn.BatchNorm1d(50)

        self.hidden4 = nn.Linear(50, 10)
        self.bn4 = nn.BatchNorm1d(10)

        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)

        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D [B,C], got shape={tuple(x.shape)}")
        b, c = x.size()
        if c != self.input_dim:
            raise ValueError(f"Expected feature dim={self.input_dim}, got {c}")

        gate = self.fc_1(x).view(b, c)
        x = x * gate

        x = self.relu(self.bn1(self.hidden1(x)))
        x = self.relu(self.bn2(self.hidden2(x)))
        x = self.relu(self.bn3(self.hidden3(x)))
        x = self.relu(self.bn4(self.hidden4(x)))

        x = self.sigmoid(self.output(x))
        return x.squeeze(-1)


def init_weights(net: Discriminator) -> None:
    init.normal_(net.hidden1.weight, mean=0.0, std=0.02)
    init.normal_(net.hidden2.weight, mean=0.0, std=0.02)
    init.normal_(net.hidden3.weight, mean=0.0, std=0.02)
    init.normal_(net.hidden4.weight, mean=0.0, std=0.02)
    init.normal_(net.output.weight, mean=0.0, std=0.02)

    init.normal_(net.fc_1[0].weight, mean=0.0, std=0.2)

    init.constant_(net.hidden1.bias, 0.0)
    init.constant_(net.hidden2.bias, 0.0)
    init.constant_(net.hidden3.bias, 0.0)
    init.constant_(net.hidden4.bias, 0.0)
    init.constant_(net.output.bias, 0.0)


@torch.no_grad()
def compute_metrics(pred01: torch.Tensor, y_true: torch.Tensor) -> Tuple[float, float, float]:
    pred01 = pred01.to(torch.int64).view(-1)
    y_true = y_true.to(torch.int64).view(-1)

    if pred01.numel() != y_true.numel():
        raise ValueError(f"pred and y length mismatch: {pred01.numel()} vs {y_true.numel()}")

    TP = int(((pred01 == 1) & (y_true == 1)).sum().item())
    FP = int(((pred01 == 1) & (y_true == 0)).sum().item())
    TN = int(((pred01 == 0) & (y_true == 0)).sum().item())
    FN = int(((pred01 == 0) & (y_true == 1)).sum().item())

    total = pred01.numel()
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    upload_rate_percent = (TP + FP) / total * 100.0 if total > 0 else 0.0

    return upload_rate_percent, accuracy, precision


@torch.no_grad()
def evaluate(net: Discriminator, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    net.eval()
    out = net(x)
    pred01 = (out >= THRESHOLD).to(torch.int64)
    return compute_metrics(pred01, y)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    # -------- Paths --------
    ensure_dir(MODEL_DIR)
    model_save_dir = os.path.join(MODEL_DIR, MODEL_SUBDIR)
    ensure_dir(model_save_dir)
    best_ckpt_path = os.path.join(model_save_dir, BEST_CKPT_NAME)

    # -------- Load data --------
    train_x = torch.load("small_model_output_results_train_data")
    train_y = torch.load("small_model_output_results_label_train_data")
    test_x = torch.load("small_model_output_results_test_data")
    test_y = torch.load("small_model_output_results_label_test_data")

    if train_x.dim() != 2 or train_x.size(1) != INPUT_DIM:
        raise ValueError(f"train_x must be [N,{INPUT_DIM}], got {tuple(train_x.shape)}")
    if test_x.dim() != 2 or test_x.size(1) != INPUT_DIM:
        raise ValueError(f"test_x must be [N,{INPUT_DIM}], got {tuple(test_x.shape)}")

    train_y = train_y.view(-1).to(torch.float32)
    test_y = test_y.view(-1).to(torch.float32)

    # -------- Build model --------
    net = Discriminator(input_dim=INPUT_DIM)
    init_weights(net)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    # -------- DataLoader --------
    dataset = Data.TensorDataset(train_x, train_y)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # =====================
    # NEW: wandb init
    # =====================
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "input_dim": INPUT_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "betas": BETAS,
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "threshold": THRESHOLD,
            "num_workers": NUM_WORKERS,
        },
        mode=WANDB_MODE,  # set "offline" if needed
    )
    # optional: watch gradients/params (can be heavy)
    # wandb.watch(net, log="gradients", log_freq=100)

    # -------- Train --------
    best_acc = 0.0
    global_step = 0

    with trange(EPOCHS) as pbar:
        for epoch in pbar:
            pbar.set_description(f"epoch {epoch}")

            for step, (batch_x, batch_y) in enumerate(loader):
                net.train()
                out = net(batch_x)
                loss_val = criterion(out, batch_y)

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Evaluate on full test set (same frequency as original)
                upload, acc, precision = evaluate(net, test_x, test_y)

                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=float(loss_val.item()), acc=acc * 100.0, l_r=lr_now)

                # NEW: log to wandb every step
                wandb.log(
                    {
                        "train/loss": float(loss_val.item()),
                        "test/accuracy": acc,
                        "test/precision": precision,
                        "test/upload_rate_percent": upload,
                        "lr": lr_now,
                        "epoch": epoch,
                        "step_in_epoch": step,
                    },
                    step=global_step,
                )
                global_step += 1

                # Save best
                if acc >= best_acc:
                    best_acc = acc
                    torch.save(net.state_dict(), best_ckpt_path)

                    # NEW: also log best checkpoint as an artifact (optional but useful)
                    artifact = wandb.Artifact("best_discriminator", type="model")
                    artifact.add_file(best_ckpt_path)
                    wandb.log_artifact(artifact)

            # NEW: log best_acc after each epoch
            wandb.log({"best_acc_so_far": best_acc, "epoch_end": epoch}, step=global_step)

    print("best_acc", best_acc)
    wandb.finish()


if __name__ == "__main__":
    main()
