import argparse
import csv
from pathlib import Path

import accelerate
import ml_collections
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from experiment_utils import build_finetune_model, load_config, set_seed
from rrcm_tune import consistency_loss, load_data, train_one_epoch_consistency


def run_baseline_step(config, data_dir):
    dataset = load_data(
        name="cifar10",
        data_dir=data_dir,
        image_size=32,
        mode="train",
        value_range="0.5,0.5",
        augmentation_type="weak",
    )
    loader = DataLoader(Subset(dataset, list(range(4))), batch_size=2, shuffle=False, num_workers=0, drop_last=True)
    device = torch.device("cpu")
    model = build_finetune_model(config, device=device, tiny_random=True)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad and n not in ["linear_head.weight", "linear_head.bias"]],
        lr=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss()

    img, y = next(iter(loader))
    img = img.to(device)
    y = y.to(device)
    img_repeated = torch.cat([img, img], dim=0)
    y_repeated = torch.cat([y, y], dim=0)
    logits = model(img_repeated, noise_aug=0.5, noise=None)
    clsloss = loss_fn(logits, y_repeated)
    closs = consistency_loss(logits.chunk(2), lbd=10.0, eta=0.5)
    loss = clsloss + closs
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "method": "baseline_smoke_tiny_random",
        "status": "ok",
        "dataset_len": len(dataset),
        "batch_size": img.shape[0],
        "classification_loss": float(clsloss.detach()),
        "consistency_loss": float(closs.detach()),
        "proto_loss": "",
        "margin_loss": "",
        "total_loss": float(loss.detach()),
        "notes": "Tiny random model smoke test; not a robustness result.",
    }


def run_margin_step(config, data_dir):
    dataset = load_data(
        name="cifar10",
        data_dir=data_dir,
        image_size=32,
        mode="train",
        value_range="0.5,0.5",
        augmentation_type="weak",
    )
    loader = DataLoader(Subset(dataset, list(range(2))), batch_size=2, shuffle=False, num_workers=0, drop_last=True)
    accelerator = accelerate.Accelerator(cpu=True)
    device = accelerator.device
    model = build_finetune_model(config, device=device, tiny_random=True)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad and n not in ["linear_head.weight", "linear_head.bias"]],
        lr=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss()
    margin_config = ml_collections.ConfigDict(
        {
            "use_margin_aware_loss": True,
            "use_proto_loss": True,
            "use_margin_loss": True,
            "lambda_proto": 0.1,
            "lambda_margin": 0.1,
            "prototype_margin": 0.2,
            "prototype_temperature": 0.1,
            "prototype_source": "classifier_weight",
            "normalize_features": True,
            "log_geometry_metrics": True,
        }
    )
    metrics = train_one_epoch_consistency(
        accelerator,
        model,
        loader,
        optimizer,
        loss_fn,
        mixup_fn=None,
        noise_aug=0.5,
        lbd=10.0,
        eta=0.5,
        margin_config=margin_config,
    )
    total_loss = float(metrics["avg_loss"]) + float(metrics["avg_closs"])
    total_loss += 0.1 * float(metrics["avg_proto_loss"]) + 0.1 * float(metrics["avg_margin_loss"])
    return {
        "method": "margin_aware_smoke_tiny_random",
        "status": "ok",
        "dataset_len": len(dataset),
        "batch_size": 2,
        "classification_loss": metrics["avg_loss"],
        "consistency_loss": metrics["avg_closs"],
        "proto_loss": metrics["avg_proto_loss"],
        "margin_loss": metrics["avg_margin_loss"],
        "total_loss": total_loss,
        "notes": "Tiny random model smoke test; not a robustness result.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_margin_aware_rrcm.py")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output", default="results/smoke_tests.csv")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    rows = [run_baseline_step(config, args.data_dir)]
    set_seed(args.seed)
    config = load_config(args.config)
    rows.append(run_margin_step(config, args.data_dir))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
