import argparse
import csv
from pathlib import Path

import torch

from experiment_utils import build_finetune_model, load_config, make_cifar10_loader, resolve_device, set_seed


@torch.no_grad()
def accuracy_at_noise(model, loader, device, noise_aug):
    model.eval()
    correct = 0
    total = 0
    for img, labels in loader:
        img = img.to(device)
        labels = labels.to(device)
        logits = model(img, noise_aug=noise_aug)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / total * 100 if total else 0.0


@torch.no_grad()
def smoothed_stability(model, loader, device, noise_aug, samples, batch_repeats):
    model.eval()
    clean_match = 0.0
    majority_match = 0.0
    total = 0
    for img, _ in loader:
        img = img.to(device)
        clean_pred = model(img, noise_aug=0.0).argmax(dim=1)
        for idx in range(img.shape[0]):
            counts = None
            remaining = samples
            while remaining > 0:
                current = min(batch_repeats, remaining)
                repeated = img[idx : idx + 1].repeat(current, 1, 1, 1)
                pred = model(repeated, noise_aug=noise_aug).argmax(dim=1)
                bincount = torch.bincount(pred, minlength=model.linear_head.out_features).float()
                counts = bincount if counts is None else counts + bincount
                remaining -= current
            clean_match += counts[clean_pred[idx]].item() / samples
            majority_match += counts.max().item() / samples
            total += 1
    return {
        "clean_prediction_stability": clean_match / total if total else 0.0,
        "majority_prediction_stability": majority_match / total if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_margin_aware_rrcm.py")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--method", default="method")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output", default="results/metrics_summary.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sigmas", default="0.12,0.25,0.50,0.75,1.0")
    parser.add_argument("--stability-sigmas", default="0.25,0.50,0.75")
    parser.add_argument("--stability-samples", type=int, default=0)
    parser.add_argument("--stability-batch-repeats", type=int, default=64)
    parser.add_argument("--tiny-random", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    device = resolve_device(args.device)
    model = build_finetune_model(
        config,
        checkpoint=args.checkpoint,
        step=args.step,
        device=device,
        tiny_random=args.tiny_random,
    )
    loader = make_cifar10_loader(
        config,
        data_dir=args.data_dir,
        split="test",
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )

    rows = []
    rows.append(
        {
            "method": args.method,
            "metric": "clean_accuracy",
            "sigma": 0.0,
            "value": accuracy_at_noise(model, loader, device, 0.0),
            "num_samples": args.num_samples,
            "checkpoint": args.checkpoint or "",
            "notes": "debug tiny random model" if args.tiny_random else "",
        }
    )
    for sigma in [float(x) for x in args.sigmas.split(",") if x.strip()]:
        rows.append(
            {
                "method": args.method,
                "metric": f"accuracy_noise_sigma_{sigma:g}",
                "sigma": sigma,
                "value": accuracy_at_noise(model, loader, device, sigma),
                "num_samples": args.num_samples,
                "checkpoint": args.checkpoint or "",
                "notes": "debug tiny random model" if args.tiny_random else "",
            }
        )

    if args.stability_samples > 0:
        for sigma in [float(x) for x in args.stability_sigmas.split(",") if x.strip()]:
            stability = smoothed_stability(
                model,
                loader,
                device,
                sigma,
                args.stability_samples,
                args.stability_batch_repeats,
            )
            for metric, value in stability.items():
                rows.append(
                    {
                        "method": args.method,
                        "metric": f"{metric}_sigma_{sigma:g}",
                        "sigma": sigma,
                        "value": value,
                        "num_samples": args.num_samples,
                        "checkpoint": args.checkpoint or "",
                        "notes": "smoothed prediction stability, not certification",
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
