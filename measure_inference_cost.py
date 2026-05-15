import argparse
import csv
import time
from pathlib import Path

import torch

from experiment_utils import build_finetune_model, load_config, make_cifar10_loader, resolve_device, set_seed


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure(model, batch, device, warmup, iterations):
    model.eval()
    batch = batch.to(device)
    for _ in range(warmup):
        _ = model(batch, noise_aug=0.0)
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = model(batch, noise_aug=0.0)
    synchronize(device)
    elapsed = time.perf_counter() - start
    images = batch.shape[0] * iterations
    latency_ms = elapsed / images * 1000
    throughput = images / elapsed if elapsed > 0 else 0.0
    return latency_ms, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_margin_aware_rrcm.py")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--method", default="method")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output", default="results/inference_cost.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--baseline-latency-ms", type=float, default=None)
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
        num_samples=args.batch_size,
        num_workers=0,
    )
    batch, _ = next(iter(loader))
    latency_ms, throughput = measure(model, batch, device, args.warmup, args.iterations)
    relative = ""
    if args.baseline_latency_ms and args.baseline_latency_ms > 0:
        relative = latency_ms / args.baseline_latency_ms

    row = {
        "method": args.method,
        "batch_size": batch.shape[0],
        "device": str(device),
        "num_forward_passes": 1,
        "latency_ms_per_image": latency_ms,
        "throughput_images_per_second": throughput,
        "relative_cost_vs_rrcm": relative,
        "notes": "debug tiny random model" if args.tiny_random else "",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
