import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment_utils import (
    build_finetune_model,
    classifier_weight_prototypes,
    load_config,
    make_cifar10_loader,
    resolve_device,
    set_seed,
)


@torch.no_grad()
def summarize_geometry(model, loader, device, noise_aug):
    model.eval()
    prototypes = F.normalize(classifier_weight_prototypes(model), dim=1)
    stats = {
        "correct_proto_similarity": [],
        "max_wrong_proto_similarity": [],
        "prototype_margin": [],
        "distance_to_correct_prototype": [],
        "top1_top2_logit_margin": [],
        "feature_consistency_distance": [],
        "feature_norm": [],
    }

    for img, labels in loader:
        img = img.to(device)
        labels = labels.to(device)
        clean_logits, clean_features = model(img, noise_aug=0.0, return_features=True)
        logits, features = model(img, noise_aug=noise_aug, return_features=True)

        z = F.normalize(features, dim=1)
        sim = z @ prototypes.T
        sim_correct = sim.gather(1, labels[:, None]).squeeze(1)
        wrong_sim = sim.masked_fill(F.one_hot(labels, num_classes=sim.shape[1]).bool(), torch.finfo(sim.dtype).min)
        sim_wrong_max = wrong_sim.max(dim=1).values
        correct_proto = prototypes[labels]
        top2 = logits.topk(k=2, dim=1).values
        clean_z = F.normalize(clean_features, dim=1)

        stats["correct_proto_similarity"].append(sim_correct.cpu())
        stats["max_wrong_proto_similarity"].append(sim_wrong_max.cpu())
        stats["prototype_margin"].append((sim_correct - sim_wrong_max).cpu())
        stats["distance_to_correct_prototype"].append((z - correct_proto).pow(2).sum(dim=1).sqrt().cpu())
        stats["top1_top2_logit_margin"].append((top2[:, 0] - top2[:, 1]).cpu())
        stats["feature_consistency_distance"].append((clean_z - z).pow(2).sum(dim=1).sqrt().cpu())
        stats["feature_norm"].append(features.norm(dim=1).cpu())

    return {key: torch.cat(value).mean().item() for key, value in stats.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_margin_aware_rrcm.py")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--method", default="method")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output", default="results/geometry_summary.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sigmas", default="0,0.12,0.25,0.50,0.75,1.0")
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
    for sigma in [float(x) for x in args.sigmas.split(",") if x.strip()]:
        row = {
            "method": args.method,
            "sigma": sigma,
            "num_samples": args.num_samples,
            "checkpoint": args.checkpoint or "",
            "notes": "debug tiny random model" if args.tiny_random else "",
        }
        row.update(summarize_geometry(model, loader, device, sigma))
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
