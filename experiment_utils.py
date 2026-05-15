import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import rrcm.utils as utils
from rrcm_tune import FinetuneModel, load_data, reload_forward


def load_config(config_path):
    config_path = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _strip_module_prefix(state_dict):
    return {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _tiny_nnet_config(config):
    config.nnet.image_size = 32
    config.nnet.patch_size = 4
    config.nnet.embed_dim = 64
    config.nnet.hidden_dim = 128
    config.nnet.output_dim = 32
    config.nnet.depth = 1
    config.nnet.num_heads = 4
    config.nnet.mlp_ratio = 2
    config.nnet.use_checkpoint = False
    return config


def build_finetune_model(
    config,
    *,
    checkpoint=None,
    step=None,
    device=None,
    tiny_random=False,
    strict=True,
):
    if device is None:
        device = resolve_device("auto")

    if tiny_random:
        config = _tiny_nnet_config(config)
        num_scales = 20
    else:
        if step is None:
            raise ValueError("--step is required for non-tiny checkpoint runs")
        ema_scale_fn = utils.create_ema_and_scales_fn(**config.ema_scale)
        _, num_scales, _ = ema_scale_fn(int(step))

    diffusion_kwargs = dict(config.diffusion)
    diffusion_kwargs["device"] = device.type
    model = utils.create_model(**config.nnet)
    model.forward = reload_forward(model, layernorm=bool(config.train.layernorm))
    diffusion = utils.create_diffusion(**diffusion_kwargs)
    diffusion.set_scale(num_scales)

    class_num = 10 if config.dataset.name == "cifar10" else 1000
    ftmodel = FinetuneModel(
        model,
        diffusion,
        class_num,
        task="consistency",
        num_scales=num_scales,
        sigma_max=config.diffusion.sigma_max,
        sigma_min=config.diffusion.sigma_min,
    ).to(device)

    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = _strip_module_prefix(state_dict)
        ftmodel.load_state_dict(state_dict, strict=strict)

    return ftmodel


def make_cifar10_loader(
    config,
    *,
    data_dir,
    split="test",
    batch_size=128,
    num_samples=None,
    num_workers=0,
):
    dataset = load_data(
        name="cifar10",
        data_dir=data_dir,
        image_size=config.dataset.image_size,
        mode=split,
        value_range=config.dataset.value_range,
        augmentation_type="weak",
    )
    if num_samples is not None:
        dataset = Subset(dataset, list(range(min(int(num_samples), len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def classifier_weight_prototypes(model):
    return model.linear_head.weight
