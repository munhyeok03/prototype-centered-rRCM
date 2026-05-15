"""
Train a diffusion model on images.
"""

import os
import math
import builtins
import numpy as np
from PIL import Image
from tqdm import tqdm
from functools import partial

from absl import flags
from absl import app
from ml_collections import config_flags
from timm.models.layers import trunc_normal_
from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import accelerate

from absl import logging
import rrcm.utils as utils


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def reload_forward(self, layernorm):

    def forward(x, timesteps, **kwargs):
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1).to(self.dtype)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = x.to(self.dtype)

        for blk in self.blocks:
            x = blk(x)
        x = x[:, 0]

        if layernorm:
            x = self.norm(x)

        return x

    return forward


class ShortSideResizeCenterCrop(nn.Module):

    def __init__(self, image_size):
        super().__init__()
        self.image_size =image_size

    def forward(self, pil_image):
        image_size = self.image_size
        if pil_image.size[0] == image_size and pil_image.size[1] == image_size:
            return np.array(pil_image)

        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def load_data(
    *,
    name,
    data_dir,
    image_size,
    mode="train",
    value_range = "0.5,0.5",
    augmentation_type="weak",
    autoaug = "",
    reprob = 0.0,
    **kwargs,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    if value_range == "0,1":
        normalize = transforms.Normalize(mean = [0, 0, 0],
                                        std=[1, 1, 1])
    elif value_range == "0.5,0.5":
        normalize = transforms.Normalize(mean = [0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean = [0.49139968, 0.48215827, 0.44653124],
                                        std=[0.24703233, 0.24348505, 0.26158768])

    logging.info(f"normalize: {normalize}")

    if name == "cifar10":
        if mode == "train":
            if augmentation_type == "strong":
                data_aug = transforms.Compose([
                    transforms.RandomResizedCrop(image_size, scale=(0.4, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif augmentation_type == "autoaug":
                data_aug = create_transform(
                    input_size=image_size,
                    is_training=True,
                    color_jitter=None,
                    auto_augment=autoaug,
                    interpolation='bicubic',
                    re_prob=reprob,
                    re_mode="pixel",
                    re_count=1,
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                )
            else:
                data_aug = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            if image_size != 32:
                data_aug = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                data_aug = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])

        dataset = datasets.CIFAR10(data_dir, train=(mode == "train"), transform=data_aug, download=False)

    elif name == "imagenet":
        if mode == "train":
            data_dir = os.path.join(data_dir, "train")
            if augmentation_type == "weak":
                data_aug = transforms.Compose([
                    ShortSideResizeCenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    normalize,
                ])
            elif augmentation_type == "strong":
                data_aug = transforms.Compose([
                    transforms.RandomResizedCrop(image_size, scale=(0.4, 1.),  interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif augmentation_type == "extreme":
                data_aug = transforms.Compose([
                    transforms.RandomResizedCrop(image_size, scale=(0.08, 1.), interpolation=3),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)  # not strengthened
                    ], p=0.5),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

            elif augmentation_type == "autoaug":
                data_aug = create_transform(
                    input_size=image_size,
                    is_training=True,
                    color_jitter=None,
                    auto_augment=autoaug,
                    interpolation='bicubic',
                    re_prob=reprob,
                    re_mode="pixel",
                    re_count=1,
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                )
        else:
            data_dir = os.path.join(data_dir, "val")
            data_aug = transforms.Compose([
                ShortSideResizeCenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        # make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
        # https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
        dataset = datasets.ImageFolder(data_dir, transform=data_aug)

    return dataset


class FinetuneModel(nn.Module):

    def __init__(self, base_model: nn.Module, diffusion, class_num, task="linear", num_scales=81, sigma_max=80.0, sigma_min=0.002):
        super().__init__()
        self.base_model = base_model
        self.diffusion = diffusion
        self.task = task
        self.num_scales = num_scales
        print("num scales:", num_scales)

        self.linear_head = nn.Linear(base_model.embed_dim, class_num)
        self.linear_head.weight.detach().zero_()
        self.linear_head.bias.detach().zero_()
        trunc_normal_(self.linear_head.weight, std=2e-5)

        if task == "linear":
            for name, param in self.base_model.named_parameters():
                param.requires_grad = False

        self.sigma_min=sigma_min
        self.sigma_max=sigma_max
        self.rho = 7
        self.sigmas = torch.tensor([self.sigma_max ** (1 / self.rho) + i / (num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            ) for i in range(num_scales)])
        self.sigmas = self.sigmas**self.rho

    def get_sigma(self, n):
        sigma = self.sigma_max ** (1 / self.rho) + n / (self.num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
            )
        sigma = sigma**self.rho
        return sigma
    
    def find_nearest(self, t):
        self.sigmas = self.sigmas.to(t.device)
        idx = (t[:, None].expand(-1, self.num_scales) - self.sigmas[None, :].expand(t.shape[0], -1)).abs().argmin(dim=1)
        return idx

    def forward(self, x, noise_aug=0.0, noise=None, return_features=False):

        # Augment with Gaussian noise during training
        if noise_aug != 0:
            if noise is None:
                noise = torch.randn_like(x)
            # assert noise_aug in [0.5, 1.0, 2.0]

            idx = self.find_nearest(torch.full((x.shape[0], ), noise_aug, device=x.device))
            x = x + noise * noise_aug
        else:
            idx = torch.full((x.shape[0], ), self.num_scales-1, device=x.device)


        if self.task == "linear":
            with torch.no_grad():
                h = self.forward_features(x, idx)
        else:
            h = self.forward_features(x, idx)

        logits = self.linear_head(h)
        if return_features:
            return logits, h
        return logits

    def forward_features(self, x, idx):
        t = self.get_sigma(idx) # retrieve `t` (noise level used in diffusion process: xt=x0+t*noise) with timestep `idx`` 
        h = self.diffusion.rcm_tune_denoise(self.base_model, x, t, indices=None) # run denoising
        return h

    def train(self, mode=None):
        self.base_model.eval()
        if self.task == "finetune":
            self.base_model.train()
        self.linear_head.train()

    def eval(self, mode=None):
        self.base_model.eval()
        self.linear_head.eval()


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent


def train_one_epoch(accelerator:accelerate.Accelerator, ftmodel, train_data, optimizer, loss_fn, mixup_fn):
    device = accelerator.device
    avg_acc = []
    avg_loss =  []

    ftmodel.train()
    for batch in train_data:
        img, y = batch
        img = img.to(device)
        y = y.to(device)

        if mixup_fn is not None:
            img, y = mixup_fn(img, y)

        with accelerator.autocast():
            pred = ftmodel(img)
            loss = loss_fn(pred, y) + entropy(torch.softmax(pred, dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all = accelerator.gather(loss)
        avg_loss.append(loss_all.mean().cpu().item())
        if mixup_fn is None:
            acc = (pred.argmax(dim=1) == y).sum() / img.shape[0] * 100
            acc_all = accelerator.gather(acc)
            avg_acc.append(acc_all.mean().cpu().item())

    if accelerator.process_index == 0:
        logging_dict = {
            "avg_acc": f"{sum(avg_acc)/len(avg_acc)}" if mixup_fn is None else -1, 
            "avg_loss":f"{sum(avg_loss)/len(avg_loss)}",
        }
        logging.info(logging_dict)
        return logging_dict
    else:
        return None


def consistency_loss(logits, lbd, eta=0.5, loss='default'):
    """
    Consistency regularization for certified robustness.

    Parameters
    ----------
    logits : List[torch.Tensor]
        A list of logit batches of the same shape, where each
        is sampled from f(x + noise) with i.i.d. noises.
        len(logits) determines the number of noises, i.e., m > 1.
    lbd : float
        Hyperparameter that controls the strength of the regularization.
    eta : float (default: 0.5)
        Hyperparameter that controls the strength of the entropy term.
        Currently used only when loss='default'.
    loss : {'default', 'xent', 'kl', 'mse'} (optional)
        Which loss to minimize to obtain consistency.
        - 'default': The default form of loss.
            All the values in the paper are reproducible with this option.
            The form is equivalent to 'xent' when eta = lbd, but allows
            a larger lbd (e.g., lbd = 20) when eta is smaller (e.g., eta < 1).
        - 'xent': The cross-entropy loss.
            A special case of loss='default' when eta = lbd. One should use
            a lower lbd (e.g., lbd = 3) for better results.
        - 'kl': The KL-divergence between each predictions and their average.
        - 'mse': The mean-squared error between the first two predictions.

    """

    m = len(logits)

    softmax = [F.softmax(logit, dim=1) for logit in logits]
    avg_softmax = sum(softmax) / m

    loss_kl = [kl_div(logit, avg_softmax) for logit in logits]
    loss_kl = sum(loss_kl) / m

    if loss == 'default':
        loss_ent = entropy(avg_softmax)
        consistency = lbd * loss_kl + eta * loss_ent
    elif loss == 'xent':
        loss_ent = entropy(avg_softmax)
        consistency = lbd * (loss_kl + loss_ent)
    elif loss == 'kl':
        consistency = lbd * loss_kl
    elif loss == 'mse':
        sm1, sm2 = softmax[0], softmax[1]
        loss_mse = ((sm2 - sm1) ** 2).sum(1)
        consistency = lbd * loss_mse
    else:
        raise NotImplementedError()

    return consistency.mean()


def kl_div(input, targets):
    return F.kl_div(F.log_softmax(input, dim=1), targets, reduction='none').sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent


def _cfg_get(config, name, default):
    if config is None:
        return default
    try:
        return config.get(name, default)
    except AttributeError:
        return getattr(config, name, default)


def classifier_weight_prototypes(ftmodel):
    return ftmodel.linear_head.weight


def margin_aware_prototype_loss(
    features,
    labels,
    prototypes,
    *,
    temperature=0.1,
    margin=0.2,
    normalize_features=True,
    use_proto_loss=True,
    use_margin_loss=True,
):
    if temperature <= 0:
        raise ValueError("prototype_temperature must be positive")

    if normalize_features:
        features = F.normalize(features.flatten(1), dim=1)
        prototypes = F.normalize(prototypes.flatten(1), dim=1)
    else:
        features = features.flatten(1)
        prototypes = prototypes.flatten(1)

    sim = features @ prototypes.T
    hard_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels

    proto_loss = features.new_zeros(())
    if use_proto_loss:
        sim_scaled = sim / temperature
        if labels.ndim == 2:
            proto_loss = -(labels * F.log_softmax(sim_scaled, dim=1)).sum(dim=1).mean()
        else:
            proto_loss = F.cross_entropy(sim_scaled, hard_labels)

    margin_loss = features.new_zeros(())
    if use_margin_loss:
        sim_correct = sim.gather(1, hard_labels[:, None]).squeeze(1)
        wrong_sim = sim.masked_fill(
            F.one_hot(hard_labels, num_classes=sim.shape[1]).bool(),
            torch.finfo(sim.dtype).min,
        )
        sim_wrong_max = wrong_sim.max(dim=1).values
        margin_loss = F.relu(margin - sim_correct + sim_wrong_max).mean()
    else:
        sim_correct = sim.gather(1, hard_labels[:, None]).squeeze(1)
        wrong_sim = sim.masked_fill(
            F.one_hot(hard_labels, num_classes=sim.shape[1]).bool(),
            torch.finfo(sim.dtype).min,
        )
        sim_wrong_max = wrong_sim.max(dim=1).values

    proto_margin = sim_correct - sim_wrong_max
    return {
        "proto_loss": proto_loss,
        "margin_loss": margin_loss,
        "sim_correct": sim_correct.detach().mean(),
        "sim_wrong_max": sim_wrong_max.detach().mean(),
        "proto_margin": proto_margin.detach().mean(),
    }


def _feature_consistency_distance(features, chunks):
    feature_chunks = features.detach().chunk(chunks)
    if len(feature_chunks) < 2:
        return features.new_zeros(())
    f0 = F.normalize(feature_chunks[0].flatten(1), dim=1)
    distances = []
    for f in feature_chunks[1:]:
        f = F.normalize(f.flatten(1), dim=1)
        distances.append((f0 - f).pow(2).sum(dim=1).sqrt().mean())
    return torch.stack(distances).mean()


def confidence_consistency_loss(logits_m, labels, m=2, beta_conf=0.1, eps=1e-6):
    """
    Label-conditioned noisy-view confidence consistency loss.
    L_conf = mean_i[-log(mean_k p_{y_i,k} + eps) + beta_conf * var_k p_{y_i,k}]
    Uses m views already computed by the consistency training pass — no extra forward passes.
    Training-time only; inference path unchanged.
    """
    B = logits_m.shape[0] // m
    p = F.softmax(logits_m, dim=1)               # [m*B, C]
    p_view = p.view(m, B, -1)                    # [m, B, C]
    y_hard = labels.argmax(1) if labels.ndim == 2 else labels  # [B]
    idx = torch.arange(B, device=logits_m.device)
    p_y = p_view[:, idx, y_hard]                 # [m, B]
    mean_py = p_y.mean(0)                        # [B]
    var_py = p_y.var(0, unbiased=False)          # [B]
    loss = (-torch.log(mean_py + eps) + beta_conf * var_py).mean()
    return loss, mean_py.detach().mean(), var_py.detach().mean()


def train_one_epoch_consistency(accelerator:accelerate.Accelerator, ftmodel, train_data, optimizer, loss_fn, mixup_fn, noise_aug, lbd, eta, margin_config=None):

    m = 2
    device = accelerator.device
    use_margin_aware_loss = bool(_cfg_get(margin_config, "use_margin_aware_loss", False))
    use_proto_loss = bool(_cfg_get(margin_config, "use_proto_loss", True))
    use_margin_loss = bool(_cfg_get(margin_config, "use_margin_loss", True))
    lambda_proto = float(_cfg_get(margin_config, "lambda_proto", 0.1))
    lambda_margin = float(_cfg_get(margin_config, "lambda_margin", 0.1))
    prototype_margin = float(_cfg_get(margin_config, "prototype_margin", 0.2))
    prototype_temperature = float(_cfg_get(margin_config, "prototype_temperature", 0.1))
    prototype_source = str(_cfg_get(margin_config, "prototype_source", "classifier_weight"))
    normalize_features = bool(_cfg_get(margin_config, "normalize_features", True))
    log_geometry_metrics = bool(_cfg_get(margin_config, "log_geometry_metrics", True))
    use_conf_loss = bool(_cfg_get(margin_config, "use_confidence_consistency_loss", False))
    lambda_conf = float(_cfg_get(margin_config, "lambda_conf", 0.05))
    beta_conf_val = float(_cfg_get(margin_config, "beta_conf", 0.1))
    conf_eps = float(_cfg_get(margin_config, "conf_eps", 1e-6))

    avg_acc = []
    avg_loss =  []
    avg_closs = []
    avg_proto_loss = []
    avg_margin_loss = []
    avg_proto_margin = []
    avg_sim_correct = []
    avg_sim_wrong_max = []
    avg_feature_consistency = []
    avg_conf_loss = []
    avg_conf_py_mean = []
    avg_conf_py_var = []

    ftmodel.train()

    for batch in train_data:
        img, y = batch
        img = img.to(device)
        y = y.to(device)

        if mixup_fn is not None:
            img, y = mixup_fn(img, y)

        img_repeated = torch.cat([img for i in range(m)], dim=0)
        y_repeated = torch.cat([y for i in range(m)], dim=0)

        if use_margin_aware_loss:
            logits, features = ftmodel(img_repeated, noise_aug=noise_aug, noise=None, return_features=True)
        else:
            logits = ftmodel(img_repeated, noise_aug=noise_aug, noise=None)
            features = None
        clsloss = loss_fn(logits, y_repeated)
        closs = consistency_loss(logits.chunk(m),  lbd=lbd, eta=eta)

        loss = clsloss + closs

        margin_metrics = None
        if use_margin_aware_loss:
            if prototype_source != "classifier_weight":
                raise NotImplementedError("Only prototype_source='classifier_weight' is currently supported")
            base_ftmodel = accelerator.unwrap_model(ftmodel)
            prototypes = classifier_weight_prototypes(base_ftmodel).to(device=features.device, dtype=features.dtype)
            margin_metrics = margin_aware_prototype_loss(
                features,
                y_repeated,
                prototypes,
                temperature=prototype_temperature,
                margin=prototype_margin,
                normalize_features=normalize_features,
                use_proto_loss=use_proto_loss,
                use_margin_loss=use_margin_loss,
            )
            loss = loss + lambda_proto * margin_metrics["proto_loss"] + lambda_margin * margin_metrics["margin_loss"]

        conf_loss_val = logits.new_zeros(())
        py_mean_diag = logits.new_zeros(())
        py_var_diag = logits.new_zeros(())
        if use_conf_loss and mixup_fn is None:
            conf_loss_val, py_mean_diag, py_var_diag = confidence_consistency_loss(
                logits, y, m=m, beta_conf=beta_conf_val, eps=conf_eps)
            loss = loss + lambda_conf * conf_loss_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if mixup_fn is None:
            acc = (logits.argmax(dim=1) == y_repeated).sum() / img_repeated.shape[0] * 100 
            acc_all = accelerator.gather(acc)
        clsloss_all = accelerator.gather(clsloss)
        closs_all =  accelerator.gather(closs)
        if margin_metrics is not None:
            proto_loss_all = accelerator.gather(margin_metrics["proto_loss"].detach())
            margin_loss_all = accelerator.gather(margin_metrics["margin_loss"].detach())
            proto_margin_all = accelerator.gather(margin_metrics["proto_margin"].detach())
            sim_correct_all = accelerator.gather(margin_metrics["sim_correct"].detach())
            sim_wrong_max_all = accelerator.gather(margin_metrics["sim_wrong_max"].detach())
            feature_consistency_all = accelerator.gather(_feature_consistency_distance(features, m).detach())
        conf_loss_all = accelerator.gather(conf_loss_val.detach())
        conf_py_mean_all = accelerator.gather(py_mean_diag)
        conf_py_var_all = accelerator.gather(py_var_diag)
        if accelerator.process_index == 0:
            if mixup_fn is None :
                avg_acc.append(acc_all.mean().cpu().item())
            avg_loss.append(clsloss_all.mean().cpu().item())
            avg_closs.append(closs_all.mean().cpu().item())
            if margin_metrics is not None:
                avg_proto_loss.append(proto_loss_all.mean().cpu().item())
                avg_margin_loss.append(margin_loss_all.mean().cpu().item())
                if log_geometry_metrics:
                    avg_proto_margin.append(proto_margin_all.mean().cpu().item())
                    avg_sim_correct.append(sim_correct_all.mean().cpu().item())
                    avg_sim_wrong_max.append(sim_wrong_max_all.mean().cpu().item())
                    avg_feature_consistency.append(feature_consistency_all.mean().cpu().item())
            if use_conf_loss and mixup_fn is None:
                avg_conf_loss.append(conf_loss_all.mean().cpu().item())
                avg_conf_py_mean.append(conf_py_mean_all.mean().cpu().item())
                avg_conf_py_var.append(conf_py_var_all.mean().cpu().item())

            logging_dict = {
                "avg_acc": f"{sum(avg_acc)/len(avg_acc)}" if mixup_fn is None else -1,
                "avg_loss":f"{sum(avg_loss)/len(avg_loss)}",
                "avg_closs": sum(avg_closs)/len(avg_closs),
            }
            if margin_metrics is not None:
                logging_dict.update({
                    "avg_proto_loss": sum(avg_proto_loss)/len(avg_proto_loss),
                    "avg_margin_loss": sum(avg_margin_loss)/len(avg_margin_loss),
                })
                if log_geometry_metrics:
                    logging_dict.update({
                        "avg_proto_margin": sum(avg_proto_margin)/len(avg_proto_margin),
                        "avg_sim_correct": sum(avg_sim_correct)/len(avg_sim_correct),
                        "avg_sim_wrong_max": sum(avg_sim_wrong_max)/len(avg_sim_wrong_max),
                        "avg_feature_consistency": sum(avg_feature_consistency)/len(avg_feature_consistency),
                    })
            if len(avg_conf_loss) > 0:
                logging_dict.update({
                    "avg_conf_loss": sum(avg_conf_loss)/len(avg_conf_loss),
                    "avg_conf_py_mean": sum(avg_conf_py_mean)/len(avg_conf_py_mean),
                    "avg_conf_py_var": sum(avg_conf_py_var)/len(avg_conf_py_var),
                })
            print(logging_dict)

    if accelerator.process_index == 0:
        logging_dict = {
            "avg_acc": f"{sum(avg_acc)/len(avg_acc)}" if mixup_fn is None else -1,
            "avg_loss":f"{sum(avg_loss)/len(avg_loss)}",
            "avg_closs": sum(avg_closs)/len(avg_closs),
        }
        if len(avg_proto_loss) > 0:
            logging_dict.update({
                "avg_proto_loss": sum(avg_proto_loss)/len(avg_proto_loss),
                "avg_margin_loss": sum(avg_margin_loss)/len(avg_margin_loss),
            })
            if log_geometry_metrics:
                logging_dict.update({
                    "avg_proto_margin": sum(avg_proto_margin)/len(avg_proto_margin),
                    "avg_sim_correct": sum(avg_sim_correct)/len(avg_sim_correct),
                    "avg_sim_wrong_max": sum(avg_sim_wrong_max)/len(avg_sim_wrong_max),
                    "avg_feature_consistency": sum(avg_feature_consistency)/len(avg_feature_consistency),
                })
        if len(avg_conf_loss) > 0:
            logging_dict.update({
                "avg_conf_loss": sum(avg_conf_loss)/len(avg_conf_loss),
                "avg_conf_py_mean": sum(avg_conf_py_mean)/len(avg_conf_py_mean),
                "avg_conf_py_var": sum(avg_conf_py_var)/len(avg_conf_py_var),
            })

        logging.info(logging_dict)
        return logging_dict
    else:
        return None

@torch.no_grad()
def evaluation(accelerator, ftmodel, test_data, num_test_sample, noise_aug=0, dist_eval=False):
    device = accelerator.device
    test_metric = {
        "correct": 0,
        "total": 0,
    }

    ftmodel.eval()
    for batch in test_data:
        img, y = batch
        img = img.to(device)
        y = y.to(device)
        pred = ftmodel(img, noise_aug=noise_aug)
        pred_label = pred.argmax(dim=1)
        correct = (pred_label == y)

        if dist_eval:
            correct = accelerator.gather(correct)

        test_metric["correct"] += correct.sum().cpu().item()
        test_metric["total"] += correct.shape[0]

    test_avg = test_metric["correct"] / test_metric["total"] * 100
    return test_avg


def train(args):

    dist_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(split_batches=True, kwargs_handlers=[dist_kwargs])
    device = accelerator.device
    utils.setup_for_distributed(accelerator.process_index==0)

    if accelerator.is_main_process:
        workdir =  "/".join( args.path.split("/")[:-1]) if args.task == "consistency" else "/".join( args.path.split("/")[:-2])

        logname = args.name.split("/")[-1].split(".")[0]
        try:
            utils.set_logger(log_level='info', fname=os.path.join(workdir, f'{logname}.log'))
        except PermissionError as ex:
            os.makedirs("./temp", exist_ok=True)
            print("Permission error occurred, save ckpt to ./temp")
            utils.set_logger(log_level='info', fname=os.path.join("./temp", f'{logname}.log'))

        logging.info(f"working dir: {workdir}, logging to {logname}.log")
        logging.info(args)

        dest =  workdir
        def save_ckpt(test_acc, best_acc, prefix="", save=True):
            if test_acc > best_acc:
                if save:
                    torch.save(
                        ftmodel.state_dict(),
                        os.path.join(dest, prefix+args.name)
                    )
                return test_acc
            else:
                return best_acc

    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    model, diffusion = utils.create_model(**args.nnet), utils.create_diffusion(**args.diffusion)
    model.forward = reload_forward(model, layernorm=args.train.layernorm)

    state_dict = None
    if args.path:
        state_dict = torch.load(args.path, map_location="cpu")
        if args.path.split(".")[-1] == "pth":
            # load pre-trained RCM weights
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logging.info(missing)
            logging.info(unexpected)
            assert len(missing) == 0
            logging.info(f"Use pre-trained weights:{args.path}")
            steps = int(args.path.split("/")[-2].split(".ckpt")[0])
        elif args.path.split(".")[-1] == "pt":
            steps = int(args.path.split("/")[-1].split("_")[-2][:-1])*1000 # e.g., linear_300k_debug

    print("Pretrained steps:", steps)
    ema_scale_fn = utils.create_ema_and_scales_fn(
        **args.ema_scale
    )
    _, num_scales, _ = ema_scale_fn(steps)
    logging.info(f"Total discretization steps: {num_scales}")

    class_num = 10 if args.dataset.name == "cifar10" else 1000
    ftmodel = FinetuneModel(model, diffusion, class_num,
                                task=args.task,
                                num_scales = num_scales,
                                sigma_max = args.diffusion.sigma_max,
                                sigma_min = args.diffusion.sigma_min,
                        ).to(device)

    if args.path and args.path.split(".")[-1] == "pt":
        new_state_dict = {}
        for k,v in state_dict.items():
            new_state_dict[k[len("module."):]] = v
        missing, unexpected = ftmodel.load_state_dict(new_state_dict, strict=False)
        logging.info(missing)
        logging.info(unexpected)
        assert len(missing) == 0
        logging.info(f"Use fine-tuned weights:{args.path}")

    if args.optimizer.scale_lr:
        args.optimizer.param.lr = args.optimizer.param.lr * args.dataset.batch_size / 256
        logging.info(f"Scale lr according to batch size, current lr:{args.optimizer.param.lr}")

    ignore = ["linear_head.weight", "linear_head.bias"] if args.task == "consistency" else []
    if args.optimizer.lr_layer_decay > 0:
        param_groups = utils.param_groups_lrd(ftmodel, 
                                            lr=args.optimizer.param.lr,
                                            weight_decay=args.optimizer.param.weight_decay,
                                            no_weight_decay_list=model.no_weight_decay(),
                                            layer_decay=args.optimizer.lr_layer_decay,
                                            ignore = ignore
                                        )
        logging.info("Using layer-wise lr decay")
    else:
        param_groups = [param for n, param in ftmodel.named_parameters() if param.requires_grad and n not in ignore]

    optimizer = utils.get_optimizer(args.optimizer.name, param_groups, **args.optimizer.param)
    lr_scheduler = utils.customized_lr_scheduler(optimizer, **args.lr_scheduler)

    if args.dataset.mixup > 0 or args.dataset.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.dataset.mixup, cutmix_alpha=args.dataset.cutmix, cutmix_minmax=args.dataset.cutmix_minmax,
            prob=args.dataset.mixup_prob, switch_prob=args.dataset.mixup_switch_prob, mode=args.dataset.mixup_mode,
            label_smoothing=args.dataset.label_smoothing, num_classes=10 if args.dataset.name == "cifar10" else 1000)
        logging.info("Using Mixup")
    else:
        mixup_fn = None

    train_dataset = load_data(**args.dataset, mode="train")
    test_dataset = load_data(**args.dataset, mode="test")
    train_data  = DataLoader(
        train_dataset, batch_size=args.dataset.batch_size, num_workers=8,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    test_data  = DataLoader(
        test_dataset, batch_size=args.dataset.batch_size, num_workers=8,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    print("test data num:", len(test_dataset))
    if args.dist_eval:
        ftmodel, train_data, test_data, optimizer = accelerator.prepare(ftmodel, train_data, test_data, optimizer)
    else:
        ftmodel, train_data, optimizer = accelerator.prepare(ftmodel, train_data, optimizer)

    if mixup_fn is not None:
        loss_fn = SoftTargetCrossEntropy()
    else:
        loss_fn = nn.CrossEntropyLoss()

    epochs = args.train.epochs
    if accelerator.process_index == 0:
        pbar = tqdm(total=epochs)
    else:
        pbar = None

    best_acc = [-1, -1, -1, -1, -1]
    for epoch in range(epochs):
        if args.task == "linear" or args.task == "finetune":
            train_logging_dict = train_one_epoch(accelerator, ftmodel, train_data, optimizer, loss_fn, mixup_fn)
        elif args.task == "consistency":
            train_logging_dict = train_one_epoch_consistency(accelerator, ftmodel, train_data, optimizer, loss_fn, mixup_fn=mixup_fn,
                                                             noise_aug=args.noise_aug, lbd=args.lbd, eta=args.eta,
                                                             margin_config=args,
                                                            )

        lr_scheduler.step()
        test_avg = evaluation(accelerator, ftmodel, test_data, len(test_dataset), noise_aug=0, dist_eval=args.dist_eval)
        noisy_test_avg = evaluation(accelerator, ftmodel, test_data, len(test_dataset), noise_aug=0.5, dist_eval=args.dist_eval)
        noisy_test_avg05 = evaluation(accelerator, ftmodel, test_data, len(test_dataset), noise_aug=1.0, dist_eval=args.dist_eval)
        noisy_test_avg10 = evaluation(accelerator, ftmodel, test_data, len(test_dataset), noise_aug=2.0,  dist_eval=args.dist_eval)

        if accelerator.process_index == 0:
            test_results = [test_avg, (noisy_test_avg+noisy_test_avg05+noisy_test_avg10)/3, noisy_test_avg, noisy_test_avg05, noisy_test_avg10]
            prefix = ["clean", "nmean", "n05", "n10", "n20"]
            for i in range(len(best_acc)):
                best_acc[i] = save_ckpt(test_results[i], best_acc[i], prefix=prefix[i], save=args.save_ckpt)

            grad_norm, param_norm = utils._compute_norms(accelerator.unwrap_model(ftmodel).named_parameters())
            logging.info(
                {
                    "epoch": epoch,
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "lr": lr_scheduler.get_last_lr(), 
                    "test_avg": test_avg, "best_test_acc": str(best_acc), 
                    "noisy_test_avg": noisy_test_avg,
                    "noisy_test_avg05": noisy_test_avg05,
                    "noisy_test_avg10": noisy_test_avg10,
                }
            )
            pbar.update(1)

        accelerator.wait_for_everyone()


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("path", None, "path to pre-trained model")
flags.DEFINE_string("task", "linear", "task to perform: [linear/finetune]")
flags.DEFINE_string("name", None, "name of folder for saving ckpts")
flags.DEFINE_float("noise_aug", 0.0, "magnitude of adopting noise augmentation")
flags.DEFINE_float("lbd", 0.0, "enabled in when task==consistency, weight of consistency regularization")
flags.DEFINE_float("eta", 0.0, "enabled in when task==consistency, weight of entropy loss")

flags.DEFINE_boolean("save_ckpt", True, "save ckpt or not")
flags.mark_flags_as_required(["config", "path"])


def main(argv):
    config = FLAGS.config
    config.path = FLAGS.path
    config.task = FLAGS.task
    config.name = FLAGS.name
    config.noise_aug = FLAGS.noise_aug

    # enabled when task == consistency
    config.lbd = FLAGS.lbd
    config.eta = FLAGS.eta

    config.save_ckpt = FLAGS.save_ckpt
    train(config)

if __name__ == "__main__":
    app.run(main)
