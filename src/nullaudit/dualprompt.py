"""Native ViT DualPrompt pool and PGP key-prefix projection (no weight downloads)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import VisionTransformer


class PrefixAttention(nn.Module):
    """Insert selected prefixes after the frozen attention key/value projections."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.prefix = None
        self.query_features = None

    def forward(self, query, key, value, need_weights=False, **kwargs):
        layer = self.original
        q, k, v = F.linear(query, layer.in_proj_weight, layer.in_proj_bias).chunk(3, -1)
        self.query_features = q.detach().reshape(-1, q.shape[-1])
        if self.prefix is not None:
            k = torch.cat([self.prefix[:, 0], k], 1)
            v = torch.cat([self.prefix[:, 1], v], 1)
        batch, length, width = q.shape
        heads = layer.num_heads
        def split(tensor):
            return tensor.reshape(batch, -1, heads, width // heads).transpose(1, 2)
        weights = (split(q) @ split(k).transpose(-1, -2) / (width // heads) ** 0.5).softmax(-1)
        output = (weights @ split(v)).transpose(1, 2).reshape(batch, length, width)
        return layer.out_proj(output), weights.mean(1) if need_weights else None


class DualPromptPGP(nn.Module):
    """Frozen ViT, general prefixes, retrieved expert pool, keys and class head."""

    def __init__(self, config, backbone_state=None):
        super().__init__()
        self.config = config
        vit = config["vit"]
        self.backbone = VisionTransformer(**vit).requires_grad_(False)
        if backbone_state:
            self.backbone.load_state_dict(torch.load(backbone_state, map_location="cpu", weights_only=True))
        width = vit["hidden_dim"]
        self.general_layers = config["general_layers"]
        self.expert_layers = config["expert_layers"]
        if set(self.general_layers) & set(self.expert_layers) or max(self.general_layers + self.expert_layers) >= vit["num_layers"]:
            raise ValueError("General/expert layers must be disjoint valid ViT layer indices")
        self.general = nn.Parameter(torch.empty(len(self.general_layers), 2, config["general_length"], width))
        self.experts = nn.Parameter(torch.empty(len(self.expert_layers), 2, config["pool_size"], config["expert_length"], width))
        self.keys = nn.Parameter(torch.empty(config["pool_size"], width))
        for parameter in (self.general, self.experts, self.keys):
            nn.init.uniform_(parameter, -1, 1)
        self.head = nn.Linear(width, config["classes"])
        self.observers = {}
        for index in self.general_layers + self.expert_layers:
            layer = self.backbone.encoder.layers[index]
            observer = PrefixAttention(layer.self_attention)
            layer.self_attention = observer
            self.observers[index] = observer
        self.protection = {}
        self.selected = None

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, images):
        images = F.interpolate(images, (self.config["vit"]["image_size"],) * 2, mode="bilinear", align_corners=False)
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tokens = self.backbone._process_input((images - mean) / std)
        tokens = torch.cat([self.backbone.class_token.expand(len(images), -1, -1), tokens], 1)
        return self.backbone.encoder(tokens)[:, 0]

    def forward(self, images, task_id=None, training_mask=False):
        for observer in self.observers.values():
            observer.prefix = None
        with torch.no_grad():
            query = F.normalize(self.encode(images), dim=-1)
        similarity = query @ F.normalize(self.keys, dim=-1).T
        top_k = self.config.get("top_k", 1)
        selected = similarity.topk(top_k, dim=1).indices
        if self.config.get("batchwise_prompt", True):
            counts = torch.bincount(selected.flatten(), minlength=len(self.keys))
            selected = counts.topk(top_k).indices[None].expand(len(images), -1)
        if training_mask:
            start = int(task_id) * top_k
            if start + top_k > len(self.keys):
                raise ValueError("Task-specific expert range exceeds the registered pool")
            selected = torch.arange(start, start + top_k, device=images.device)[None].expand(len(images), -1)
        self.selected = selected
        for slot, index in enumerate(self.general_layers):
            self.observers[index].prefix = self.general[slot][None].expand(len(images), -1, -1, -1)
        for slot, index in enumerate(self.expert_layers):
            chosen = self.experts[slot][:, selected].permute(1, 0, 2, 3, 4)
            self.observers[index].prefix = chosen.reshape(len(images), 2, top_k * self.config["expert_length"], -1)
        pull = (F.normalize(self.keys, dim=-1)[selected] * query[:, None]).sum() / len(images)
        return self.head(self.encode(images)), pull

    def begin_task(self, task_id):
        """Copy the previous expert range and reset optimization outside this method."""
        if task_id and self.config.get("shared_prompt_pool", True):
            k = self.config.get("top_k", 1)
            with torch.no_grad():
                self.experts[:, :, task_id * k:(task_id + 1) * k].copy_(
                    self.experts[:, :, (task_id - 1) * k:task_id * k])

    @torch.no_grad()
    def update_memory(self, images):
        """Incremental feature-space SVD energy memory, threshold 0.50 in the source."""
        self.eval()
        self(images)
        threshold = self.config.get("memory_threshold", 0.5)
        for index, observer in self.observers.items():
            features = observer.query_features
            covariance = features.T @ features / len(features)
            old = self.protection.get(str(index), covariance[:, :0])
            total = torch.linalg.svdvals(covariance).square().sum().clamp_min(1e-12)
            residual = covariance - old @ (old.T @ covariance)
            u, singular, _ = torch.linalg.svd(residual)
            explained = 1 - singular.square().sum() / total
            energy = explained + singular.square().cumsum(0) / total
            rank = int((energy < threshold).sum()) if old.shape[1] == 0 else int((energy < threshold).sum()) + int(explained < threshold)
            rank = min(rank, u.shape[1] - old.shape[1])
            basis = torch.cat([old, u[:, :rank]], 1)
            self.protection[str(index)] = torch.linalg.qr(basis, mode="reduced").Q

    def project_gradients(self):
        """Project key-prefix gradients only, matching the official DualPrompt-PGP path."""
        for slot, index in enumerate(self.general_layers):
            if str(index) in self.protection and self.general.grad is not None:
                basis = self.protection[str(index)]
                gradient = self.general.grad[slot, 0]
                gradient.sub_((gradient @ basis) @ basis.T)
        for slot, index in enumerate(self.expert_layers):
            if str(index) in self.protection and self.experts.grad is not None:
                basis = self.protection[str(index)]
                for expert in self.selected.unique():
                    gradient = self.experts.grad[slot, 0, expert]
                    gradient.sub_((gradient @ basis) @ basis.T)


def train(config, data, output, backbone_state=None):
    torch.manual_seed(config.get("seed", 59))
    model = DualPromptPGP(config, backbone_state)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = []
    for task in sorted(np.unique(data["task"]).tolist()):
        model.begin_task(int(task))
        selected = data["task"] == task
        x, y = torch.tensor(data["x"][selected]), torch.tensor(data["y"][selected]).long()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.get("learning_rate", 0.03))
        losses = []
        for _ in range(config.get("epochs", 20)):
            for batch in torch.randperm(len(x)).split(config.get("batch_size", 24)):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits, pull = model(x[batch], task_id=task, training_mask=True)
                allowed = y.unique()
                logits = logits.masked_fill(~torch.isin(torch.arange(logits.shape[1]), allowed)[None], float("-inf"))
                # Masked classes must not receive label smoothing mass.
                value = F.cross_entropy(logits[:, allowed], torch.searchsorted(allowed, y[batch]), label_smoothing=config.get("label_smoothing", 0.1))
                value = value - config.get("pull_coefficient", 1.0) * pull
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite DualPrompt loss")
                value.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.get("clip_gradient", 1.0))
                if task:
                    model.project_gradients()
                optimizer.step()
                losses.append(float(value.detach()))
        model.update_memory(x)
        history.append(dict(task=int(task), loss=float(np.mean(losses)), memory_ranks={k: v.shape[1] for k, v in model.protection.items()}))
    torch.save(dict(config=config, state=model.state_dict(), protection=model.protection), output / "model.pt")
    (output / "training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-state", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a saved DualPrompt model; output is NPZ")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = dict(np.load(args.data, allow_pickle=False))
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = DualPromptPGP(saved["config"]).eval()
        model.load_state_dict(saved["state"])
        with torch.inference_mode():
            logits, _ = model(torch.tensor(data["x"]))
        np.savez_compressed(args.output, logits=logits.numpy(), prediction=logits.argmax(-1).numpy(), selected_experts=model.selected.numpy())
    else:
        _, history = train(config, data, args.output, args.backbone_state)
        print(json.dumps(history[-1]))


if __name__ == "__main__":
    main()
