"""PGP, VPT-NS and DMNSP projection mechanisms on an actual CLIP backbone.

The backbone is configured locally and may load local pretrained weights. Small
random CLIP configurations are functional fixtures, not pretrained comparisons.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .comparisons import SampledRSS, state_bytes


class ObservedAttention(nn.Module):
    """CLIP attention with either true key/value prefixes or VPT observations."""

    def __init__(self, original, method, prompt_length):
        super().__init__()
        self.original, self.method, self.length = original, method, prompt_length
        dim = original.embed_dim
        if method == "pgp":
            self.prefix = nn.Parameter(torch.randn(2, prompt_length, dim) * 0.02)
        self.capture = {}

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        if attention_mask is not None or causal_attention_mask is not None:
            raise ValueError("This wrapper is for unmasked CLIP vision attention only")
        layer = self.original
        batch, tokens, width = hidden_states.shape
        q = layer.q_proj(hidden_states)
        k, v = layer.k_proj(hidden_states), layer.v_proj(hidden_states)
        if self.method == "pgp":
            k = torch.cat([self.prefix[0][None].expand(batch, -1, -1), k], 1)
            v = torch.cat([self.prefix[1][None].expand(batch, -1, -1), v], 1)
        shape = lambda value: value.reshape(batch, -1, layer.num_heads, layer.head_dim).transpose(1, 2)
        query, key, value = shape(q), shape(k), shape(v)
        weights = ((query * layer.scale) @ key.transpose(-1, -2)).softmax(-1)
        result = layer.out_proj((weights @ value).transpose(1, 2).reshape(batch, tokens, width))
        if self.method == "pgp":
            self.capture = {"query": q.detach().reshape(-1, width)}
        else:
            original_tokens = tokens - self.length
            # VPT-NS's two consistency matrices: Q W_k and original-to-prompt attention.
            qwk = (q[:, :original_tokens] @ layer.k_proj.weight).detach().reshape(-1, width)
            attention = weights[:, :, :original_tokens, -self.length:].detach().reshape(-1, self.length)
            self.capture = {"right": qwk, "left": attention}
        return result, weights if output_attentions else None


class Adapter(nn.Module):
    """Residual bottleneck adapter; up projection starts at zero."""

    def __init__(self, width, bottleneck):
        super().__init__()
        self.down = nn.Linear(width, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.activation = None

    def forward(self, value):
        self.activation = value.detach().reshape(-1, value.shape[-1])
        return self.up(F.gelu(self.down(value)))


def null_projector(covariance, multiplier=1.1, eta=1.0):
    """VPT-NS times-threshold null space and Frobenius-normalized soft projector."""
    _, singular, vh = torch.linalg.svd(covariance, full_matrices=True)
    selected = singular <= singular[-1] * multiplier + 1e-10
    null = vh[selected]
    matrix = null.T @ null
    matrix = matrix / matrix.norm().clamp_min(1e-12)
    return eta * matrix + (1 - eta) * torch.eye(len(matrix), device=matrix.device)


class PromptLearner(nn.Module):
    """Frozen CLIP + vision prefixes/deep prompts or two-modality adapters."""

    def __init__(self, config, tokens, method, prompt_length=2, backbone_state=None):
        super().__init__()
        from transformers import CLIPConfig, CLIPModel
        self.method, self.length, self.configuration = method, prompt_length, config
        self.clip = CLIPModel(CLIPConfig.from_dict(config)).requires_grad_(False)
        if backbone_state:
            self.clip.load_state_dict(torch.load(backbone_state, map_location="cpu", weights_only=True))
        self.register_buffer("text_tokens", torch.as_tensor(tokens, dtype=torch.long))
        self.protection = {}
        self.attention = []
        self.visual_adapters, self.text_adapters = nn.ModuleList(), nn.ModuleList()
        self.prompts = nn.ParameterList()
        self.handles = []
        vision_layers = self.clip.vision_model.encoder.layers
        if method in ("pgp", "vpt_ns"):
            for index, layer in enumerate(vision_layers):
                observed = ObservedAttention(layer.self_attn, method, prompt_length)
                layer.self_attn = observed
                self.attention.append(observed)
                if method == "vpt_ns":
                    prompt = nn.Parameter(torch.randn(prompt_length, self.clip.config.vision_config.hidden_size) * 0.02)
                    self.prompts.append(prompt)
                    def insert(module, args, layer_index=index):
                        hidden = args[0]
                        if layer_index:
                            hidden = hidden[:, :-self.length]
                        value = self.prompts[layer_index][None].expand(len(hidden), -1, -1)
                        return (torch.cat([hidden, value], 1), *args[1:])
                    self.handles.append(layer.register_forward_pre_hook(insert))
        elif method == "dmnsp":
            for layers, adapters in ((vision_layers, self.visual_adapters),
                                     (self.clip.text_model.encoder.layers, self.text_adapters)):
                for layer in layers:
                    width = layer.embed_dim
                    adapter = Adapter(width, max(2, width // 2))
                    adapters.append(adapter)
                    def adapt(module, inputs, result, branch=adapter):
                        value = result[0]
                        return (value + branch(value), *result[1:])
                    self.handles.append(layer.register_forward_hook(adapt))
        else:
            raise ValueError(method)

    def forward(self, images):
        """Use actual text-token class prototypes and normalized CLIP similarities."""
        images = F.interpolate(images, (self.clip.config.vision_config.image_size,) * 2, mode="bilinear", align_corners=False)
        mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        std = images.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        vision = self.clip.vision_model(pixel_values=(images - mean) / std).pooler_output
        text = self.clip.text_model(input_ids=self.text_tokens).pooler_output
        vision = F.normalize(self.clip.visual_projection(vision), dim=-1)
        text = F.normalize(self.clip.text_projection(text), dim=-1)
        return self.clip.logit_scale.exp() * vision @ text.T

    def update_memory(self, images):
        """Collect only the completed current task's unlabeled features."""
        self.eval()
        with torch.no_grad():
            self(images)
        for index, attention in enumerate(self.attention):
            if self.method == "pgp":
                feature = attention.capture["query"]
                old = self.protection.get(str(index), feature.new_zeros(feature.shape[1], feature.shape[1]))
                covariance = old + feature.T @ feature / len(feature)
                self.protection[str(index)] = covariance
            else:
                for side, feature in attention.capture.items():
                    key = f"{index}-{side}"
                    old = self.protection.get(key, feature.new_zeros(feature.shape[1], feature.shape[1]))
                    self.protection[key] = old + feature.T @ feature / len(feature)
        if self.method == "dmnsp":
            for index, adapter in enumerate(self.visual_adapters):
                feature = adapter.activation
                u, _, _ = torch.linalg.svd(feature.T @ feature, full_matrices=True)
                null = u[:, 1:]
                key = str(index)
                self.protection[key] = torch.cat([self.protection[key], null], 1) if key in self.protection else null

    def project_gradients(self):
        """PGP right-projection and source-defined DMNSP visual-only multi-layer updates."""
        if not self.protection:
            return
        if self.method == "pgp":
            for index, attention in enumerate(self.attention):
                covariance = self.protection[str(index)]
                u, singular, _ = torch.linalg.svd(covariance)
                energy = singular.square().cumsum(0) / singular.square().sum().clamp_min(1e-12)
                rank = min(len(singular) - 1, max(1, int((energy < 0.95).sum())))
                basis = u[:, :rank]
                gradient = attention.prefix.grad
                gradient.sub_((gradient @ basis) @ basis.T)
        elif self.method == "dmnsp":
            # Official DMNSP repeatedly projects through each stored layer's null
            # basis and uses exp(-top-10%-similarity)*30 per source/target layer.
            for adapter in self.visual_adapters:
                feature = adapter.activation
                leading = torch.linalg.svd(feature.T @ feature).U[:, 0]
                for basis in self.protection.values():
                    similarity = F.normalize(basis, dim=0).T @ F.normalize(leading, dim=0)
                    top = similarity.topk(max(1, int(len(similarity) * 0.1))).values.mean()
                    coefficient = torch.exp(-top) * 30
                    down, up = adapter.down.weight.grad, adapter.up.weight.grad
                    down.copy_(((down @ basis) @ basis.T) * coefficient)
                    up.copy_((basis @ (basis.T @ up)) * coefficient)

    def project_updates(self, previous):
        """VPT-NS applies two-sided projection to the optimizer update itself."""
        if self.method != "vpt_ns" or not self.protection:
            return
        with torch.no_grad():
            for index, parameter in enumerate(self.prompts):
                left = null_projector(self.protection[f"{index}-left"])
                right = null_projector(self.protection[f"{index}-right"])
                delta = previous[index] - parameter
                parameter.copy_(previous[index] - left @ delta @ right)


def train(config, tokens, method, data, output, epochs=1, seed=17, backbone_state=None):
    """Train the independent backbone path on a declared sequential NPZ stream."""
    torch.manual_seed(seed)
    tokens = torch.as_tensor(tokens, dtype=torch.long)
    model = PromptLearner(config, tokens, method, backbone_state=backbone_state)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with SampledRSS() as memory:
        for task_id in sorted(np.unique(data["task"]).tolist()):
            selected = np.flatnonzero(data["task"] == task_id)
            x, y = torch.tensor(data["x"][selected]), torch.tensor(data["y"][selected]).long()
            loss_values, updates = [], []
            for _ in range(epochs):
                order = rng.permutation(len(x))
                for start in range(0, len(x), 2):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    batch = order[start:start + 2]
                    loss = F.cross_entropy(model(x[batch]), y[batch])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(method)
                    loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                    model.project_gradients()
                    previous = [p.detach().clone() for p in parameters]
                    old_prompts = [p.detach().clone() for p in model.prompts]
                    optimizer.step()
                    model.project_updates(old_prompts)
                    updates.append(sum(float((a - p).norm()) for a, p in zip(previous, parameters)))
                    loss_values.append(float(loss.detach()))
            model.update_memory(x)
            row = dict(task=task_id, loss=float(np.mean(loss_values)), update_norm=float(np.mean(updates)),
                       protection_bytes=state_bytes(model.protection),
                       trainable_method_bytes=sum(p.numel() * p.element_size() for p in parameters))
            rows.append(row)
            torch.save(dict(state=model.state_dict(), protection=model.protection, config=config,
                            tokens=tokens, method=method), output / f"task_{task_id:04d}.pt")
            del x, y
    torch.save(dict(state=model.state_dict(), protection=model.protection, config=config,
                    tokens=tokens, method=method), output / "model.pt")
    report = dict(method=method, tasks=rows, peak_process_rss_bytes_sampled=memory.peak,
                  random_backbone=backbone_state is None, seed=seed)
    (output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Local Hugging Face CLIPConfig JSON")
    parser.add_argument("--text-tokens", required=True, help="C x sequence-length integer NPY, ordered by class id")
    parser.add_argument("--backbone-state", help="Optional local CLIP state_dict; no automatic weights download")
    parser.add_argument("--data", required=True, help="Prepared train NPZ with x/y/task")
    parser.add_argument("--method", choices=("pgp", "vpt_ns", "dmnsp"), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tokens = np.load(args.text_tokens, allow_pickle=False)
    data = dict(np.load(args.data, allow_pickle=False))
    train(config, tokens, args.method, data, args.output, args.epochs, args.seed, args.backbone_state)


if __name__ == "__main__":
    main()
