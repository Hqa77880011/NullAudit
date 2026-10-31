"""Two-task native DualPrompt/PGP learning, retrieval and checkpoint checks."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .dualprompt import DualPromptPGP, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(59)
    cfg = json.loads(Path("configs/dualprompt_pgp_cifar100.json").read_text())
    cfg.update(classes=4, general_length=2, expert_length=2, pool_size=2,
               memory_threshold=0.99, epochs=1, batch_size=2, learning_rate=0.001)
    cfg["vit"] = dict(image_size=16, patch_size=4, num_layers=5, num_heads=2,
                      hidden_dim=16, mlp_dim=32, num_classes=4)
    rng = np.random.default_rng(59)
    x = torch.tensor(rng.uniform(0, 1, (8, 3, 16, 16)).astype(np.float32))
    model = DualPromptPGP(cfg)
    critical = ["general", "experts", "keys", "head.weight"]
    initial = {key: dict(model.named_parameters())[key].detach().clone() for key in critical}
    value, pull = model(x[:2], task_id=0, training_mask=True)
    loss = F.cross_entropy(value, torch.tensor([0, 1])) - pull
    loss.backward()
    for key in critical:
        gradient = dict(model.named_parameters())[key].grad
        assert gradient is not None and gradient.isfinite().all() and gradient.abs().sum() > 0, key
    torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001).step()
    assert all(not torch.equal(initial[key], dict(model.named_parameters())[key]) for key in critical)
    assert all(p.grad is None for p in model.backbone.parameters())
    model.update_memory(x[:4])
    assert any(b.shape[1] for b in model.protection.values())
    model.begin_task(1)
    torch.testing.assert_close(model.experts[:, :, 1], model.experts[:, :, 0])
    model.zero_grad(set_to_none=True)
    logits, pull = model(x[4:6], task_id=1, training_mask=True)
    assert (model.selected == 1).all()
    (F.cross_entropy(logits, torch.tensor([2, 3])) - pull).backward()
    values_before = model.general.grad[:, 1].clone()
    model.project_gradients()
    torch.testing.assert_close(values_before, model.general.grad[:, 1], rtol=0, atol=0)
    for slot, layer in enumerate(model.general_layers):
        basis = model.protection[str(layer)]
        assert torch.linalg.norm(model.general.grad[slot, 0] @ basis) < 1e-5
    assert model.experts.grad[:, :, 0].abs().sum() == 0
    model.eval()
    for observer in model.observers.values():
        observer.prefix = None
    with torch.no_grad():
        query = F.normalize(model.encode(x[:1]), dim=-1)[0]
        model.keys[0].copy_(query)
        model.keys[1].copy_(-query)
        model(x[:1])
    assert model.selected.item() == 0  # Retrieval does not require an inference task ID.
    data = dict(x=x.numpy(), y=np.repeat(np.arange(4), 2), task=np.repeat([0, 1], 4))
    np.savez_compressed(args.output / "train.npz", **data)
    trained, history = train(cfg, data, args.output / "native")
    heldout = torch.tensor(rng.uniform(0, 1, (4, 3, 16, 16)).astype(np.float32))
    trained.eval()
    with torch.inference_mode():
        expected = trained(heldout)[0]
    buffer = io.BytesIO()
    torch.save(trained.state_dict(), buffer)
    buffer.seek(0)
    restored = DualPromptPGP(cfg).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    buffer.close()
    with torch.inference_mode():
        actual = restored(heldout)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (args.output / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    report = dict(functional_inputs_only=True, tasks=history, critical_updates=critical,
                  native_vit=True, general_and_expert_pool=True, task_mask_and_unlabeled_retrieval=True,
                  key_only_projection=True, exact_reload=True, heldout_accuracy=float((actual.argmax(-1) == torch.arange(4)).float().mean()))
    (args.output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
