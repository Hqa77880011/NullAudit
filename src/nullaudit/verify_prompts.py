"""Actual tiny CLIP vision/text, prompt and adapter projection verification."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

from .core import continual_metrics
from .prompts import PromptLearner, null_projector, train
from .vision import classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                              num_attention_heads=2, image_size=16, patch_size=4)
    text = CLIPTextConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
                         eos_token_id=63, bos_token_id=1, pad_token_id=0)
    config = CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=16).to_dict()
    tokens = torch.tensor([[1, i + 5, 63, 0] for i in range(4)])
    (args.output / "clip.json").write_text(json.dumps(config), encoding="utf-8")
    np.save(args.output / "class_tokens.npy", tokens.numpy())
    rng = np.random.default_rng(17)
    data = dict(x=rng.uniform(0, 1, (16, 3, 16, 16)).astype(np.float32),
                y=np.tile(np.arange(4), 4), task=np.tile(np.arange(4), 4) // 2)
    np.savez_compressed(args.output / "train.npz", **data)
    held_out = dict(x=rng.uniform(0, 1, (8, 3, 16, 16)).astype(np.float32),
                    y=np.tile(np.arange(4), 2), task=np.tile(np.arange(4), 2) // 2)
    np.savez_compressed(args.output / "heldout.npz", **held_out)
    x = torch.tensor(held_out["x"][:4])
    reports = []
    for method in ("pgp", "vpt_ns", "dmnsp"):
        torch.manual_seed(17)
        initial = PromptLearner(config, tokens, method)
        before = {name: p.detach().clone() for name, p in initial.named_parameters() if p.requires_grad}
        model, report = train(config, tokens, method, data, args.output / method, epochs=2)
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        assert report["tasks"][-1]["update_norm"] > 0
        if method == "dmnsp":
            keys = [name for name in trainable if "adapters" in name and name.endswith("weight")]
            assert any(name.startswith("visual") for name in keys) and any(name.startswith("text") for name in keys)
        else:
            keys = [name for name in trainable if "prefix" in name or name.startswith("prompts")]
        assert keys and all(not torch.equal(before[name], trainable[name]) for name in keys)
        assert all(trainable[name].grad is not None and torch.isfinite(trainable[name].grad).all()
                   and trainable[name].grad.abs().sum() > 0 for name in keys)
        model.eval()
        with torch.inference_mode():
            expected = model(x)
        saved = torch.load(args.output / method / "model.pt", weights_only=True)
        restored = PromptLearner(saved["config"], saved["tokens"], saved["method"]).eval()
        restored.load_state_dict(saved["state"])
        restored.protection = saved["protection"]
        with torch.inference_mode():
            actual = restored(x)
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-6)
        assert torch.isfinite(actual).all()
        matrix = []
        for checkpoint in sorted((args.output / method).glob("task_*.pt")):
            restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state"])
            with torch.inference_mode():
                pred = restored(torch.tensor(held_out["x"])).argmax(-1).numpy()
            matrix.append([float(np.mean(pred[held_out["task"] == task] == held_out["y"][held_out["task"] == task])) for task in (0, 1)])
        report.update(continual_metrics(matrix), inference=classification_metrics(actual.numpy(), held_out["y"][:4]),
                      critical_parameter_updates=keys, reload_max_abs=float((actual - expected).abs().max()))
        (args.output / method / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
    p = null_projector(torch.diag(torch.tensor([4., 1., 0.])))
    torch.testing.assert_close(p, torch.diag(torch.tensor([0., 0., 1.])))
    assert len(reports) == 3
    print(json.dumps({"functional_inputs_only": True, "backbone": "HuggingFace CLIP with actual text tokens",
                      "methods": [r["method"] for r in reports], "reload": "1e-6", "critical_updates": True}))


if __name__ == "__main__":
    main()
