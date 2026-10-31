"""CIFAR-format I/O, distinct baseline state and equal-byte functional checks."""

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

from . import cifar, cli, core, frontier
from .comparisons import METHODS, feature_basis, orthogonal_append, state_bytes


def invoke(entry, arguments):
    original = sys.argv
    sys.argv = [entry.__module__, *map(str, arguments)]
    try:
        entry()
    finally:
        sys.argv = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    release = output / "cifar/cifar-100-python"
    release.mkdir(parents=True)
    for name, repeats in (("train", 4), ("test", 2)):
        labels = np.tile(np.arange(4), repeats).tolist()
        payload = {"data": rng.integers(0, 256, (len(labels), 3072), dtype=np.uint8),
                   "fine_labels": labels, "coarse_labels": [v // 2 for v in labels]}
        with (release / name).open("wb") as stream:
            pickle.dump(payload, stream)
    invoke(cifar.main, ["--root", output / "cifar", "--output", output / "cifar.npz",
                       "--ordered-classes", "--classes-per-task", 2])
    packed = dict(np.load(output / "cifar.npz"))
    assert packed["x"].shape == (24, 3, 32, 32)
    assert (packed["split"][-8:] == 2).all()
    assert set(packed["task"]) == {0, 1}
    # Use a spatially pooled view only for this cheap method-state check.
    packed["x"] = packed["x"].mean((2, 3))
    np.savez_compressed(output / "small.npz", **packed)
    cli.prepare(output / "small.npz", output / "prepared", 17)
    cfg = json.loads(Path("configs/default.json").read_text())
    cfg.update(input_dim=3, hidden=8, classes=4, epochs=1, epochs_per_task=1,
               maximum_rank=2, fixed_rank=1, memory_bytes=8192,
               batch_size=2, jacobian_examples=2, jacobian_outputs=2, functional_inputs_only=True)
    data = cli.Arrays(output / "prepared/train.npz")
    for method in METHODS:
        torch.manual_seed(17)
        initial = core.Model(cfg, method)
        before = {k: v.detach().clone() for k, v in initial.named_parameters()}
        cli.train(cfg, output / "prepared", output / method, method)
        payload = torch.load(output / method / "model.pt", weights_only=True)
        current = core.Model(cfg, method).eval()
        current.load_state_dict(payload["state"])
        for name in ("network.1.weight", "network.3.weight"):
            assert not torch.equal(before[name], dict(current.named_parameters())[name]), (method, name)
        restored = core.Model(cfg, method).eval()
        restored.load_state_dict(torch.load(output / method / "model.pt", weights_only=True)["state"])
        sample = torch.tensor(data.data["x"][:2])
        with torch.inference_mode():
            torch.testing.assert_close(current(sample), restored(sample), rtol=1e-6, atol=1e-6)
        history = [json.loads(line) for line in (output / method / "history.jsonl").read_text().splitlines()]
        assert all(np.isfinite(row["train_loss"]) and row["gradient_norm"] > 0 for row in history)
        if method in ("lwf", "prior_rsj"):
            assert history[-1]["regularizer_mean"] > 0, method
        if method == "packnet":
            first = torch.load(output / method / "task_0000.pt", weights_only=True)
            old = first["protection"]["owners"] == 0
            first_vector = core.flatten(tuple(first["state"].values()))
            final_vector = core.flatten(tuple(payload["state"].values()))
            torch.testing.assert_close(first_vector[old], final_vector[old], rtol=0, atol=0)
        report = cli.evaluate(output / method / "model.pt", output / "prepared/test.npz", output / method / "evaluation")
        assert np.isfinite(report["average_task_accuracy"])
        assert report["process_peak_rss_bytes_sampled"] >= report["process_rss_start_bytes"]
        assert state_bytes(payload["protection"]) <= cfg["memory_bytes"] or method == "exact_jacobian"
    q = orthogonal_append(torch.empty(3, 0), torch.tensor([1., 0, 0]), 2)
    q = orthogonal_append(q, torch.tensor([1., 1, 0]), 2)
    gradient = torch.tensor([1., 2, 3])
    projected = gradient - q @ (q.T @ gradient)
    torch.testing.assert_close(projected, torch.tensor([0., 0, 3.]))
    activation = torch.diag(torch.tensor([3., 1, 0]))
    gpm = feature_basis(activation, torch.empty(3, 0), 0.8, 3)
    assert gpm.shape == (3, 1)
    config = output / "config.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    invoke(frontier.main, ["--config", config, "--data", output / "prepared", "--output", output / "frontier",
                          "--budgets", 4096, 8192, "--methods", "nullaudit", "prior_rsj", "ogd", "gpm", "replay"])
    rows = json.loads((output / "frontier/frontier.json").read_text())
    assert len(rows) == 10 and sum(r["status"] == "computed" for r in rows) == 9
    assert [r["method"] for r in rows if r["status"] == "infeasible"] == ["nullaudit"]
    for row in rows:
        if row["status"] == "computed":
            assert row["protected_bytes"] <= row["byte_cap"]
    result = {"functional_inputs_only": True, "methods": METHODS, "cifar_release_format": True,
              "gradient_update_reload": True, "equal_byte_configurations": len(rows),
              "measured_peak_rss_separate_from_protection_state": True}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
