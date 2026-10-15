# NullAudit

Implementation of **Residual-guided adaptive function-space projection for replay-free continual learning**.

Streams output Jacobian rows into Frequent Directions, selects protection rank from the residual bound and relaxes it when projected gradients retain too little new-task signal.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/NullAudit.git
cd NullAudit
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Use Split CIFAR-100 with paired task-order seeds, or a controlled Jacobian-spectrum dataset. Training examples are accessed only while their current task is active.

x is image/features, y is fine class, task the ordered task ID and group the original source object. Test task IDs share the registered sequence. All labels use one persistent output head.

```bash
python -m nullaudit prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m nullaudit train --config configs/default.json --data data/prepared --output runs/main
python -m nullaudit suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. nullaudit, fixed_rank, no_gate, fine_tune and ewc.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m nullaudit evaluate --checkpoint runs/main/nullaudit/model.pt --data data/prepared/test.npz --output results/nullaudit
python -m nullaudit plot --inputs runs/main/nullaudit results/nullaudit --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Average task accuracy, forgetting, new-task accuracy, task accuracy matrix, protected memory, gradient retention, mean rank and measured training time.

## Implementation conventions

The custom driver uses SGD without momentum so the actual update is the projected gradient. It stores task checkpoints for evaluator-only test curves. The sketch uses past parameter-point Jacobians without retaining raw examples; first-order drift estimates therefore retain that local linearization boundary. Memory_bytes budgets sketch/basis working storage, separate from model and dataset memory.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m nullaudit.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
```

## CIFAR-100 preparation

```bash
python -m nullaudit.cifar --output data/raw.npz
```

This downloads the author dataset through TorchVision, reserves 10% of its training images for validation, and preserves the original test set. It writes fine/coarse labels, source IDs and task metadata. Paired/audited transformation fields and ontology constraints must be added from the registered pair/ontology protocol before using those objectives.

## Evaluator-only first-order audit

```bash
python -m nullaudit.audit --checkpoint runs/main/nullaudit/model.pt --sketch runs/main/nullaudit/sketch_task_0009.pt --old-data data/old_audit.npz --new-data data/new_audit.npz --ranks 0 4 8 --step-sizes 0.001 0.01 0.1 --output results/drift_audit.json
```

Use the sketch from the same final task as the model checkpoint. The audit inputs stay with the evaluator and never enter sketch training. It compares actual old-logit changes against residual-derived quantities across ranks and step sizes.

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.
