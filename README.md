# NullAudit

Implementation of **Residual-guided adaptive function-space projection for replay-free continual learning**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The CIFAR adapter downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), reserves 10% of training examples for validation, preserves the test set and records sequential class-task IDs.

```bash
python -m nullaudit.cifar --output data/raw.npz --seed 17
python -m nullaudit prepare --source data/raw.npz --output data/prepared --seed 17
```

## Training

Use the same `configs/default.json` throughout. Its seed is `17`; preserve the prepared splits and use separate output directories for new seeds.

```bash
python -m nullaudit train --config configs/default.json --data data/prepared --output runs/main
```

## Evaluation

```bash
python -m nullaudit evaluate --checkpoint runs/main/nullaudit/model.pt --data data/prepared/test.npz --output results/nullaudit
```

## Comparisons and ablations

`suite` trains the implemented methods on the same prepared split. Use `--methods` to select a subset, then evaluate each method checkpoint with the evaluation command above.

```bash
python -m nullaudit suite --config configs/default.json --data data/prepared --output runs/comparison
```

## Outputs

Each training method directory contains `model.pt` and `history.jsonl`. Evaluation writes `predictions.npz` and `metrics.json`; plots read the supplied run directories.

```bash
python -m nullaudit plot --inputs runs/main/nullaudit results/nullaudit --output figures
```

## Implemented scope and memory conventions

This is an implementation written from the manuscript, separate from its original experiment artifacts. The model is a two-layer MLP. Tasks arrive in increasing declared `task` order. The learner moves only the current task's examples to its compute device, then drops those tensors at the task boundary; its protected state consists of a streaming Frequent-Directions sketch/basis (or EWC diagonal/anchor), never replay examples. The offline NPZ dataset remains the input provider in host memory and is excluded from method-state memory.

| Paper module / endpoint | Entry | Command |
| --- | --- | --- |
| Streaming Jacobian FD sketch and residual energy | `core.FrequentDirections` | train command above |
| Residual-based rank selection and gradient-retention gate | `core.fit` | same train command |
| Average task accuracy, maximum-past forgetting, new-task accuracy and accuracy matrix | `core.continual_metrics`, `core.metrics` | evaluate command above |
| Stored tensor bytes, retention, rank, update time and predicted first-order quantity | `core.fit` history / config | same train/evaluate commands |
| Evaluator-only realized drift versus rank/step size | `audit` | command below |
| Fixed-rank, no-gate, fine-tuning and EWC controls; plots | `core.METHODS`, `cli.plot` | suite / plot commands above |

```bash
python -m nullaudit.audit --checkpoint runs/main/nullaudit/model.pt --sketch runs/main/nullaudit/sketch_task_0009.pt --old-data data/prepared/validation.npz --new-data data/prepared/train.npz --ranks 0 4 8 --step-sizes 0.001 0.01 --output results/drift_audit.json
```

Select a sketch filename and ranks that actually exist in the run. `old-data` and `new-data` must be evaluator exports for the declared old/new tasks; they are never consumed by the training controller. The audit restores the same original model before each update and reports actual old-logit drift and the residual-derived first-order quantity, not a verified global bound. It must be repeated at multiple step sizes to inspect curvature.

`maximum_rank` and `memory_bytes` allocate sketch rows using an eight-buffer float32 working allowance. `protected_bytes` records retained sketch/basis/singular-value tensors, the discarded-energy scalar and, for EWC, the diagonal/anchor tensors. It is not an OS peak-memory measurement or an exact SVD workspace cap. The model, input provider, gradients and Python runtime are excluded; RSS is separately reported during evaluation. Dense SVD workspace, task metadata and full system peak memory have not been profiled. The default 64 MiB is therefore a protected-state/analytic-working-budget setting, not an empirically audited total process limit.

Accuracy metrics are fractions. Forgetting is maximum accuracy after learning each old task minus final accuracy, averaged over old tasks. New-task accuracy averages the diagonal of the task accuracy matrix. ECE uses integer-indexed bins so confidences at floating-point boundaries are counted exactly once. The CIFAR adapter preserves released test images; no test labels tune ranks or gates.

The exact Split CIFAR backbone and prior sketched-Jacobian replication, OGD/GPM/LwF/PackNet/prompt methods/replay/exact-Jacobian oracle, equal-byte frontier, multi-seed study and full residual-to-realized-drift calibration are not implemented or claimed by the included controls.

## Functional verification

```bash
python -m nullaudit.verify --output runs/check
```

Actually run on 2026-11-01 with Python 3.13.5, torch 2.7.1+cu118 and numpy 2.1.3, CPU, seed 17, one torch thread. The check prepares a generated two-task NPZ, runs all five methods for one epoch per task, checks finite loss/gradients, model updates, exact restored-checkpoint logits and the retained-state budget, then runs a separate four-row rank/step evaluator audit and plots. It checks FD-basis orthonormality and the residual upper bound against an explicit small Jacobian. A hand accuracy matrix [[0.8, 0.2], [0.5, 0.7]] yields final average 0.6, forgetting 0.3 and new-task accuracy 0.75; the ECE boundary fixture gives 0.25. Generated examples validate functionality, not paper performance. No full CIFAR experiment was run.

Use process-local `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MKL_THREADING_LAYER=SEQUENTIAL` on this host to select sequential MKL when starting Python.