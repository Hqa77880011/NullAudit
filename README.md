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
python -m nullaudit.cifar --root data/cifar --download --output data/raw.npz --seed 17
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

The default model is a two-layer MLP. Tasks follow declared `task` order. Training transfers the current task to the compute device and drops its tensors at task boundaries. Protected state consists of a Frequent-Directions sketch/basis or EWC diagonal/anchor. The host-memory NPZ input provider is counted separately from method state.

```bash
python -m nullaudit.audit --checkpoint runs/main/nullaudit/model.pt --sketch runs/main/nullaudit/sketch_task_0009.pt --old-data data/prepared/validation.npz --new-data data/prepared/train.npz --ranks 0 4 8 --step-sizes 0.001 0.01 --output results/drift_audit.json
```

Select a sketch filename and ranks that actually exist in the run. `old-data` and `new-data` must be evaluator exports for the declared old/new tasks; they are never consumed by the training controller. The audit restores the same original model before each update and reports actual old-logit drift and the residual-derived first-order quantity, not a verified global bound. It must be repeated at multiple step sizes to inspect curvature.

`maximum_rank` and `memory_bytes` allocate sketch rows with an eight-buffer float32 working allowance. `protected_bytes` counts retained sketch/basis/singular-value tensors, discarded energy and EWC diagonal/anchor tensors. Model weights, input arrays, gradients, SVD workspace and Python runtime are outside this payload budget; evaluation reports RSS separately.

Accuracy metrics are fractions. Forgetting is maximum accuracy after learning each old task minus final accuracy, averaged over old tasks. New-task accuracy averages the diagonal of the task accuracy matrix. ECE uses integer-indexed bins so confidences at floating-point boundaries are counted exactly once. The CIFAR adapter preserves released test images; no test labels tune ranks or gates.

Common-backbone comparisons share the MLP, class order, CE, batch schedule and seed. `prior_rsj` uses Gaussian `S J` rows and `lambda/2 * ||K(theta-anchor)||^2`, recentering its accumulated quadratic at the latest task solution. Its Jacobian sampling matches NullAudit; `fixed_rank` uses hard projection.

## Source-defined comparison methods

Baseline equations are independently implemented; source references are listed below.

| Method | Mechanism | Primary source / official implementation |
|---|---|---|
| `prior_rsj` | Gaussian task-Jacobian sketch, recentered quadratic regularization | [Heckel 2022, Sec.3](https://proceedings.mlr.press/v151/heckel22a/heckel22a.pdf) |
| `ogd` | Ground-truth-logit gradients, incremental Gram-Schmidt and projection | [OGD-GTL](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf) |
| `gpm` | Activation residual SVD and layer-input gradient projection | [official GPM](https://github.com/sahagobinda/GPM/blob/main/main_pmnist.py) |
| `lwf` | Frozen previous network, old-class temperature distillation on current images | [official LwF](https://github.com/lizhitwo/LearningWithoutForgetting) |
| `packnet` | Magnitude pruning, persistent ownership, old-weight freezing and mask refitting | [official PackNet](https://github.com/arunmallya/packnet/blob/master/src/prune.py) |
| `replay` | Explicit bounded old-example buffer mixed into current batches | Declared uniform retained-plus-current replay control |
| `exact_jacobian` | Every sampled-example logit gradient, unbounded basis | Unbounded diagnostic oracle |

GPM uses MLP activations and freezes biases after task one. PackNet stores parameter ownership, prunes by magnitude (default fraction 0.5), refits once, and uses requested-task masks. Replay uniformly resamples retained-plus-current records. LwF counts its frozen teacher/class metadata as protected state. The exact-Jacobian oracle uses an unbounded basis outside the state cap.

## Equal-byte frontier and measured memory

```bash
python -m nullaudit.frontier --config configs/default.json --data data/prepared --methods nullaudit prior_rsj ogd gpm lwf packnet replay --budgets 16777216 33554432 67108864 --output runs/equal-byte
```

`frontier.csv/json/png` use identical **byte caps** and plot actual retained bytes; unused capacity is not padded. Infeasible configurations remain marked `infeasible`. New controls' retained payload counts tensors, deduplicated aliases and serialized primitive metadata. Python object/allocator overhead, the common model, provider arrays and optimizer are outside this payload definition. `process_peak_rss_bytes_sampled` is measured separately every 2 ms during training and includes those process allocations; it may miss shorter spikes. The existing eight-buffer sketch allowance remains analytic, not an empirical RSS cap.

## Prompt and vision-language paths

Install `python -m pip install -e ".[prompts]"`. The independent path uses an actual locally configured Hugging Face CLIP vision/text backbone, class-ordered tokenizer IDs (`C x sequence_length` NPY) and optionally a local pretrained CLIP state_dict. It automatically downloads no weights.

```bash
python -m nullaudit.prompts --config data/clip-config.json --text-tokens data/class-tokens.npy --backbone-state data/clip-state.pt --data data/prepared/train.npz --method pgp --epochs 5 --output runs/pgp
python -m nullaudit.prompts --config data/clip-config.json --text-tokens data/class-tokens.npy --backbone-state data/clip-state.pt --data data/prepared/train.npz --method vpt_ns --epochs 5 --output runs/vpt-ns
python -m nullaudit.prompts --config data/clip-config.json --text-tokens data/class-tokens.npy --backbone-state data/clip-state.pt --data data/prepared/train.npz --method dmnsp --epochs 5 --output runs/dmnsp
```

`pgp` learns key/value prefixes and projects gradients against query-feature directions, using one shared prefix per layer; see [PGP](https://github.com/JingyangQiao/prompt-gradient-projection/tree/main/dualprompt-pgp). `vpt_ns` uses deep prompts, `Q W_k` and original-to-prompt attention covariances, and projects the optimizer update on both sides with the times-threshold/Frobenius convention from [VPT-NS](https://github.com/zugexiaodui/VPTinNSforCL/blob/main/train_eval.py). `dmnsp` adds bottleneck adapters to both CLIP modalities and token-derived class prototypes. Its visual gradients use multi-layer null projections with `exp(-top-10%-similarity)*30`; text adapters train normally, following [DMNSP](https://github.com/RL-VIG/DMNSP/blob/master/continual_clip/models.py). Top-k is at least one. Report trainable parameters and projection state separately for these backbone-specific runs.

## Backbone checks

```bash
python -m nullaudit.verify_comparisons --output runs/comparisons-check
python -m nullaudit.verify_prompts --output runs/prompts-check
```

For PyTorch-only checks use process-local `USE_TF=0`, `USE_FLAX=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `MKL_THREADING_LAYER=SEQUENTIAL`.

## CPU checks

```bash
python -m nullaudit.verify --output runs/check
```

Use process-local `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MKL_THREADING_LAYER=SEQUENTIAL` on this host to select sequential MKL when starting Python.

## Native DualPrompt-PGP pool

`nullaudit.dualprompt` uses a frozen ViT encoder and unprompted CLS retrieval features, general key/value prefixes at layers 0/1, expert pools at layers 2/3/4, cosine-key top-k retrieval with batch voting, training-task expert ranges, previous-task expert initialization, a trainable class head and key-pulling loss. Inference retrieves experts without task labels. PGP projects key-prefix gradients; value-prefix gradients remain unchanged. Optimizer state resets at task boundaries.

```bash
python -m nullaudit.dualprompt --config configs/dualprompt_pgp_cifar100.json --data data/prepared/train.npz --backbone-state data/torchvision-vit-b16-state.pt --output runs/dualprompt-pgp
python -m nullaudit.dualprompt --config configs/dualprompt_pgp_cifar100.json --data data/prepared/test.npz --checkpoint runs/dualprompt-pgp/model.pt --output runs/dualprompt-pgp/test.npz
python -m nullaudit.verify_dualprompt --output runs/native-pgp-check
```

`configs/dualprompt_pgp_cifar100.json` follows the [CIFAR-100/10-task recipe](https://github.com/JingyangQiao/prompt-gradient-projection/blob/main/dualprompt-pgp/configs/cifar100_10_dualprompt_pgp.py): ViT-B/16 at 224, pool 10, top-k 1, prompt length 5, Adam LR 0.03, batch 24, 20 epochs/task, zero decay, clipping 1, smoothing 0.1 and seed 59. Sources: [retrieval](https://github.com/JingyangQiao/prompt-gradient-projection/blob/main/dualprompt-pgp/prompt.py), [prefix attachment](https://github.com/JingyangQiao/prompt-gradient-projection/blob/main/dualprompt-pgp/vision_transformer.py), [projection](https://github.com/JingyangQiao/prompt-gradient-projection/blob/main/dualprompt-pgp/engine.py). This implementation uses torchvision ViT state dictionaries, ImageNet normalization and feature-axis query-covariance SVD. Smoothing covers current-task classes. Supply a compatible local pretrained state; omitting it initializes random weights.
