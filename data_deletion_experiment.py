"""Pilot experiment for low-degree data-deletion approximations.

The experiment defines a deterministic learning map A(x) by fitting a weighted
ridge-classification head on frozen MobileNetV3-Small features. The scalar
observable is

    f(x) = validation cross-entropy after fitting the head with data weights x.

Only a small, designated set of training examples has variable weights. All
other training examples remain fixed. At x = 1, this script computes the
first-, second-, and third-order derivatives of f, builds Taylor polynomials,
and compares them with exact head re-fitting after low-Hamming-weight deletions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.func import jacrev
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: str
    output_dir: str
    seed: int
    base_per_class: int
    variable_per_class: int
    validation_per_class: int
    projection_dim: int
    ridge: float
    max_derivative_order: int
    hamming_weights: tuple[int, ...]
    trials_per_weight: int
    batch_size: int
    num_workers: int
    jacrev_chunk_size: int | None
    device: str
    download: bool
    recompute_features: bool


class FrozenMobileNetFeatures(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small up to global average pooling."""

    def __init__(self) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        self.features = model.features.eval()
        self.avgpool = model.avgpool
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        activations = self.features(images)
        return torch.flatten(self.avgpool(activations), 1)


class WeightedRidgeObservable:
    """Differentiable A(x) followed by scalar validation cross-entropy."""

    def __init__(
        self,
        base_features: torch.Tensor,
        base_labels: torch.Tensor,
        variable_features: torch.Tensor,
        variable_labels: torch.Tensor,
        validation_features: torch.Tensor,
        validation_labels: torch.Tensor,
        ridge: float,
        num_classes: int,
    ) -> None:
        self.base_features = base_features
        self.variable_features = variable_features
        self.validation_features = validation_features
        self.base_labels = base_labels
        self.variable_labels = variable_labels
        self.validation_labels = validation_labels
        self.base_targets = F.one_hot(base_labels, num_classes).to(base_features.dtype)
        self.variable_targets = F.one_hot(variable_labels, num_classes).to(
            base_features.dtype
        )
        self.ridge = ridge

        penalty = torch.eye(
            base_features.shape[1],
            dtype=base_features.dtype,
            device=base_features.device,
        )
        # The final feature is a constant intercept and is not regularized.
        penalty[-1, -1] = 0.0
        self.penalty = penalty
        self.base_gram = base_features.T @ base_features
        self.base_rhs = base_features.T @ self.base_targets

    def fit_head(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim != 1 or weights.numel() != self.variable_features.shape[0]:
            raise ValueError("weights must have one entry per variable example")

        sample_count = self.base_features.shape[0] + weights.sum()
        weighted_features = weights[:, None] * self.variable_features
        weighted_targets = weights[:, None] * self.variable_targets
        gram = (
            self.base_gram + self.variable_features.T @ weighted_features
        ) / sample_count
        rhs = (
            self.base_rhs + self.variable_features.T @ weighted_targets
        ) / sample_count
        return torch.linalg.solve(gram + self.ridge * self.penalty, rhs)

    def __call__(self, weights: torch.Tensor) -> torch.Tensor:
        head = self.fit_head(weights)
        logits = self.validation_features @ head
        return F.cross_entropy(logits, self.validation_labels)

    def diagnostics(self, weights: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            head = self.fit_head(weights)
            train_features = torch.cat(
                [self.base_features, self.variable_features], dim=0
            )
            train_labels = torch.cat([self.base_labels, self.variable_labels], dim=0)
            train_accuracy = (
                (train_features @ head).argmax(dim=1) == train_labels
            ).double().mean()
            validation_accuracy = (
                (self.validation_features @ head).argmax(dim=1)
                == self.validation_labels
            ).double().mean()
            return {
                "train_accuracy": float(train_accuracy),
                "validation_accuracy": float(validation_accuracy),
                "validation_loss": float(self(weights)),
            }


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Measure derivative decay and Taylor deletion error."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/mobilenet_pilot")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--base-per-class", type=int, default=10)
    parser.add_argument("--variable-per-class", type=int, default=1)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument(
        "--max-derivative-order", type=int, default=3, choices=(1, 2, 3, 4)
    )
    parser.add_argument(
        "--hamming-weights",
        default="1,2,3,4",
        help="Comma-separated deletion-set sizes.",
    )
    parser.add_argument("--trials-per-weight", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--jacrev-chunk-size",
        type=int,
        default=16,
        help="Lower values use less memory; 0 disables chunking.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--recompute-features", action="store_true")
    args = parser.parse_args()

    hamming_weights = tuple(
        sorted({int(value) for value in args.hamming_weights.split(",") if value})
    )
    if not hamming_weights or hamming_weights[0] < 1:
        parser.error("--hamming-weights must contain positive integers")
    if args.base_per_class < 1 or args.variable_per_class < 1:
        parser.error("base and variable counts must be positive")
    if args.validation_per_class < 1:
        parser.error("validation count must be positive")
    if args.projection_dim < 1:
        parser.error("--projection-dim must be positive")
    if args.ridge <= 0:
        parser.error("--ridge must be positive")
    if args.trials_per_weight < 1:
        parser.error("--trials-per-weight must be positive")

    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    return ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        base_per_class=args.base_per_class,
        variable_per_class=args.variable_per_class,
        validation_per_class=args.validation_per_class,
        projection_dim=args.projection_dim,
        ridge=args.ridge,
        max_derivative_order=args.max_derivative_order,
        hamming_weights=hamming_weights,
        trials_per_weight=args.trials_per_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        jacrev_chunk_size=(
            None if args.jacrev_chunk_size == 0 else args.jacrev_chunk_size
        ),
        device=selected_device,
        download=args.download,
        recompute_features=args.recompute_features,
    )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_indices(
    labels: Iterable[int],
    base_per_class: int,
    variable_per_class: int,
    validation_per_class: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    labels_tensor = torch.as_tensor(list(labels), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    groups: list[tuple[list[int], list[int], list[int]]] = []
    required = base_per_class + variable_per_class + validation_per_class

    for class_index in range(10):
        candidates = torch.where(labels_tensor == class_index)[0]
        if candidates.numel() < required:
            raise ValueError(f"class {class_index} has fewer than {required} examples")
        candidates = candidates[torch.randperm(candidates.numel(), generator=generator)]
        base_end = base_per_class
        variable_end = base_end + variable_per_class
        validation_end = variable_end + validation_per_class
        groups.append(
            (
                candidates[:base_end].tolist(),
                candidates[base_end:variable_end].tolist(),
                candidates[variable_end:validation_end].tolist(),
            )
        )

    def merge_and_shuffle(position: int) -> list[int]:
        merged = [index for group in groups for index in group[position]]
        order = torch.randperm(len(merged), generator=generator).tolist()
        return [merged[index] for index in order]

    return merge_and_shuffle(0), merge_and_shuffle(1), merge_and_shuffle(2)


def extract_features(
    dataset: CIFAR10,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    extractor = FrozenMobileNetFeatures().to(device).eval()
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            features = extractor(images.to(device, non_blocking=True))
            feature_batches.append(features.cpu())
            label_batches.append(labels.cpu())
            print(
                f"feature batch {batch_index}/{len(loader)}",
                end="\r" if batch_index < len(loader) else "\n",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(f"extracted {len(indices)} feature vectors in {elapsed:.1f}s")
    return torch.cat(feature_batches), torch.cat(label_batches)


def load_or_extract_features(
    config: ExperimentConfig,
    dataset: CIFAR10,
    indices: list[int],
    output_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"mobilenet_features_seed{config.seed}_"
        f"b{config.base_per_class}_x{config.variable_per_class}_"
        f"v{config.validation_per_class}.pt"
    )
    cache_path = cache_dir / cache_name

    if cache_path.exists() and not config.recompute_features:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload["indices"].tolist() == indices:
            print(f"loaded cached features: {cache_path}")
            return payload["features"], payload["labels"]
        print("feature cache index mismatch; recomputing")

    features, labels = extract_features(
        dataset,
        indices,
        torch.device(config.device),
        config.batch_size,
        config.num_workers,
    )
    torch.save(
        {
            "indices": torch.tensor(indices, dtype=torch.long),
            "features": features,
            "labels": labels,
        },
        cache_path,
    )
    print(f"saved feature cache: {cache_path}")
    return features, labels


def project_features(
    base: torch.Tensor,
    variable: torch.Tensor,
    validation: torch.Tensor,
    requested_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    base = base.double()
    variable = variable.double()
    validation = validation.double()
    mean = base.mean(dim=0, keepdim=True)
    centered_base = base - mean
    dimension = min(requested_dim, centered_base.shape[0] - 1, centered_base.shape[1])
    if dimension < 1:
        raise ValueError("not enough base examples to construct a feature projection")

    _, _, right_vectors = torch.linalg.svd(centered_base, full_matrices=False)
    components = right_vectors[:dimension].T.contiguous()
    projected_base = centered_base @ components
    projected_variable = (variable - mean) @ components
    projected_validation = (validation - mean) @ components

    scale = projected_base.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    projected_base = projected_base / scale
    projected_variable = projected_variable / scale
    projected_validation = projected_validation / scale

    def add_intercept(features: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(features.shape[0], 1, dtype=features.dtype)
        return torch.cat([features, ones], dim=1)

    return (
        add_intercept(projected_base),
        add_intercept(projected_variable),
        add_intercept(projected_validation),
        dimension,
    )


def tensor_symmetry_error(tensor: torch.Tensor) -> float | None:
    if tensor.ndim == 2:
        difference = tensor - tensor.T
    elif tensor.ndim == 3:
        difference = tensor - tensor.permute(1, 0, 2)
    else:
        return None
    denominator = torch.linalg.vector_norm(tensor).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(difference) / denominator)


def compute_derivatives(
    function: Callable[[torch.Tensor], torch.Tensor],
    expansion_point: torch.Tensor,
    maximum_order: int,
    chunk_size: int | None,
) -> tuple[dict[int, torch.Tensor], list[dict[str, float | int | str | None]]]:
    derivatives: dict[int, torch.Tensor] = {}
    rows: list[dict[str, float | int | str | None]] = []
    derivative_function: Callable[[torch.Tensor], torch.Tensor] = function
    previous_coefficient_rms: float | None = None

    for order in range(1, maximum_order + 1):
        started = time.perf_counter()
        derivative_function = jacrev(derivative_function, chunk_size=chunk_size)
        tensor = derivative_function(expansion_point).detach()
        elapsed = time.perf_counter() - started
        derivatives[order] = tensor

        frobenius = float(torch.linalg.vector_norm(tensor))
        rms = float(tensor.square().mean().sqrt())
        factorial = math.factorial(order)
        coefficient_frobenius = frobenius / factorial
        coefficient_rms = rms / factorial
        ratio = (
            coefficient_rms / previous_coefficient_rms
            if previous_coefficient_rms not in (None, 0.0)
            else None
        )
        previous_coefficient_rms = coefficient_rms
        rows.append(
            {
                "order": order,
                "shape": "x".join(str(value) for value in tensor.shape),
                "entries": tensor.numel(),
                "raw_frobenius_norm": frobenius,
                "raw_rms": rms,
                "taylor_coefficient_frobenius_norm": coefficient_frobenius,
                "taylor_coefficient_rms": coefficient_rms,
                "coefficient_rms_ratio_to_previous_order": ratio,
                "symmetry_relative_error": tensor_symmetry_error(tensor),
                "seconds": elapsed,
            }
        )
        print(
            f"order {order}: shape={tuple(tensor.shape)}, "
            f"coefficient RMS={coefficient_rms:.6g}, time={elapsed:.1f}s"
        )

    return derivatives, rows


def directional_taylor_term(
    derivative: torch.Tensor, delta: torch.Tensor, order: int
) -> torch.Tensor:
    contracted = derivative
    for _ in range(order):
        contracted = torch.tensordot(contracted, delta, dims=([0], [0]))
    return contracted / math.factorial(order)


def sampled_subsets(
    dimension: int,
    size: int,
    maximum_trials: int,
    generator: torch.Generator,
) -> list[tuple[int, ...]]:
    total = math.comb(dimension, size)
    if total <= maximum_trials:
        return list(itertools.combinations(range(dimension), size))

    subsets: set[tuple[int, ...]] = set()
    while len(subsets) < maximum_trials:
        subset = torch.randperm(dimension, generator=generator)[:size]
        subsets.add(tuple(sorted(subset.tolist())))
    return sorted(subsets)


def evaluate_deletions(
    function: Callable[[torch.Tensor], torch.Tensor],
    expansion_point: torch.Tensor,
    derivatives: dict[int, torch.Tensor],
    hamming_weights: tuple[int, ...],
    trials_per_weight: int,
    variable_labels: torch.Tensor,
    class_names: list[str],
    seed: int,
) -> list[dict[str, float | int | str]]:
    generator = torch.Generator().manual_seed(seed + 10_000)
    baseline = float(function(expansion_point).detach())
    dimension = expansion_point.numel()
    rows: list[dict[str, float | int | str]] = []

    for hamming_weight in hamming_weights:
        if hamming_weight > dimension:
            print(
                f"skipping Hamming weight {hamming_weight}; variable dimension is {dimension}"
            )
            continue
        subsets = sampled_subsets(
            dimension, hamming_weight, trials_per_weight, generator
        )
        for trial, subset in enumerate(subsets):
            weights = expansion_point.clone()
            weights[list(subset)] = 0.0
            delta = weights - expansion_point
            exact = float(function(weights).detach())
            terms = {
                order: float(directional_taylor_term(tensor, delta, order))
                for order, tensor in derivatives.items()
                if order <= 3
            }
            approximations: dict[int, float] = {}
            running = baseline
            for order in range(1, 4):
                if order in terms:
                    running += terms[order]
                    approximations[order] = running

            actual_change = exact - baseline
            scale = max(abs(actual_change), 1e-12)
            deleted_labels = [int(variable_labels[index]) for index in subset]
            deleted_classes = [class_names[label] for label in deleted_labels]
            row: dict[str, float | int | str] = {
                "hamming_weight": hamming_weight,
                "trial": trial,
                "deleted_coordinate_indices": ";".join(map(str, subset)),
                "deleted_label_indices": ";".join(map(str, deleted_labels)),
                "deleted_classes": ";".join(deleted_classes),
                "baseline_f": baseline,
                "exact_f": exact,
                "actual_change": actual_change,
            }
            for order in range(1, 4):
                if order not in approximations:
                    continue
                approximation = approximations[order]
                absolute_error = abs(exact - approximation)
                row[f"term_{order}"] = terms[order]
                row[f"p{order}"] = approximation
                row[f"absolute_error_p{order}"] = absolute_error
                row[f"relative_error_to_change_p{order}"] = absolute_error / scale
            rows.append(row)

    return rows


def summarize_deletions(
    rows: list[dict[str, float | int | str]], maximum_order: int
) -> list[dict[str, float | int]]:
    summaries: list[dict[str, float | int]] = []
    hamming_weights = sorted({int(row["hamming_weight"]) for row in rows})
    for hamming_weight in hamming_weights:
        group = [row for row in rows if row["hamming_weight"] == hamming_weight]
        summary: dict[str, float | int] = {
            "hamming_weight": hamming_weight,
            "trials": len(group),
        }
        for order in range(1, min(maximum_order, 3) + 1):
            errors = [float(row[f"absolute_error_p{order}"]) for row in group]
            relative_errors = [
                float(row[f"relative_error_to_change_p{order}"]) for row in group
            ]
            summary[f"mean_absolute_error_p{order}"] = statistics.fmean(errors)
            summary[f"median_absolute_error_p{order}"] = statistics.median(errors)
            summary[f"max_absolute_error_p{order}"] = max(errors)
            summary[f"mean_relative_error_to_change_p{order}"] = statistics.fmean(
                relative_errors
            )
        if maximum_order >= 2:
            summary["fraction_p2_better_than_p1"] = statistics.fmean(
                float(row["absolute_error_p2"]) < float(row["absolute_error_p1"])
                for row in group
            )
        if maximum_order >= 3:
            summary["fraction_p3_better_than_p2"] = statistics.fmean(
                float(row["absolute_error_p3"]) < float(row["absolute_error_p2"])
                for row in group
            )
            summary["fraction_strictly_monotone_p3_p2_p1"] = statistics.fmean(
                float(row["absolute_error_p3"])
                < float(row["absolute_error_p2"])
                < float(row["absolute_error_p1"])
                for row in group
            )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_log2_decay(derivative_rows: list[dict]) -> dict[str, float] | None:
    points = [
        (float(row["order"]), float(row["taylor_coefficient_rms"]))
        for row in derivative_rows
        if float(row["taylor_coefficient_rms"]) > 0
    ]
    if len(points) < 2:
        return None
    mean_x = statistics.fmean(point[0] for point in points)
    log_values = [(x, math.log2(y)) for x, y in points]
    mean_y = statistics.fmean(point[1] for point in log_values)
    denominator = sum((x - mean_x) ** 2 for x, _ in log_values)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in log_values) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in log_values)
    total = sum((y - mean_y) ** 2 for _, y in log_values)
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    return {
        "log2_slope": slope,
        "effective_decay_c": -slope,
        "r_squared": r_squared,
        "number_of_orders": len(points),
    }


def save_plots(
    output_dir: Path,
    derivative_rows: list[dict],
    deletion_summaries: list[dict],
    maximum_order: int,
) -> None:
    orders = [int(row["order"]) for row in derivative_rows]
    raw_rms = [float(row["raw_rms"]) for row in derivative_rows]
    coefficient_rms = [
        float(row["taylor_coefficient_rms"]) for row in derivative_rows
    ]
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(orders, raw_rms, marker="o", label=r"RMS of $D^r f$")
    axis.plot(
        orders,
        coefficient_rms,
        marker="s",
        label=r"RMS of $D^r f/r!$",
    )
    if all(value > 0 for value in raw_rms + coefficient_rms):
        axis.set_yscale("log")
    axis.set_xlabel("Derivative order r")
    axis.set_ylabel("Magnitude")
    axis.set_title("Derivative and Taylor-coefficient magnitude")
    axis.set_xticks(orders)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "derivative_decay.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    hamming_weights = [int(row["hamming_weight"]) for row in deletion_summaries]
    plotted_values: list[float] = []
    for order in range(1, min(maximum_order, 3) + 1):
        means = [
            float(row[f"mean_absolute_error_p{order}"])
            for row in deletion_summaries
        ]
        plotted_values.extend(means)
        axis.plot(hamming_weights, means, marker="o", label=f"degree {order}")
    if plotted_values and all(value > 0 for value in plotted_values):
        axis.set_yscale("log")
    axis.set_xlabel("Deletion-set Hamming weight")
    axis.set_ylabel("Mean absolute approximation error")
    axis.set_title("Taylor error versus deletion-set size")
    axis.set_xticks(hamming_weights)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "taylor_error_by_hamming_weight.png", dpi=180)
    plt.close(figure)


def print_summary(
    diagnostics: dict[str, float],
    derivative_rows: list[dict],
    deletion_summaries: list[dict],
    decay_fit: dict[str, float] | None,
    output_dir: Path,
) -> None:
    print("\nBaseline")
    print(f"  train accuracy:      {diagnostics['train_accuracy']:.3f}")
    print(f"  validation accuracy: {diagnostics['validation_accuracy']:.3f}")
    print(f"  validation f(x):     {diagnostics['validation_loss']:.6f}")
    print("\nQuestion 1: derivative/Taylor-coefficient magnitude")
    for row in derivative_rows:
        print(
            f"  r={row['order']}: raw RMS={float(row['raw_rms']):.6g}, "
            f"coefficient RMS={float(row['taylor_coefficient_rms']):.6g}"
        )
    if decay_fit:
        print(
            "  descriptive log2 slope="
            f"{decay_fit['log2_slope']:.3f} "
            f"(effective c={decay_fit['effective_decay_c']:.3f}, "
            f"R^2={decay_fit['r_squared']:.3f})"
        )
    print("\nQuestions 2 and 3: mean absolute error by Hamming weight")
    for row in deletion_summaries:
        errors = ", ".join(
            f"P{order}={float(row[f'mean_absolute_error_p{order}']):.6g}"
            for order in range(1, 4)
            if f"mean_absolute_error_p{order}" in row
        )
        print(f"  k={row['hamming_weight']} ({row['trials']} trials): {errors}")
    print(f"\nArtifacts written to: {output_dir.resolve()}")


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {config.device}")
    print(f"torch: {torch.__version__}; torchvision: {torchvision.__version__}")
    weights = MobileNet_V3_Small_Weights.DEFAULT
    dataset = CIFAR10(
        root=config.data_dir,
        train=True,
        download=config.download,
        transform=weights.transforms(),
    )
    base_indices, variable_indices, validation_indices = stratified_indices(
        dataset.targets,
        config.base_per_class,
        config.variable_per_class,
        config.validation_per_class,
        config.seed,
    )
    all_indices = base_indices + variable_indices + validation_indices
    features, labels = load_or_extract_features(
        config, dataset, all_indices, output_dir
    )

    base_end = len(base_indices)
    variable_end = base_end + len(variable_indices)
    base_features = features[:base_end]
    variable_features = features[base_end:variable_end]
    validation_features = features[variable_end:]
    base_labels = labels[:base_end]
    variable_labels = labels[base_end:variable_end]
    validation_labels = labels[variable_end:]
    (
        base_features,
        variable_features,
        validation_features,
        actual_projection_dim,
    ) = project_features(
        base_features,
        variable_features,
        validation_features,
        config.projection_dim,
    )
    print(
        f"base={len(base_indices)}, variable={len(variable_indices)}, "
        f"validation={len(validation_indices)}, PCA dimension={actual_projection_dim}"
    )

    observable = WeightedRidgeObservable(
        base_features,
        base_labels,
        variable_features,
        variable_labels,
        validation_features,
        validation_labels,
        config.ridge,
        num_classes=10,
    )
    expansion_point = torch.ones(len(variable_indices), dtype=torch.float64)
    diagnostics = observable.diagnostics(expansion_point)
    derivatives, derivative_rows = compute_derivatives(
        observable,
        expansion_point,
        config.max_derivative_order,
        config.jacrev_chunk_size,
    )
    deletion_rows = evaluate_deletions(
        observable,
        expansion_point,
        derivatives,
        config.hamming_weights,
        config.trials_per_weight,
        variable_labels,
        dataset.classes,
        config.seed,
    )
    deletion_summaries = summarize_deletions(
        deletion_rows, config.max_derivative_order
    )
    decay_fit = fit_log2_decay(derivative_rows)

    write_csv(output_dir / "derivative_decay.csv", derivative_rows)
    write_csv(output_dir / "deletion_trials.csv", deletion_rows)
    write_csv(output_dir / "deletion_summary.csv", deletion_summaries)
    save_plots(
        output_dir,
        derivative_rows,
        deletion_summaries,
        config.max_derivative_order,
    )
    manifest = {
        "config": {
            **asdict(config),
            "hamming_weights": list(config.hamming_weights),
        },
        "environment": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "definition": {
            "learning_map": "weighted ridge head on frozen MobileNetV3-Small features",
            "observable_f": "mean cross-entropy on a fixed held-out validation subset",
            "expansion_point": "all variable-example weights equal to one",
            "deletion": "set selected variable-example weights from one to zero",
            "feature_projection": "PCA fit only on the fixed base-training subset",
            "actual_projection_dim": actual_projection_dim,
        },
        "indices": {
            "base": base_indices,
            "variable": variable_indices,
            "validation": validation_indices,
        },
        "variable_labels": variable_labels.tolist(),
        "class_names": dataset.classes,
        "baseline": diagnostics,
        "descriptive_decay_fit": decay_fit,
    }
    with (output_dir / "experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print_summary(
        diagnostics,
        derivative_rows,
        deletion_summaries,
        decay_fit,
        output_dir,
    )


if __name__ == "__main__":
    main()
