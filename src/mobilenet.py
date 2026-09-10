import torch
import torch.nn as nn
import numpy as np
import pickle

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import (
    mobilenet_v3_small,
)

import math

from .polys import Poly, set_degree, get_degree
from data_deletion_experiment import WeightedRidgeObservable



torch.manual_seed(1)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        return image, label, index


class LearningAlgorithm:
    def __init__(
        self,
        model,
        dataset,
        x,
        device,
        num_epochs,
        batch_size=64,
        learning_rate=1e-3,
    ):
        if len(dataset) != len(x):
            raise ValueError(
                f"x must contain one weight per training example: "
                f"len(dataset)={len(dataset)}, "
                f"len(x)={len(x)}"
            )

        self.model = model.to(device)
        self.dataset = IndexedDataset(dataset)
        self.x = x

        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.train_criterion = nn.CrossEntropyLoss(
            reduction="none"
        )

        self.optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                self.model.parameters(),
            ),
            lr=self.learning_rate,
        )

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()

            running_loss = 0.0

            for images, labels, indices in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                batch_weights = self.x[indices].to(
                    self.device
                )

                self.optimizer.zero_grad()

                outputs = self.model(images)

                # One loss per training example
                losses = self.train_criterion(
                    outputs,
                    labels,
                )

                # x controls the contribution of each example
                weighted_loss = (
                    batch_weights * losses
                ).mean()

                weighted_loss.backward()
                self.optimizer.step()

                running_loss += weighted_loss.item()

            average_loss = (
                running_loss / len(self.train_loader)
            )

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}: "
                f"weighted training loss = "
                f"{average_loss:.4f}"
            )

        return self.model

    def evaluate(self, dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.model.eval()

        total_loss = 0.0
        total_examples = 0

        eval_criterion = nn.CrossEntropyLoss(
            reduction="sum"
        )

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = eval_criterion(
                    outputs,
                    labels,
                )

                total_loss += loss.item()
                total_examples += images.size(0)

        return total_loss / total_examples












def precompute(observable: WeightedRidgeObservable, k: int = 50):
    #the learning algorithm needs tobe somehow specified as a circuit
    #With n inputs and p outputs.

    #Chose the ring with s = 5
    s = 5
    set_degree(s)

    n = observable.variable_features.shape[0]

    # unit vectors on the complex sphere in C^n; must be complex,
    # not real, for the estimator's moment-unbiasedness
    rng = np.random.default_rng()
    directions = rng.standard_normal((k, n)) + 1j * rng.standard_normal((k, n))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    variable_features = observable.variable_features.detach().cpu().numpy()
    variable_targets = observable.variable_targets.detach().cpu().numpy()
    base_gram = observable.base_gram.detach().cpu().numpy()
    base_rhs = observable.base_rhs.detach().cpu().numpy()
    penalty = observable.penalty.detach().cpu().numpy()
    base_count = observable.base_features.shape[0]

    outputs = []
    for psi in directions:
        # weights(z) = 1 + z*psi: expand around x0 = 1 (all variable
        # examples included), matching this repo's convention
        # (EXPERIMENT.md) -- deletion moves a weight from 1 toward 0.
        weights = np.array(
            [Poly([1.0, psi[i]]) for i in range(n)], dtype=object
        )

        sample_count = base_count + weights.sum()
        weighted_features = weights[:, None] * variable_features
        weighted_targets = weights[:, None] * variable_targets

        gram = (
            base_gram + variable_features.T @ weighted_features
        ) / sample_count
        rhs = (
            base_rhs + variable_features.T @ weighted_targets
        ) / sample_count

        theta = solve_linear_system(gram + observable.ridge * penalty, rhs)
        outputs.append(theta)

    return directions, outputs

def poly_cross_entropy(logits, label_index: int) -> Poly:
    # logits: sequence of Poly, one per class. Numerically-stabilized
    # log-sum-exp, but shifted by a *real* constant (the unperturbed,
    # z=0 logit) since Poly has no ordering to take a max of directly.
    max_logit = max(float(logit.coeffs[0].real) for logit in logits)
    shifted = [logit - max_logit for logit in logits]
    log_sum_exp = sum(term.exp() for term in shifted).log()
    return log_sum_exp - shifted[label_index]


def predict(directions, outputs, deleted_indices, observable: WeightedRidgeObservable):
    k, n = directions.shape
    s = get_degree()

    delta = np.zeros(n)
    delta[list(deleted_indices)] = -1.0

    validation_features = observable.validation_features.detach().cpu().numpy()
    validation_labels = observable.validation_labels.detach().cpu().numpy()
    num_validation = validation_features.shape[0]

    # order_samples[r][i] = <psi_i, delta>^r * q_{i,r}, the r-th-order
    # contribution of sketch i, before averaging across the k sketches.
    order_samples = [[] for _ in range(s + 1)]
    for psi, theta in zip(directions, outputs):
        logits = validation_features @ theta  # (num_validation, num_classes) of Poly

        measurement = Poly(0.0)
        for row in range(num_validation):
            measurement = measurement + poly_cross_entropy(
                logits[row], int(validation_labels[row])
            )
        measurement = measurement * (1.0 / num_validation)

        # <psi, delta>, conjugate-linear in psi. Needed so this pairs
        # against the holomorphic-in-psi sketch (precompute substituted
        # z*psi, never psi-bar) to give a nonzero, phase-invariant
        # expectation -- see the paper's symmetric-subspace-projector
        # argument, Sec 3.2.
        inner_product = np.vdot(psi, delta)

        weight = 1.0 + 0j
        for order in range(s + 1):
            order_samples[order].append(weight * measurement.coeffs[order])
            weight *= inner_product

    # E[<psi,delta>^r * q_r(psi)] = T_r[delta,...,delta] / n^(r), where
    # n^(r) = n(n+1)...(n+r-1) is the rising factorial (verified against
    # the known Beta(1, n-1) moments of |psi_1|^2 on the complex sphere).
    # q_r is already a Taylor coefficient (i.e. T_r/r!), so multiplying
    # the sample mean by n^(r)/r! recovers the r-th order Taylor term.
    estimate = 0.0
    rising_factorial = 1.0
    for order in range(s + 1):
        if order > 0:
            rising_factorial *= (n + order - 1)
        mean_sample = np.mean(order_samples[order])
        taylor_term = (rising_factorial / math.factorial(order)) * mean_sample
        estimate += taylor_term.real  # imaginary part is finite-k noise

    return estimate

class MobileNet:
    def __init__(self, weights):
        self.weights = weights

        self.model = mobilenet_v3_small(
            weights=self.weights
        )

        # Freeze pretrained feature extractor
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Replace ImageNet's 1000-class output
        # with CIFAR-10's 10 classes
        self.in_features = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(
            self.in_features,
            10,
        )

        # --------------------------------------------------
        # Dataset
        # --------------------------------------------------

        transform = weights.transforms()

        self.train_dataset = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        self.test_dataset = CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        print(f"Classifier initialized: { self.model.classifier}")

    def save(self, path):
        # --------------------------------------------------
        # Pytorches save
        # --------------------------------------------------
        torch.save(
            self.model.state_dict(),
            path,
        )


    def load(self, path):
        state_dict = torch.load(
            path,
            map_location=device,
        )
        self.model.load_state_dict(state_dict)

    def init_classifier(self, train_dataset, x):
        algorithm = LearningAlgorithm(
            model=self.model,
            dataset=train_dataset,
            x=x,
            device=device,
            num_epochs=5,
            batch_size=64,
            learning_rate=1e-3,
        )
        self.classifier = algorithm
        return self.classifier

    def train_classifier(self):
        self.classifier.train()
        return self.classifier

    def evaluate_classifier(self):
        # --------------------------------------------------
        # Evaluate
        # --------------------------------------------------

        test_loss = self.classifier.evaluate(
            self.test_dataset
        )

        print("Test loss:", test_loss)
        return test_loss
