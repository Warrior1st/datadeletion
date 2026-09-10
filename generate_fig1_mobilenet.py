import os
import math
import random
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


import src.autograd_poly as autograd_poly
import src.polys as polys
from src.mobilenet import MobileNet

# Initialize the model.
mobile_net = MobileNet(MobileNet_V3_Small_Weights)
mobile_net.save("data/fig1/init.pkl")

random.seed(42)
polys.set_degree(100)

num_steps = 1000
num_samples = 12

# Load the dataset.
#We load directly from CIFAR10
# --------------------------------------------------
# Dataset
# --------------------------------------------------

transform = MobileNet_V3_Small_Weights.transforms()

train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

# Load or sample random directions.
print(f"Sampling {num_samples} directions")
for samp in range(num_samples):
    direction = [random.gauss(0, 1) + random.gauss(0, 1) * 1j for _ in range(num_steps)]
    direction_norm = sum(abs(direction_i)**2 for direction_i in direction) ** 0.5
    for i in range(num_steps):
        direction[i] /= direction_norm
    with open(f"data/fig1/direction_{samp}.pkl", "wb") as direction_file:
        pickle.dump(direction, direction_file)
print("Done sampling directions.\n")

# Push directions through training.
for samp in range(num_samples):
    print(f"Training model {samp+1:4d} / {num_samples:4d}")

    # Re-initialize the model.
    mobile_net = MobileNet(autograd_poly.Value)
    mobile_net.load("data/fig1/init.pkl")

    with open(f"data/fig1/direction_{samp}.pkl", "rb") as direction_file:
        direction = pickle.load(direction_file)

    # Construct downweight polynomials.
    downweights = [polys.Poly([0, direction_i]) for direction_i in direction]

    #init the classifier
    classifier = mobile_net.init_classifier(train_dataset, downweights)

    # Classifier is our learning algo, train it.
    mobile_net.train_classifier(docs[:num_steps], downweights=downweights)
    gpt.save(f"data/fig1/model_{samp}.pkl")

# Compute estimates of log derivative norms
estimates = []
for samp in range(num_samples):
    gpt = GPT(autograd_poly.Value, uchars)
    gpt.load(f'data/fig1/model_{samp}.pkl')

    # Choose a document to test on.
    test_doc = docs[random.randint(0, num_steps-1)]
    test_loss = gpt.compute_loss(test_doc).data # type: ignore

    derivative_mags = [abs(coeff) for coeff in test_loss.coeffs]
    est = [math.log(math.comb(num_steps + r - 1, r) ** 0.5 * mag, 2) for r, mag in enumerate(derivative_mags)]
    estimates.append(est)

if not os.path.exists('results'):
    os.mkdir('results')
with open('results/fig1.pkl', 'wb') as results_file:
    pickle.dump(estimates, results_file)
print("========= Results =========")
print(estimates)

degree = polys.get_degree()
rs = list(range(1, degree + 1))
avg_estimates = [sum(est[r] for est in estimates) / num_samples for r in rs]

def as_real(x, tol=1e-12):
    z = complex(x)
    if abs(z.imag) > tol:
        raise ValueError(f"Expected near-real value for plotting, got {z}")
    return float(z.real)

for est in estimates:
    plt.plot(rs, [as_real(est[r]) for r in rs])
plt.savefig('results/fig1.png')
