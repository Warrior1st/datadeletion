import os
import math
import random
import pickle
import matplotlib.pyplot as plt

import src.autograd_poly as autograd_poly
import src.polys as polys
from src.microgpt import GPT


random.seed(42)
polys.set_degree(100)

num_steps = 10
num_samples = 12

# Load the dataset.
docs = [line.strip() for line in open('data/train.txt') if line.strip()]
uchars = sorted(set("".join(docs)))
random.shuffle(docs)

if not os.path.exists('data/fig1'):
    os.mkdir('data/fig1')

# Initialize the model.
gpt = GPT(autograd_poly.Value, uchars)
gpt.save("data/fig1/init.pkl")

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
    gpt = GPT(autograd_poly.Value, uchars)
    gpt.load("data/fig1/init.pkl")

    with open(f"data/fig1/direction_{samp}.pkl", "rb") as direction_file:
        direction = pickle.load(direction_file)

    # Construct downweight polynomials.
    downweights = [polys.Poly([0, direction_i]) for direction_i in direction]

    # Train model.
    gpt.train(docs[:num_steps], downweights=downweights)
    gpt.save(f"data/fig1/model_{samp}.pkl")

# Compute estimates of log derivative norms
estimates = []
test_losses = []

for samp in range(num_samples):
    gpt = GPT(autograd_poly.Value, uchars)
    gpt.load(f'data/fig1/model_{samp}.pkl')

    # Choose a document to test on.
    test_doc = docs[random.randint(0, num_steps-1)]
    test_loss = gpt.compute_loss(test_doc).data # type: ignore

    test_losses.append(test_loss)
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


print("\n========= Polynomial approximation errors =========")

t_values = [
    -0.02,
    -0.01,
    -0.005,
    0.005,
    0.01,
    0.02,
]

def truncate_eval(poly, t, degree):
    return sum(
        poly.coeffs[r] * (t ** r)
        for r in range(
            min(degree + 1, len(poly.coeffs))
        )
    )

for samp, test_loss in enumerate(test_losses):

    print(f"\nSample {samp}")

    for t in t_values:

        full_value = test_loss.eval(t)

        p1 = truncate_eval(test_loss, t, 1)
        p2 = truncate_eval(test_loss, t, 2)
        p3 = truncate_eval(test_loss, t, 3)

        err1 = abs(full_value - p1)
        err2 = abs(full_value - p2)
        err3 = abs(full_value - p3)

        print(
            f"t={t:+.3f} | "
            f"linear={err1:.3e} | "
            f"degree2={err2:.3e} | "
            f"degree3={err3:.3e}"
        )



print("\n========= Low Hamming weight estimations =========")

# x has Hamming weight 1: delete document 0
j = 0
deletions = [j]

x = [0] * num_steps
x[j] = 1

test_doc = docs[0]

# True f(x)
true_gpt = GPT(autograd_poly.Value, uchars)
true_gpt.load("data/fig1/init.pkl")

true_gpt.train(
    docs[:num_steps],
    downweights=x
)

true_fx_poly = true_gpt.compute_loss(test_doc).data
true_fx = true_fx_poly.coeffs[0].real

print(f"True f(x): {true_fx:.10f}")


# --------------------------------------------------
# Estimate f(x) using more and more random samples
# --------------------------------------------------

taylor_sums = [0.0 for _ in range(degree)]

f_zero = None

for samp in range(num_samples):

    # Load sampled direction
    with open(
        f"data/fig1/direction_{samp}.pkl",
        "rb"
    ) as direction_file:
        direction = pickle.load(direction_file)

    # Load corresponding polynomial model
    gpt = GPT(autograd_poly.Value, uchars)
    gpt.load(f"data/fig1/model_{samp}.pkl")

    # <psi, 1_D>
    # Since deletions = [j], this is just conjugate(direction[j])
    psi_inner_product = sum(
        direction[i].conjugate()
        for i in deletions
    )

    # Directional Taylor coefficients
    taylor = gpt.compute_loss(test_doc).data.coeffs

    # Constant term f(0)
    if f_zero is None:
        f_zero = taylor[0].real

    # Add THIS sample's contribution
    for r in range(1, degree + 1):
        taylor_sums[r - 1] += (
            psi_inner_product ** r
            * taylor[r]
            * math.comb(num_steps + r - 1, r)
        ).real

    # Number of random samples used so far
    m = samp + 1

    # Average estimate of each Taylor coefficient
    taylor_approximations = [
        total / m
        for total in taylor_sums
    ]

    # x has value 1 on the deleted coordinate,
    # so evaluate the Taylor approximation at x=1
    estimated_fx = (
        f_zero
        + sum(taylor_approximations)
    )

    error = abs(estimated_fx - true_fx)

    print(
        f"samples={m:2d} | "
        f"estimate={estimated_fx:.10f} | "
        f"true={true_fx:.10f} | "
        f"error={error:.6e}"
    )