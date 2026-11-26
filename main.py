import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TF

from fgsm import fgsm_attack
from pgd import pgd_attack
from defense import sanitize_input


# 1. PREPARE THE ENVIRONMENT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("results", exist_ok=True)


# 2. LOAD PRETRAINED RESNET-18

print("\nLoading ResNet-18...")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).to(device)
model.eval()

categories = weights.meta["categories"]

print("ResNet-18 loaded successfully!")
print("Number of output classes:", len(categories))


# 3. IMAGE NORMALIZATION SETTINGS

mean = torch.tensor(
    [0.485, 0.456, 0.406], device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225], device=device
).view(1, 3, 1, 1)


# 4. LOAD AND PREPROCESS THE IMAGE

image_path = "dog.jpg"

if not os.path.exists(image_path):
    image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    print("Downloading sample image...")
    urllib.request.urlretrieve(image_url, image_path)

image = Image.open(image_path).convert("RGB")
image = TF.resize(image, 256)
image = TF.center_crop(image, [224, 224])

# Keep image pixels in [0, 1]. Normalization is applied
# inside predict() and the attack functions.
x = TF.to_tensor(image).unsqueeze(0).to(device)

print("Image loaded successfully!")



# 5. HELPER FUNCTIONS


def predict(image_tensor):
    """Return the predicted ImageNet class ID and confidence."""
    with torch.no_grad():
        normalized_image = (image_tensor - mean) / std
        output = model(normalized_image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

    return predicted_class.item(), confidence.item() * 100


def tensor_to_numpy(image_tensor):
    """Convert a [1, 3, H, W] tensor in [0, 1] to an RGB image."""
    return (
        image_tensor.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )


def save_tensor_image(image_tensor, filename):
    """Save a [1, 3, H, W] image tensor as a PNG file."""
    image_array = tensor_to_numpy(image_tensor)
    image_array = (image_array * 255).round().clip(0, 255).astype("uint8")
    Image.fromarray(image_array).save(filename)


def plot_image_difference(original_tensor, adversarial_tensor,
                           attack_name, output_filename):
    """Save original, adversarial, and magnified difference images."""
    original_array = tensor_to_numpy(original_tensor)
    adversarial_array = tensor_to_numpy(adversarial_tensor)

    difference = np.abs(adversarial_array - original_array)

    # Magnification is for display only; it does not change the attack.
    magnified_difference = np.clip(difference * 8, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(adversarial_array)
    axes[1].set_title(f"{attack_name} Adversarial Image")
    axes[1].axis("off")

    axes[2].imshow(magnified_difference)
    axes[2].set_title("Absolute Pixel Difference (Magnified 8x)")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_attack_history(history, attack_name, filename_prefix):
    """Save graphs and print pixel-change measurements."""
    iterations = [item["iteration"] for item in history]
    mean_changes = [item["mean_change"] for item in history]
    changed_pixels = [item["changed_pixels"] for item in history]

    # Graph 1: mean absolute pixel change
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, mean_changes, marker="o")
    ax.set_title(f"{attack_name}: Mean Absolute Pixel Change")
    ax.set_xlabel("Attack Step")
    ax.set_ylabel("Mean Absolute Pixel Change")
    ax.set_xticks(iterations)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        f"results/{filename_prefix}_mean_pixel_change.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    # Graph 2: number of pixels changed above the selected threshold
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, changed_pixels, marker="o")
    ax.set_title(f"{attack_name}: Number of Changed Pixels")
    ax.set_xlabel("Attack Step")
    ax.set_ylabel("Changed Pixel Count")
    ax.set_xticks(iterations)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        f"results/{filename_prefix}_changed_pixels.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    print(f"\n========== {attack_name.upper()} PIXEL CHANGE DETAILS ==========")
    for item in history:
        print(
            f"Step {item['iteration']}: "
            f"mean change = {item['mean_change']:.6f}, "
            f"maximum change = {item['max_change']:.6f}, "
            f"changed pixels = {item['changed_pixels']}"
        )


# 6. ORIGINAL IMAGE PREDICTION

original_class, original_confidence = predict(x)

print("\n========== ORIGINAL IMAGE ==========")
print("Prediction:", categories[original_class])
print("Confidence:", round(original_confidence, 2), "%")

# The original model prediction is used as a pseudo-label.
# It is not a verified ground-truth label.
attack_label = torch.tensor([original_class], device=device)


# 7. FGSM ATTACK

print("\n========== FGSM ATTACK ==========")

epsilon = 8 / 255

x_fgsm, fgsm_history = fgsm_attack(
    model=model,
    image=x,
    label=attack_label,
    epsilon=epsilon,
    mean=mean,
    std=std,
    return_history=True
)

fgsm_class, fgsm_confidence = predict(x_fgsm)

print("FGSM prediction:", categories[fgsm_class])
print("FGSM confidence:", round(fgsm_confidence, 2), "%")
print("Prediction changed:", fgsm_class != original_class)
print(
    "Maximum pixel change:",
    (x_fgsm - x).abs().max().item()
)

plot_image_difference(
    x, x_fgsm, "FGSM", "results/fgsm_pixel_comparison.png"
)
plot_attack_history(fgsm_history, "FGSM", "fgsm")
save_tensor_image(x_fgsm, "results/fgsm_image.png")


# 8. PGD ATTACK

print("\n========== PGD ATTACK ==========")

x_pgd, pgd_history = pgd_attack(
    model=model,
    image=x,
    label=attack_label,
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=10,
    mean=mean,
    std=std,
    return_history=True
)

pgd_class, pgd_confidence = predict(x_pgd)

print("PGD prediction:", categories[pgd_class])
print("PGD confidence:", round(pgd_confidence, 2), "%")
print("Prediction changed:", pgd_class != original_class)
print(
    "Maximum pixel change:",
    (x_pgd - x).abs().max().item()
)

plot_image_difference(
    x, x_pgd, "PGD", "results/pgd_pixel_comparison.png"
)
plot_attack_history(pgd_history, "PGD", "pgd")
save_tensor_image(x_pgd, "results/pgd_image.png")


# 9. APPLY INPUT SANITIZATION DEFENSE

print("\n========== FGSM + SANITIZATION ==========")

x_fgsm_sanitized = sanitize_input(x_fgsm)
sanitized_fgsm_class, sanitized_fgsm_confidence = predict(
    x_fgsm_sanitized
)

print("Prediction after sanitization:", categories[sanitized_fgsm_class])
print("Confidence:", round(sanitized_fgsm_confidence, 2), "%")
print("Matches original prediction:", sanitized_fgsm_class == original_class)

save_tensor_image(
    x_fgsm_sanitized, "results/fgsm_sanitized.png"
)


print("\n========== PGD + SANITIZATION ==========")

x_pgd_sanitized = sanitize_input(x_pgd)
sanitized_pgd_class, sanitized_pgd_confidence = predict(
    x_pgd_sanitized
)

print("Prediction after sanitization:", categories[sanitized_pgd_class])
print("Confidence:", round(sanitized_pgd_confidence, 2), "%")
print("Matches original prediction:", sanitized_pgd_class == original_class)

save_tensor_image(
    x_pgd_sanitized, "results/pgd_sanitized.png"
)


# 10. FINAL PREDICTION COMPARISON

print("\n========== FINAL COMPARISON ==========")

comparison_results = [
    ("Original", original_class),
    ("FGSM", fgsm_class),
    ("FGSM + Sanitization", sanitized_fgsm_class),
    ("PGD", pgd_class),
    ("PGD + Sanitization", sanitized_pgd_class)
]

print(f"{'Condition':<25} {'Predicted Class':<25} {'Matches Original'}")
print("-" * 75)

for condition, class_id in comparison_results:
    matches = class_id == original_class
    print(
        f"{condition:<25} "
        f"{categories[class_id]:<25} "
        f"{matches}"
    )


# 11. DISPLAY ALL FIVE EXPERIMENTAL IMAGES

images = [
    x,
    x_fgsm,
    x_fgsm_sanitized,
    x_pgd,
    x_pgd_sanitized
]

titles = [
    "Original Image",
    "FGSM Attack",
    "FGSM + Sanitization",
    "PGD Attack",
    "PGD + Sanitization"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for index, (image_tensor, title) in enumerate(zip(images, titles)):
    axes[index].imshow(tensor_to_numpy(image_tensor))
    axes[index].set_title(title)
    axes[index].axis("off")

# Hide the unused sixth subplot.
axes[5].axis("off")

fig.tight_layout()
fig.savefig(
    "results/attack_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)

# 12. FINISH

print("\nAll experiments completed successfully!")
print("Images and graphs are saved inside the results folder.")
