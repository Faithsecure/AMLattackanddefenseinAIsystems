
import torch
import torch.nn.functional as F


def pgd_attack(
    model, image, label, epsilon, alpha, steps, mean, std,
    return_history=False
):
    """
    Projected Gradient Descent (PGD) attack.
    Generates an adversarial image and optionally
    records pixel changes at every attack step.
    """

    original_image = image.detach().clone()
    adversarial_image = original_image.clone()

    history = []

    for step in range(steps):
        adversarial_image = adversarial_image.detach()
        adversarial_image.requires_grad_(True)

        normalized_image = (adversarial_image - mean) / std

        output = model(normalized_image)
        loss = F.cross_entropy(output, label)

        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            adversarial_image = (
                adversarial_image
                + alpha * adversarial_image.grad.sign()
            )

            perturbation = adversarial_image - original_image
            perturbation = perturbation.clamp(-epsilon, epsilon)

            adversarial_image = (
                original_image + perturbation
            ).clamp(0, 1)

            difference = (
                adversarial_image - original_image
            ).abs()

            changed_pixels = (
                difference > (1 / 255)
            ).any(dim=1).sum().item()

            history.append({
                "iteration": step + 1,
                "mean_change": difference.mean().item(),
                "max_change": difference.max().item(),
                "changed_pixels": changed_pixels
            })

    adversarial_image = adversarial_image.detach()

    if return_history:
        return adversarial_image, history

    return adversarial_image
