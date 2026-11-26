
import torch
import torch.nn.functional as F


def fgsm_attack(
    model, image, label, epsilon, mean, std,
    return_history=False
):
    """
    Fast Gradient Sign Method (FGSM).
    Generates an adversarial image and optionally
    returns pixel-change measurements.
    """

    image_for_gradient = image.detach().clone()
    image_for_gradient.requires_grad_(True)

    normalized_image = (image_for_gradient - mean) / std

    output = model(normalized_image)
    loss = F.cross_entropy(output, label)

    model.zero_grad(set_to_none=True)
    loss.backward()

    gradient_sign = image_for_gradient.grad.sign()

    adversarial_image = (
        image_for_gradient + epsilon * gradient_sign
    )

    adversarial_image = adversarial_image.clamp(0, 1)

    difference = (
        adversarial_image.detach() - image.detach()
    ).abs()

    changed_pixels = (
        difference > (1 / 255)
    ).any(dim=1).sum().item()

    history = [{
        "iteration": 1,
        "mean_change": difference.mean().item(),
        "max_change": difference.max().item(),
        "changed_pixels": changed_pixels
    }]

    adversarial_image = adversarial_image.detach()

    if return_history:
        return adversarial_image, history

    return adversarial_image
