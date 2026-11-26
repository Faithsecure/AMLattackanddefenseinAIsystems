
import torch
from torchvision.transforms import functional as TF


def sanitize_input(image):
    """
    Input sanitization using Gaussian blur.

    Accepts an image tensor with pixel values between 0 and 1.
    """

    sanitized_image = TF.gaussian_blur(
        image,
        kernel_size=[3, 3],
        sigma=[0.8, 0.8]
    )

    sanitized_image = sanitized_image.clamp(0, 1)

    return sanitized_image
