import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

from ..utils.image import get_crop_images, remove_outliers_and_average, remove_outliers_and_average_circular


def get_3angle(image, dino, val_preprocess, device):
    """
    Single-image orientation inference.

    Args:
        image: PIL Image
        dino: DINOv2_MLP model
        val_preprocess: AutoImageProcessor instance
        device: torch device

    Returns:
        Tensor of [azimuth, polar-90, rotation-180, confidence]
    """
    image_inputs = val_preprocess(images=image)
    image_inputs["pixel_values"] = torch.from_numpy(
        np.array(image_inputs["pixel_values"])
    ).to(device)
    with torch.no_grad():
        dino_pred = dino(image_inputs)

    gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1)
    gaus_pl_pred = torch.argmax(dino_pred[:, 360:360 + 180], dim=-1)
    gaus_ro_pred = torch.argmax(dino_pred[:, 360 + 180:360 + 180 + 360], dim=-1)
    confidence = F.softmax(dino_pred[:, -2:], dim=-1)[0][0]

    angles = torch.zeros(4)
    angles[0] = gaus_ax_pred
    angles[1] = gaus_pl_pred - 90
    angles[2] = gaus_ro_pred - 180
    angles[3] = confidence
    return angles


def get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, device):
    """
    Augmented orientation inference using multiple crops.

    Args:
        origin_img: Original PIL Image
        rm_bkg_img: Background-removed PIL Image
        dino: DINOv2_MLP model
        val_preprocess: AutoImageProcessor instance
        device: torch device

    Returns:
        Tensor of [azimuth, polar-90, rotation-180, confidence]
    """
    image = get_crop_images(origin_img, num=3) + get_crop_images(rm_bkg_img, num=3)
    image_inputs = val_preprocess(images=image)
    image_inputs["pixel_values"] = torch.from_numpy(
        np.array(image_inputs["pixel_values"])
    ).to(device)
    with torch.no_grad():
        dino_pred = dino(image_inputs)

    gaus_ax_pred = torch.argmax(dino_pred[:, 0:360], dim=-1).to(torch.float32)
    gaus_pl_pred = torch.argmax(dino_pred[:, 360:360 + 180], dim=-1).to(torch.float32)
    gaus_ro_pred = torch.argmax(dino_pred[:, 360 + 180:360 + 180 + 360], dim=-1).to(torch.float32)

    gaus_ax_pred = remove_outliers_and_average_circular(gaus_ax_pred)
    gaus_pl_pred = remove_outliers_and_average(gaus_pl_pred)
    gaus_ro_pred = remove_outliers_and_average(gaus_ro_pred)

    confidence = torch.mean(F.softmax(dino_pred[:, -2:], dim=-1), dim=0)[0]

    angles = torch.zeros(4)
    angles[0] = gaus_ax_pred
    angles[1] = gaus_pl_pred - 90
    angles[2] = gaus_ro_pred - 180
    angles[3] = confidence
    return angles
