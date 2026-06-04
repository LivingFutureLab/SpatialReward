import random
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Any


def resize_foreground(image: Image.Image, ratio: float) -> Image.Image:
    """Crop RGBA foreground, pad to square, then add padding for the given ratio."""
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    fg = image[y1:y2, x1:x2]
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    return Image.fromarray(new_image)


def remove_background(
    image: Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image.Image:
    """Remove background using rembg."""
    import rembg

    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def background_preprocess(input_image: Image.Image, do_remove_background: bool) -> Image.Image:
    """Remove background and resize foreground if requested."""
    import rembg

    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    return input_image


def random_crop(image: Image.Image, crop_scale=(0.8, 0.95)) -> Image.Image:
    """Randomly crop an image within the given scale range."""
    assert isinstance(image, Image.Image), "Input must be PIL.Image.Image"
    assert len(crop_scale) == 2 and 0 < crop_scale[0] <= crop_scale[1] <= 1

    width, height = image.size
    crop_width = random.randint(int(width * crop_scale[0]), int(width * crop_scale[1]))
    crop_height = random.randint(int(height * crop_scale[0]), int(height * crop_scale[1]))
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    return image.crop((left, top, left + crop_width, top + crop_height))


def get_crop_images(img: Image.Image, num: int = 3):
    """Generate multiple random crops of an image."""
    return [random_crop(img) for _ in range(num)]


def remove_outliers_and_average(tensor: torch.Tensor, threshold: float = 1.5):
    """IQR-based outlier removal and mean for linear values."""
    assert tensor.dim() == 1, "Input tensor must be 1-dimensional"

    q1 = torch.quantile(tensor, 0.25)
    q3 = torch.quantile(tensor, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    non_outliers = tensor[(tensor >= lower_bound) & (tensor <= upper_bound)]
    if len(non_outliers) == 0:
        return tensor.mean().item()
    return non_outliers.mean().item()


def remove_outliers_and_average_circular(tensor: torch.Tensor, threshold: float = 1.5):
    """IQR-based outlier removal for circular/angular values."""
    assert tensor.dim() == 1, "Input tensor must be 1-dimensional"

    radians = tensor * torch.pi / 180.0
    x_coords = torch.cos(radians)
    y_coords = torch.sin(radians)

    mean_x = torch.mean(x_coords)
    mean_y = torch.mean(y_coords)

    differences = torch.sqrt(
        (x_coords - mean_x) ** 2 + (y_coords - mean_y) ** 2
    )

    q1 = torch.quantile(differences, 0.25)
    q3 = torch.quantile(differences, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    non_outliers = tensor[(differences >= lower_bound) & (differences <= upper_bound)]

    if len(non_outliers) == 0:
        mean_angle = torch.atan2(mean_y, mean_x) * 180.0 / torch.pi
        mean_angle = (mean_angle + 360) % 360
        return mean_angle

    radians = non_outliers * torch.pi / 180.0
    x_coords = torch.cos(radians)
    y_coords = torch.sin(radians)

    mean_x = torch.mean(x_coords)
    mean_y = torch.mean(y_coords)

    mean_angle = torch.atan2(mean_y, mean_x) * 180.0 / torch.pi
    mean_angle = (mean_angle + 360) % 360
    return mean_angle


class ImageCrops(torch.utils.data.Dataset):
    """Dataset that yields cropped image regions for CLIP classification."""

    def __init__(self, image: Image.Image, objects, transform):
        self._image = image.convert("RGB")
        self._objects = objects
        self.transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, idx):
        box, _ = self._objects[idx]
        img_crop = self._image.crop(box[:4])
        return self.transform(img_crop), 0
