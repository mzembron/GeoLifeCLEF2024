from typing import Dict, List, Optional, Tuple, Union

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL.Image import Image
from torch import Tensor, cat

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.utils import IMAGENET_NORMALIZE


class BYOLView1Transform:
    def __init__(
        self,
        input_size: int = 224,
        channels: int = 4,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        solarization_prob: float = 0.0,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        if channels == 4:
            color_jitter = ColorJitter4Channel(
                brightness=cj_strength * cj_bright,
                contrast=cj_strength * cj_contrast,
                saturation=cj_strength * cj_sat,
                hue=cj_strength * cj_hue,
            )
        else:
            color_jitter = T.ColorJitter(
                brightness=cj_strength * cj_bright,
                contrast=cj_strength * cj_contrast,
                saturation=cj_strength * cj_sat,
                hue=cj_strength * cj_hue,
            )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            if channels == 4:
                transform += [T.Normalize(mean=normalize["mean"]+[sum(normalize["mean"])/3], std=normalize["std"]+[sum(normalize["std"])/3])]
            else:
                transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class BYOLView2Transform:
    def __init__(
        self,
        input_size: int = 224,
        channels: int = 4,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.1,
        solarization_prob: float = 0.2,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):  
        if channels == 4:
            color_jitter = ColorJitter4Channel(
                brightness=cj_strength * cj_bright,
                contrast=cj_strength * cj_contrast,
                saturation=cj_strength * cj_sat,
                hue=cj_strength * cj_hue,
            )
        else:
            color_jitter = T.ColorJitter(
                brightness=cj_strength * cj_bright,
                contrast=cj_strength * cj_contrast,
                saturation=cj_strength * cj_sat,
                hue=cj_strength * cj_hue,
            )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)), 
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees), 
            T.RandomHorizontalFlip(p=hf_prob), 
            T.RandomVerticalFlip(p=vf_prob), 
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            if channels == 4:
                transform += [T.Normalize(mean=normalize["mean"]+[sum(normalize["mean"])/3], std=normalize["std"]+[sum(normalize["std"])/3])]
            else:
                transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class ColorJitter4Channel(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img):
        img = TF.pil_to_tensor(img)
        if img.shape[0] != 4:
            # print("/n", img.size, "/n")
            raise ValueError(f"Input image must have 4 channels (RGBA) {img.shape}")

        img_rgb, img_nir = img[:3], img[3]  #split tensor into 3 and 1 channels
        img_rgb = self.color_jitter(img_rgb)

        img = cat((img_rgb, img_nir.unsqueeze(0)), dim=0) #connect back img_rgb and img_nir 

        return TF.to_pil_image(img)