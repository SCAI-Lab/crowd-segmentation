import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import v2

from dataset.scene_class import train_id_to_color


class TorchDataset(data.Dataset):
    def __init__(self, root: str, split: str = "train") -> None:
        super().__init__()
        self.root = os.path.expanduser(root)

        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(0, 180)),
                v2.PILToTensor(),
            ]
        )

        config_path = os.path.join(self.root, "config.txt")
        if not os.path.exists(config_path):
            raise ValueError(f"There exists no config file in the root path {self.root}")

        with open(config_path, "r") as f:
            dir_names = [x.strip() for x in f.readlines()]

        self.images = []
        self.targets = []
        for dir in dir_names:
            image_dir = os.path.join(self.root, dir, "input")
            mask_dir = os.path.join(self.root, dir, "label")
            datasize = len(os.listdir(image_dir)) - 1
            file_names = [
                str(x).zfill(6) for x in range(150, datasize)
            ]  # The first 150 images don't have enough knowledge of the past

            self.images += [os.path.join(image_dir, x + ".png") for x in file_names]
            self.targets += [os.path.join(mask_dir, x + ".png") for x in file_names]

        assert len(self.images) == len(self.targets)

    # @classmethod
    # def decode_target(cls, target):
    #     return train_id_to_color[target]
    @classmethod
    def encode_target(cls, target):
        target_numpy = np.array(target)

        target_indecies = [
            np.all(target_numpy.T[:, :] == rgb_val, axis=2) * idx
            for idx, rgb_val in enumerate(train_id_to_color)
        ]
        encoded_target = np.sum(np.array(target_indecies), axis=0).astype(np.uint8)
        return torch.from_numpy(encoded_target)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        decoded_img = np.zeros((255, 255, 3))
        for i in range(len(train_id_to_color)):
            decoded_img[mask == i, :] = train_id_to_color[i]
        return decoded_img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert("L")
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)
