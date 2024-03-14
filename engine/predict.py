import argparse
import os
import sys
from argparse import Namespace
from glob import glob
from typing import Optional

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../DeepLabV3Plus"))

import time

from dataset.loader import TorchDataset
from DeepLabV3Plus.datasets import Cityscapes, VOCSegmentation
from DeepLabV3Plus.network import convert_to_separable_conv, modeling
from DeepLabV3Plus.utils import set_bn_momentum
from engine.train import setup_logger
from engine.utils.predict_img import predict_img
from network.grayscale_layer import GrayscaleLayer


def main(
    cfg_path: str,
    input: str,
    save_location: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    version: str = "default",
) -> None:
    logger = setup_logger(f"prediction_logger_{version}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        opts = Namespace(**cfg)

    opts.input = input

    if opts.dataset.lower() == "voc":
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == "cityscapes":
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == "daav":
        opts.num_classes = 7
        decode_fn = TorchDataset.decode_target

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device}")

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ["png", "jpeg", "jpg", "JPEG"]:
            files = glob(os.path.join(opts.input, "**/*.%s" % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model (all models are 'constructed at network.modeling)
    model = modeling.__dict__[opts.model](num_classes=21, output_stride=opts.output_stride)
    if opts.separable_conv and "plus" in opts.model:
        convert_to_separable_conv(model.classifier)
    set_bn_momentum(model.backbone, momentum=0.01)

    if opts.dataset == "daav":
        # Load pretrained model
        file_path = os.path.join(
            os.path.dirname(__file__), "../dataset/deeplabv3plus_mobilenet_voc_os16.pth"
        )
        model.load_state_dict(torch.load(file_path)["model_state"])

        if opts.pretrained:
            # Ensure the model's weights are not updated during training
            for param in model.parameters():
                param.requires_grad = False

        # Add layer to front to use grayscale images instead of RGB
        model.backbone.low_level_features.insert(0, GrayscaleLayer())

        # Add additional layer to back to use correct number of classes while using pretrained model
        model.classifier.classifier.insert(
            4, nn.Conv2d(21, opts.num_classes, kernel_size=(1, 1), stride=(1, 1))
        )

    if checkpoint_path is not None:
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % checkpoint_path)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if opts.dataset in ["voc", "cityscapes"]:
        if opts.crop_val:
            transform = transforms.Compose(
                [
                    transforms.Resize(opts.crop_size),
                    transforms.CenterCrop(opts.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    if save_location is not None:
        os.makedirs(save_location, exist_ok=True)

    with torch.no_grad():
        start = time.time()
        model = model.eval()
        if opts.dataset == "daav":
            for idx, img_path in enumerate(tqdm(image_files)):
                ext = os.path.basename(img_path).split(".")[-1]
                img_name = os.path.basename(img_path)[: -len(ext) - 1]

                pred_img_path = f"{save_location}/{img_name}.png"
                predict_img(
                    model=model,
                    device=device,
                    input_img_path=img_path,
                    predict_img_path=pred_img_path,
                )
        else:
            for img_path in tqdm(image_files):
                ext = os.path.basename(img_path).split(".")[-1]
                img_name = os.path.basename(img_path)[: -len(ext) - 1]
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0)  # To tensor of NCHW
                img = img.to(device)

                pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
                colorized_preds = decode_fn(pred).astype("uint8")
                colorized_preds = Image.fromarray(colorized_preds)
                if save_location:
                    colorized_preds.save(os.path.join(save_location, img_name + ".png"))

        duration = time.time() - start
        print(f"Average calculation time: {duration/len(image_files)}")


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yml")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to the input data to predict",
        default="",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the location where the predictions should be saved at",
        default="",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="Location where the checkpoint to load is saved at.",
        default="",
    )

    opts = parser.parse_args()
    main(
        cfg_path=config_path,
        input=opts.input,
        save_location=opts.output,
        checkpoint_path=opts.checkpoint,
    )
