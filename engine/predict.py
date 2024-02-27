import os
import sys
from argparse import Namespace
from glob import glob

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../DeepLabV3Plus"))

from dataset.loader import TorchDataset
from DeepLabV3Plus.datasets import Cityscapes, VOCSegmentation
from DeepLabV3Plus.network import convert_to_separable_conv, modeling, set_bn_momentum
from engine.train import setup_logger


def main(cfg_path: str, input: str, version: str = "default") -> None:
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
        opts.num_classes = 6
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
    model = modeling.__dict__[opts.model](
        num_classes=opts.num_classes, output_stride=opts.output_stride
    )
    if opts.separable_conv and "plus" in opts.model:
        convert_to_separable_conv(model.classifier)
    set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
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

    if opts.save_val_results and opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split(".")[-1]
            img_name = os.path.basename(img_path)[: -len(ext) - 1]
            if opts.dataset == "daav":
                img = Image.open(img_path).convert("L")
            else:
                img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
            colorized_preds = decode_fn(pred).astype("uint8")
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + ".png"))

    if __name__ == "__main__":
        main()
