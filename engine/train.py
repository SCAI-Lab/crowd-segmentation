import logging
import os
import random
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../DeepLabV3Plus"))

from torch.utils.data import DataLoader, random_split

from dataset.loader import TorchDataset
from DeepLabV3Plus.main import get_dataset, validate
from DeepLabV3Plus.network import convert_to_separable_conv, modeling
from DeepLabV3Plus.utils import (
    Denormalize,
    FocalLoss,
    PolyLR,
    grayscale_to_rgb,
    mkdir,
    set_bn_momentum,
)
from DeepLabV3Plus.utils.visualizer import Visualizer
from engine.loss_weight import get_loss_weights
from engine.stream_seg_metrics import StreamSegMetrics
from network.grayscale_layer import GrayscaleLayer


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    with open(os.path.join(os.path.dirname(__file__), "../config/logging.yml"), "r") as f:
        logging_cfg = yaml.safe_load(f)

    formatter = logging.Formatter(fmt=logging_cfg["format"], datefmt=logging_cfg["datefmt"])

    handler = logging.FileHandler(logging_cfg["filename"])
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def main(cfg_path: str, version: str = "default") -> None:
    logger = setup_logger(f"training_logger_{version}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        opts = Namespace(**cfg)

    if opts.dataset.lower() == "daav":
        opts.num_classes = opts.num_classes
    elif opts.dataset.lower() == "voc":
        opts.num_classes = 21
    elif opts.dataset.lower() == "cityscapes":
        opts.num_classes = 19

    vis = Visualizer(port=opts.vis_port, env=opts.vis_env) if opts.enable_vis else None

    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == "voc" and not opts.crop_val:
        opts.val_batch_size = 1

    if opts.dataset == "daav":
        dataset = TorchDataset(opts.data_root)
        if not os.path.isdir(opts.validation_dir):
            logger.info("Using random split for training and validation set.")
            train_dst, val_dst = random_split(dataset=dataset, lengths=(0.8, 0.2))
        else:
            logger.info(f"Using {opts.validation_dir} as location of validation set")
            train_dst = dataset
            val_dst = TorchDataset(opts.validation_dir)

        train_loader = DataLoader(
            dataset=train_dst,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2
        )

    elif opts.dataset == "voc" or opts.dataset == "cityscapes":
        train_dst, val_dst = get_dataset(opts=opts)
        train_loader = DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2, drop_last=True
        )  # drop_last=True to ignore single-image batches.
        val_loader = DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2
        )

    else:
        raise Exception(f"There is currently no model implemented with the name {opts.dataset}")

    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)}")

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

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(
        params=[
            {"params": model.backbone.parameters(), "lr": 0.1 * opts.lr},
            {"params": model.classifier.parameters(), "lr": opts.lr},
        ],
        lr=opts.lr,
        momentum=0.9,
        weight_decay=opts.weight_decay,
    )
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == "poly":
        scheduler = PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == "focal_loss":
        criterion = FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == "cross_entropy":
        weights_np = get_loss_weights(
            cfg_file_path=opts.data_root, num_classes=opts.num_classes
        ).astype(float)
        weights = torch.from_numpy(weights_np).to(device).float()
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean", weight=weights)

    def save_ckpt(path):
        """save current model"""
        torch.save(
            {
                "cur_itrs": cur_itrs,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
            },
            path,
        )
        logger.info(f"Model saved as {path}")

    mkdir("checkpoints")
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint["best_score"]
            logger.info(f"Training state restored from {opts.ckpt}")
        logger.info(f"Model restored from {opts.ckpt}")
        del checkpoint  # free memory
    else:
        logger.info("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = (
        np.random.randint(0, len(val_loader), opts.vis_num_samples, np.int32)
        if opts.enable_vis
        else None
    )  # sample idxs for visualization
    denorm = Denormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=metrics,
            ret_samples_ids=vis_sample_id,
        )
        logging.info(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for images, labels in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar("Loss", cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                logger.info(
                    f"Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={interval_loss}"
                )
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt(
                    f"checkpoints/latest_{opts.model}_{opts.dataset}_os{opts.output_stride}_{version}.pth"
                )
                logger.info("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts,
                    model=model,
                    loader=val_loader,
                    device=device,
                    metrics=metrics,
                    ret_samples_ids=vis_sample_id,
                )
                logger.info(metrics.to_str(val_score))
                if val_score["Mean IoU"] > best_score:  # save best model
                    best_score = val_score["Mean IoU"]
                    save_ckpt(
                        f"checkpoints/best_{opts.model}_{opts.dataset}_os{opts.output_stride}_{version}.pth"
                    )

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score["Overall Acc"])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score["Mean IoU"])
                    vis.vis_table("[Val] Class IoU", val_score["Class IoU"])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (
                            (denorm(img) * 255).astype(np.uint8)
                            if opts.dataset != "daav"
                            else np.array(grayscale_to_rgb(img)).T
                        )
                        if opts.dataset == "daav":
                            target = (
                                train_dst.dataset.decode_target(target)
                                .transpose(2, 0, 1)
                                .astype(np.uint8)
                            )
                            lbl = (
                                train_dst.dataset.decode_target(lbl)
                                .transpose(2, 0, 1)
                                .astype(np.uint8)
                            )
                        else:
                            target = (
                                train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            )
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate(
                            (img, target, lbl), axis=2
                        )  # concat along width
                        vis.vis_image(f"Sample {k}", concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yml")
    main(cfg_path=config_path)
