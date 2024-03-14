import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from matplotlib import ticker
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../DeepLabV3Plus"))

from utils.denormalise import Denormalize
from utils.grayscale_to_rgb import grayscale_to_rgb


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists("results"):
            os.mkdir("results")
        denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append((images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (
                        np.array(grayscale_to_rgb(image.astype(np.uint8).T))
                        if opts.dataset == "daav"
                        else (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    )
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save("results/%d_image.png" % img_id)
                    Image.fromarray(target).save("results/%d_target.png" % img_id)
                    Image.fromarray(pred).save("results/%d_pred.png" % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis("off")
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(ticker.NullLocator())
                    ax.yaxis.set_major_locator(ticker.NullLocator())
                    plt.savefig(
                        "results/%d_overlay.png" % img_id, bbox_inches="tight", pad_inches=0
                    )
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples
