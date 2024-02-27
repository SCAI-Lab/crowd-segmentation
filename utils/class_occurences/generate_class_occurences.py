import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.scene_class import classes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_class_occurences(dir_path: str) -> None:
    label_path = f"{dir_path}/label"
    label_list = os.listdir(label_path)

    input_list = os.listdir(f"{dir_path}/input")
    if len(label_list) != len(input_list) or label_list[-1] != input_list[-1]:
        raise Exception(
            f"Label and input list are not correct Length label {len(label_list)}, Length input {len(input_list)}"
        )
    data = np.zeros((len(classes), len(label_list)))

    for idx, label in enumerate(label_list):
        img = Image.open(f"{label_path}/{label}").convert("RGB")
        img_array = np.asarray(img)

        if img_array.shape != (256, 256, 3):
            raise Exception(f"Image shape {img_array.shape}")

        for cls in classes:
            cls_id = cls.id
            cls_col = list(cls.color)

            occ = np.all(img_array == cls_col, axis=-1)
            sum = np.sum(occ)
            data[cls_id, idx] = sum

    np.savetxt(f"{dir_path}/occurences.csv", data.T, delimiter=";")
    logger.info(f"{dir_path}/occurences.csv saved successfully")


def bulk_generation(data_dir: str) -> None:
    excluded_dirs = ["label_test"]
    folders = os.listdir(data_dir)
    if "label" in folders and "input" in folders:
        try:
            logger.info(f"Generating {data_dir}/occurences.csv")
            get_class_occurences(data_dir)
        except Exception as e:
            logger.error(f"{data_dir}/occurences.csv failed to generate")
            logger.error(e)
    else:
        for folder in folders:
            if folder not in excluded_dirs:
                bulk_generation(data_dir=f"{data_dir}/{folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input path",
        default="",
    )

    opts = parser.parse_args()
    bulk_generation(data_dir=opts.input)
