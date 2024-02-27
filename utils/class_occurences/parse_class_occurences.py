import argparse
import csv
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def parse_occurence_files(data_dir: str) -> None:
    occurence_file_list = parse_recursively(data_dir=data_dir, occ_files=[])
    max_val_array = np.zeros(6, dtype=int)
    percentage_array = np.zeros(6)
    bincount = np.zeros(7)

    for file in occurence_file_list:
        with open(f"{file}/occurences.csv", "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=";")
            for line in csvreader:
                data = [float(elem) for elem in line]

                bincount += data

                data = data[1:]

                max_value = np.argmax(data)
                if np.sum(data) > 0:
                    percentage_array += np.array(data) / np.sum(data)

                max_val_array[int(max_value)] += 1

    percentage_array = percentage_array / np.sum(percentage_array)

    logger.info(np.sum(bincount) / (7 * bincount))

    logger.info(max_val_array)
    logger.info(percentage_array)


def parse_recursively(data_dir: str, occ_files: list[str]) -> list[str]:
    if not os.path.isdir(data_dir):
        return occ_files
    files = os.listdir(data_dir)
    if "occurences.csv" in files:
        occ_files.append(data_dir)
    else:
        for file in files:
            if file in ["input", "label", "lidar_2d", "lidar_3d", "overview"]:
                continue

            occ_files = parse_recursively(data_dir=f"{data_dir}/{file}", occ_files=occ_files)

    return occ_files


def get_loss_weights(cfg_file_path: str, num_classes: int) -> np.ndarray:

    with open(f"{cfg_file_path}/config.txt", "r") as f:
        occ_files = [line.rstrip("\n") for line in f]

    bincount = np.zeros(7)
    for file in occ_files:
        with open(f"{cfg_file_path}/{file}/occurences.csv") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=";")
            for line in csvreader:
                data = [float(elem) for elem in line]
                bincount += data

    return np.sum(bincount) / (num_classes * bincount)


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
    parse_occurence_files(data_dir=opts.input)
