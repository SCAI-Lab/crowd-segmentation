import csv

import numpy as np


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
