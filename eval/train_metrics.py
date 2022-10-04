import csv
import json
import os
import pickle as pkl
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from eval.utils.log_utils import find_output_folder_from_slurm_log, train_failed


def read_log(dir_name: str):
    base_path = "dpr_logs/"
    dir_path = os.path.join(base_path, dir_name)
    files = os.listdir(dir_path)
    for file in files:
        if ".out" in file:
            file_path = os.path.join(dir_path, file)

    with open(file_path, "r") as log:
        log_lines = log.read().splitlines()

    return log_lines


def order_by_epoch(log_lines: list):
    epochs = {}
    epoch_patt = "\*\*\*\*\* Epoch [0-9]+ \*\*\*\*\*"
    epoch = []
    match_id = 0
    epoch_id = -1
    for line in log_lines:
        match = re.search(epoch_patt, line)
        if match:
            last_epoch_id = epoch_id
            epoch_id = line.split(" ")[-2]
            match_id += 1
            if epoch:
                if last_epoch_id not in epochs:
                    epochs[last_epoch_id] = []

                epochs[last_epoch_id].append(epoch)

            epoch = []
        if match_id > 0:
            epoch.append(line)

    return epochs


def date_from_str(string):
    return datetime.strptime(string, "%Y-%m-%d")


def time_from_str(string):
    return datetime.strptime(string, "%H-%M-%S")


def find_target_logs(start: str, end: str, dir: str, target_log_name: str):
    list_dir = os.listdir(dir)
    start_date = date_from_str(start)
    end_date = date_from_str(end)
    target_dir_path = [
        os.path.join(dir, folder)
        for folder in list_dir
        if start_date <= date_from_str(folder) <= end_date
    ]

    target_paths = dict()
    for (idx, path) in enumerate(target_dir_path):

        logs = os.listdir(path)
        final_target_path = []
        for log in logs:
            final_target_path.append(os.path.join(path, log, target_log_name))
        target_paths[idx] = final_target_path
    return target_paths


def get_dict_epochs(log_lines: list):
    epochs = {}
    epoch_patt = "\*\*\*\*\* Epoch [0-9]+ \*\*\*\*\*"
    epoch_id = -1
    for line in log_lines:
        match = re.search(epoch_patt, line)
        if match:
            epoch_id = int(line.split(" ")[-2])
            if epoch_id not in epochs:
                epochs[epoch_id] = []
                current_epoch = epochs[epoch_id]

        if epoch_id > -1:
            current_epoch.append(line)

    return epochs


def get_epoch_range(dict_epoch):
    epoch_ids = [key for key in dict_epoch.keys() if isinstance(key, int)]
    if not epoch_ids:
        return (None, None)
    end_epoch = max(epoch_ids)
    start_epoch = min(epoch_ids)
    return (start_epoch, end_epoch)


def find_sublogs_to_merge(ranges):
    best_log_idx = []
    interval = [
        end - start if start is not None and end is not None else -1
        for (start, end) in ranges
    ]
    best_interval_idx = np.argmax(interval)
    min_start = min([start for start, _ in ranges if start is not None])

    best_idx, best_size = -1, -1
    for idx, size in enumerate(interval):
        if ranges[idx][0] == min_start:
            if size >= best_size:
                best_idx = idx

    idx_to_merge = [best_idx]
    best_start, best_end = ranges[best_idx]
    for idx, range in enumerate(ranges):
        start, end = range
        if range != (None, None):
            if end > best_end and start > best_start:
                idx_to_merge.append(idx)

            if start > best_end:
                idx_to_merge.append(idx)

    return idx_to_merge


def find_relevant_logs(paths: str, index: int):
    list_dict = []
    ranges = []
    for path in paths:

        with open(path) as log:
            lines = log.read().splitlines()
            dict_epoch = get_dict_epochs(lines)

            if index == 0:
                for (line_idx, line) in enumerate(lines):
                    match = re.search("(\*\*\*\*\* Epoch 0 \*\*\*\*\*)", line)
                    if match:

                        dict_epoch["start"] = lines[:line_idx]

        list_dict.append(dict_epoch)
        ranges.append(get_epoch_range(dict_epoch))

    idx_to_merge = find_sublogs_to_merge(ranges)
    relevant_log = [list_dict[idx] for idx in idx_to_merge]

    return relevant_log


def merge_outputs(start: str, end: str, dir: str, target_log_name: str):

    target_paths_dict = find_target_logs(start, end, dir, target_log_name)

    all_relevant_dict = []
    for (idx, paths) in target_paths_dict.items():
        all_relevant_dict.extend(find_relevant_logs(paths, idx))

    merge_dict = dict()

    if len(all_relevant_dict) > 1:
        for idx in range(len(all_relevant_dict) - 1):

            pt = idx + 1
            current_dict = all_relevant_dict[idx]
            next_dict = all_relevant_dict[pt]

            intersection = set(current_dict.keys()).intersection(set(next_dict.keys()))
            if intersection:
                for int_idx in intersection:
                    current_dict.pop(int_idx)

            merge_dict.update(current_dict)
    else:
        merge_dict.update(all_relevant_dict[0])

    return merge_dict


def parse_log(dir_name: str = None, clean_epochs: dict = None):
    if dir_name:
        log_lines = read_log(dir_name)
        epochs = order_by_epoch(log_lines)
        clean_epochs = {}
        for _id, epoch in epochs.items():
            clean_epochs[_id] = epoch[-1]

    train_loss = dict()
    nll_validation_loss = dict()
    avg_validation_rank = dict()
    correct_predictions = dict()
    correct_predictions_ratio = dict()
    for _id, epoch_log in clean_epochs.items():
        for log in epoch_log:
            if "Av Loss per epoch=" in log:
                train_loss[_id] = float(log.split("=")[-1])
            if "NLL Validation: loss =" in log:
                nll_validation_loss[_id] = float(log.split(" ")[7][:-1])
                correct_predictions_ratio[_id] = float(log.split(" ")[-1])
            if "Av.rank validation: average rank" in log:
                avg_validation_rank[_id] = float(log.split(" ")[7][:-1])
            if "epoch total correct predictions=" in log:
                correct_predictions[_id] = int(log.split("=")[-1])

    return (
        train_loss,
        nll_validation_loss,
        avg_validation_rank,
        correct_predictions,
        correct_predictions_ratio,
    )


def plot_metrics(
    train_loss, nll_validation_loss, avg_validation_rank, correct_predictions
):

    fig = px.line(
        x=list(train_loss.keys())[: len(nll_validation_loss)],
        y=[
            list(train_loss.values())[: len(nll_validation_loss)],
            nll_validation_loss.values(),
        ],
    )
    fig.show()

    if avg_validation_rank:
        fig = px.line(x=avg_validation_rank.keys(), y=avg_validation_rank.values())
        fig.show()


def is_valid_output_log(lines: List[str]) -> bool:
    for line in lines:
        match = re.search(
            "(Epoch: (\d{1}|\d{2}|\d{3}): Step: .+, loss=.+, lr=.+)", line
        )
        if match:
            return True

    return False


def main():

    slurm_dir = "/media/simon/Samsung_T5/CEDAR/data/datasets/dpr_log/slurm_log"
    outputs_dir = "/media/simon/Samsung_T5/CEDAR/data/datasets/dpr_log/outputs"
    # ckpt_dir = "/media/simon/Samsung_T5/CEDAR/data/dpr-ckpt"

    # clean_failed_train(
    #     slurm_dir=Path(slurm_dir),
    #     outputs_dir=Path(outputs_dir),
    #     ckpt_dir=Path(ckpt_dir),
    # )
    # # with open(slurm_logs_path, 'r') as slurms :

    # # merge_outputs()

    # QA-fr

    slurm_dir_path = Path(slurm_dir)
    outputs_dir = Path(outputs_dir)

    for folder in outputs_dir.iterdir():
        for subfolder in folder.iterdir():
            log_path = subfolder.joinpath("train_dense_encoder.log")
            if log_path.exists():
                with log_path.open() as log:
                    llines = log.read().splitlines()
                    if not is_valid_output_log(llines):
                        print(log_path)
                        # shutil.rmtree(log_path.parent)
                        # print(log_path.parent)

    valid_outputs_logs = []
    for folder in slurm_dir_path.iterdir():
        if folder.name.startswith("run"):
            folder_path = slurm_dir_path.joinpath(folder)
            err_file, out_file = [file for file in sorted(folder_path.iterdir())]

            if not train_failed(err_file):
                outputs_files = find_output_folder_from_slurm_log(
                    slurm_out_to_parse=out_file, outputs_dir=outputs_dir
                )
                valid_outputs_logs.extend(
                    [file.joinpath("train_dense_encoder.log") for file in outputs_files]
                )

    print(valid_outputs_logs)

    pass


if __name__ == "__main__":
    main()

    # slurm_dir = "/media/simon/Samsung_T5/CEDAR/data/datasets/dpr_log/slurm_log"
    # outputs_dir = "/media/simon/Samsung_T5/CEDAR/data/datasets/dpr_log/outputs"
    # slurm_dir_path = Path(slurm_dir)
    # outputs_dir = Path(outputs_dir)

    # test_dir = outputs_dir.joinpath("test")
    # test_dir.mkdir(exist_ok=True)

    # test_file = test_dir.joinpath("test.txt")

    # with test_file.open("w") as f:
    #     f.write("test")
