import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def get_date_in_log_lines(line: str) -> Tuple[str]:
    """
    Find and extract the date of a log's line.
    """
    search = re.search("(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),\d+", line)
    date = search.group()
    day, hour = date.split(" ")
    hour = hour.split(",")[0]

    return day, hour


def find_output_folder_from_slurm_log(
    slurm_out_to_parse: Path, outputs_dir: Path
) -> List[Path]:
    """
    Find ouptputs subfolders to delete beteween a range of date.
    """
    with slurm_out_to_parse.open() as out_log:
        lines = out_log.read().splitlines()
        start, end = lines[0], lines[-1]
        start_day, start_hour = get_date_in_log_lines(start)
        end_day, end_hour = get_date_in_log_lines(end)

        output_folders = [
            folder
            for folder in outputs_dir.iterdir()
            if datetime.strptime(end_day, "%Y-%M-%d")
            >= datetime.strptime(folder.name, "%Y-%M-%d")
            >= datetime.strptime(start_day, "%Y-%M-%d")
        ]

        folders_path = []
        for folder in output_folders:

            subfolders = [
                subfolder
                for subfolder in folder.iterdir()
                if datetime.strptime(f"{end_day} {end_hour}", "%Y-%m-%d %H:%M:%S")
                >= datetime.strptime(
                    f"{folder.name} {subfolder.name}", "%Y-%m-%d %H-%M-%S"
                )
                >= datetime.strptime(f"{start_day} {start_hour}", "%Y-%m-%d %H:%M:%S")
            ]
            folders_path.extend(subfolders)

    return folders_path


def train_failed(err_file: Path):
    with err_file.open() as err_log:
        lines = err_log.read().splitlines()
        for line in lines:
            match = re.match(".*(error *:).*", line.lower())
            if match:
                return True

    return False


def clean_failed_train(
    slurm_dir: Path, outputs_dir: Path, ckpt_dir: Path
) -> List[Tuple[Path]]:
    """
    Clean logs and ckpt folders of failed train.
    """
    folders_to_delete = []
    for folder in slurm_dir.iterdir():
        if folder.name.startswith("run"):
            folder_path = slurm_dir.joinpath(folder)
            err_file, out_file = [file for file in sorted(folder_path.iterdir())]

            if train_failed(err_file):
                outputs_to_clean = find_output_folder_from_slurm_log(
                    out_file, outputs_dir
                )
                folders_to_delete.append(
                    folder,
                    outputs_to_clean,
                )

    return folders_to_delete


# if __name__ == "__main__":
#     slurm_dir = "/media/simon/Samsung_T5/CEDAR/data/datasets/dpr_log/slurm_log"
#     outputs_dir = "/media/simon/Samsung_T5/CEDAR/data/datasets/dpr_log/outputs"
#     ckpt_dir = "/media/simon/Samsung_T5/CEDAR/data/dpr-ckpt"

#     clean_failed_train(
#         slurm_dir=Path(slurm_dir),
#         outputs_dir=Path(outputs_dir),
#         ckpt_dir=Path(ckpt_dir),
#     )
