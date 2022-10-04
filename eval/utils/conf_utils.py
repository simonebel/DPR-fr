import logging
import os
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

from dpr.options import setup_logger


def get_log():
    """
    Custom log
    """
    logger = logging.getLogger("StatCheck")
    setup_logger(logger)
    return logger


def set_cfg(cfg: DictConfig) -> DictConfig:
    """
    Set config for evaluation.
    """
    current_date = datetime.now().strftime("%Y_%m_%d")
    eval_dir = Path(os.path.join(cfg.eval_base, cfg.type, current_date))

    if not eval_dir.exists():
        os.makedirs(eval_dir, exist_ok=True)

    num_pefix = str(len(os.listdir(eval_dir)) + 1)

    current_eval_dir = os.path.join(eval_dir, num_pefix)
    os.makedirs(Path(current_eval_dir), exist_ok=True)
    cfg.dest_path = current_eval_dir

    for dataset_name in cfg.datasets.keys():
        if dataset_name.startswith(cfg.type):
            current_dataset = cfg.datasets[dataset_name]
            cfg.test_set_path = current_dataset.file

    return cfg
