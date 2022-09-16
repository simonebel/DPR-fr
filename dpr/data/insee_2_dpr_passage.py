import argparse
import pickle as pkl
import csv
import os
import logging
from tqdm import tqdm
from dpr.options import setup_logger
from .insee_2_dpr import generate_positive_ctx

logger = logging.getLogger(__name__)
setup_logger(logger)

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="The path to the insee tables ")
parser.add_argument("--out_path", type=str, help="The path to the insee psgs ")

def convert_table_to_dpr_passage(tables: list, out_path: str):
    logger.info("Start conversion and writing out...")
    with open(os.path.join(out_path, "psgs_insee.tsv"), "w") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        writer.writerow(["id", "text", "title"])
        for table in tqdm(tables):
            id = table["id"].decode()
            ctx = generate_positive_ctx(table)
            writer.writerow([id, ctx["text"], ctx["title"]])


def main():
    args = parser.parse_args()
    tables = pkl.load(open(args.path, "rb"))
    convert_table_to_dpr_passage(tables, args.out_path)


if __name__ == "__main__":
    main()
