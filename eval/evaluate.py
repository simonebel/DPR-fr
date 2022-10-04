import json
import logging
import os
import pickle as pkl
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
from haystack.document_stores import FAISSDocumentStore
from haystack.schema import Document
from omegaconf import DictConfig

from eval.convert_dpr_original_checkpoint_to_pytorch import convert
from eval.models import DensePassageRetriever
from eval.utils.conf_utils import get_log, set_cfg
from eval.utils.score_utils import mean_average_precision, precision_at_k, recall_at_k

logger = get_log()


def convert_tables_to_faiss_dicts(tables: List[Dict], size: int = None) -> List[Dict]:
    """
    Convert Insee tables dataset to haystack format
    """
    dicts = []
    featured_tables = tables[:size] if size else tables
    for table in featured_tables:
        description = table["description"] if "description" in table else ""
        text = "{} {} {} {} {}".format(
            table["title"],
            " ".join(table["header_rows"].values()),
            " ".join(table["header_columns"].values()),
            table["comment"],
            description,
        )
        doc = {"content": text, "meta": {"id": table["id"].decode()}}
        dicts.append(doc)

    return dicts


def index_embeddings(
    faiss_dir_path: str, tables: list, encoder_ckpt_path: str, flush_faiss: bool = False
) -> DensePassageRetriever:
    """
    Init a retriever and index the embeddings with Faiss
    """
    if flush_faiss:
        shutil.rmtree(faiss_dir_path)

    if not os.path.isdir(faiss_dir_path):
        os.makedirs(faiss_dir_path)

    faiss_index = os.path.join(faiss_dir_path, "faiss.index")
    faiss_db = os.path.join(faiss_dir_path, "faiss.db")

    question_encoder_path = os.path.join(encoder_ckpt_path, "question_encoder")
    question_ctx_path = os.path.join(encoder_ckpt_path, "ctx_encoder")

    if os.path.isfile(faiss_index):

        document_store = FAISSDocumentStore.load(faiss_index)

    else:
        document_store = FAISSDocumentStore(
            similarity="dot_product", sql_url=f"sqlite:///{faiss_db}"
        )

        dicts = convert_tables_to_faiss_dicts(tables)
        document_store.write_documents(dicts)

        retriever = DensePassageRetriever(
            query_embedding_model=question_encoder_path,
            passage_embedding_model=question_ctx_path,
            max_seq_len_query=64,
            max_seq_len_passage=256,
            batch_size=8,
            use_gpu=True,
            embed_title=True,
            use_fast_tokenizers=True,
        )

    if not os.path.isfile(faiss_index):
        document_store.update_embeddings(retriever)
        document_store.save(faiss_index)

    return retriever


def retrives_topk_documents(query, retriever, top_k: int = 20) -> List[Document]:
    """
    Retrieve the top k documents given a query
    """
    candidate_documents = retriever.retrieve(
        query=query,
        top_k=top_k,
    )
    return candidate_documents


def evaluate(test_file: Path, retriever: DensePassageRetriever) -> Tuple[float]:
    """
    Compute precision, recall and map for a given test set and for the online test set
    """
    with open(test_file) as json_file:
        test = json.load(json_file)
        questions = [ex["question"] for ex in test]
        golds = [ex["gold"] for ex in test]

    answers = []
    for question, gold in zip(questions, golds):
        res = retrives_topk_documents(question, retriever)
        res_ids = [doc.meta["id"] for doc in res]
        answers.append({"gold": gold, "predicted": res_ids})

    mean_avg_p = mean_average_precision(answers)
    p_k_vals = [
        precision_at_k(20, gold, predicted)
        for gold, predicted in map(lambda x: (x["gold"], x["predicted"]), answers)
    ]
    r_k_vals = [
        recall_at_k(20, gold, predicted)
        for gold, predicted in map(lambda x: (x["gold"], x["predicted"]), answers)
    ]

    print(f"Retriever - Recall at 20: {np.mean(r_k_vals)}")
    print(f"Retriever - Precision at 20: {np.mean(p_k_vals)}")
    print(f"Retriever - Mean Average Precision: {mean_avg_p}")

    return np.mean(r_k_vals), np.mean(p_k_vals), mean_avg_p


def generate_dense_and_evaluate(
    ckpt_path: Path,
    dest_path: Path,
    table_path: Path,
    test_set_path: Path,
    test_online_path: Path,
    flush_faiss: bool = False,
) -> DensePassageRetriever:
    """
    Full evaluation pipeline :
    - Convert the ckpt to HF
    - Generate dense embeddings for every table in redis
    - Index the embeddings with faiss
    - Evaluate the model on the specific dataset test set and on the online 'vdf' test set
    """

    faiss_dir = Path(dest_path, "faiss")
    ckpt_out = Path(dest_path, "ckpt")

    convert(ckpt_path, ckpt_out)

    tables = pkl.load(open(table_path, "rb"))

    retriever = index_embeddings(
        faiss_dir_path=faiss_dir,
        tables=tables,
        encoder_ckpt_path=ckpt_out,
        flush_faiss=flush_faiss,
    )

    if test_set_path:
        evaluate(test_set_path, retriever)

    evaluate(test_online_path, retriever)

    return retriever


@hydra.main(config_path="../conf", config_name="statcheck_val")
def main(cfg: DictConfig):

    cfg = set_cfg(cfg)
    retriever = generate_dense_and_evaluate(
        ckpt_path=Path(cfg.src_ckpt),
        dest_path=cfg.dest_path,
        table_path=Path(cfg.ctx_sources.insee_tables.file),
        test_set_path=Path(cfg.test_set_path),
        test_online_path=Path(cfg.datasets.online_test.file),
    )


if __name__ == "__main__":
    main()
