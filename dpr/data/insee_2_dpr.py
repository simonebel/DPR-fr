import json
import os
import logging
import argparse
from tqdm import tqdm

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.retriever.sparse import BM25Retriever

from dpr.options import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)

DATA_PATH = "/media/simon/Samsung_T5/CEDAR/data/datasets/insee_dataset.json"
SEED = 2022

parser = argparse.ArgumentParser(
    description="This script is used to convert an insee datasets to a DPR format"
)

parser.add_argument("--data_path", type=str)
parser.add_argument("--out_path", type=str)


def build_ES_index(documents_dicts: list):
    """
    Build Elastic Search Index and retriever
    """
    logger.info("Building Elastic Search Index")
    document_store = ElasticsearchDocumentStore(
        host="localhost", username="", password="", index="document"
    )
    # document_store = InMemoryDocumentStore()
    document_store.write_documents(documents_dicts)
    retriever = BM25Retriever(document_store=document_store)
    return retriever


def build_ES_document(ex: dict, id: str, positive_ctx: dict):
    """
    Convert insee example to Elastic Search Store document
    """
    es_doc = {
        "content": "{} {}".format(positive_ctx["title"], positive_ctx["text"]),
        "meta": {
            "publication_id": ex["publication_id"],
            "redis_id": id,
            "title": positive_ctx["title"],
            "positive_ctx": positive_ctx,
        },
    }

    return es_doc


def flat_headers(header_dict: dict):
    """
    Sequentially flat a header rows or columns
    """
    orderable_header_dict = {int(rx): header for rx, header in header_dict.items()}
    flat_header = [
        orderable_header_dict[rx] for rx in sorted(orderable_header_dict.keys())
    ]
    return " ".join(flat_header).strip()


def generate_positive_ctx(table: dict):
    """
    Generate positive_ctx dict from table dict
    """
    title = table["title"]
    comment = table["comment"] if "comment" in table else ""
    description = table["description"] if "description" in table else ""

    flat_hr = flat_headers(table["header_rows"])
    flat_hc = flat_headers(table["header_columns"])

    text = "{}. {}.".format(flat_hr, flat_hc)
    if comment:
        text = "{} {}.".format(text, comment.strip())
    if description:
        text = "{} {}.".format(text, description.strip())

    positive_ctx = {"title": title, "text": text}

    return positive_ctx


def convert_insee_2_dicts(data_path: str):
    """
    Convert insee summaries dataset to a dpr dataset and generate list of document for Elastic Search Retriever
    """
    logger.info("Converting Insee dataset to Haystack dicts")
    data = json.load(open(data_path, "r"))
    dicts = []
    for ex in tqdm(data.values()):

        gold_tables = ex["gold_tables"]
        for id, table in gold_tables.items():
            positive_ctx = generate_positive_ctx(table)

            es_doc = build_ES_document(ex, id, positive_ctx)
            dicts.append(es_doc)

    return dicts


def get_hard_negative_ctxs(
    retriever: BM25Retriever,
    question: dict,
    publication_id_ctx: str,
    table_id: str,
    n_hard_negative: int,
):
    BM25_pool = retriever.retrieve(query=question, top_k=n_hard_negative)
    hard_negative_ctxs = []
    for document in BM25_pool:
        if (
            document.meta["publication_id"] != publication_id_ctx
            and table_id != document.meta["redis_id"]
        ):
            hard_negative_ctxs.append(document.meta["positive_ctx"])
            break

    return hard_negative_ctxs


def convert_insee_2_dpr(data_path: str, retriever: BM25Retriever):
    """
    Convert insee summaries dataset to a dpr dataset and generate list of document for Elastic Search Retriever
    """
    logger.info("Converting Insee dataset to DPR dataset ")
    dpr_dataset = []
    data = json.load(open(data_path, "r"))
    for ex in tqdm(data.values()):
        question = ex["query"]
        gold_tables = ex["gold_tables"]
        publication_id_ctx = ex["publication_id"]
        for id, table in gold_tables.items():
            positive_ctx = generate_positive_ctx(table)
            hard_negative_ctxs = get_hard_negative_ctxs(
                retriever, question, publication_id_ctx, id, 30
            )

            dpr_dataset.append(
                {
                    "question": question,
                    "answers": [],
                    "positive_ctxs": [positive_ctx],
                    "negative_ctxs": [],
                    "hard_negative_ctxs": hard_negative_ctxs,
                }
            )

    return dpr_dataset


def split_and_save_dataset(dataset: list, out_path: str):
    from sklearn.model_selection import train_test_split
    import random

    random.seed(SEED)

    logger.info("Splitting and saving...")

    questions = map(lambda x: x["question"], dataset)
    map_question_id = {question: idx for idx, question in enumerate(questions)}
    q_ids = list(map_question_id.values())
    train_ids, test_ids, _, _ = train_test_split(
        q_ids, q_ids, test_size=0.1, random_state=SEED
    )

    train_ids, dev_ids, _, _ = train_test_split(
        train_ids, train_ids, test_size=0.1, random_state=SEED
    )

    train = [ex for ex in dataset if map_question_id[ex["question"]] in train_ids]
    random.shuffle(train)
    logger.info("Train set size: {}".format(len(train)))

    dev = [ex for ex in dataset if map_question_id[ex["question"]] in dev_ids]
    random.shuffle(dev)
    logger.info("Dev set size: {}".format(len(dev)))

    test = [ex for ex in dataset if map_question_id[ex["question"]] in test_ids]
    random.shuffle(test)
    logger.info("Test set size: {}".format(len(test)))

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    json.dump(
        train, open("{}/train.json".format(out_path), "w"), indent=4, ensure_ascii=False
    )
    json.dump(
        dev, open("{}/dev.json".format(out_path), "w"), indent=4, ensure_ascii=False
    )
    json.dump(
        test, open("{}/test.json".format(out_path), "w"), indent=4, ensure_ascii=False
    )


def main():
    args = parser.parse_args()

    if not args.data_path or not args.out_path:
        logger.error("You must input data_patah and out_path, exiting")
        quit()

    dicts = convert_insee_2_dicts(args.data_path)
    BM25_retriever = build_ES_index(dicts)
    dpr_dataset = convert_insee_2_dpr(args.data_path, BM25_retriever)
    split_and_save_dataset(dpr_dataset, args.out_path)


if __name__ == "__main__":
    main()
