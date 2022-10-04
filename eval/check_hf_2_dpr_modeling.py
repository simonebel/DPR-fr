import csv
import logging

import torch
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from transformers import CamembertTokenizerFast

from eval.models import DPRQuestionEncoder
from eval.models.haystack_dense_retriever import DensePassageRetriever
from eval.utils import get_log

logger = get_log()


def generate_hf_embeddings(queries):

    tokenizer = CamembertTokenizerFast.from_pretrained(
        pretrained_model_name_or_path="camembert-base",
        do_lower_case=True,
        use_fast=True,
    )
    with torch.no_grad():
        model = DPRQuestionEncoder.from_pretrained(
            "/media/simon/Samsung_T5/CEDAR/data/dpr-ckpt/question_encoder/"
        )
        inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
        embeddings = model(**inputs).pooler_output

    return embeddings


def generate_haystack_embeddings(queries):

    retriever = DensePassageRetriever(
        document_store=InMemoryDocumentStore(),
        query_embedding_model="/media/simon/Samsung_T5/CEDAR/data/dpr-ckpt/question_encoder/",
        passage_embedding_model="/media/simon/Samsung_T5/CEDAR/data/dpr-ckpt/ctx_encoder/",
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=8,
        use_gpu=False,
        embed_title=True,
        use_fast_tokenizers=True,
    )

    embeddings = retriever.embed_queries(queries)
    return embeddings


def main():

    with open(
        "/media/simon/Samsung_T5/CEDAR/data/datasets/insee_ref_dpr/test.csv", "r"
    ) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        queries = [line[0] for line in reader]

    queries = queries[:5]

    hf_embeddings = generate_hf_embeddings(queries)
    haystack_embeddings = torch.from_numpy(generate_haystack_embeddings(queries))

    max_absolute_diff = torch.max(torch.abs(hf_embeddings - haystack_embeddings)).item()

    logger.info(f"max_absolute_diff = {max_absolute_diff}")

    success = torch.allclose(hf_embeddings, haystack_embeddings, atol=1e-3)
    if not success:
        raise Exception("Not Same output")
    else:
        logger.info("Sanity check success, same output")


if __name__ == "__main__":
    main()
