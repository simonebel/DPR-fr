import glob
import json
import logging
import pickle
import time
import zlib
from typing import List, Tuple, Dict, Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from dpr.utils.data_utils import RepTokenSelector
from dpr.data.qa_validation import (
    calculate_matches,
    calculate_chunked_matches,
    calculate_matches_from_meta,
)
from dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from dpr.models import init_biencoder_components
from dpr.models.biencoder import (
    BiEncoder,
    _select_span_with_token,
)
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)

logger = logging.getLogger()
setup_logger(logger)


from dense_retriever import DenseRPCRetriever, LocalFaissRetriever, get_all_passages


@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    saved_state = load_states_from_checkpoint(cfg.model_file)

    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )

    logger.info("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    encoder_path = cfg.encoder_path
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
    )
    encoder.eval()

    model_to_load = get_model_obj(encoder)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    questions = []
    questions_text = []
    question_answers = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)

    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    total_queries = len(qa_src)
    for i in range(total_queries):
        qa_sample = qa_src[i]
        question, answers = qa_sample.query, qa_sample.answers
        questions.append(question)
        question_answers.append(answers)

    logger.info("questions len %d", len(questions))
    logger.info("questions_text len %d", len(questions_text))

    if cfg.rpc_retriever_cfg_file:
        index_buffer_sz = 1000
        retriever = DenseRPCRetriever(
            encoder,
            cfg.batch_size,
            tensorizer,
            cfg.rpc_retriever_cfg_file,
            vector_size,
            use_l2_conversion=cfg.use_l2_conversion,
        )
    else:
        index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
        logger.info("Local Index class %s ", type(index))
        index_buffer_sz = index.buffer_size
        index.init_index(vector_size)
        retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)

    logger.info("Using special token %s", qa_src.special_query_token)
    questions_tensor = retriever.generate_question_vectors(
        questions, query_token=qa_src.special_query_token
    )

    if qa_src.selector:
        logger.info("Using custom representation token selector")
        retriever.selector = qa_src.selector

    index_path = cfg.index_path
    if cfg.rpc_retriever_cfg_file and cfg.rpc_index_id:
        retriever.load_index(cfg.rpc_index_id)
    elif index_path and index.index_exists(index_path):
        logger.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        # send data for indexing
        id_prefixes = []
        ctx_sources = []
        for ctx_src in cfg.ctx_datatsets:
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix)
            ctx_sources.append(ctx_src)
            logger.info("ctx_sources: %s", type(ctx_src))

        logger.info("id_prefixes per dataset: %s", id_prefixes)

        # index all passages
        ctx_files_patterns = cfg.encoded_ctx_files

        logger.info("ctx_files_patterns: %s", ctx_files_patterns)
        if ctx_files_patterns:
            assert len(ctx_files_patterns) == len(
                id_prefixes
            ), "ctx len={} pref leb={}".format(
                len(ctx_files_patterns), len(id_prefixes)
            )
        else:
            assert (
                index_path or cfg.rpc_index_id
            ), "Either encoded_ctx_files or index_path pr rpc_index_id parameter should be set."

        input_paths = []
        path_id_prefixes = []
        for i, pattern in enumerate(ctx_files_patterns):
            pattern_files = glob.glob(pattern)
            print(pattern_files)
            pattern_id_prefix = id_prefixes[i]
            input_paths.extend(pattern_files)
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
        logger.info("Reading all passages data from files: %s", input_paths)

        retriever.index_encoded_data(
            input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes
        )
        if index_path:
            retriever.index.serialize(index_path)

    # get top k results
    top_results_and_scores = retriever.get_top_docs(
        questions_tensor.numpy(), cfg.n_docs
    )

    if cfg.use_rpc_meta:
        questions_doc_hits = validate_from_meta(
            question_answers,
            top_results_and_scores,
            cfg.validation_workers,
            cfg.match,
            cfg.rpc_meta_compressed,
        )
        if cfg.out_file:
            save_results_from_meta(
                questions,
                question_answers,
                top_results_and_scores,
                questions_doc_hits,
                cfg.out_file,
                cfg.rpc_meta_compressed,
            )
    else:
        all_passages = get_all_passages(ctx_sources)
        if cfg.validate_as_tables:

            questions_doc_hits = validate_tables(
                all_passages,
                question_answers,
                top_results_and_scores,
                cfg.validation_workers,
                cfg.match,
            )

        else:
            questions_doc_hits = validate(
                all_passages,
                question_answers,
                top_results_and_scores,
                cfg.validation_workers,
                cfg.match,
            )

        if cfg.out_file:
            save_results(
                all_passages,
                questions_text if questions_text else questions,
                question_answers,
                top_results_and_scores,
                questions_doc_hits,
                cfg.out_file,
            )

    if cfg.kilt_out_file:
        kilt_ctx = next(
            iter([ctx for ctx in ctx_sources if isinstance(ctx, KiltCsvCtxSrc)]), None
        )
        if not kilt_ctx:
            raise RuntimeError("No Kilt compatible context file provided")
        assert hasattr(cfg, "kilt_out_file")
        kilt_ctx.convert_to_kilt(qa_src.kilt_gold_file, cfg.out_file, cfg.kilt_out_file)


if __name__ == "__main__":
    main()
