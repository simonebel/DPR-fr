#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""
import logging
from pathlib import Path
from typing import List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from transformers import CamembertModel

from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder, _select_span_with_token
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger
from dpr.utils.data_utils import RepTokenSelector, Tensorizer
from dpr.utils.model_utils import (
    get_model_obj,
    load_states_from_checkpoint,
    setup_for_distributed_mode,
)
from eval.models import DPRQuestionEncoder
from eval.utils import get_log

logger = get_log()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def generate_question_vectors_dpr_orig(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    n = len(questions)
    query_vectors = []

    question_encoder.to(device)

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    batch_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token)
                        for q in batch_questions
                    ]
                else:
                    batch_tensors = [
                        tensorizer.text_to_tensor(" ".join([query_token, q]))
                        for q in batch_questions
                    ]
            elif isinstance(batch_questions[0], T):
                batch_tensors = [q for q in batch_questions]
            else:
                batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]

            q_ids_batch = torch.stack(batch_tensors, dim=0).to(device)
            q_seg_batch = torch.zeros_like(q_ids_batch).to(device)
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor


def generate_question_vectors_dpr_hf(questions, tensorizer, bsz):
    n = len(questions)
    query_vectors = []
    question_ckpt_path = Path(
        "/media/simon/Samsung_T5/CEDAR/data/dpr-ckpt/question_encoder"
    )
    question_encoder = DPRQuestionEncoder.from_pretrained(question_ckpt_path)
    question_encoder.to(device)
    question_encoder.eval()

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]
            batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]
            q_ids_batch = torch.stack(batch_tensors, dim=0).to(device)
            q_seg_batch = torch.zeros_like(q_ids_batch).to(device)
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            out = question_encoder(q_ids_batch, q_attn_mask, q_seg_batch).pooler_output
            query_vectors.extend(out.cpu())

    return torch.stack(query_vectors)


def generate_question_vectors_camembert(questions, tensorizer, bsz):
    n = len(questions)
    query_vectors = []
    question_encoder = CamembertModel.from_pretrained("camembert-base")
    question_encoder.to(device)
    question_encoder.eval()

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]
            batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]
            q_ids_batch = torch.stack(batch_tensors, dim=0).to(device)
            q_seg_batch = torch.zeros_like(q_ids_batch).to(device)
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            out = question_encoder(q_ids_batch, q_attn_mask, q_seg_batch).pooler_output
            query_vectors.extend(out.cpu())

    return torch.stack(query_vectors)


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
    logger.info("Using special token %s", qa_src.special_query_token)

    questions_tensor = generate_question_vectors_dpr_orig(
        encoder,
        tensorizer,
        questions,
        cfg.batch_size,
        query_token=qa_src.special_query_token,
    )

    questions_tensor_hf = generate_question_vectors_dpr_hf(
        questions, tensorizer, cfg.batch_size
    )

    logger.info(f"Orginal DPR emb: {questions_tensor.size()}")
    logger.info(f"HF DPR emb: {questions_tensor_hf.size()}")

    max_absolute_diff = torch.max(
        torch.abs(questions_tensor_hf - questions_tensor)
    ).item()
    logger.info(f"max_absolute_diff = {max_absolute_diff}")
    success = torch.allclose(questions_tensor_hf, questions_tensor, atol=1e-3)
    if not success:
        raise Exception("Not Same output")
    else:
        logger.info("Sanity check success, same output")

    # questions_tensor_camembert = generate_question_vectors_camembert(
    #     questions, tensorizer, cfg.batch_size
    # )
    # logger.info(questions_tensor_camembert)


if __name__ == "__main__":
    main()
