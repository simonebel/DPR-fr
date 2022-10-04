# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import os
from pathlib import Path
from typing import Dict, NamedTuple

import torch
from torch.serialization import default_restore_location
from transformers import CamembertConfig, CamembertModel, DPRConfig

from eval.models.hf_dpr import DPRContextEncoder, DPRQuestionEncoder
from eval.utils import get_log

logger = get_log()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)

TYPE = ["ctx_encoder", "question_encoder"]


def load_states_from_checkpoint(model_file: Path) -> NamedTuple:
    """
    Load the checkpoint.
    """
    logger.info(f"Reading saved model from {model_file}")

    state_dict = torch.load(
        model_file, map_location=lambda s, l: default_restore_location(s, "cpu")
    )

    return CheckpointState(**state_dict)


def convert_bi_encoder_fairseq_to_pytorch(
    model_dict: Dict,
    encoder_prefix: str,
) -> Dict:
    """
    Map the state dictionnary's key-value from an original fairseq checkpoint to a hugging face roberta's state dictionnary.
    """
    logger.info("Mapping state dict")

    fairseq_roberta = f"{encoder_prefix}.fairseq_roberta."
    fairseq_sent_enc_prefix = fairseq_roberta + "model.encoder.sentence_encoder."

    hf_encoder_roberta = "camembert_model"
    hf_embeddings = f"{hf_encoder_roberta}.embeddings"

    camembert_config = CamembertConfig()
    state_dict = {
        f"{hf_embeddings}.position_ids": CamembertModel(
            CamembertConfig.from_pretrained("camembert-base")
        ).embeddings.position_ids,
        f"{hf_embeddings}.word_embeddings.weight": model_dict[
            fairseq_sent_enc_prefix + "embed_tokens.weight"
        ],
        f"{hf_embeddings}.position_embeddings.weight": model_dict[
            fairseq_sent_enc_prefix + "embed_positions.weight"
        ],
        f"{hf_embeddings}.token_type_embeddings.weight": torch.zeros(
            (1, camembert_config.hidden_size)
        ),
        f"{hf_embeddings}.LayerNorm.weight": model_dict[
            fairseq_sent_enc_prefix + "layernorm_embedding.weight"
        ],
        f"{hf_embeddings}.LayerNorm.bias": model_dict[
            fairseq_sent_enc_prefix + "layernorm_embedding.bias"
        ],
    }

    for i in range(camembert_config.num_hidden_layers):
        # Encoder: start of layer
        fairseq_layer_prefix = fairseq_sent_enc_prefix + f"layers.{i}."
        roberta_layer = f"{hf_encoder_roberta}.encoder.layer.{i}."

        self_attn = roberta_layer + "attention.self."
        state_dict[self_attn + "query.weight"] = model_dict[
            fairseq_layer_prefix + "self_attn.q_proj.weight"
        ]
        state_dict[self_attn + "query.bias"] = model_dict[
            fairseq_layer_prefix + "self_attn.q_proj.bias"
        ]
        state_dict[self_attn + "key.weight"] = model_dict[
            fairseq_layer_prefix + "self_attn.k_proj.weight"
        ]
        state_dict[self_attn + "key.bias"] = model_dict[
            fairseq_layer_prefix + "self_attn.k_proj.bias"
        ]
        state_dict[self_attn + "value.weight"] = model_dict[
            fairseq_layer_prefix + "self_attn.v_proj.weight"
        ]
        state_dict[self_attn + "value.bias"] = model_dict[
            fairseq_layer_prefix + "self_attn.v_proj.bias"
        ]

        # self-attention output
        self_output = roberta_layer + "attention.output."
        state_dict[self_output + "dense.weight"] = model_dict[
            fairseq_layer_prefix + "self_attn.out_proj.weight"
        ]
        state_dict[self_output + "dense.bias"] = model_dict[
            fairseq_layer_prefix + "self_attn.out_proj.bias"
        ]
        state_dict[self_output + "LayerNorm.weight"] = model_dict[
            fairseq_layer_prefix + "self_attn_layer_norm.weight"
        ]
        state_dict[self_output + "LayerNorm.bias"] = model_dict[
            fairseq_layer_prefix + "self_attn_layer_norm.bias"
        ]

        # intermediate
        intermediate = roberta_layer + "intermediate."
        state_dict[intermediate + "dense.weight"] = model_dict[
            fairseq_layer_prefix + "fc1.weight"
        ]
        state_dict[intermediate + "dense.bias"] = model_dict[
            fairseq_layer_prefix + "fc1.bias"
        ]

        # output
        roberta_output = roberta_layer + "output."
        state_dict[roberta_output + "dense.weight"] = model_dict[
            fairseq_layer_prefix + "fc2.weight"
        ]
        state_dict[roberta_output + "dense.bias"] = model_dict[
            fairseq_layer_prefix + "fc2.bias"
        ]
        state_dict[roberta_output + "LayerNorm.weight"] = model_dict[
            fairseq_layer_prefix + "final_layer_norm.weight"
        ]
        state_dict[roberta_output + "LayerNorm.bias"] = model_dict[
            fairseq_layer_prefix + "final_layer_norm.bias"
        ]

    return state_dict


class DPRState:
    """
    Base class of an encoder state
    """

    def __init__(self, src_file: Path):
        self.src_file = src_file

    def load_dpr_model(self):
        raise NotImplementedError

    @staticmethod
    def from_type(comp_type: str, *args, **kwargs) -> "DPRState":
        if comp_type.startswith("c"):
            return DPRContextEncoderState(*args, **kwargs)
        if comp_type.startswith("q"):
            return DPRQuestionEncoderState(*args, **kwargs)
        else:
            raise ValueError(
                "Component type must be either 'ctx_encoder', 'question_encoder' or 'reader'."
            )


class DPRContextEncoderState(DPRState):
    """
    Class to convert the checkpoint of the context encoder.
    """

    def load_dpr_model(self):
        model = DPRContextEncoder(
            DPRConfig(**CamembertConfig.get_config_dict("camembert-base")[0])
        )
        logger.info(f"Loading DPR context encoder from {self.src_file}")

        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.ctx_encoder, "ctx_model"
        state_dict = convert_bi_encoder_fairseq_to_pytorch(
            saved_state.model_dict, prefix
        )
        encoder.load_state_dict(state_dict)
        return model


class DPRQuestionEncoderState(DPRState):
    """
    Class to convert the checkpoint of the question encoder.
    """

    def load_dpr_model(self):
        model = DPRQuestionEncoder(
            DPRConfig(**CamembertConfig.get_config_dict("camembert-base")[0])
        )
        logger.info(f"Loading DPR question encoder from {self.src_file}")
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = model.question_encoder, "question_model"
        state_dict = convert_bi_encoder_fairseq_to_pytorch(
            saved_state.model_dict, prefix
        )
        encoder.load_state_dict(state_dict)
        return model


def convert(src_file: Path, dest_dir: Path) -> None:
    """
    Convert the checkpoints of each encoder from src_file to dest_dir.
    """

    logger.info(
        f"Starting convertion with source {src_file} and destination {dest_dir}"
    )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    for comp_type in TYPE:
        dpr_state = DPRState.from_type(comp_type, src_file=src_file)
        model = dpr_state.load_dpr_model()
        type_dest_dir = Path(os.path.join(dest_dir, comp_type))
        model.save_pretrained(type_dest_dir)
        model.from_pretrained(type_dest_dir)

    logger.info("Convertion done")


def main():
    parser = argparse.ArgumentParser(
        description="Script to convert a original roberta-based dpr ckpt to hf"
    )

    parser.add_argument(
        "--src",
        type=str,
        help="Path to the original dpr checkpoint file.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Path to the output checkpoint directory.",
    )
    args = parser.parse_args()

    src_file = Path(args.src)
    dest_dir = f"converted-{src_file.name}" if args.dest is None else args.dest
    dest_dir = Path(dest_dir)

    assert src_file.exists()

    convert(src_file, dest_dir)


if __name__ == "__main__":
    main()
