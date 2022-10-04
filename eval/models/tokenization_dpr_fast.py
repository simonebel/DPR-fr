from transformers import CamembertTokenizerFast
from transformers.utils import (
    TensorType,
    add_end_docstrings,
    add_start_docstrings,
    logging,
)

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": (
            "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/vocab.txt"
        ),
        "facebook/dpr-ctx_encoder-multiset-base": (
            "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "facebook/dpr-ctx_encoder-single-nq-base": (
            "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/tokenizer.json"
        ),
        "facebook/dpr-ctx_encoder-multiset-base": (
            "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/tokenizer.json"
        ),
    },
}


class DPRContextEncoderTokenizerFast(CamembertTokenizerFast):
    r"""
    Construct a "fast" DPRContextEncoder tokenizer (backed by HuggingFace's *tokenizers* library).
    [`DPRContextEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.
    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = DPRContextEncoderTokenizer


class DPRQuestionEncoderTokenizerFast(CamembertTokenizerFast):
    r"""
    Constructs a "fast" DPRQuestionEncoder tokenizer (backed by HuggingFace's *tokenizers* library).
    [`DPRQuestionEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.
    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = DPRQuestionEncoderTokenizer
