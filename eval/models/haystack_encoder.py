import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import torch
import transformers
from haystack.errors import ModelingError
from haystack.modeling.model.language_model import (
    HUGGINGFACE_TO_HAYSTACK,
    LanguageModel,
    _guess_language,
    capitalize_model_type,
    get_language_model_class,
)
from torch.serialization import default_restore_location
from transformers import AutoConfig, AutoModel, PreTrainedModel


class DPREncoder(LanguageModel):
    """
    A DPREncoder model that wraps Hugging Face's implementation.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Union[Path, str],
        model_type: str,
        language: str = None,
        n_added_tokens: int = 0,
        use_auth_token: Optional[Union[str, bool]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load a pretrained model by supplying one of the following:
        * The name of a remote model on s3 (for example, "facebook/dpr-question_encoder-single-nq-base").
        * A local path of a model trained using transformers (for example, "some_dir/huggingface_model").
        * A local path of a model trained using Haystack (for example, "some_dir/haystack_model").
        :param pretrained_model_name_or_path: The path of the base pretrained language model whose weights are used to initialize DPRQuestionEncoder.
        :param model_type: the type of model (see `HUGGINGFACE_TO_HAYSTACK`)
        :param language: the model's language. If not given, it will be inferred. Defaults to english.
        :param n_added_tokens: unused for `DPREncoder`
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param model_kwargs: any kwarg to pass to the model at init
        """
        super().__init__(model_type=model_type)
        self.role = "question" if "question" in model_type.lower() else "context"
        self._encoder = None
        model_classname = f"DPR{self.role.capitalize()}Encoder"
        try:
            model_class: Type[PreTrainedModel] = getattr(transformers, model_classname)
        except AttributeError as e:
            raise ModelingError(f"Model class of type '{model_classname}' not found.")

        haystack_lm_config = (
            Path(pretrained_model_name_or_path) / "language_model_config.json"
        )
        if os.path.exists(haystack_lm_config):
            logger.info("Path Exist")
            self._init_model_haystack_style(
                haystack_lm_config=haystack_lm_config,
                model_name_or_path=pretrained_model_name_or_path,
                model_class=model_class,
                model_kwargs=model_kwargs or {},
                use_auth_token=use_auth_token,
            )
        else:
            self._init_model_transformers_style(
                model_name_or_path=pretrained_model_name_or_path,
                model_class=model_class,
                model_kwargs=model_kwargs or {},
                use_auth_token=use_auth_token,
                language=language,
            )

    def _init_model_haystack_style(
        self,
        haystack_lm_config: Path,
        model_name_or_path: Union[str, Path],
        model_class: Type[PreTrainedModel],
        model_kwargs: Dict[str, Any],
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Init a Haystack-style DPR model.
        :param haystack_lm_config: path to the language model config file
        :param model_name_or_path: name or path of the model to load
        :param model_class: The HuggingFace model class name
        :param model_kwargs: any kwarg to pass to the model at init
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        original_model_config = AutoConfig.from_pretrained(haystack_lm_config)
        haystack_lm_model = Path(model_name_or_path) / "language_model.bin"

        original_model_type = original_model_config.model_type
        if original_model_type and "dpr" in original_model_type.lower():
            dpr_config = transformers.DPRConfig.from_pretrained(
                haystack_lm_config, use_auth_token=use_auth_token
            )
            self.model = model_class.from_pretrained(
                haystack_lm_model,
                config=dpr_config,
                use_auth_token=use_auth_token,
                **model_kwargs,
            )

        else:
            self.model = self._init_model_through_config(
                model_config=original_model_config,
                model_class=model_class,
                model_kwargs=model_kwargs,
            )
            original_model_type = capitalize_model_type(original_model_type)
            language_model_class = get_language_model_class(original_model_type)
            if not language_model_class:
                raise ValueError(
                    f"The type of model supplied ({model_name_or_path} , "
                    f"({original_model_type}) is not supported by Haystack. "
                    f"Supported model categories are: {', '.join(HUGGINGFACE_TO_HAYSTACK.keys())}"
                )
            # Instantiate the class for this model
            self.model.base_model.bert_model = language_model_class(
                pretrained_model_name_or_path=model_name_or_path,
                model_type=original_model_type,
                use_auth_token=use_auth_token,
                **model_kwargs,
            ).model

        self.language = self.model.config.language

    def _init_model_transformers_style(
        self,
        model_name_or_path: Union[str, Path],
        model_class: Type[PreTrainedModel],
        model_kwargs: Dict[str, Any],
        use_auth_token: Optional[Union[str, bool]] = None,
        language: Optional[str] = None,
    ):
        """
        Init a Transformers-style DPR model.
        :param model_name_or_path: name or path of the model to load
        :param model_class: The HuggingFace model class name
        :param model_kwargs: any kwarg to pass to the model at init
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param language: the model's language. If not given, it will be inferred. Defaults to english.
        """
        original_model_config = AutoConfig.from_pretrained(
            model_name_or_path, use_auth_token=use_auth_token
        )
        if "dpr" in original_model_config.model_type.lower():

            try:
                files_in_model_path = os.listdir(model_name_or_path)
                model_file = [
                    file for file in files_in_model_path if "pytorch_model.bin" in file
                ]
                model_file = Path(os.path.join(model_name_or_path, model_file[0]))
                state_dict = torch.load(
                    model_file,
                    map_location=lambda s, l: default_restore_location(s, "cpu"),
                )
                first_layer = list(state_dict.keys())[0]
                is_camembert = (
                    first_layer.split(".")[0]
                    if "camembert" in list(state_dict.keys())[0]
                    else False
                )

            except:
                raise ModelingError(f"Ckpt in '{model_name_or_path}' not found.")

            if is_camembert:
                from dpr_modeling_hf.dpr import DPRContextEncoder, DPRQuestionEncoder

                model_map = {
                    "ctx_encoder": DPRContextEncoder,
                    "question_encoder": DPRQuestionEncoder,
                }

                self.model = model_map[is_camembert].from_pretrained(
                    str(model_name_or_path)
                )

                language = "french"

            else:
                # "pretrained dpr model": load existing pretrained DPRQuestionEncoder model
                self.model = model_class.from_pretrained(
                    str(model_name_or_path),
                    use_auth_token=use_auth_token,
                    **model_kwargs,
                )
        else:
            # "from scratch": load weights from different architecture (e.g. bert) into DPRQuestionEncoder
            # but keep config values from original architecture
            # TODO test for architectures other than BERT, e.g. Electra
            self.model = self._init_model_through_config(
                model_config=original_model_config,
                model_class=model_class,
                model_kwargs=model_kwargs,
            )
            self.model.base_model.bert_model = AutoModel.from_pretrained(
                str(model_name_or_path),
                use_auth_token=use_auth_token,
                **vars(original_model_config),
            )
        self.language = language or _guess_language(str(model_name_or_path))

    def _init_model_through_config(
        self,
        model_config: AutoConfig,
        model_class: Type[PreTrainedModel],
        model_kwargs: Optional[Dict[str, Any]],
    ):
        """
        Init a DPR model using a config object.
        """
        if model_config.model_type.lower() != "bert":
            logger.warning(
                f"Using a model of type '{model_config.model_type}' which might be incompatible with DPR encoders. "
                f"Only Bert-based encoders are supported. They need input_ids, token_type_ids, attention_mask as input tensors."
            )
        config_dict = vars(model_config)
        if model_kwargs:
            config_dict.update(model_kwargs)
        return model_class(config=transformers.DPRConfig(**config_dict))

    @property
    def encoder(self):
        if not self._encoder:
            self._encoder = (
                self.model.question_encoder
                if self.role == "question"
                else self.model.ctx_encoder
            )
        return self._encoder

    def save_config(self, save_dir: Union[Path, str]) -> None:
        """
        Save the configuration of the language model in Haystack format.
        :param save_dir: the path to save the model at
        """
        # For DPR models, transformers overwrites the model_type with the one set in DPRConfig
        # Therefore, we copy the model_type from the model config to DPRConfig
        setattr(transformers.DPRConfig, "model_type", self.model.config.model_type)
        super().save_config(save_dir=save_dir)

    def save(
        self, save_dir: Union[str, Path], state_dict: Optional[Dict[Any, Any]] = None
    ) -> None:
        """
        Save the model `state_dict` and its configuration file so that it can be loaded again.
        :param save_dir: The directory in which the model should be saved.
        :param state_dict: A dictionary containing the whole state of the module including names of layers.
                           By default, the unchanged state dictionary of the module is used.
        """
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model itself

        if "dpr" not in self.model.config.model_type.lower():
            prefix = "question" if self.role == "question" else "ctx"

            state_dict = model_to_save.state_dict()
            if state_dict:
                for key in list(
                    state_dict.keys()
                ):  # list() here performs a copy and allows editing the dict
                    new_key = key

                    if key.startswith(f"{prefix}_encoder.bert_model.model."):
                        new_key = key.split("_encoder.bert_model.model.", 1)[1]

                    elif key.startswith(f"{prefix}_encoder.bert_model."):
                        new_key = key.split("_encoder.bert_model.", 1)[1]

                    state_dict[new_key] = state_dict.pop(key)

        super().save(save_dir=save_dir, state_dict=state_dict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: Optional[torch.Tensor],
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: bool = True,
    ):
        """
        Perform the forward pass of the DPR encoder model.
        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, number_of_hard_negative, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, number_of_hard_negative_passages, max_seq_len].
        :param attention_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size,  number_of_hard_negative_passages, max_seq_len].
        :param output_hidden_states: whether to add the hidden states along with the pooled output
        :param output_attentions: unused
        :return: Embeddings for each token in the input sequence.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.encoder.config.output_hidden_states
        )

        model_output = self.model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            return_dict=return_dict,
        )

        if output_hidden_states:
            return model_output.pooler_output, model_output.hidden_states
        return model_output.pooler_output, None
