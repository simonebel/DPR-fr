from transformers import CamembertConfig
from transformers.models.roberta.modeling_roberta import RobertaEncoder


class CamembertEncoder(RobertaEncoder):
    """
    This class overrides [`RobertaEncoder`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = CamembertConfig
