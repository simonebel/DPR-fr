from transformers.models.roberta.modeling_roberta import RobertaEncoder
from transformers import CamembertConfig

class CamembertEncoder( RobertaEncoder):
    """
    This class overrides [`RobertaEncoder`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = CamembertConfig

