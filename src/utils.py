# ---------------------------------------------------------
# Author: Dr Pantelis Georgiades
#         Computation-based Science and Technology Resarch
#         Centre (CaSToRC) - The Cyprus Institute
# License: MIT
# ---------------------------------------------------------

import os
import json
from pathlib import Path
from typing import Union, Dict

# ---------------------------------------------------------

def load_configs(config: Union[str, Dict, Path]) -> Dict:
    """
    Load configuration from multiple sources
    
    :param config: dict configuration or path to json configuration
    :return: dictionary configuration
    """
    try:
        if config is None:
            raise ValueError("config should not be empty")
        if isinstance(config, Dict):
            return config
        if isinstance(config, str) or isinstance(config, Path):
            if not os.path.isfile(str(config)):
                return ValueError(
                    "configuration path [{0}] is not valid".format(
                        str(config)
                    ))
            with open(str(config), "r") as f:
                return json.load(f)
        raise ValueError("don't know how to handle config [{0}]".format(config))
    except Exception as e:
        raise ValueError(f"failed to load [{config}]")
