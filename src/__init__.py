# ---------------------------------------------------------
# Author: Dr Pantelis Georgiades
#         Computation-based Science and Technology Resarch
#         Centre (CaSToRC) - The Cyprus Institute
# License: MIT
# ---------------------------------------------------------

import os
import pathlib

# ---------------------------------------------------------

from .load_data import load_dataset, im_view
from .utils import load_configs

# ---------------------------------------------------------
# Configs
# Get the path
current_path = pathlib.Path(__file__).parent.resolve()

configs_dir = current_path / "configs"

configs = [
    load_configs(str(c))
    for c in configs_dir.glob("*.json")
]