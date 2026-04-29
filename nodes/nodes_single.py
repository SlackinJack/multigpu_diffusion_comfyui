from comfy.utils import ProgressBar
from torchvision.transforms import ToPILImage, ToTensor


from .data_types import *
from .nodes_general import *
from ..modules.host_manager import *
from ..multigpu_diffusion.modules.utils import *


SINGLE_CONFIGS = {
    "device_id":            ("INT", { "default": 0, "min": 0 }),
    "deep_cache":           BOOLEAN_DEFAULT_FALSE,
    "deep_cache_interval":  ("INT", { "default": 3, "min": 1 }),
    "deep_cache_id":        ("INT", { "default": 0, "min": 0 }),
}


SINGLE_CONFIGS_OPTIONAL = {

}


class SingleConfig:
    @classmethod
    def INPUT_TYPES(s): return { "required": SINGLE_CONFIGS, "optional": SINGLE_CONFIGS_OPTIONAL, }
    RETURN_TYPES, FUNCTION, CATEGORY = BACKEND_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)
