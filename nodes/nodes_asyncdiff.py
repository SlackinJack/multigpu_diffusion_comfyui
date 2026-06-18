from comfy.utils import ProgressBar
from torchvision.transforms import ToPILImage, ToTensor


from .data_types import *
from .nodes_general import *
from ..modules.host_manager import *
from ..multigpu_diffusion.modules.utils import *


ASYNCDIFF_CONFIGS = {
    "model_n":      ("INT", { "default": 2, "min": 1 }),
    "stride":       ("INT", { "default": 1, "min": 1 }),
    "synced_steps": ("INT", { "default": 10, "min": 0 }),
    "time_shift":   ("INT", { "default": 0, "min": 0 }),
    "shifted_steps": ("INT", { "default": 0, "min": 0 }),
    "cached_step":  ("INT", { "default": 1, "min": 1 }),
}


ASYNCDIFF_CONFIGS_OPTIONAL = {

}


class AsyncDiffConfig:
    @classmethod
    def INPUT_TYPES(s): return { "required": ASYNCDIFF_CONFIGS, "optional": ASYNCDIFF_CONFIGS_OPTIONAL, }
    RETURN_TYPES, FUNCTION, CATEGORY = BACKEND_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)
