from comfy.utils import ProgressBar
from torchvision.transforms import ToPILImage, ToTensor


from .data_types import *
from .nodes_general import *
from ..modules.host_manager import *
from ..multigpu_diffusion.modules.utils import *


BALANCED_CONFIGS = {
}


BALANCED_CONFIGS_OPTIONAL = {
    
}


class BalancedConfig:
    @classmethod
    def INPUT_TYPES(s): return { "required": BALANCED_CONFIGS, "optional": BALANCED_CONFIGS_OPTIONAL, }
    RETURN_TYPES, FUNCTION, CATEGORY = BACKEND_CONFIG, "get_config", ROOT_CATEGORY_CONFIG
    def get_config(self, **kwargs): return (kwargs,)
