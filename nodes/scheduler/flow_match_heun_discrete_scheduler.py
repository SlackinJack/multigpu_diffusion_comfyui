from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


class FlowMatchHeunDiscreteScheduler:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "num_train_timesteps":      NUM_TRAIN_TIMESTEPS,
            "shift":                    ("FLOAT", { "default": 1.00000, "min": 0.00001, "step": 0.00001 }),
        },
    }
    RETURN_TYPES, FUNCTION, CATEGORY = FM_SCHEDULER, "get", ROOT_CATEGORY_SCHEDULERS
    def get(self, **kwargs):
        scheduler_config = { "scheduler": "FM Heun" }
        for k, v in kwargs.items():
            if k in ["num_train_timesteps", "shift"]:
                scheduler_config[k] = v
            # elif k in [""]:
            #     if v != "default": scheduler_config[k] = v
            # elif trilean(v) != None:
            #     scheduler_config[k] = trilean(v)
        return (scheduler_config,)
