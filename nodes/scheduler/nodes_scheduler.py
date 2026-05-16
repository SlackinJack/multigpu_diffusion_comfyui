from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


class FlowMatchScheduler:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": { "scheduler": (["FM Euler", "FM Heun"], {"default": "FM Euler"}) },
        "optional": { "config": SCHEDULER_CONFIG }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = FM_SCHEDULER, "get", ROOT_CATEGORY_SCHEDULERS
    def get(self, scheduler, config=None):
        scheduler_config = { "scheduler": scheduler }
        if config is not None:
            for k, v in config.items():
                scheduler_config[k] = v
        return (scheduler_config,)


class FlowMatchSchedulerConfig:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "num_train_timesteps":      ("INT", { "default": 1000, "min": 0 }),
            "shift":                    ("FLOAT", { "default": 1.00000, "min": 0.00001, "step": 0.00001 }),
            "use_dynamic_shifting":     TRILEAN_WITH_DEFAULT,
            "base_shift":               ("FLOAT", { "default": 0.50000, "min": 0.00000, "step": 0.00001 }),
            "max_shift":                ("FLOAT", { "default": 1.15000, "min": 0.00000, "step": 0.00001 }),
            "base_image_seq_len":       ("INT", { "default": 256, "min": INT_MIN, "max": INT_MAX }),
            "max_image_seq_len":        ("INT", { "default": 4096, "min": INT_MIN, "max": INT_MAX }),
            "invert_sigmas":            TRILEAN_WITH_DEFAULT,
            "shift_terminal":           ("FLOAT", { "default": 0.00000, "min": 0.00000, "max": INT_MAX, "step": 0.00001 }),
            "use_karras_sigmas":        TRILEAN_WITH_DEFAULT,
            "use_exponential_sigmas":   TRILEAN_WITH_DEFAULT,
            "use_beta_sigmas":          TRILEAN_WITH_DEFAULT,
            "time_shift_type":          (["default", "exponential", "linear"], { "default": "default" }),
            "stochastic_sampling":      TRILEAN_WITH_DEFAULT,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER_CONFIG, "get", ROOT_CATEGORY_SCHEDULERS
    def get(self, **kwargs):
        scheduler_config = {}
        for k, v in kwargs.items():
            if k in ["num_train_timesteps", "shift", "base_shift", "max_shift", "base_image_seq_len", "max_image_seq_len", "shift_terminal"]:
                scheduler_config[k] = v
            elif k in ["time_shift_type"]:
                if v != "default": scheduler_config[k] = v
            elif trilean(v) != None:
                scheduler_config[k] = trilean(v)
        return (scheduler_config,)
