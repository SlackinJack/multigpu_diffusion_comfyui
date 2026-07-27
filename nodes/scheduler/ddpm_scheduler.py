from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


class DDPMScheduler:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "num_train_timesteps":      NUM_TRAIN_TIMESTEPS,
            "beta_start":               ("FLOAT", { "default": 0.00010, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "beta_end":                 ("FLOAT", { "default": 0.02000, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "beta_schedule":            (["default", "linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"], { "default": "default" }),
            # "trained_betas"
            "variance_type":            (["default", "fixed_small", "fixed_small_log", "fixed_large", "fixed_large_log", "learned", "learned_range"], { "default": "default" }),
            "clip_sample":              TRILEAN_WITH_DEFAULT,
            "clip_sample_range":        ("FLOAT", { "default": 1.00000, "min": INT_MIN, "max": INT_MAX, "step": 0.00001 }),
            "prediction_type":          (["default", "epsilon", "sample", "v_prediction"], { "default": "default" }),
            "thresholding":             TRILEAN_WITH_DEFAULT,
            "dynamic_thresholding_ratio": ("FLOAT", { "default": 0.99500, "min": INT_MIN, "max": INT_MAX, "step": 0.00001 }),
            "sample_max_value":         ("FLOAT", { "default": 1.00000, "min": INT_MIN, "max": INT_MAX, "step": 0.00001 }),
            "timestep_spacing":         (["default", "leading", "linspace", "trailing"], { "default": "default" }),
            "steps_offset":             STEPS_OFFSET,
            "rescale_betas_zero_snr":   TRILEAN_WITH_DEFAULT,
        },
        "optional": {
            "timesteps":                TIMESTEPS,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER, "get", ROOT_CATEGORY_SCHEDULERS
    def get(self, **kwargs):
        scheduler_config = { "scheduler": "DDPM" }
        for k, v in kwargs.items():
            if k in ["num_train_timesteps", "beta_start", "beta_end", "clip_sample_range", "dynamic_thresholding_ratio", "sample_max_value", "steps_offset", "timesteps"]:
                scheduler_config[k] = v
            elif k in ["beta_schedule", "variance_type", "prediction_type", "timestep_spacing"]:
                if v != "default": scheduler_config[k] = v
            elif trilean(v) != None:
                scheduler_config[k] = trilean(v)
        return (scheduler_config,)
