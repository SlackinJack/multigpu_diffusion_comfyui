from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


class EulerDiscreteScheduler:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "num_train_timesteps":      NUM_TRAIN_TIMESTEPS,
            "beta_start":               ("FLOAT", { "default": 0.00010, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "beta_end":                 ("FLOAT", { "default": 0.02000, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "beta_schedule":            (["default", "linear", "scaled_linear", "squaredcos_cap_v2"], { "default": "default" }),
            # "trained_betas"
            "prediction_type":          (["default", "epsilon", "sample", "v_prediction"], { "default": "default" }),
            "interpolation_type":       (["default", "linear", "log_linear"], { "default": "default" }),
            "use_karras_sigmas":        TRILEAN_WITH_DEFAULT,
            "use_exponential_sigmas":   TRILEAN_WITH_DEFAULT,
            "use_beta_sigmas":          TRILEAN_WITH_DEFAULT,
            # "sigma_min"
            # "sigma_max"
            "timestep_spacing":         (["default", "leading", "linspace", "trailing"], { "default": "default" }),
            "timestep_type":            (["default", "discrete", "continuous"], { "default": "default" }),
            "steps_offset":             STEPS_OFFSET,
            "rescale_betas_zero_snr":   TRILEAN_WITH_DEFAULT,
            "final_sigma_type":         (["default", "zero", "sigma_min"], { "default": "default" }),
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER, "get", ROOT_CATEGORY_SCHEDULERS
    def get(self, **kwargs):
        scheduler_config = { "scheduler": "Euler" }
        for k, v in kwargs.items():
            if k in ["num_train_timesteps", "beta_start", "beta_end", "steps_offset"]:
                scheduler_config[k] = v
            elif k in ["beta_schedule", "prediction_type", "interpolation_type", "timestep_spacing", "timestep_type", "final_sigma_type"]:
                if v != "default": scheduler_config[k] = v
            elif trilean(v) != None:
                scheduler_config[k] = trilean(v)
        return (scheduler_config,)
