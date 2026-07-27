from ..data_types import *
from ..nodes_host import get_current_manager
from ...multigpu_diffusion.modules.utils import *


class DPMSolverMultistepScheduler:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "num_train_timesteps":      NUM_TRAIN_TIMESTEPS,
            "beta_start":               ("FLOAT", { "default": 0.00010, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "beta_end":                 ("FLOAT", { "default": 0.02000, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "beta_schedule":            (["default", "linear", "scaled_linear", "squaredcos_cap_v2"], { "default": "default" }),
            # "trained_betas"
            "solver_order":             ("INT", { "default": 2, "min": 1, "max": 3 }),
            "prediction_type":          (["default", "epsilon", "sample", "v_prediction", "flow_prediction"], { "default": "default" }),
            "thresholding":             TRILEAN_WITH_DEFAULT,
            "dynamic_thresholding_ratio": ("FLOAT", { "default": 0.99500, "min": INT_MIN, "max": INT_MAX, "step": 0.00001 }),
            "sample_max_value":         ("FLOAT", { "default": 1.00000, "min": INT_MIN, "max": INT_MAX, "step": 0.00001 }),
            "algorithm_type":           (["default", "dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"], { "default": "default" }),
            "solver_type":              (["default", "midpoint", "heun"], { "default": "default" }),
            "lower_order_final":        TRILEAN_WITH_DEFAULT,
            "euler_at_final":           TRILEAN_WITH_DEFAULT,
            "use_karras_sigmas":        TRILEAN_WITH_DEFAULT,
            "use_exponential_sigmas":   TRILEAN_WITH_DEFAULT,
            "use_beta_sigmas":          TRILEAN_WITH_DEFAULT,
            "use_lu_sigmas":            TRILEAN_WITH_DEFAULT,
            "use_flow_sigmas":          TRILEAN_WITH_DEFAULT,
            "flow_shift":               ("FLOAT", { "default": 1.00000, "min": INT_MIN, "max": INT_MAX, "step": 0.00001 }),
            "final_sigmas_type":        (["default", "zero", "sigma_min"], { "default": "default" }),
            # "lambda_min_clipped"
            "variance_type":            (["default", "learned", "learned_range"], { "default": "default" }),
            "timestep_spacing":         (["default", "leading", "linspace", "trailing"], { "default": "default" }),
            "steps_offset":             STEPS_OFFSET,
            "rescale_betas_zero_snr":   TRILEAN_WITH_DEFAULT,
            "use_dynamic_shifting":     TRILEAN_WITH_DEFAULT,
            "time_shift_type":          (["default", "exponential"], { "default": "default" }),
        },
        "optional": {
            "timesteps":                TIMESTEPS,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER, "get", ROOT_CATEGORY_SCHEDULERS
    def get(self, **kwargs):
        scheduler_config = { "scheduler": "DPM++ 2M" }
        for k, v in kwargs.items():
            if k in ["num_train_timesteps", "beta_start", "beta_end", "solver_order", "dynamic_thresholding_ratio", "sample_max_value", "flow_shift", "steps_offset", "timesteps"]:
                scheduler_config[k] = v
            elif k in ["beta_schedule", "prediction_type", "algorithm_type", "solver_type", "final_sigmas_type", "variance_type", "timestep_spacing", "time_shift_type"]:
                if v != "default": scheduler_config[k] = v
            elif trilean(v) != None:
                scheduler_config[k] = trilean(v)
        return (scheduler_config,)
