from .data_types import *
from .nodes_host import get_current_manager
from ..multigpu_diffusion.modules.utils import *


class SchedulerSelector:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "scheduler": (["ddim", "ddpm", "deis", "dpm_2", "dpm_2_a", "dpm_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "euler", "euler_a", "heun", "ipndm", "lms", "pndm", "tcd", "unipc"], { "default": "ddim" }),
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs): return (kwargs,)


class AdvancedSchedulerSelector:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "scheduler":                (["ddim", "ddpm", "deis", "dpm_2", "dpm_2_a", "dpm_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde", "euler", "euler_a", "heun", "ipndm", "lms", "pndm", "tcd", "unipc"], { "default": "ddim" }),
            "timestep_spacing":         (["default", "leading", "linspace", "trailing"], { "default": "default" }),
            "beta_schedule":            (["default", "linear", "scaled_linear", "squaredcos_cap_v2"], { "default": "default" }),
            "beta_start":               ("FLOAT", { "default": 0.00010, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "beta_end":                 ("FLOAT", { "default": 0.02000, "min": 0.00000, "max": 1.00000, "step": 0.00001 }),
            "use_karras_sigmas":        TRILEAN_WITH_DEFAULT,
            "rescale_betas_zero_snr":   TRILEAN_WITH_DEFAULT,
            "use_exponential_sigmas":   TRILEAN_WITH_DEFAULT,
            "use_beta_sigmas":          TRILEAN_WITH_DEFAULT,

        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = SCHEDULER, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs):
        scheduler_config = {}
        for k, v in kwargs.items():
            if k in ["scheduler", "beta_start", "beta_end"]:
                scheduler_config[k] = v
            elif k in ["timestep_spacing", "beta_schedule"]:
                if v != "default": scheduler_config[k] = v
            elif trilean(v) != None:
                scheduler_config[k] = trilean(v)
        return (scheduler_config,)


class AdvancedFMSchedulerSelector:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "shift":                    ("FLOAT", { "default": 1.00000, "min": 0.00000, "step": 0.00001 }),
            "use_dynamic_shifting":     TRILEAN_WITH_DEFAULT,
            "base_shift":               ("FLOAT", { "default": 0.50000, "min": 0.00000, "step": 0.00001 }),
            "max_shift":                ("FLOAT", { "default": 1.15000, "min": 0.00000, "step": 0.00001 }),
            "base_image_seq_len":       ("INT", { "default": 256, "min": INT_MIN, "max": INT_MAX }),
            "max_image_seq_len":        ("INT", { "default": 4096, "min": INT_MIN, "max": INT_MAX }),
            "invert_sigmas":            TRILEAN_WITH_DEFAULT,
            "shift_terminal":           ("FLOAT", { "default": 0.00000, "min": 0.00000, "step": 0.00001 }),
            "use_karras_sigmas":        TRILEAN_WITH_DEFAULT,
            "use_exponential_sigmas":   TRILEAN_WITH_DEFAULT,
            "use_beta_sigmas":          TRILEAN_WITH_DEFAULT,
            "time_shift_type":          (["exponential", "linear"], { "default": "exponential" }),
            "stochastic_sampling":      TRILEAN_WITH_DEFAULT,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = FM_EULER_SCHEDULER, "get", ROOT_CATEGORY_GENERAL
    def get(self, **kwargs):
        scheduler_config = {"scheduler": "fm_euler"}
        for k, v in kwargs.items():
            if k in ["shift", "base_shift", "max_shift", "base_image_seq_len", "max_image_seq_len", "shift_terminal", "time_shift_type"]:
                scheduler_config[k] = v
            elif trilean(v) != None:
                scheduler_config[k] = trilean(v)
        return (scheduler_config,)
