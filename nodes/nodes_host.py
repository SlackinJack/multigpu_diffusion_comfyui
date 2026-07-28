import os
import time


from comfy.utils import ProgressBar


from .data_types import *
from ..modules.host_manager import HostManager
from ..multigpu_diffusion.modules.utils import *


hm = HostManager()


def get_current_manager():
    global hm
    return hm


class CreateHost:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "port": PORT,
            "master_port": MASTER_PORT,
            "backend": BACKEND,
            "cuda_visible_devices": ("STRING", { "default": "", "multiline": False }),
            "transformers_version": (["4", "5"], { "default": "4" }),
            "s33d": SEED,
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = HOST, "create_host", ROOT_CATEGORY_GENERAL
    def create_host(self, **kwargs):
        kwargs.pop("s33d")
        kwargs["cuda_visible_devices"] = kwargs["cuda_visible_devices"].replace(" ", "")
        global hm
        pbar = ProgressBar(100)
        host = hm.launch_host(kwargs, pbar=pbar)
        return (host,)


class CloseHost:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "host": HOST,
            "obj": ("*",),
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = ("*",), "destroy_host", ROOT_CATEGORY_GENERAL
    def destroy_host(self, host, obj):
        global hm
        hm.close_host_process(host, "Closed by node")
        return (obj,)


class ApplyPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "host": HOST,
                "backend_config": BACKEND_CONFIG,
                "pipeline_type": SUPPORTED_MODEL_LIST,
                "variant": VARIANT,
                "checkpoint": MODEL,
                "vae_fp16": BOOLEAN_DEFAULT_FALSE,
                "enable_vae_tiling": BOOLEAN_DEFAULT_FALSE,
                "enable_vae_slicing": BOOLEAN_DEFAULT_FALSE,
                "enable_attention_slicing": BOOLEAN_DEFAULT_FALSE,
                "xformers_efficient": BOOLEAN_DEFAULT_FALSE,
                "sd_fuse_qkv_projections": BOOLEAN_DEFAULT_FALSE,
            },
            "optional": {
                "lora": LORA,
                "transformer": MODEL,
                "vae": MODEL,
                "control_net": MODEL,
                "ip_adapter": MODEL,
                "text_encoder": MODEL,
                "text_encoder_2": MODEL,
                "text_encoder_3": MODEL,
                # "motion_module": MODEL,
                # "motion_adapter": MODEL,
                # "motion_adapter_lora": MOTION_ADAPTER_LORA,
                "compile_config": COMPILE_CONFIG,
                "quantization_config": QUANT_CONFIG,
                "torch_config": TORCH_CONFIG,
                "group_offload_config": GROUP_OFFLOAD_CONFIG,
                "attn_backend_config": ATTN_BACKEND_CONFIG,
            },
        }
    RETURN_TYPES, FUNCTION, CATEGORY = HOST, "apply_pipeline", ROOT_CATEGORY_CONFIG
    def apply_pipeline(self, **kwargs):
        global hm
        host = kwargs.pop("host")
        data = {}
        for k,v in kwargs.items():
            if k not in ["checkpoint", "transformer", "vae", "control_net", "ip_adapter", "motion_module", "motion_adapter", "text_encoder", "text_encoder_2", "text_encoder_3"]:
                data[k] = v
                continue
            else:
                if v.get("checkpoint") is not None:
                    data[k] = { "checkpoint": os.path.join(get_models_dir(), v["checkpoint"]) }
                    continue
                else:
                    data[k] = { "model": os.path.join(get_models_dir(), v["model"]), "config": os.path.join(get_models_dir(), v["config"]) }
                    continue

        pbar = ProgressBar(100)
        response = hm.post_to_address(host, "apply", data, pbar=pbar)
        if response is None or response.status_code != 200:
            hm.close_host_process(host, "Failed to initialize pipeline", with_assert="Failed to initialize pipeline.\n\nCheck console for details.")
        return (host,)


class SleepHost:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "host": HOST,
            "time": ("INT", { "min": 0, "max": INT_MAX, "step": 1 }),
            "obj": ("*",),
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "*",), "sleep_host", ROOT_CATEGORY_CONFIG
    def sleep_host(self, host, time, obj):
        global hm
        pbar = ProgressBar(100)
        data = {"sleep": True, "time": time}
        response = hm.post_to_address(host, "sleep", data, pbar=pbar)
        # TODO: maybe do something with response
        return (host, obj,)


class OffloadPipeline:
    @classmethod
    def INPUT_TYPES(s): return {
        "required": {
            "host": HOST,
            "obj": ("*",),
        }
    }
    RETURN_TYPES, FUNCTION, CATEGORY = ("MD_HOST", "*",), "offload_pipeline", ROOT_CATEGORY_CONFIG
    def offload_pipeline(self, host, wait_for_offload, obj):
        global hm
        pbar = ProgressBar(100)
        response = hm.get_from_address(host, "offload", pbar=pbar)
        # TODO: maybe do something with response
        return (host, obj,)
